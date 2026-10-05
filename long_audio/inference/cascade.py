# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-B: text-only segmentation over Stage-A descriptions.

Reads ``<model>.descriptions.jsonl`` (produced by ``describe.py``),
groups consecutive descriptions into ``window_minutes``-long windows
(default 10 min for parity with the direct-audio chunk runner), and
asks a ``TextLLMAdapter`` to emit segmentation JSON per window. Output
is written in the SAME shape as ``chunk_runner.run_inference`` — same
``chunk_*.json`` + ``summary.json`` layout — so
``scripts/eval_segmentation.py --dataset {ego4d,egolife,sins}`` consumes cascade
runs identically.

The downstream LLM is swappable: any class implementing
``TextLLMAdapter`` (Gemini text, OpenAI, Claude, local, …) plugs in.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import json_repair

from long_audio.inference.chunk_runner import ChunkRecord, RunSummary
from long_audio.inference.describe import load_descriptions
from long_audio.inference.models.base import Decoder, TextLLMAdapter
from long_audio.inference.prompt import (
    ChunkContext,
    ContextMode,
    render_prompt_from_descriptions,
)
from long_audio.inference.schema import (
    TimeUnit,
    make_segmentation_schema,
    parse_segments,
)


def _window_descriptions(
    descriptions: list[dict],
    window_seconds: float,
) -> list[list[dict]]:
    """Group descriptions into consecutive windows of ``window_seconds``.

    Each description is assigned to the window containing its
    ``start_s``. Windows are right-open so a description at the boundary
    goes to the next window. Empty windows are dropped (Stage-B has
    nothing to segment in a span with no captions).
    """
    windows: dict[int, list[dict]] = {}
    for d in descriptions:
        idx = int(float(d["start_s"]) // window_seconds)
        windows.setdefault(idx, []).append(d)
    return [windows[k] for k in sorted(windows)]


def _next_prev_label(new_segs_abs: list[dict]) -> str | None:
    if not new_segs_abs:
        return None
    return new_segs_abs[-1]["label"]


def run_cascade_segmentation(
    descriptions_path: Path | str,
    text_model: TextLLMAdapter,
    output_dir: Path | str,
    labels: tuple[str, ...],
    *,
    label_hints: dict[str, str] | None = None,
    window_minutes: float = 10.0,
    decoder: Decoder = "structured",
    max_windows: int | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    context_mode: ContextMode = "none",
    time_unit: TimeUnit = "second",
    audio_path: Path | str | None = None,
    audio_duration_s: float | None = None,
    dataset_name: str = "dataset",
    t0_iso: str = "1970-01-01T00:00:00Z",
    with_description: bool = False,
) -> RunSummary:
    """Run Stage-B segmentation over a JSONL of Stage-A descriptions.

    Args:
        descriptions_path: ``<model>.descriptions.jsonl`` produced by
            ``describe.describe_audio_chunks``.
        text_model: a loaded ``TextLLMAdapter``. ``load()`` is called if needed.
        output_dir: where to write ``chunk_*.json`` + ``summary.json``.
        labels: dataset label tuple.
        window_minutes: Stage-B window size; default 10 min for parity
            with the direct-audio chunk runner.
        decoder: ``"structured"`` (schema-guided) or ``"freeform"``.
        max_windows: cap for smoke tests.
        audio_path / audio_duration_s: passed through to the run summary
            for parity with chunk_runner output (eval scripts read
            ``audio_duration_s`` to set the timeline length).

    Returns the ``RunSummary``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptions = load_descriptions(descriptions_path)
    if not descriptions:
        raise ValueError(f"No descriptions found in {descriptions_path}")

    text_model.load()
    if decoder == "structured" and not text_model.supports_structured():
        raise RuntimeError(
            f"TextLLMAdapter {text_model.name!r} does not support structured "
            "decoding. Pass decoder='freeform' to run free-text "
            "(output will go through json_repair; no chunk-wide-guess fallback)."
        )

    schema = make_segmentation_schema(
        labels=labels, time_unit=time_unit, with_description=with_description,
    )
    window_seconds = window_minutes * 60.0
    windows = _window_descriptions(descriptions, window_seconds)
    if max_windows is not None:
        windows = windows[:max_windows]
    n_windows = len(windows)

    # Derive timeline end: max description end_s, unless caller overrides.
    timeline_end_s = max(float(d["end_s"]) for d in descriptions)
    if audio_duration_s is None:
        audio_duration_s = timeline_end_s

    prev_label: str | None = None
    chunk_files: list[str] = []
    total_segments = 0
    total_inference_s = 0.0
    chunks_with_segments = 0
    chunks_with_zero_segments = 0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_run_start = time.time()

    for i, win_descs in enumerate(windows):
        chunk_id = f"chunk_{i:05d}"
        win_start_s = min(float(d["start_s"]) for d in win_descs)
        win_end_s = max(float(d["end_s"]) for d in win_descs)
        # Snap window start to the nominal window grid so prompts line up
        # cleanly even if Stage-A skipped descriptions at the boundary.
        nominal_start_s = i * window_seconds
        chunk_start_s = min(win_start_s, nominal_start_s) if i > 0 else win_start_s
        chunk_end_s = win_end_s

        ctx = ChunkContext(
            chunk_index=i, n_chunks=n_windows,
            chunk_start_wallclock_s=chunk_start_s,
            chunk_end_wallclock_s=chunk_end_s,
            t0_iso=t0_iso, dataset_name=dataset_name,
        )
        prompt = render_prompt_from_descriptions(
            ctx, win_descs, labels,
            label_hints=label_hints,
            context_mode=context_mode, prev_label=prev_label,
            time_unit=time_unit,
            with_description=with_description,
        )

        out = text_model.generate(
            prompt,
            schema=schema, decoder=decoder,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        total_inference_s += out.latency_s or 0.0

        raw_json = out.raw_json if out.raw_json is not None else json_repair.loads(out.raw_text)
        segs_chunk_local = parse_segments(
            raw_json, chunk_start_s=0.0, labels=labels, time_unit=time_unit,
        )

        segs_abs = []
        for s in segs_chunk_local:
            abs_seg = {
                "label": s["label"],
                "start": s["start"] + chunk_start_s,
                "end": s["end"] + chunk_start_s,
            }
            if "description" in s:
                abs_seg["description"] = s["description"]
            segs_abs.append(abs_seg)
        total_segments += len(segs_abs)
        if segs_chunk_local:
            chunks_with_segments += 1
        else:
            chunks_with_zero_segments += 1

        record = ChunkRecord(
            chunk_id=chunk_id, chunk_index=i, n_chunks=n_windows,
            chunk_start_s=chunk_start_s, chunk_end_s=chunk_end_s,
            decoder=decoder,
            n_segments=len(segs_abs), segments_abs=segs_abs,
            raw_text=out.raw_text, raw_json=raw_json,
            prompt=prompt, latency_s=out.latency_s, metadata=out.metadata,
            thinking_trace=out.thinking_trace,
        )
        chunk_path = output_dir / f"{chunk_id}.json"
        chunk_path.write_text(json.dumps(asdict(record), indent=2))
        chunk_files.append(str(chunk_path))

        prev_label = _next_prev_label(segs_abs)

    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_s = time.time() - t_run_start
    summary = RunSummary(
        audio_path=str(audio_path) if audio_path else "",
        output_dir=str(output_dir),
        model_name=text_model.name,
        decoder=decoder,
        chunk_minutes=window_minutes,
        n_chunks=n_windows,
        n_chunks_completed=len(chunk_files),
        n_segments_total=total_segments,
        audio_duration_s=audio_duration_s,
        total_inference_s=total_inference_s,
        rt_factor=audio_duration_s / wall_s if wall_s else 0.0,
        started_at=started_at, ended_at=ended_at,
        config={
            "stage": "cascade_segment",
            "window_minutes": window_minutes,
            "decoder": decoder,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "dataset_name": dataset_name,
            "t0_iso": t0_iso,
            "context_mode": context_mode,
            "time_unit": time_unit,
            "descriptions_path": str(descriptions_path),
        },
        chunk_files=chunk_files,
        parse_stats={
            "chunks_with_segments": chunks_with_segments,
            "chunks_with_zero_segments": chunks_with_zero_segments,
            "timestamp_readability": (
                chunks_with_segments / max(1, chunks_with_segments + chunks_with_zero_segments)
            ),
        },
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2)
    )
    return summary
