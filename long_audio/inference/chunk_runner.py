"""End-to-end chunk runner for long-audio segmentation inference.

Slices a recording into contiguous chunks, calls a ``ModelAdapter`` per
chunk (threading the previous predicted label as context), and writes
per-chunk JSON plus a stitched ``summary.json``. Model-agnostic — swap
FakeAdapter for tests, real vLLM/HF adapters for production.

Entry point: :func:`run_inference`. Helpers: :func:`plan_chunks` (pure
planning), :func:`make_chunk_record` (per-chunk post-processing),
:func:`build_run_summary` (final aggregation).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import json_repair
import numpy as np
import soundfile

from long_audio.inference.chunking import compute_n_chunks
from long_audio.inference.models.base import Decoder, ModelAdapter
from long_audio.inference.prompt import (
    ChunkContext,
    ContextMode,
    render_prompt,
    render_simple_classification_prompt,
)
from long_audio.inference.schema import (
    TimeUnit,
    make_segmentation_schema,
    parse_segments,
)


@dataclass
class ChunkRecord:
    """Per-chunk artifact written to <output_dir>/<chunk_id>.json."""

    chunk_id: str
    chunk_index: int
    n_chunks: int
    chunk_start_s: float
    chunk_end_s: float
    decoder: Decoder
    n_segments: int
    segments_abs: list[dict]
    raw_text: str
    raw_json: dict | None
    prompt: str
    latency_s: float | None
    metadata: dict = field(default_factory=dict)
    # Extended-reasoning trace when the model was run with thinking mode
    # (Gemini `thinking_budget>0`, Qwen 3 / Gemma `enable_thinking=True`).
    # None when thinking was off or the adapter did not surface a trace.
    thinking_trace: str | None = None


@dataclass
class RunSummary:
    """Aggregate run output written to <output_dir>/summary.json.

    Per N2 (DECISION.md §9): predictions are stored RAW at chunk
    granularity. Stitching / gap-filling / coalescing now happens at
    eval time in ``scripts/eval_run*.py``. ``stitched_segments`` is
    intentionally NOT a field here — old run dirs whose summary.json
    has that key will fail loudly when eval scripts load them, which
    is the desired signal to re-run inference (SINS is cheap; Ego4D
    was a one-shot we already have v4 of).
    """

    audio_path: str
    output_dir: str
    model_name: str
    decoder: Decoder
    chunk_minutes: float
    n_chunks: int
    n_chunks_completed: int
    n_segments_total: int
    audio_duration_s: float
    total_inference_s: float
    rt_factor: float
    started_at: str
    ended_at: str
    config: dict
    chunk_files: list[str]
    parse_stats: dict = field(default_factory=dict)
    eval_scores: dict | None = None


# ---------------------------------------------------------------------------
# Planning (pure — no model, no I/O writes)
# ---------------------------------------------------------------------------


@dataclass
class ChunkPlan:
    """One chunk's planning info: WHAT to infer on, WITHOUT the model call.

    The sync loop iterates these one at a time.
    """

    chunk_id: str
    chunk_index: int
    n_chunks: int
    chunk_start_s: float    # relative to manifest t=0
    chunk_end_s: float
    start_sample: int       # in file (with manifest audio_offset baked in)
    stop_sample: int        # in file
    ctx: ChunkContext       # for prompt rendering


@dataclass
class AudioPlan:
    """Everything ``run_inference`` needs to know about one file.

    Produced by ``plan_chunks`` and consumed by the sync loop.
    """

    audio_path: Path
    sr: int
    file_duration_s: float
    offset_sample: int
    usable_frames: int
    duration_s: float
    chunk_samples: int
    n_chunks: int           # after ``max_chunks`` cap, if any
    chunks: list[ChunkPlan]


def plan_chunks(
    audio_path: Path | str,
    *,
    chunk_minutes: float,
    max_chunks: int | None = None,
    sample_rate_target: int = 16000,
    dataset_name: str = "dataset",
    t0_iso: str = "1970-01-01T00:00:00Z",
    audio_offset_s: float = 0.0,
    audio_window_s: float | None = None,
) -> AudioPlan:
    """Compute the chunk grid for ``audio_path`` without touching the model.

    Runs the same audio-validity checks as ``run_inference`` (sample
    rate, mono, offset past EOF) so planning fails loudly at the same
    boundary as the sync loop.
    """
    audio_path = Path(audio_path)
    info = soundfile.info(str(audio_path))
    if info.samplerate != sample_rate_target:
        raise ValueError(
            f"{audio_path} sample rate is {info.samplerate}, "
            f"expected {sample_rate_target}. Resample first."
        )
    if info.channels != 1:
        raise ValueError(
            f"{audio_path} has {info.channels} channels, expected mono."
        )

    sr = info.samplerate
    file_duration_s = info.frames / sr
    offset_sample = int(round(audio_offset_s * sr))
    if offset_sample >= info.frames:
        raise ValueError(
            f"{audio_path}: audio_offset_s={audio_offset_s:.2f}s is past file "
            f"duration {file_duration_s:.2f}s — nothing to infer on."
        )
    if audio_window_s is None:
        usable_frames = info.frames - offset_sample
    else:
        usable_frames = min(
            int(round(audio_window_s * sr)), info.frames - offset_sample
        )
    duration_s = usable_frames / sr
    chunk_samples = int(round(chunk_minutes * 60 * sr))
    n_chunks_full = compute_n_chunks(usable_frames, chunk_samples, sr)
    n_chunks = min(n_chunks_full, max_chunks) if max_chunks else n_chunks_full

    chunks: list[ChunkPlan] = []
    for i in range(n_chunks):
        rel_start = i * chunk_samples
        rel_stop = min(rel_start + chunk_samples, usable_frames)
        start_sample = offset_sample + rel_start
        stop_sample = offset_sample + rel_stop
        chunk_start_s = rel_start / sr
        chunk_end_s = rel_stop / sr
        chunks.append(ChunkPlan(
            chunk_id=f"chunk_{i:05d}",
            chunk_index=i,
            n_chunks=n_chunks,
            chunk_start_s=chunk_start_s,
            chunk_end_s=chunk_end_s,
            start_sample=start_sample,
            stop_sample=stop_sample,
            ctx=ChunkContext(
                chunk_index=i, n_chunks=n_chunks,
                chunk_start_wallclock_s=chunk_start_s,
                chunk_end_wallclock_s=chunk_end_s,
                t0_iso=t0_iso, dataset_name=dataset_name,
            ),
        ))

    return AudioPlan(
        audio_path=audio_path,
        sr=sr,
        file_duration_s=file_duration_s,
        offset_sample=offset_sample,
        usable_frames=usable_frames,
        duration_s=duration_s,
        chunk_samples=chunk_samples,
        n_chunks=n_chunks,
        chunks=chunks,
    )


def read_chunk_audio(
    audio_path: Path | str,
    plan: ChunkPlan,
    *,
    dtype: str = "float32",
) -> np.ndarray:
    """Read the audio slice for one chunk. Returns a 1-D array."""
    audio, _ = soundfile.read(
        str(audio_path), start=plan.start_sample, stop=plan.stop_sample,
        always_2d=False, dtype=dtype,
    )
    return audio


def render_chunk_prompt(
    plan: ChunkPlan,
    *,
    labels: tuple[str, ...],
    label_hints: dict[str, str] | None = None,
    prompt_mode: Literal["structured", "simple"] = "structured",
    context_mode: ContextMode = "none",
    prev_label: str | None = None,
    time_unit: TimeUnit = "second",
    with_description: bool = False,
) -> str:
    """Render one chunk's prompt for the sync loop."""
    if prompt_mode == "simple":
        return render_simple_classification_prompt(plan.ctx, labels)
    return render_prompt(
        plan.ctx, labels, label_hints=label_hints,
        context_mode=context_mode, prev_label=prev_label,
        time_unit=time_unit,
        with_description=with_description,
    )


# ---------------------------------------------------------------------------
# Response -> ChunkRecord
# ---------------------------------------------------------------------------


def make_chunk_record(
    plan: ChunkPlan,
    *,
    decoder: Decoder,
    prompt: str,
    raw_text: str,
    raw_json: dict | None,
    latency_s: float | None,
    metadata: dict,
    labels: tuple[str, ...],
    time_unit: TimeUnit = "second",
    thinking_trace: str | None = None,
) -> ChunkRecord:
    """Parse a single model response into a ``ChunkRecord`` with
    absolute-time segments.
    """
    # Structured adapters return raw_json directly; free-text adapters
    # go through json_repair, which tolerates fences, smart quotes,
    # single quotes, comments, and truncations. parse_segments drops
    # anything that isn't a {"segments": [...]} dict — output that
    # isn't shaped like the schema surfaces in chunks_with_zero_segments.
    raw_json = raw_json if raw_json is not None else json_repair.loads(raw_text)
    segs_chunk_local = parse_segments(
        raw_json, chunk_start_s=0.0, labels=labels, time_unit=time_unit,
    )

    # Convert to absolute coords. No gap-fill / no merge at this stage
    # — predictions are stored raw and stitched at eval time. Description
    # field (if emitted) is carried through; absent → key not present.
    segs_abs = []
    for s in segs_chunk_local:
        abs_seg = {
            "label": s["label"],
            "start": s["start"] + plan.chunk_start_s,
            "end": s["end"] + plan.chunk_start_s,
        }
        if "description" in s:
            abs_seg["description"] = s["description"]
        segs_abs.append(abs_seg)

    return ChunkRecord(
        chunk_id=plan.chunk_id, chunk_index=plan.chunk_index,
        n_chunks=plan.n_chunks,
        chunk_start_s=plan.chunk_start_s, chunk_end_s=plan.chunk_end_s,
        decoder=decoder,
        n_segments=len(segs_abs), segments_abs=segs_abs,
        raw_text=raw_text, raw_json=raw_json,
        prompt=prompt, latency_s=latency_s, metadata=metadata,
        thinking_trace=thinking_trace,
    )


def write_chunk_record(
    record: ChunkRecord,
    *,
    output_dir: Path,
    write_raw_text: bool = True,
) -> Path:
    """Persist a ``ChunkRecord`` to ``<output_dir>/<chunk_id>.json`` (and the
    raw text to ``<output_dir>/raw/<chunk_id>.txt`` if enabled). Returns the
    chunk JSON path so callers can accumulate ``chunk_files``.
    """
    chunk_path = output_dir / f"{record.chunk_id}.json"
    chunk_path.write_text(json.dumps(asdict(record), indent=2))
    if write_raw_text:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{record.chunk_id}.txt").write_text(record.raw_text)
    return chunk_path


def _next_prev_label(new_segs_abs: list[dict]) -> str | None:
    """The last predicted label from this chunk, to seed the next chunk's
    ``context_mode='prev'`` block. Returns None if no segments parsed."""
    if not new_segs_abs:
        return None
    return new_segs_abs[-1]["label"]


# ---------------------------------------------------------------------------
# Summary building (shared)
# ---------------------------------------------------------------------------


def build_run_summary(
    audio_plan: AudioPlan,
    records: list[ChunkRecord],
    *,
    model_name: str,
    decoder: Decoder,
    chunk_minutes: float,
    output_dir: Path,
    config: dict,
    started_at: str,
    ended_at: str,
    total_inference_s: float,
    wall_s: float,
) -> RunSummary:
    """Assemble a ``RunSummary`` from per-chunk records. Callers persist it
    via ``write_run_summary`` — kept separate so callers can inject
    ``eval_scores`` before writing (see ``scripts/run_e2e.py``).
    """
    chunk_files = [str(output_dir / f"{r.chunk_id}.json") for r in records]
    total_segments = sum(r.n_segments for r in records)
    chunks_with_segments = sum(1 for r in records if r.n_segments > 0)
    chunks_with_zero_segments = sum(1 for r in records if r.n_segments == 0)
    audio_processed_s = min(
        audio_plan.duration_s, audio_plan.n_chunks * chunk_minutes * 60,
    )
    return RunSummary(
        audio_path=str(audio_plan.audio_path),
        output_dir=str(output_dir),
        model_name=model_name, decoder=decoder, chunk_minutes=chunk_minutes,
        n_chunks=audio_plan.n_chunks, n_chunks_completed=len(records),
        n_segments_total=total_segments,
        audio_duration_s=audio_processed_s,
        total_inference_s=total_inference_s,
        rt_factor=audio_processed_s / wall_s if wall_s else 0.0,
        started_at=started_at, ended_at=ended_at,
        config=config,
        chunk_files=chunk_files,
        parse_stats={
            "chunks_with_segments": chunks_with_segments,
            "chunks_with_zero_segments": chunks_with_zero_segments,
            "timestamp_readability": (
                chunks_with_segments / max(1, chunks_with_segments + chunks_with_zero_segments)
            ),
        },
    )


def write_run_summary(summary: RunSummary, output_dir: Path) -> Path:
    """Persist a ``RunSummary`` to ``<output_dir>/summary.json``."""
    path = output_dir / "summary.json"
    path.write_text(json.dumps(asdict(summary), indent=2))
    return path


# ---------------------------------------------------------------------------
# Sync executor (per-file, per-chunk loop with ``prev_label`` threading)
# ---------------------------------------------------------------------------


def run_inference(
    audio_path: Path | str,
    model: ModelAdapter,
    output_dir: Path | str,
    labels: tuple[str, ...],
    *,
    label_hints: dict[str, str] | None = None,
    chunk_minutes: float = 10.0,
    decoder: Decoder = "structured",
    max_chunks: int | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    dataset_name: str = "dataset",
    t0_iso: str = "1970-01-01T00:00:00Z",
    sample_rate_target: int = 16000,
    write_raw_text: bool = True,
    prompt_mode: Literal["structured", "simple"] = "structured",
    context_mode: ContextMode = "none",
    time_unit: TimeUnit = "second",
    audio_offset_s: float = 0.0,
    audio_window_s: float | None = None,
    with_description: bool = False,
) -> RunSummary:
    """Run end-to-end inference on a single mono audio file.

    Args:
        audio_path: mono FLAC/WAV. Must already be downmixed.
        model: a loaded ``ModelAdapter``. ``model.load()`` is called if needed.
        output_dir: where to write per-chunk + summary JSON.
        chunk_minutes: chunk size; AF3 trained at 10, Qwen claims up to 600.
        decoder: ``"structured"`` (schema-guided) or ``"freeform"`` (text).
        max_chunks: cap chunk count (useful for smoke tests).
        max_new_tokens: per-call generation cap.
        temperature: 0.0 = greedy.
        dataset_name / t0_iso: passed to the prompt template.
        sample_rate_target: resample target if file SR mismatches. Currently
            we error out on mismatch to avoid silently mishandling.
        write_raw_text: store the model's raw text alongside JSON for debug.
        audio_offset_s: seek into the raw FLAC at this many seconds before
            chunk 0 starts. Used for Ego4D where manifest t=0 is offset
            into the raw audio (per Stage 3's ``audio_offset_s`` field —
            mean ~55s, max ~1.7h). Defaults to 0 (read from start).
        audio_window_s: cap how many seconds of audio (starting from
            ``audio_offset_s``) to chunk. Used to clamp Ego4D inference to
            the manifest-derived ``duration`` so we don't infer past EOF.
            Defaults to None (use the FLAC's full duration past the offset).

    Returns the RunSummary written to ``<output_dir>/summary.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_plan = plan_chunks(
        audio_path,
        chunk_minutes=chunk_minutes,
        max_chunks=max_chunks,
        sample_rate_target=sample_rate_target,
        dataset_name=dataset_name,
        t0_iso=t0_iso,
        audio_offset_s=audio_offset_s,
        audio_window_s=audio_window_s,
    )
    sr = audio_plan.sr

    # Load the model once.
    model.load()
    if decoder == "structured" and not model.supports_structured():
        raise RuntimeError(
            f"Adapter {model.name!r} does not support structured decoding. "
            "Pass --decoder freeform to run it in free-text mode (output will "
            "go through json_repair; no chunk-wide-guess fallback)."
        )

    # Build the schema once with the requested label set + time_unit.
    schema = make_segmentation_schema(
        labels=labels, time_unit=time_unit, with_description=with_description,
    )

    prev_label: str | None = None
    records: list[ChunkRecord] = []
    total_inference_s = 0.0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_run_start = time.time()

    for plan in audio_plan.chunks:
        audio = read_chunk_audio(audio_plan.audio_path, plan)
        prompt = render_chunk_prompt(
            plan, labels=labels, label_hints=label_hints,
            prompt_mode=prompt_mode, context_mode=context_mode,
            prev_label=prev_label, time_unit=time_unit,
            with_description=with_description,
        )
        out = model.generate(
            audio, sr, prompt,
            schema=schema, decoder=decoder,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        total_inference_s += out.latency_s or 0.0

        record = make_chunk_record(
            plan, decoder=decoder, prompt=prompt,
            raw_text=out.raw_text, raw_json=out.raw_json,
            latency_s=out.latency_s, metadata=out.metadata,
            labels=labels, time_unit=time_unit,
            thinking_trace=out.thinking_trace,
        )
        write_chunk_record(record, output_dir=output_dir, write_raw_text=write_raw_text)
        records.append(record)

        # Progress emit — surfaced to SLURM stdout so a stalled run is
        # visible before it times out. Includes per-chunk latency,
        # cumulative inference time, and wall time, so it's also a useful
        # signal for catching unexpected per-chunk slowdowns mid-run.
        print(
            f"[chunk_runner] {plan.chunk_index + 1}/{audio_plan.n_chunks} "
            f"{plan.chunk_id} latency={out.latency_s or 0.0:.1f}s "
            f"segs={record.n_segments} "
            f"cum_inference={total_inference_s:.0f}s "
            f"wall={time.time() - t_run_start:.0f}s",
            flush=True,
        )

        prev_label = _next_prev_label(record.segments_abs)

    # No stitching here — eval-time scripts read chunk_*.json and apply
    # gap-fill / merging policy per dataset (DECISION.md §9 N2).
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_s = time.time() - t_run_start
    summary = build_run_summary(
        audio_plan, records,
        model_name=model.name, decoder=decoder, chunk_minutes=chunk_minutes,
        output_dir=output_dir,
        config={
            "chunk_minutes": chunk_minutes, "decoder": decoder,
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "dataset_name": dataset_name, "t0_iso": t0_iso,
            "sample_rate": sr, "audio_duration_s": audio_plan.duration_s,
            "prompt_mode": prompt_mode, "context_mode": context_mode,
            "time_unit": time_unit,
            "audio_offset_s": audio_offset_s,
            "audio_window_s": audio_plan.duration_s,
            "file_duration_s": audio_plan.file_duration_s,
        },
        started_at=started_at, ended_at=ended_at,
        total_inference_s=total_inference_s,
        wall_s=wall_s,
    )
    write_run_summary(summary, output_dir)
    return summary
