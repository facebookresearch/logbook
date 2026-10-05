"""Stage-A: per-chunk audio captioning for the cascade pipeline.

Slices the input audio into ``chunk_seconds``-long windows and asks an
audio ``ModelAdapter`` to produce one short description per window. The
canonical prompt is fixed: ``"Describe the audio clip in one sentence."``
— stays well within the Qwen-Omni / AF-Next prompt distribution.
Single-sentence is the intentional design: 60 descriptions per 10-min
Stage-B window × N models is already a lot of context for the
downstream LLM.

OUTPUT LAYOUT (one file per audio per model, hence ``one-file-per-audio``):
    <output_dir>/<model_name>.descriptions.jsonl
        One JSON object per line:
            {"chunk_idx": int, "start_s": float, "end_s": float,
             "description": str, "latency_s": float, "metadata": dict}
    <output_dir>/<model_name>.describe_summary.json
        Run-level metadata + counts.

The runner is model-agnostic — it only talks to ``ModelAdapter`` /
``ModelOutput``, so swapping in af3-hf, af-next, qwen2.5-omni,
qwen3-omni, or the fake adapter is a constructor change.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile

from long_audio.inference.chunking import compute_n_chunks
from long_audio.inference.models.base import ModelAdapter


DESCRIBE_PROMPT = "Describe the audio clip in one sentence."


@dataclass
class DescribeSummary:
    """Aggregate run output written to ``<model_name>.describe_summary.json``."""

    audio_path: str
    output_dir: str
    model_name: str
    chunk_seconds: float
    prompt: str
    n_chunks: int
    n_chunks_completed: int
    audio_duration_s: float
    total_inference_s: float
    rt_factor: float
    started_at: str
    ended_at: str
    descriptions_path: str
    config: dict = field(default_factory=dict)


def describe_audio_chunks(
    audio_path: Path | str,
    model: ModelAdapter,
    output_dir: Path | str,
    *,
    chunk_seconds: float = 10.0,
    batch_size: int = 64,
    max_chunks: int | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    sample_rate_target: int = 16000,
    audio_offset_s: float = 0.0,
    audio_window_s: float | None = None,
) -> DescribeSummary:
    """Run Stage-A on one mono audio file.

    Args:
        audio_path: mono FLAC/WAV at ``sample_rate_target``.
        model: a loaded ``ModelAdapter``. ``model.load()`` is called if needed.
        output_dir: where to write the per-model JSONL + describe_summary.
        chunk_seconds: caption window size. Default 10s.
        batch_size: number of chunks per ``generate_batch`` call. With
            10-s chunks Stage A produces ~60×-100× the chunks of a
            10-min E2E run, so batching is the difference between a
            multi-hour and a few-minute pass. Adapters that don't
            implement true batching fall back to a loop via
            ``ModelAdapter.generate_batch`` default. Default 64 (clean
            power-of-2 above 60, since the parity argument "10-min audio
            works, so 60×10-s should too" places the natural ceiling
            near there). For vLLM adapters, the LLM constructor's
            ``max_num_seqs`` must be >= this for real GPU concurrency
            (otherwise the API accepts the batch but vLLM serializes).
        max_chunks: cap chunk count for smoke tests.
        max_new_tokens: per-call generation cap. 2048 to match the rest
            of the inference pipeline — one consistent ceiling everywhere
            beats per-call tuning. Caption text is still short (one
            sentence per the prompt); the cap just removes a foot-gun
            for unusually verbose models.
        temperature: 0.0 = greedy.
        sample_rate_target: error out on mismatch (avoid silent resampling).

    Returns the ``DescribeSummary`` written next to the JSONL.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            f"duration {file_duration_s:.2f}s — nothing to describe."
        )
    if audio_window_s is None:
        usable_frames = info.frames - offset_sample
    else:
        usable_frames = min(
            int(round(audio_window_s * sr)), info.frames - offset_sample
        )
    duration_s = usable_frames / sr
    chunk_samples = int(round(chunk_seconds * sr))
    n_chunks_full = compute_n_chunks(usable_frames, chunk_samples, sr)
    n_chunks = min(n_chunks_full, max_chunks) if max_chunks else n_chunks_full

    model.load()

    descriptions_path = output_dir / f"{model.name}.descriptions.jsonl"
    total_inference_s = 0.0
    n_completed = 0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_run_start = time.time()

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    with descriptions_path.open("w") as fp:
        for batch_lo in range(0, n_chunks, batch_size):
            batch_hi = min(batch_lo + batch_size, n_chunks)
            audios: list = []
            spans: list[tuple[int, float, float]] = []
            for i in range(batch_lo, batch_hi):
                rel_start = i * chunk_samples
                rel_stop = min(rel_start + chunk_samples, usable_frames)
                start_sample = offset_sample + rel_start
                stop_sample = offset_sample + rel_stop
                audio, _ = soundfile.read(
                    str(audio_path), start=start_sample, stop=stop_sample,
                    always_2d=False, dtype="float32",
                )
                audios.append(audio)
                # Per-chunk timestamps are *relative to manifest t=0*
                # (= file_t=0 - audio_offset_s) so eval coords don't need
                # to know the offset.
                spans.append((i, rel_start / sr, rel_stop / sr))

            prompts = [DESCRIBE_PROMPT] * len(audios)
            outs = model.generate_batch(
                audios, sr, prompts,
                schema=None, decoder="freeform",
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            for (idx, chunk_start_s, chunk_end_s), out in zip(spans, outs):
                total_inference_s += out.latency_s or 0.0
                record = {
                    "chunk_idx": idx,
                    "start_s": chunk_start_s,
                    "end_s": chunk_end_s,
                    "description": (out.raw_text or "").strip(),
                    "latency_s": out.latency_s,
                    "metadata": out.metadata,
                }
                fp.write(json.dumps(record) + "\n")
                n_completed += 1
            fp.flush()
            # Per-batch progress emit — Stage A produces 60-100x more chunks
            # than E2E so per-chunk would be too noisy. Surfaces to SLURM
            # stdout so a stalled batch is visible before the job times out.
            n_batches_total = (n_chunks + batch_size - 1) // batch_size
            n_batches_done = batch_lo // batch_size + 1
            print(
                f"[describe] {n_completed}/{n_chunks} chunks done "
                f"(batch {n_batches_done}/{n_batches_total}) "
                f"cum_inference={total_inference_s:.0f}s "
                f"wall={time.time() - t_run_start:.0f}s",
                flush=True,
            )

    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_s = time.time() - t_run_start
    audio_processed_s = min(duration_s, n_chunks * chunk_seconds)
    summary = DescribeSummary(
        audio_path=str(audio_path),
        output_dir=str(output_dir),
        model_name=model.name,
        chunk_seconds=chunk_seconds,
        prompt=DESCRIBE_PROMPT,
        n_chunks=n_chunks,
        n_chunks_completed=n_completed,
        audio_duration_s=audio_processed_s,
        total_inference_s=total_inference_s,
        rt_factor=audio_processed_s / wall_s if wall_s else 0.0,
        started_at=started_at,
        ended_at=ended_at,
        descriptions_path=str(descriptions_path),
        config={
            "chunk_seconds": chunk_seconds,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "sample_rate": sr,
            "audio_duration_s": duration_s,
            "audio_offset_s": audio_offset_s,
            "file_duration_s": file_duration_s,
        },
    )
    (output_dir / f"{model.name}.describe_summary.json").write_text(
        json.dumps(asdict(summary), indent=2)
    )
    return summary


def load_descriptions(jsonl_path: Path | str) -> list[dict]:
    """Read a ``<model>.descriptions.jsonl`` file into memory."""
    out = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
