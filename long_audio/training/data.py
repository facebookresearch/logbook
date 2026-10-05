# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Ego4D SFT training-data builder: manifest + audio -> chat-format examples.

Produces SFT examples for teaching an audio-LLM temporal activity
segmentation on Ego4D. Each example is one chunk of one video; prompt +
target rendering, chunking grid, and timestamp grammar all mirror
:mod:`long_audio.inference.chunk_runner` so the target distribution is
byte-compatible with inference output. Gold segments come from
midpoint-resolved narration slices (no same-label coalesce, so each
~5-min slice stays its own segment; longer chunks → more boundaries).
``Qwen3OmniSFTCollator`` handles mel-spectrogram encoding + label
masking for the training entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from long_audio.datasets.ego4d.dataset import Ego4DDataset
from long_audio.datasets.ego4d.prompt import render_ego4d_prompt
from long_audio.datasets.ego4d.schema import ATUS_LABELS
from long_audio.inference.chunking import compute_n_chunks
from long_audio.inference.prompt import ChunkContext, ContextMode, _fmt_mmss
from long_audio.inference.schema import TimeUnit, parse_segments
from long_audio.utils.events import midpoint_boundaries

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Gold-segment derivation (slices -> disjoint per-slice timeline).
# ---------------------------------------------------------------------------


def disjoint_slice_segments(slices: Sequence[dict], duration: float) -> list[dict]:
    """Convert sliding-window ``slices`` into a disjoint, gap-free timeline.

    Thin wrapper over :func:`long_audio.utils.events.midpoint_boundaries` (the
    shared midpoint-rule core, also used by Ego4D ``actions`` assembly). Unlike
    the ``actions`` path it keeps one segment per slice (no coalescing, to
    preserve the per-slice segmentation signal + descriptions) and carries the
    slice ``summary`` through as ``description``.

    Args:
        slices: per-pass slice dicts with ``start``, ``end``, ``event``,
            ``summary`` (post-OLMo annotated-manifest schema).
        duration: manifest-window length in seconds; the output covers
            ``[0, duration]`` contiguously.

    Returns:
        ``[{"start", "end", "event", "description"}, ...]`` sorted by start,
        one per input slice.

    Raises:
        ValueError: if a midpoint boundary produces a degenerate span. By the
            annotated-manifest contract (same slices produced the contiguous
            ``actions``) this should never fire.
    """
    return midpoint_boundaries(
        list(slices),
        float(duration),
        carry={"event": "event", "description": "summary"},
        snap_edges=True,
        name="disjoint_slice_segments",
    )


def _clip_shift(gold: Sequence[dict], c_start: float, c_end: float) -> list[dict]:
    """Clip gold segments to ``[c_start, c_end]`` and shift to chunk-relative.

    Segments with <=0 overlap are dropped; the remainder stays contiguous.
    """
    out: list[dict] = []
    for g in gold:
        s = max(float(g["start"]), c_start)
        e = min(float(g["end"]), c_end)
        if e - s <= 1e-6:
            continue
        out.append(
            {
                "start": s - c_start,
                "end": e - c_start,
                "event": g["event"],
                "description": g["description"],
            }
        )
    return out


def _fmt_ts(units: int, time_unit: TimeUnit) -> str:
    """Format a grid-unit boundary as the schema's timestamp string.

    ``units`` is seconds for ``second`` mode (MM:SS) and whole minutes for
    ``minute`` mode (bare integer string).
    """
    if time_unit == "minute":
        return str(int(units))
    return _fmt_mmss(float(units))


def quantize_and_format(
    rel_segs: Sequence[dict],
    chunk_dur_s: float,
    time_unit: TimeUnit,
    *,
    with_description: bool = True,
) -> list[dict]:
    """Quantize chunk-relative float segments onto the ``time_unit`` grid.

    Produces contiguous, gap-free, non-overlapping segments covering
    ``[0, total]`` on the grid (1 s for ``second``, 60 s for ``minute``),
    matching the grammar ``make_segmentation_schema`` enforces. Segments that
    round to zero length on the grid are absorbed into their neighbour (keeps
    contiguity; may drop a very short slice's description).

    Returns:
        ``[{"start": str, "end": str, "event": str[, "description": str]}, ...]``.
    """
    grid = 60.0 if time_unit == "minute" else 1.0
    total = max(1, int(round(chunk_dur_s / grid)))
    if not rel_segs:
        return []

    # Internal cut points (grid units), monotonic non-decreasing in [0, total].
    cuts = [0]
    for seg in rel_segs[:-1]:
        c = int(round(float(seg["end"]) / grid))
        c = min(max(c, cuts[-1]), total)
        cuts.append(c)
    cuts.append(total)

    spans: list[tuple[int, int, str, str]] = []
    for i, seg in enumerate(rel_segs):
        a, b = cuts[i], cuts[i + 1]
        if b <= a:  # collapsed on the grid — drop; neighbours stay contiguous
            continue
        spans.append((a, b, seg["event"], seg["description"]))

    if not spans:  # everything collapsed (e.g. sub-minute chunk) -> longest wins
        best = max(rel_segs, key=lambda s: float(s["end"]) - float(s["start"]))
        spans = [(0, total, best["event"], best["description"])]

    out: list[dict] = []
    for a, b, event, desc in spans:
        d: dict = {
            "start": _fmt_ts(a, time_unit),
            "end": _fmt_ts(b, time_unit),
            "event": event,
        }
        if with_description:
            d["description"] = desc
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# The training example + the PyTorch Dataset.
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    """One chunk-level SFT example (processor-independent).

    ``audio`` is a mono float32 array at ``SAMPLE_RATE`` (None when the dataset
    was built with ``load_audio=False`` — useful for format/count tests without
    the FLACs on disk). ``target_text`` is the exact JSON the model should emit;
    ``target_obj`` is its parsed form; ``gold_segments`` are the chunk-relative
    float segments before grid quantization (for debugging / eval alignment).
    """

    uid: str
    chunk_index: int
    n_chunks: int
    split: str
    chunk_start_s: float
    chunk_end_s: float
    prompt: str
    target_text: str
    target_obj: dict
    n_segments: int
    gold_segments: list[dict]
    time_unit: TimeUnit = "minute"
    audio: np.ndarray | None = None
    sample_rate: int = SAMPLE_RATE

    def conversation(self) -> list[dict]:
        """Qwen-Omni chat messages: user(audio+prompt) -> assistant(target).

        The audio object is the raw float32 array; the processor expands the
        ``<|AUDIO|>`` placeholder into mel-frame tokens at encode time.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": self.audio},
                    {"type": "text", "text": self.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.target_text}],
            },
        ]


class Ego4DSFTDataset(Dataset):
    """PyTorch Dataset of chunk-level Ego4D SFT examples.

    Builds a flat index of ``(uid, chunk_index)`` over the videos in ``split``
    (via :class:`Ego4DDataset` for split + filter handling), then materializes
    one :class:`SFTExample` per chunk on ``__getitem__``.

    Args:
        manifest_path: path to ``annotated_manifest.json``.
        split: ``"train"`` / ``"val"`` / ``"test"`` / ``None`` (all).
        chunk_minutes: chunk length; 5/10/20/30 supported (default 10).
        time_unit: ``"minute"`` (default, per the chunked-inference decision) or
            ``"second"``. Must match the inference/eval ``--time-unit``.
        pass_id: which narration pass supplies the gold timeline (default "1").
        with_description: emit a ``description`` per segment (default True).
        context_mode: prompt prior-context mode (default "none").
        load_audio: read the FLAC chunk into ``SFTExample.audio`` (default True).
            Set False for format/segment-count tests without audio on disk.
        min_events / min_duration_s / require_both_passes: filters forwarded to
            :class:`Ego4DDataset`. Defaults are training-lenient (keep more data)
            rather than the strict eval-set defaults.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = "train",
        chunk_minutes: float = 10.0,
        time_unit: TimeUnit = "minute",
        pass_id: str = "1",
        with_description: bool = True,
        context_mode: ContextMode = "none",
        load_audio: bool = True,
        min_events: int = 1,
        min_duration_s: float = 0.0,
        require_both_passes: bool = True,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        self.split = split
        self.chunk_minutes = float(chunk_minutes)
        self.time_unit: TimeUnit = time_unit
        self.pass_id = pass_id
        self.with_description = with_description
        self.context_mode: ContextMode = context_mode
        self.load_audio = load_audio
        self.sample_rate = sample_rate

        self._ds = Ego4DDataset(
            manifest_path=self.manifest_path,
            split=split,
            min_events=min_events,
            min_duration_s=min_duration_s,
            require_both_passes=require_both_passes,
            pass_filter=pass_id,
        )
        self.chunk_samples = int(round(self.chunk_minutes * 60 * self.sample_rate))
        self._index: list[tuple[int, int, int]] = []  # (video_idx, chunk_idx, n_chunks)
        self._build_index()

    # -- index -------------------------------------------------------------

    def _usable_frames(self, video: dict) -> int:
        """Frames the runner would chunk over: ``round(duration * sr)``.

        Matches ``run_inference``'s ``usable_frames`` (``audio_window_s`` =
        manifest ``duration``, which was patched to the true FLAC length) without
        opening the file, so the index builds fast and needs no audio on disk.
        """
        return int(round(float(video["duration"]) * self.sample_rate))

    def _build_index(self) -> None:
        self._index.clear()
        for vi, v in enumerate(self._ds.videos):
            usable = self._usable_frames(v)
            try:
                n_chunks = compute_n_chunks(usable, self.chunk_samples, self.sample_rate)
            except ValueError:
                # Whole video shorter than the min-inference floor — skip it
                # (the eval-set min_duration_s normally prevents this).
                continue
            for ci in range(n_chunks):
                self._index.append((vi, ci, n_chunks))

    def __len__(self) -> int:
        return len(self._index)

    # -- example materialization ------------------------------------------

    def _gold_for_video(self, video: dict) -> list[dict]:
        slices = video["passes"][self.pass_id]["slices"]
        return disjoint_slice_segments(slices, float(video["duration"]))

    def _read_chunk_audio(
        self, video: dict, rel_start: int, rel_stop: int
    ) -> np.ndarray:
        offset_sample = int(round(float(video["audio_offset_s"]) * self.sample_rate))
        audio, _ = soundfile.read(
            str(video["audio_path"]),
            start=offset_sample + rel_start,
            stop=offset_sample + rel_stop,
            always_2d=False,
            dtype="float32",
        )
        return audio

    def build_example(self, video_idx: int, chunk_idx: int, n_chunks: int) -> SFTExample:
        v = self._ds.videos[video_idx]
        usable = self._usable_frames(v)
        rel_start = chunk_idx * self.chunk_samples
        rel_stop = min(rel_start + self.chunk_samples, usable)
        chunk_start_s = rel_start / self.sample_rate
        chunk_end_s = rel_stop / self.sample_rate
        chunk_dur_s = chunk_end_s - chunk_start_s

        gold = self._gold_for_video(v)
        rel_segs = _clip_shift(gold, chunk_start_s, chunk_end_s)
        seg_objs = quantize_and_format(
            rel_segs, chunk_dur_s, self.time_unit,
            with_description=self.with_description,
        )
        target_obj = {"segments": seg_objs}
        target_text = json.dumps(target_obj, ensure_ascii=False)

        ctx = ChunkContext(
            chunk_index=chunk_idx,
            n_chunks=n_chunks,
            chunk_start_wallclock_s=chunk_start_s,
            chunk_end_wallclock_s=chunk_end_s,
            dataset_name="ego4d",
        )
        prompt = render_ego4d_prompt(
            ctx,
            context_mode=self.context_mode,
            time_unit=self.time_unit,
            with_description=self.with_description,
        )

        audio = None
        if self.load_audio:
            audio = self._read_chunk_audio(v, rel_start, rel_stop)

        return SFTExample(
            uid=v["uid"],
            chunk_index=chunk_idx,
            n_chunks=n_chunks,
            split=v.get("split", ""),
            chunk_start_s=chunk_start_s,
            chunk_end_s=chunk_end_s,
            prompt=prompt,
            target_text=target_text,
            target_obj=target_obj,
            n_segments=len(seg_objs),
            gold_segments=rel_segs,
            time_unit=self.time_unit,
            audio=audio,
            sample_rate=self.sample_rate,
        )

    def __getitem__(self, idx: int) -> SFTExample:
        video_idx, chunk_idx, n_chunks = self._index[idx]
        return self.build_example(video_idx, chunk_idx, n_chunks)

    def iter_examples(self) -> Iterator[SFTExample]:
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# Processor collator (Qwen3-Omni). Requires the real AutoProcessor; used by the
# training entry point, not by the format/count unit tests.
# ---------------------------------------------------------------------------


# Response-start marker in Qwen chat templates. Locating this in the tokenized
# full sequence lets us compute the prompt-vs-response mask from a SINGLE
# processor pass (audio expansion is applied to the placeholder token in the
# same call). The alternative — tokenizing the prompt separately and using its
# length — is BPE-boundary-unsafe: `\n{first char of JSON}` can merge into a
# different token than what appears in the prefix-only tokenization, off-by-one
# masking the response's first token. See training code review, finding #2.
QWEN_RESPONSE_MARKER = "<|im_start|>assistant\n"


def _locate_marker_ends(input_ids: "torch.Tensor", marker_ids: Sequence[int]) -> list[int]:
    """For each row, return the index just past the LAST occurrence of ``marker_ids``.

    Returns -1 for rows where the marker is not found — the caller should raise
    (missing marker = chat-template drift, unsafe to train on). Scans from the
    end because Qwen chat templates put the assistant-open marker last, so this
    is O(seq_len) in practice.
    """
    n_rows, n_cols = input_ids.shape
    m = len(marker_ids)
    if m == 0 or m > n_cols:
        return [-1] * n_rows
    marker = torch.as_tensor(marker_ids, dtype=input_ids.dtype, device=input_ids.device)
    ends: list[int] = []
    for row in range(n_rows):
        row_ids = input_ids[row]
        end = -1
        for start in range(n_cols - m, -1, -1):
            if torch.equal(row_ids[start : start + m], marker):
                end = start + m
                break
        ends.append(end)
    return ends


class Qwen3OmniSFTCollator:
    """Encode :class:`SFTExample` batches with a Qwen3-Omni processor.

    Uses ``processor.apply_chat_template`` + ``processor(...)`` to build
    ``input_ids`` / ``attention_mask`` / ``input_features`` /
    ``feature_attention_mask`` and a ``labels`` tensor that is ``-100`` on
    everything except the assistant-response tokens (so the audio placeholder,
    system, and user turns contribute no loss). See
    ``docs/research/qwen3_omni_hf_training.md`` for the submodule/processor
    ground truth.

    Loss-masking correctness notes (from code review):

      * The response span is located by scanning ``input_ids`` for the
        ``<|im_start|>assistant\\n`` marker id sequence (see
        :data:`QWEN_RESPONSE_MARKER`) — safer than tokenizing the prompt
        separately and using its length, which is BPE-boundary-unsafe.
      * Padding is masked via ``attention_mask == 0`` (not
        ``input_ids == pad_id``) so that any content token which happens to
        share the pad id — e.g. the response-terminating EOS when
        ``pad_token == eos_token`` — is *not* zeroed out. The constructor also
        fails fast if ``pad_token_id == eos_token_id`` upstream.

    ``processor`` is typically ``AutoProcessor.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct")`` (-> ``Qwen3OmniMoeProcessor``).
    """

    def __init__(self, processor: Any, sample_rate: int = SAMPLE_RATE):
        self.processor = processor
        self.sample_rate = sample_rate
        tokenizer = processor.tokenizer
        self._marker_ids: tuple[int, ...] = tuple(
            tokenizer.encode(QWEN_RESPONSE_MARKER, add_special_tokens=False)
        )
        if not self._marker_ids:
            raise RuntimeError(
                "Qwen3OmniSFTCollator: failed to tokenize response-start marker "
                f"{QWEN_RESPONSE_MARKER!r}; only Qwen-style chat templates supported."
            )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is not None and eos_id is not None and pad_id == eos_id:
            raise RuntimeError(
                f"Qwen3OmniSFTCollator: tokenizer pad_token_id ({pad_id}) equals "
                "eos_token_id — masking pads in labels would also zero out the "
                "response-terminating EOS and the model would never learn to stop. "
                "Add a distinct pad token before constructing the collator."
            )
        self._verified = False

    def __call__(self, examples: Sequence[SFTExample]) -> dict:
        full_texts: list[str] = []
        audios: list[np.ndarray] = []
        for ex in examples:
            if ex.audio is None:
                raise ValueError(
                    f"SFTExample(uid={ex.uid}, chunk={ex.chunk_index}) has no "
                    "audio; build the dataset with load_audio=True for training."
                )
            conv = ex.conversation()
            full_texts.append(
                self.processor.apply_chat_template(
                    conv, add_generation_prompt=False, tokenize=False
                )
            )
            audios.append(ex.audio)

        batch = self.processor(
            text=full_texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        marker_ends = _locate_marker_ends(input_ids, self._marker_ids)
        for row, m_end in enumerate(marker_ends):
            if m_end < 0:
                raise RuntimeError(
                    f"Qwen3OmniSFTCollator: response-start marker "
                    f"{QWEN_RESPONSE_MARKER!r} not found in row {row} — chat "
                    "template drift? first 32 ids: "
                    f"{input_ids[row, :32].tolist()}"
                )
            labels[row, :m_end] = -100

        # Pad-mask via attention_mask (see class docstring: safer than input_ids
        # == pad_id when pad shares the eos id).
        labels[attention_mask == 0] = -100
        batch["labels"] = labels

        # First-call cross-check: verify marker-locate agrees with the old
        # prefix-tokenization approach. Runs the audio processor twice on the
        # first batch only; the fast single-pass path is used from then on.
        if not self._verified:
            self._verify_marker_alignment(examples[0], int(marker_ends[0]))
            self._verified = True

        return batch

    def _verify_marker_alignment(self, example: SFTExample, marker_end: int) -> None:
        """Sanity: independently tokenized prefix length must equal ``marker_end``."""
        prefix_text = self.processor.apply_chat_template(
            example.conversation()[:-1], add_generation_prompt=True, tokenize=False
        )
        prefix = self.processor(
            text=[prefix_text],
            audio=[example.audio],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        prefix_len = int(prefix["attention_mask"].sum(dim=1).item())
        if prefix_len != marker_end:
            raise RuntimeError(
                f"Qwen3OmniSFTCollator: marker-locate produced end={marker_end} "
                f"but prefix-tokenization length={prefix_len}. Chat template or "
                "marker token sequence has drifted; investigate before training."
            )


# ---------------------------------------------------------------------------
# Cross-check helper + CLI.
# ---------------------------------------------------------------------------


def target_to_eval_events(example: SFTExample, labels: tuple[str, ...] = ATUS_LABELS) -> list[dict]:
    """Round-trip the target JSON through the *inference* parser.

    Proves the generated gold is byte-compatible with what
    ``chunk_runner`` produces for a real model: returns
    ``[{"label", "start", "end"[, "description"]}, ...]`` in seconds relative to
    the chunk start (``chunk_start_s=0.0``).
    """
    return parse_segments(
        example.target_obj,
        labels=labels,
        chunk_start_s=0.0,
        time_unit=example.time_unit,
    )


def split_counts(manifest_path: str | Path, **ds_kwargs: Any) -> dict[str, int]:
    """Return ``{split: n_videos}`` for the filtered manifest (video-level)."""
    out: dict[str, int] = {}
    for sp in ("train", "val", "test"):
        ds = Ego4DDataset(
            manifest_path=manifest_path,
            split=sp,
            min_events=ds_kwargs.get("min_events", 1),
            min_duration_s=ds_kwargs.get("min_duration_s", 0.0),
            require_both_passes=ds_kwargs.get("require_both_passes", True),
            pass_filter=ds_kwargs.get("pass_id", "1"),
        )
        out[sp] = len(ds.videos)
    return out


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parents[2]
    default_manifest = repo_root / "datasets" / "ego4d" / "annotated_manifest.json"

    ap = argparse.ArgumentParser(description="Inspect Ego4D SFT training examples.")
    ap.add_argument("--manifest", default=str(default_manifest))
    ap.add_argument("--split", default="train")
    ap.add_argument("--chunk-min", type=float, default=10.0)
    ap.add_argument("--time-unit", choices=["second", "minute"], default="minute")
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--limit", type=int, default=3, help="print this many examples")
    args = ap.parse_args()

    ds = Ego4DSFTDataset(
        args.manifest,
        split=args.split,
        chunk_minutes=args.chunk_min,
        time_unit=args.time_unit,
        with_description=not args.no_description,
        load_audio=False,
    )
    print(
        f"[{args.split}] chunk_min={args.chunk_min} time_unit={args.time_unit}: "
        f"{len(ds._ds.videos)} videos -> {len(ds)} chunk-examples"
    )
    seg_counts = [ds[i].n_segments for i in range(min(500, len(ds)))]
    if seg_counts:
        print(f"segments/chunk over first {len(seg_counts)}: "
              f"mean={sum(seg_counts)/len(seg_counts):.2f} "
              f"min={min(seg_counts)} max={max(seg_counts)}")
    for ex in list(ds.iter_examples())[: args.limit]:
        print("-" * 72)
        print(f"uid={ex.uid} chunk {ex.chunk_index + 1}/{ex.n_chunks} "
              f"[{ex.chunk_start_s:.0f}-{ex.chunk_end_s:.0f}s] n_segments={ex.n_segments}")
        print("TARGET:", ex.target_text[:400])


if __name__ == "__main__":
    main()
