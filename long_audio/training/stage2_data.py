"""Stage-2 cascade SFT data: fixed captions -> gold segmentation JSON.

This module builds text-only supervised-fine-tuning examples for the cascade
Stage B model. Each example is one Stage-B description window: the model sees
the exact ``render_prompt_from_descriptions(...)`` prompt used by cascade
inference and must emit the gold segmentation JSON for that same window.

Stage A is fixed. Captions are read from JSONL files written by
``long_audio.inference.describe``. The preferred SFT layout is:

    runs/cascA_inference/<describe_model>/ego4d/<split>/<uid>/<describe_model>.descriptions.jsonl

The target construction intentionally reuses the Ego4D E2E SFT helpers so the
output distribution matches existing fine-tuning/eval conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset

from long_audio.datasets.ego4d.dataset import Ego4DDataset
from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
from long_audio.inference.cascade import _window_descriptions
from long_audio.inference.describe import load_descriptions
from long_audio.inference.prompt import (
    ChunkContext,
    ContextMode,
    render_prompt_from_descriptions,
)
from long_audio.inference.schema import TimeUnit, parse_segments
from long_audio.training.data import (
    _clip_shift,
    disjoint_slice_segments,
    quantize_and_format,
)


@dataclass
class Stage2SFTExample:
    """One text-only Stage-B SFT example.

    ``prompt`` is byte-compatible with cascade inference for the same caption
    window and settings. ``target_text`` is the JSON string the assistant should
    produce; ``target_obj`` is the parsed equivalent for tests/debugging.
    """

    uid: str
    window_index: int
    n_windows: int
    split: str
    window_start_s: float
    window_end_s: float
    prompt: str
    target_text: str
    target_obj: dict
    n_segments: int
    gold_segments: list[dict]
    descriptions: list[dict]
    descriptions_path: str
    time_unit: TimeUnit = "minute"

    def conversation(self) -> list[dict]:
        """Text-only chat messages: user(prompt) -> assistant(target)."""
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.target_text}],
            },
        ]


class Stage2SFTDataset(Dataset):
    """PyTorch Dataset of Ego4D Stage-B text-only SFT examples.

    Args:
        manifest_path: path to Ego4D ``annotated_manifest.json``.
        captions_root: root containing Stage-A caption JSONLs. If omitted,
            defaults to ``runs/cascA_inference/<describe_model>`` under the repo.
            Preferred lookup is ``<root>/ego4d/<split>/<uid>/<model>.descriptions.jsonl``.
        describe_model: Stage-A model slug used in the JSONL filename, e.g.
            ``"af-next-captioner"``.
        split: ``"train"`` / ``"val"`` / ``"test"`` / ``None``.
        window_minutes: Stage-B window size. Must match cascade inference.
        time_unit: target timestamp grammar, ``"minute"`` or ``"second"``.
        pass_id: Ego4D narration pass that supplies gold segments.
        with_description: include per-segment gold descriptions in the target.
        context_mode: prompt prior-context mode. ``"prev"`` uses the previous
            gold window's final label as the teacher-forced prior label.
        skip_missing_captions: when true, silently skip videos without Stage-A
            JSONLs; when false, raise ``FileNotFoundError``.
        min_events / min_duration_s / require_both_passes: forwarded to
            :class:`Ego4DDataset`.
    """

    dataset_name = "ego4d"
    t0_iso = "1970-01-01T00:00:00Z"

    def __init__(
        self,
        manifest_path: str | Path,
        captions_root: str | Path | None = None,
        *,
        describe_model: str = "af-next-captioner",
        split: str | None = "train",
        window_minutes: float = 10.0,
        time_unit: TimeUnit = "minute",
        pass_id: str = "1",
        with_description: bool = True,
        context_mode: ContextMode = "none",
        skip_missing_captions: bool = False,
        min_events: int = 1,
        min_duration_s: float = 0.0,
        require_both_passes: bool = True,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        if captions_root is None:
            captions_root = Path(__file__).resolve().parents[2] / "runs" / "cascA_inference" / describe_model
        self.captions_root = Path(captions_root).expanduser()
        self.describe_model = describe_model
        self.split = split
        self.window_minutes = float(window_minutes)
        self.window_seconds = self.window_minutes * 60.0
        self.time_unit: TimeUnit = time_unit
        self.pass_id = pass_id
        self.with_description = with_description
        self.context_mode: ContextMode = context_mode
        self.skip_missing_captions = skip_missing_captions

        self._ds = Ego4DDataset(
            manifest_path=self.manifest_path,
            split=split,
            min_events=min_events,
            min_duration_s=min_duration_s,
            require_both_passes=require_both_passes,
            pass_filter=pass_id,
        )
        self._captions_cache: dict[int, tuple[Path, list[dict], list[list[dict]]]] = {}
        # (video_idx, window_idx, n_windows)
        self._index: list[tuple[int, int, int]] = []
        self._build_index()

    def _descriptions_path(self, uid: str) -> Path:
        filename = f"{self.describe_model}.descriptions.jsonl"
        split = self.split or ""
        candidates = []
        if split:
            candidates.extend([
                self.captions_root / self.dataset_name / split / uid / filename,
                self.captions_root / self.describe_model / self.dataset_name / split / uid / filename,
                self.captions_root / split / self.dataset_name / uid / filename,
                self.captions_root / uid / filename,
            ])
        candidates.extend([
            self.captions_root / self.dataset_name / uid / filename,
            self.captions_root / self.describe_model / self.dataset_name / uid / filename,
            self.captions_root / uid / filename,
        ])
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _load_video_windows(self, video_idx: int) -> tuple[Path, list[dict], list[list[dict]]]:
        cached = self._captions_cache.get(video_idx)
        if cached is not None:
            return cached
        video = self._ds.videos[video_idx]
        path = self._descriptions_path(video["uid"])
        if not path.exists():
            raise FileNotFoundError(f"Stage-A descriptions not found: {path}")
        descriptions = load_descriptions(path)
        if not descriptions:
            raise ValueError(f"No Stage-A descriptions found in {path}")
        windows = _window_descriptions(descriptions, self.window_seconds)
        if not windows:
            raise ValueError(f"No Stage-B windows built from {path}")
        cached = (path, descriptions, windows)
        self._captions_cache[video_idx] = cached
        return cached

    def _build_index(self) -> None:
        self._index.clear()
        for vi, v in enumerate(self._ds.videos):
            try:
                _, _, windows = self._load_video_windows(vi)
            except FileNotFoundError:
                if self.skip_missing_captions:
                    continue
                raise
            for wi in range(len(windows)):
                self._index.append((vi, wi, len(windows)))

    def __len__(self) -> int:
        return len(self._index)

    def _gold_for_video(self, video: dict) -> list[dict]:
        slices = video["passes"][self.pass_id]["slices"]
        return disjoint_slice_segments(slices, float(video["duration"]))

    def build_example(
        self,
        video_idx: int,
        window_idx: int,
        n_windows: int,
        *,
        prev_label: str | None = None,
    ) -> Stage2SFTExample:
        video = self._ds.videos[video_idx]
        descriptions_path, _, windows = self._load_video_windows(video_idx)
        win_descs = windows[window_idx]
        win_start_s = min(float(d["start_s"]) for d in win_descs)
        win_end_s = max(float(d["end_s"]) for d in win_descs)
        nominal_start_s = window_idx * self.window_seconds
        window_start_s = min(win_start_s, nominal_start_s) if window_idx > 0 else win_start_s
        window_end_s = win_end_s
        window_dur_s = window_end_s - window_start_s

        gold = self._gold_for_video(video)
        rel_segs = _clip_shift(gold, window_start_s, window_end_s)
        seg_objs = quantize_and_format(
            rel_segs,
            window_dur_s,
            self.time_unit,
            with_description=self.with_description,
        )
        target_obj = {"segments": seg_objs}
        target_text = json.dumps(target_obj, ensure_ascii=False)

        if self.context_mode == "prev" and prev_label is None and window_idx > 0:
            prev_label = self._last_gold_label(video, windows[window_idx - 1], window_idx - 1)

        ctx = ChunkContext(
            chunk_index=window_idx,
            n_chunks=n_windows,
            chunk_start_wallclock_s=window_start_s,
            chunk_end_wallclock_s=window_end_s,
            t0_iso=self.t0_iso,
            dataset_name=self.dataset_name,
        )
        prompt = render_prompt_from_descriptions(
            ctx,
            win_descs,
            ATUS_LABELS,
            label_hints=ATUS_HINTS,
            context_mode=self.context_mode,
            prev_label=prev_label,
            time_unit=self.time_unit,
            with_description=self.with_description,
        )

        return Stage2SFTExample(
            uid=video["uid"],
            window_index=window_idx,
            n_windows=n_windows,
            split=video.get("split", ""),
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            prompt=prompt,
            target_text=target_text,
            target_obj=target_obj,
            n_segments=len(seg_objs),
            gold_segments=rel_segs,
            descriptions=list(win_descs),
            descriptions_path=str(descriptions_path),
            time_unit=self.time_unit,
        )

    def _last_gold_label(
        self,
        video: dict,
        descriptions: list[dict],
        window_idx: int,
    ) -> str | None:
        win_start_s = min(float(d["start_s"]) for d in descriptions)
        win_end_s = max(float(d["end_s"]) for d in descriptions)
        nominal_start_s = window_idx * self.window_seconds
        window_start_s = min(win_start_s, nominal_start_s) if window_idx > 0 else win_start_s
        gold = self._gold_for_video(video)
        rel_segs = _clip_shift(gold, window_start_s, win_end_s)
        seg_objs = quantize_and_format(
            rel_segs,
            win_end_s - window_start_s,
            self.time_unit,
            with_description=self.with_description,
        )
        return seg_objs[-1]["event"] if seg_objs else None

    def __getitem__(self, idx: int) -> Stage2SFTExample:
        video_idx, window_idx, n_windows = self._index[idx]
        return self.build_example(video_idx, window_idx, n_windows)

    def iter_examples(self) -> Iterator[Stage2SFTExample]:
        for i in range(len(self)):
            yield self[i]


def target_to_eval_events(
    example: Stage2SFTExample,
    labels: tuple[str, ...] = ATUS_LABELS,
) -> list[dict]:
    """Round-trip a Stage-2 target JSON through the inference parser."""
    return parse_segments(
        example.target_obj,
        labels=labels,
        chunk_start_s=0.0,
        time_unit=example.time_unit,
    )
