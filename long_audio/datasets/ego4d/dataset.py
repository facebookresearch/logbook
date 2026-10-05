# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Ego4D dataset class — loads the manifest built by ``manifest_builder``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from long_audio.datasets.base import BaseDataset, DatasetItem
from long_audio.utils.paths import resolve_audio_path


class Ego4DDataset(BaseDataset):
    """Per-video Ego4D dataset with optional ``min_events`` / ``min_duration_s``
    / ``require_both_passes`` filters.

    ``__len__`` and ``__getitem__`` operate on the filtered video list.

    Args:
        manifest_path: path to the JSON written by ``manifest_builder.build_manifest``.
        min_events: minimum number of merged-runs in pass-1 ``actions`` to keep
            the video. (3 = at least 3 transitions in the timeline.)
        min_duration_s: minimum audio duration in seconds.
        pass_filter: only count event-runs from this pass when applying the
            ``min_events`` filter ("1" or "2"). Default "1" since pass-2 may be
            absent for some videos.
        require_both_passes: if True (the default), also drop videos where
            either pass is missing from ``passes``. ``True`` by default —
            most callers want the canonical eval set (paired ground-truth
            from both annotators). Pass ``False`` explicitly for the rare
            case where single-pass videos are acceptable (e.g. ad-hoc
            inference runs over the full manifest).
    """

    # Canonical evaluation-set filter. Used by audit_agreement, eval_run_ego4d,
    # and (transitively) any paper-figure regeneration script. Bump the values
    # in lockstep with the paper's reported eval set.
    EVAL_DEFAULTS: dict = {
        "min_events": 2,
        "require_both_passes": True,
        "min_duration_s": 600.0,
        "pass_filter": "1",
    }

    @classmethod
    def eval_set(
        cls, manifest_path: str | Path, split: str | None = None
    ) -> "Ego4DDataset":
        """Construct the dataset with the canonical eval-set filter applied.

        Equivalent to::

            Ego4DDataset(
                manifest_path,
                min_events=2,
                require_both_passes=True,
                min_duration_s=600.0,
                pass_filter='1',
                split=split,
            )

        ``split`` is passed through unchanged: None = all splits (use for
        human-ceiling baseline / inter-annotator stats), ``"test"`` =
        test split only (use for inference / eval runs).

        min_events=2 (drop passes with fewer than 2 events; see
        annotate_manifest.py).
        """
        return cls(manifest_path=manifest_path, split=split, **cls.EVAL_DEFAULTS)

    def __init__(
        self,
        manifest_path: str | Path,
        min_events: int = 0,
        min_duration_s: float = 0.0,
        pass_filter: str = "1",
        require_both_passes: bool = True,
        split: str | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        self.min_events = min_events
        self.min_duration_s = min_duration_s
        self.pass_filter = pass_filter
        self.require_both_passes = require_both_passes
        self.split = split  # None = no split filter; e.g. "test", "val", "train"
        self._manifest: dict | None = None
        self._videos: list[dict] | None = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"Ego4D manifest not found at {self.manifest_path}. "
                    "Run `long-audio prepare ego4d` (or "
                    "`python -m long_audio.datasets.ego4d.manifest_builder`) first."
                )
            self._manifest = json.loads(self.manifest_path.read_text())
        return self._manifest

    @property
    def videos(self) -> list[dict]:
        """Cached filtered video list."""
        if self._videos is None:
            all_videos = self.manifest["videos"]
            kept = []
            for v in all_videos:
                if self.split is not None and v.get("split") != self.split:
                    continue
                if v["duration"] < self.min_duration_s:
                    continue
                passes = v["passes"]
                if self.require_both_passes and not ("1" in passes and "2" in passes):
                    continue
                actions = passes[self.pass_filter]["actions"]
                if len(actions) < self.min_events:
                    continue
                kept.append(v)
            self._videos = kept
        return self._videos

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        v = self.videos[idx]
        return {
            "id": f"ego4d_{v['uid']}",
            "uid": v["uid"],
            "audio_path": resolve_audio_path(v["audio_path"]),
            "sample_rate": v["sample_rate"],
            "duration": v["duration"],
            "scenarios": v["scenarios"],
            "passes": v["passes"],
            "moments": v.get("moments", []),
            # ``audio_offset_s`` is the read-time seek offset into the raw FLAC
            # where manifest t=0 begins. Stage 3 (build_manifest) renamed the
            # old ``lead_shift`` field to this — the value here folds the
            # narration lead-shift AND the cross-pass intersection start so
            # consumers just need a single seek offset.
            "audio_offset_s": float(v["audio_offset_s"]),
        }

    def iter_items(self) -> Iterator[DatasetItem]:
        """Yield filtered videos in the common DatasetItem shape.

        ``ground_truth`` is the ``pass_filter`` annotator's coalesced
        ``actions`` list. The full per-pass structure (and other extras
        — ``scenarios``, ``moments``) stays accessible via
        ``__getitem__`` / ``.videos`` for eval-side code that needs
        inter-annotator agreement or scenario stratification.
        """
        for v in self.videos:
            actions = v["passes"][self.pass_filter]["actions"]
            # Manifest's coalesced `actions` already use {start, end, event};
            # passed through unchanged (matches the unified schema).
            ground_truth = [
                {"start": a["start"], "end": a["end"], "event": a["event"]}
                for a in actions
            ]
            yield DatasetItem(
                id=f"ego4d_{v['uid']}",
                audio_path=resolve_audio_path(v["audio_path"]),
                sample_rate=v["sample_rate"],
                duration=v["duration"],
                audio_offset_s=float(v["audio_offset_s"]),
                subdir=Path("ego4d") / v["uid"],
                ground_truth=ground_truth,
            )
