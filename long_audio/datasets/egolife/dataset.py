"""EgoLife dataset class - loads the manifest built by ``build_manifest``.

Single-pass variant of the Ego4D loader. Filters mirror Ego4D's
``EVAL_DEFAULTS`` (``min_duration_s=600``, ``min_events=2``) but
``require_both_passes`` is fixed False since EgoLife only ships one
annotator's captions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from long_audio.datasets.base import BaseDataset, DatasetItem
from long_audio.utils.paths import resolve_audio_path


class EgoLifeDataset(BaseDataset):
    """Per-session EgoLife dataset with optional ``min_events`` /
    ``min_duration_s`` filters.

    ``__len__`` and ``__getitem__`` operate on the filtered session list.
    A "session" is a continuous recording run split from the participant-day
    stream at any inter-clip gap > 1 s (per ``extract_audio.py``); ~193
    sessions total across the 6 participants x 7 days.

    Args:
        manifest_path: path to the JSON written by ``build_manifest.build_manifest``.
        min_events: minimum number of merged-runs in ``actions`` to keep
            the session.
        min_duration_s: minimum audio duration in seconds.
    """

    # Canonical evaluation-set filter, mirroring Ego4D's EVAL_DEFAULTS
    # for cross-dataset comparability. Bump in lockstep with Ego4D.
    EVAL_DEFAULTS: dict = {
        "min_events": 2,
        "min_duration_s": 600.0,
    }

    @classmethod
    def eval_set(cls, manifest_path: str | Path) -> "EgoLifeDataset":
        """Construct the dataset with the canonical eval-set filter applied.

        Equivalent to::

            EgoLifeDataset(
                manifest_path,
                min_events=2,
                min_duration_s=600.0,
            )
        """
        return cls(manifest_path=manifest_path, **cls.EVAL_DEFAULTS)

    def __init__(
        self,
        manifest_path: str | Path,
        min_events: int = 0,
        min_duration_s: float = 0.0,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        self.min_events = min_events
        self.min_duration_s = min_duration_s
        self._manifest: dict | None = None
        self._videos: list[dict] | None = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"EgoLife manifest not found at {self.manifest_path}. "
                    "Run the EgoLife prep pipeline first: "
                    "scripts/data/egolife/{download,extract_audio,"
                    "translate_captions,annotate_manifest,build_manifest}.py."
                )
            self._manifest = json.loads(self.manifest_path.read_text())
        return self._manifest

    @property
    def videos(self) -> list[dict]:
        """Cached filtered session list. Named `videos` to match Ego4D's shape."""
        if self._videos is None:
            all_videos = self.manifest["videos"]
            kept = []
            for v in all_videos:
                if v["duration"] < self.min_duration_s:
                    continue
                actions = v["passes"]["1"]["actions"]
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
            "id": f"egolife_{v['uid']}",
            "uid": v["uid"],
            "audio_path": resolve_audio_path(v["audio_path"]),
            "sample_rate": v["sample_rate"],
            "duration": v["duration"],
            "scenarios": v["scenarios"],
            "passes": v["passes"],
            "moments": v.get("moments", []),
            "audio_offset_s": float(v["audio_offset_s"]),
            "wall_clock_start": v["wall_clock_start"],
        }

    def iter_items(self) -> Iterator[DatasetItem]:
        """Yield filtered sessions in the common DatasetItem shape."""
        for v in self.videos:
            actions = v["passes"]["1"]["actions"]
            ground_truth = [
                {"start": a["start"], "end": a["end"], "event": a["event"]}
                for a in actions
            ]
            yield DatasetItem(
                id=f"egolife_{v['uid']}",
                audio_path=resolve_audio_path(v["audio_path"]),
                sample_rate=v["sample_rate"],
                duration=v["duration"],
                audio_offset_s=float(v["audio_offset_s"]),
                subdir=Path("egolife") / v["uid"],
                ground_truth=ground_truth,
            )
