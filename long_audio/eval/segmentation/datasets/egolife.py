# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""EgoLife segmentation-eval adapter.

EgoLife is single-annotator — only ``passes["1"]`` exists. Each uid
therefore produces exactly one eval data point. Otherwise identical to
Ego4D: same ``actions`` shape (post-coalesce, {start, end, event}), same
5-min classifier granularity → same 150 s boundary-F1 tolerance, same
``other``-excluded event-F1.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.eval.metrics import MetricConfig
from long_audio.eval.segmentation.datasets.base import VideoRecord

_REPO_ROOT = Path(__file__).resolve().parents[4]

#: Same tolerance as Ego4D (5-min classifier granularity).
EGOLIFE_BF1_TOLERANCE_S = 150.0


class EgoLifeAdapter:
    name = "egolife"
    # Default to the shipped public manifest so segmentation eval works
    # out of the box on a fresh clone. Users with the full manifest
    # (post-rehydrate) can pass --manifest explicitly.
    default_manifest_path = (
        _REPO_ROOT / "datasets" / "egolife" / "manifest.public.json"
    )
    id_prefix = "egolife_"
    has_human_baseline = False  # single-annotator

    def metric_config(self) -> MetricConfig:
        return MetricConfig(
            boundary_tolerance_s=EGOLIFE_BF1_TOLERANCE_S,
            event_exclude_labels=("other",),
        )

    def load_manifest(self, manifest_path: Path) -> dict:
        return json.loads(Path(manifest_path).read_text())

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        from long_audio.datasets.egolife.dataset import EgoLifeDataset
        ds = EgoLifeDataset.eval_set(manifest_path)
        return sorted(v["uid"] for v in ds.videos)

    def video_record(self, manifest: dict, uid: str) -> VideoRecord | None:
        v = next((x for x in manifest["videos"] if x["uid"] == uid), None)
        if v is None:
            return None
        return VideoRecord(
            uid=uid,
            duration=float(v["duration"]),
            passes={
                pkey: {"actions": pd["actions"]}
                for pkey, pd in v["passes"].items()
                if "actions" in pd
            },
        )
