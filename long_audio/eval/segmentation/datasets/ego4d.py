# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Ego4D segmentation-eval adapter.

Ego4D has two annotator passes ("1" and "2"). Each pass carries its own
``actions`` list post-coalesce. Eval scores each ``(uid, pass)`` as a
separate data point.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.eval.metrics import MetricConfig
from long_audio.eval.segmentation.datasets.base import VideoRecord

_REPO_ROOT = Path(__file__).resolve().parents[4]

#: 2.5-min boundary tolerance (half the 5-min classifier granularity).
EGO4D_BF1_TOLERANCE_S = 150.0


class Ego4DAdapter:
    name = "ego4d"
    # Default to the shipped public manifest so segmentation eval works
    # out of the box on a fresh clone. Users with the full annotated
    # manifest (post-rehydrate + annotate) can pass --manifest explicitly.
    default_manifest_path = (
        _REPO_ROOT / "datasets" / "ego4d" / "manifest.public.json"
    )
    id_prefix = "ego4d_"
    # Ego4D ships two annotator passes → inter-annotator human baseline
    # (pass1 vs pass2 both directions, pooled) is available. EgoLife + SINS
    # are single-annotator, so their adapters set this False.
    has_human_baseline = True

    def metric_config(self) -> MetricConfig:
        return MetricConfig(
            boundary_tolerance_s=EGO4D_BF1_TOLERANCE_S,
            event_exclude_labels=("other",),
        )

    def load_manifest(self, manifest_path: Path) -> dict:
        return json.loads(Path(manifest_path).read_text())

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        from long_audio.datasets.ego4d.dataset import Ego4DDataset
        ds = Ego4DDataset.eval_set(manifest_path, split="test")
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
