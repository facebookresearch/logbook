# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""SINS segmentation-eval adapter.

SINS is a single-file dataset — one long mono FLAC with a flat
``annotations`` list, no per-video / per-pass structure. The adapter
synthesizes a single VideoRecord with uid ``sins`` and pseudo-pass
``"1"``, so the same generic ``score_video`` / ``aggregate`` pipeline
handles SINS alongside Ego4D and EgoLife.

Both ``absence`` (no observed activity) and ``other`` (named bucket for
activities outside the 17-class set) are excluded from event-F1 so the
metric reflects performance on actual activities, not filler classes.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.eval.metrics import MetricConfig
from long_audio.eval.segmentation.datasets.base import VideoRecord
from long_audio.utils.events import assert_gt_contiguous

_REPO_ROOT = Path(__file__).resolve().parents[4]

#: SINS boundary tolerance. Adjust if we ever tighten SINS's boundary metric.
SINS_BF1_TOLERANCE_S = 150.0

#: Single synthetic uid for SINS. All predictions map to this uid.
SINS_UID = "sins"


class SinsAdapter:
    name = "sins"
    default_manifest_path = _REPO_ROOT / "datasets" / "SINS" / "manifest.json"
    id_prefix = ""  # SINS run_index (if any) uses bare ids; not typically used
    has_human_baseline = False  # single-annotator, single-file

    def metric_config(self) -> MetricConfig:
        return MetricConfig(
            boundary_tolerance_s=SINS_BF1_TOLERANCE_S,
            event_exclude_labels=("absence", "other"),
        )

    def load_manifest(self, manifest_path: Path) -> dict:
        manifest = json.loads(Path(manifest_path).read_text())
        if "annotations" not in manifest:
            raise ValueError(
                f"{manifest_path}: missing top-level 'annotations' field. "
                "Rebuild the manifest via scripts/data/sins/build_mono.py."
            )
        return manifest

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        return [SINS_UID]

    def video_record(self, manifest: dict, uid: str) -> VideoRecord | None:
        if uid != SINS_UID:
            return None
        # Manifest carries {activity, start, end}; conform to the
        # {event, start, end} shape ``score_video`` expects.
        actions = [
            {
                "start": float(a["start"]),
                "end": float(a["end"]),
                "event": a["event"],
            }
            for a in manifest["annotations"]
        ]
        duration = float(manifest.get("duration", actions[-1]["end"] if actions else 0.0))
        assert_gt_contiguous(
            [{"label": a["event"], "start": a["start"], "end": a["end"]} for a in actions],
            duration_s=duration, name="SINS GT",
        )
        return VideoRecord(
            uid=SINS_UID,
            duration=duration,
            passes={"1": {"actions": actions}},
        )
