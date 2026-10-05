# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the shared segmentation aggregation.

Tests the dataset-agnostic ``long_audio.eval.segmentation.aggregate``
module and the ``Ego4DAdapter`` shape.
"""

from __future__ import annotations

import unittest

from long_audio.eval.segmentation import (
    aggregate,
    get_adapter,
    headline_metrics,
    score_video,
)
from long_audio.eval.segmentation.aggregate import (
    actions_to_eval_events,
    clip_predictions,
)
from long_audio.eval.segmentation.datasets.ego4d import (
    EGO4D_BF1_TOLERANCE_S,
)

_EGO4D_CFG = get_adapter("ego4d").metric_config()


class TestHelpers(unittest.TestCase):
    def test_ego4d_metric_config(self):
        self.assertEqual(_EGO4D_CFG.boundary_tolerance_s, EGO4D_BF1_TOLERANCE_S)
        self.assertEqual(_EGO4D_CFG.event_exclude_labels, ("other",))

    def test_actions_to_eval_events(self):
        ev = actions_to_eval_events([{"start": 0, "end": 30, "event": "food"}])
        self.assertEqual(ev, [{"label": "food", "start": 0.0, "end": 30.0}])

    def test_clip_predictions(self):
        preds = [
            {"label": "a", "start": 0.0, "end": 40.0},   # trimmed to 30
            {"label": "b", "start": 35.0, "end": 50.0},  # dropped (starts past 30)
        ]
        out = clip_predictions(preds, 30.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["end"], 30.0)


def _video(uid, duration, actions1, actions2=None):
    passes = {"1": {"actions": actions1}}
    if actions2 is not None:
        passes["2"] = {"actions": actions2}
    return {"uid": uid, "duration": duration, "passes": passes}


class TestScoreAndAggregate(unittest.TestCase):
    def test_perfect_match_frame_accuracy_one(self):
        acts = [
            {"start": 0.0, "end": 30.0, "event": "productive"},
            {"start": 30.0, "end": 60.0, "event": "leisure"},
        ]
        preds = [
            {"label": "productive", "start": 0.0, "end": 30.0},
            {"label": "leisure", "start": 30.0, "end": 60.0},
        ]
        v = _video("v1", 60.0, acts, acts)
        dps = score_video(
            "v1", preds, v, audio_duration_s=60.0, n_chunks=2, config=_EGO4D_CFG,
        )
        self.assertEqual(len(dps), 2)  # two passes -> two data points
        for d in dps:
            self.assertAlmostEqual(d["scores"]["frame_level_accuracy"]["overall_accuracy"], 1.0)

        agg = aggregate(
            dps, model_name="test",
            boundary_tolerance_s=_EGO4D_CFG.boundary_tolerance_s,
        )
        # Format matches eval_segmentation.py output.
        self.assertEqual(
            set(agg.keys()),
            {"model", "n_videos", "n_data_points", "total_chunks",
             "boundary_tolerance_s", "macro", "micro", "pred_missing", "per_datapoint"},
        )
        self.assertEqual(agg["model"], "test")
        self.assertEqual(agg["n_videos"], 1)
        self.assertEqual(agg["n_data_points"], 2)
        self.assertEqual(agg["total_chunks"], 2)  # deduped by uid, not x passes
        self.assertAlmostEqual(agg["micro"]["frame_accuracy"], 1.0)
        self.assertAlmostEqual(agg["macro"]["frame_accuracy"], 1.0)
        self.assertEqual(agg["boundary_tolerance_s"], EGO4D_BF1_TOLERANCE_S)

    def test_headline_metrics_shape(self):
        acts = [{"start": 0.0, "end": 60.0, "event": "productive"}]
        preds = [{"label": "productive", "start": 0.0, "end": 60.0}]
        dps = score_video(
            "v1", preds, _video("v1", 60.0, acts), 60.0, 1, config=_EGO4D_CFG,
        )
        agg = aggregate(
            dps, "m", boundary_tolerance_s=_EGO4D_CFG.boundary_tolerance_s,
        )
        hm = headline_metrics(agg)
        self.assertEqual(
            set(hm.keys()),
            {"boundary_f1", "frame_accuracy", "event_f1", "event_error_rate"},
        )

    def test_egolife_gt_trailing_gap_clips_to_gt_end(self):
        # EgoLife GT ends at 6300s; audio runs to 6339.02s. Eval must score
        # only [0, 6300] — trailing audio is unannotated and excluded.
        acts = [
            {"start": 0.0, "end": 6000.0, "event": "working"},
            {"start": 6000.0, "end": 6300.0, "event": "eating"},
        ]
        # Perfect predictions over the GT window only.
        preds = [
            {"label": "working", "start": 0.0, "end": 6000.0},
            {"label": "eating", "start": 6000.0, "end": 6300.0},
        ]
        v = _video("egolife_A1_JAKE_DAY7_S01", 6339.02, acts)
        dps = score_video(
            "egolife_A1_JAKE_DAY7_S01", preds, v,
            audio_duration_s=6339.02, n_chunks=22,
            config=get_adapter("egolife").metric_config(),
        )
        self.assertEqual(len(dps), 1)
        dp = dps[0]
        # Eval window is [0, 6300], not [0, 6339.02].
        self.assertAlmostEqual(dp["duration_s"], 6300.0)
        # Perfect predictions over [0, 6300] → frame accuracy 1.0.
        self.assertAlmostEqual(
            dp["scores"]["frame_level_accuracy"]["overall_accuracy"], 1.0
        )
        # Predictions past 6300s are clipped and don't affect scoring.
        preds_with_overshoot = preds + [{"label": "other", "start": 6300.0, "end": 6339.02}]
        dps2 = score_video(
            "egolife_A1_JAKE_DAY7_S01", preds_with_overshoot, v,
            audio_duration_s=6339.02, n_chunks=22,
            config=get_adapter("egolife").metric_config(),
        )
        self.assertAlmostEqual(
            dps2[0]["scores"]["frame_level_accuracy"]["overall_accuracy"], 1.0
        )

    def test_total_chunks_dedup_across_videos(self):
        acts = [{"start": 0.0, "end": 60.0, "event": "productive"}]
        preds = [{"label": "productive", "start": 0.0, "end": 60.0}]
        dps = []
        dps += score_video(
            "v1", preds, _video("v1", 60.0, acts, acts), 60.0, 3, config=_EGO4D_CFG,
        )
        dps += score_video(
            "v2", preds, _video("v2", 60.0, acts, acts), 60.0, 5, config=_EGO4D_CFG,
        )
        agg = aggregate(
            dps, "m", boundary_tolerance_s=_EGO4D_CFG.boundary_tolerance_s,
        )
        self.assertEqual(agg["n_videos"], 2)
        self.assertEqual(agg["n_data_points"], 4)  # 2 videos x 2 passes
        self.assertEqual(agg["total_chunks"], 8)   # 3 + 5, each counted once


if __name__ == "__main__":
    unittest.main()
