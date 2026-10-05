# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Segmentation eval — dataset-agnostic scoring for Ego4D / EgoLife / SINS.

The generic scoring core lives in :mod:`long_audio.eval.segmentation.aggregate`;
per-dataset shape adapters live in :mod:`long_audio.eval.segmentation.datasets`.

Typical CLI usage: ``python scripts/eval_segmentation.py --dataset {ego4d|egolife|sins} <run_dir>``.
"""

from long_audio.eval.segmentation.aggregate import (
    aggregate,
    build_pred_missing,
    clip_predictions,
    evaluate_human_baseline_segmentation,
    format_headline,
    format_pred_missing_line,
    headline_metrics,
    score_one_pass,
    score_video,
    stitch_chunk_predictions,
)
from long_audio.eval.segmentation.datasets import get_adapter
from long_audio.eval.segmentation.datasets.base import DatasetAdapter, VideoRecord

__all__ = [
    "DatasetAdapter",
    "VideoRecord",
    "aggregate",
    "build_pred_missing",
    "clip_predictions",
    "evaluate_human_baseline_segmentation",
    "format_headline",
    "format_pred_missing_line",
    "get_adapter",
    "headline_metrics",
    "score_one_pass",
    "score_video",
    "stitch_chunk_predictions",
]
