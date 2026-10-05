# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation metrics for long-audio activity segmentation + classification.

Common data shape across all metrics:

    events = [{"label": str, "start": float, "end": float}, ...]   # seconds

Predictions and ground truth use the same shape. See metrics.py for
individual metric semantics.
"""

from long_audio.eval.metrics import (
    EventList,
    boundary_f1,
    event_based_f1,
    event_error_rate,
    frame_level_accuracy,
    summarize,
)

__all__ = [
    "EventList",
    "boundary_f1",
    "event_based_f1",
    "event_error_rate",
    "frame_level_accuracy",
    "summarize",
]
