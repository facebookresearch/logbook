# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Ego4D dataset: egocentric daily activity videos with narrations.

Loading-side public surface (inference + eval consumers). All data-prep
code (Stages 1-4) lives in ``scripts/data/ego4d/``.
"""

from long_audio.datasets.ego4d.dataset import Ego4DDataset
from long_audio.datasets.ego4d.prompt import render_ego4d_prompt
from long_audio.datasets.ego4d.schema import (
    ATUS_HINTS,
    ATUS_LABELS,
    ego4d_schema,
)

__all__ = [
    "Ego4DDataset",
    "ATUS_LABELS",
    "ATUS_HINTS",
    "render_ego4d_prompt",
    "ego4d_schema",
]
