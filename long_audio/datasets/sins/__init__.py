# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""SINS dataset: stationary multi-mic home recording (DCASE 2017)."""

from long_audio.datasets.sins.dataset import SINSDataset
from long_audio.datasets.sins.prompt import render_sins_prompt
from long_audio.datasets.sins.schema import SINS_LABEL_HINTS, SINS_LABELS, sins_schema

__all__ = [
    "SINSDataset",
    "SINS_LABELS",
    "SINS_LABEL_HINTS",
    "render_sins_prompt",
    "sins_schema",
]
