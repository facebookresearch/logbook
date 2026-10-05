# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""EgoLife shares Ego4D's 6-class ATUS taxonomy for cross-dataset comparability.

This module re-exports the Ego4D symbols under EgoLife-namespaced names so
callers importing ``EGOLIFE_LABELS`` / ``EGOLIFE_HINTS`` / ``egolife_schema``
keep working, but there is only one source of truth for the label set:
``long_audio/datasets/ego4d/atus.json``.
"""

from __future__ import annotations

from long_audio.datasets.ego4d.schema import (
    ATUS_HINTS as EGOLIFE_HINTS,
    ATUS_LABELS as EGOLIFE_LABELS,
    ego4d_schema as egolife_schema,
)

__all__ = ["EGOLIFE_LABELS", "EGOLIFE_HINTS", "egolife_schema"]
