# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Ego4D dataset-specific label set + per-label hints + schema helpers.

Single source of truth for the ATUS label set:
``long_audio/datasets/ego4d/atus.json`` (American Time Use Survey
v4 taxonomy). ``ATUS_HINTS`` is loaded from disk; ``ATUS_LABELS`` is
derived from its keys so the two can never drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.inference.schema import TimeUnit, make_segmentation_schema


_ATUS_PATH = Path(__file__).resolve().parent / "atus.json"


def load_atus_hints(path: Path | str = _ATUS_PATH) -> dict[str, str]:
    """Return ``{class_name: description}`` from the ATUS taxonomy."""
    raw = json.loads(Path(path).read_text())
    return {k: v["description"] for k, v in raw.items()}


ATUS_HINTS: dict[str, str] = load_atus_hints()
# Derived from HINTS' keys (Python 3.7+ dict-preserves-insertion-order
# guarantee). Adding/removing/renaming a label only happens in atus.json.
ATUS_LABELS: tuple[str, ...] = tuple(ATUS_HINTS.keys())


def ego4d_schema(
    time_unit: TimeUnit = "second",
    *,
    with_description: bool = False,
) -> dict:
    """Return the segmentation JSON schema bound to the ATUS label set."""
    return make_segmentation_schema(
        labels=ATUS_LABELS, time_unit=time_unit,
        with_description=with_description,
    )
