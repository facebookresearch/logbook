# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""SINS dataset-specific label set + per-label hints + schema helpers.

Single source of truth for the SINS label set:
``long_audio/datasets/sins/sins.json`` (17 fine-grained activity classes
from the DCASE 2017 home-recording corpus, location-flat — no room
suffix). ``SINS_LABEL_HINTS`` is loaded from disk; ``SINS_LABELS`` is
derived from its keys so the two can never drift.

Hints are paper-grounded natural-language only — no per-label statistics
(at evaluation time we can't assume the model has seen them). ``absence``
is the negative class (no human activity); ``other`` is the catch-all
for activities outside the named buckets.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.inference.schema import TimeUnit, make_segmentation_schema


_SINS_PATH = Path(__file__).resolve().parent / "sins.json"


def load_sins_hints(path: Path | str = _SINS_PATH) -> dict[str, str]:
    """Return ``{class_name: description}`` from the SINS taxonomy."""
    raw = json.loads(Path(path).read_text())
    return {k: v["description"] for k, v in raw.items()}


SINS_LABEL_HINTS: dict[str, str] = load_sins_hints()
# Derived from HINTS' keys (Python 3.7+ dict-preserves-insertion-order
# guarantee). Adding/removing/renaming a label only happens in sins.json.
SINS_LABELS: tuple[str, ...] = tuple(SINS_LABEL_HINTS.keys())


def sins_schema(
    time_unit: TimeUnit = "second",
    *,
    with_description: bool = False,
) -> dict:
    """Return the segmentation JSON schema bound to the SINS label set."""
    return make_segmentation_schema(
        labels=SINS_LABELS, time_unit=time_unit,
        with_description=with_description,
    )
