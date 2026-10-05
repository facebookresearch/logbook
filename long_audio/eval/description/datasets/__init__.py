# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset adapter registry for description-quality eval."""

from __future__ import annotations

from long_audio.eval.description.datasets.base import DatasetAdapter
from long_audio.eval.description.datasets.ego4d import Ego4DAdapter
from long_audio.eval.description.datasets.egolife import EgoLifeAdapter

_REGISTRY: dict[str, DatasetAdapter] = {
    "ego4d": Ego4DAdapter(),
    "egolife": EgoLifeAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    """Look up a description-eval dataset adapter by name.

    Raises ``KeyError`` with a helpful message if the dataset is unknown.
    SINS is not supported (no description GT — audio-only annotations).
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r} for description eval. "
            f"Available: {sorted(_REGISTRY)}. "
            "(SINS is not supported — audio-only annotations, no description GT.)"
        ) from None


__all__ = [
    "DatasetAdapter",
    "Ego4DAdapter",
    "EgoLifeAdapter",
    "get_adapter",
]
