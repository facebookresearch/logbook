# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset adapter registry for segmentation eval."""

from __future__ import annotations

from long_audio.eval.segmentation.datasets.base import DatasetAdapter, VideoRecord
from long_audio.eval.segmentation.datasets.ego4d import Ego4DAdapter
from long_audio.eval.segmentation.datasets.egolife import EgoLifeAdapter
from long_audio.eval.segmentation.datasets.sins import SinsAdapter

_REGISTRY: dict[str, DatasetAdapter] = {
    "ego4d": Ego4DAdapter(),
    "egolife": EgoLifeAdapter(),
    "sins": SinsAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    """Look up a dataset adapter by name.

    Raises ``KeyError`` with a helpful message if the dataset is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r} for segmentation eval. "
            f"Available: {sorted(_REGISTRY)}."
        ) from None


__all__ = [
    "DatasetAdapter",
    "VideoRecord",
    "Ego4DAdapter",
    "EgoLifeAdapter",
    "SinsAdapter",
    "get_adapter",
]
