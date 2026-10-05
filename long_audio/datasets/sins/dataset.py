# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""SINS dataset loader.

For dataset preparation (download / extract / per-node FLAC / mono.flac +
manifest), see ``long_audio.datasets.sins.prepare`` and the per-stage CLI
scripts under ``scripts/data/sins/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile

from long_audio.datasets.base import BaseDataset, DatasetItem


class SINSDataset(BaseDataset):
    """SINS long-audio dataset.

    Each item is one full node (~7 days of continuous audio) with
    apartment-level activity annotations.
    """

    def __init__(self, data_dir: str | Path = "datasets/SINS"):
        self.data_dir = Path(data_dir)
        self._manifest = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            manifest_path = self.data_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found at {manifest_path}. "
                    "Run the SINS prep pipeline first: "
                    "scripts/data/sins/{download,extract,build_flacs,build_mono}.py."
                )
            with manifest_path.open() as f:
                self._manifest = json.load(f)
        return self._manifest

    # SINS is one mono stream after the annotation-driven build.
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        if idx != 0:
            raise IndexError(f"SINSDataset has a single item; got idx={idx}")
        m = self.manifest
        return {
            "id": "sins_mono",
            "audio_path": Path(m["audio_path"]),
            "sample_rate": m["sample_rate"],
            "duration": m["duration"],
            "annotations": m["annotations"],
            "node_trace": m["node_trace"],
        }

    def load_audio(self, idx: int = 0) -> np.ndarray:
        """Load the mono FLAC as a numpy array."""
        item = self[idx]
        audio, sr = soundfile.read(str(item["audio_path"]))
        return audio

    def iter_items(self) -> Iterator[DatasetItem]:
        """Yield the single SINS item in the common DatasetItem shape.

        Manifest annotations already use the unified ``{start, end, event}``
        schema; passed through unchanged.
        """
        m = self.manifest
        ground_truth = [
            {"start": a["start"], "end": a["end"], "event": a["event"]}
            for a in m["annotations"]
        ]
        yield DatasetItem(
            id="sins_mono",
            audio_path=Path(m["audio_path"]),
            sample_rate=m["sample_rate"],
            duration=m["duration"],
            audio_offset_s=0.0,
            subdir=Path("sins"),
            ground_truth=ground_truth,
        )
