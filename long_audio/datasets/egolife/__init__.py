"""EgoLife dataset: 6 participants x 7 days of egocentric daily audio.

Loading-side public surface (inference + eval consumers). All data-prep
code (Stages 1-5) lives in ``scripts/data/egolife/``.
"""

from long_audio.datasets.egolife.dataset import EgoLifeDataset
from long_audio.datasets.egolife.prompt import render_egolife_prompt
from long_audio.datasets.egolife.schema import (
    EGOLIFE_HINTS,
    EGOLIFE_LABELS,
    egolife_schema,
)

__all__ = [
    "EgoLifeDataset",
    "EGOLIFE_LABELS",
    "EGOLIFE_HINTS",
    "render_egolife_prompt",
    "egolife_schema",
]
