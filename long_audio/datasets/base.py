"""Base dataset abstract class + the common iteration item."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class DatasetItem:
    """One inference-ready audio item, common shape across datasets.

    The orchestration scripts (``scripts/run_e2e.py``,
    ``scripts/run_cascade.py``) consume the ``iter_items()`` view rather
    than dataset-specific ``__getitem__`` shapes, so the runner code can
    be the same for SINS (1 item) and Ego4D (thousands).

    Dataset-specific extras (SINS ``node_trace``, Ego4D ``passes`` /
    ``moments`` / ``scenarios``) are NOT mapped here — eval-side code
    that needs them should still go through the dataset class's
    ``__getitem__`` / ``.videos`` / ``.manifest``.
    """

    id: str
    """Stable identifier. Used as a logging tag + (for multi-item datasets)
    the per-item subdir name. SINS uses ``"sins_mono"``."""

    audio_path: Path
    """Mono FLAC/WAV at ``sample_rate`` Hz."""

    sample_rate: int
    """Sampling rate of the audio file. Asserted, not resampled."""

    duration: float
    """Effective audio length in seconds, starting from ``audio_offset_s``.
    May be less than ``soundfile.info(audio_path).frames / sample_rate``
    if the manifest only covers part of the raw file (Ego4D's narration
    coverage window). The runner uses this as ``audio_window_s`` to clamp
    inference to the manifest-derived span."""

    audio_offset_s: float = 0.0
    """Seek offset into the raw FLAC where manifest t=0 begins. SINS = 0
    (manifest t=0 is file t=0 by construction). Ego4D = per-video, folds
    narration lead-shift + cross-pass intersection start."""

    subdir: Path = field(default_factory=lambda: Path("."))
    """Per-item subdir under the run output root. SINS = ``"."`` (single
    item; runs land at ``<UTC>/<model>/``). Ego4D = ``ego4d/<uid>``
    (per-video subdirs under ``<UTC>/<model>/``)."""

    ground_truth: list[dict] = field(default_factory=list)
    """Normalized GT segments, ``[{"start": float, "end": float,
    "event": str}, ...]``. For Ego4D this is one annotator's actions
    (the dataset's ``pass_filter``); the full per-pass structure stays
    on ``Ego4DDataset.videos``. The shared schema is ``{start, end,
    event}`` — same keys used by the manifests AND the model's JSON
    output, so there's a single canonical name across the whole pipeline."""


class BaseDataset(ABC):
    """Abstract base class for long-audio datasets.

    Subclasses MUST implement ``__len__``, ``__getitem__`` (dataset-shaped),
    and ``iter_items`` (common shape — what the runner consumes).
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...

    @abstractmethod
    def iter_items(self) -> Iterator[DatasetItem]: ...
