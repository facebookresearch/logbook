"""DatasetAdapter protocol for segmentation eval.

Each adapter conforms one dataset's shape (manifest path, per-video GT
extraction, uid enumeration, id prefix, metric config) so a single
generic aggregate / CLI can run on Ego4D (2 annotator passes),
EgoLife (1 pass), or SINS (single-file dataset, wrapped as 1 pseudo-pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from long_audio.eval.metrics import MetricConfig


@dataclass
class VideoRecord:
    """Uniform per-video GT shape consumed by ``score_video``.

    ``passes`` maps a pass key ("1" or "2") to that pass's
    ``[{start, end, event}, ...]`` action list. Single-pass datasets
    supply ``{"1": [...]}``. SINS synthesizes a whole-audio "video" wrapping
    its single ``annotations`` list as pseudo-pass "1".

    ``duration`` is the manifest's declared audio duration in seconds; the
    caller clips against ``min(duration, audio_duration_s)`` at eval time.
    """
    uid: str
    duration: float
    passes: dict[str, dict]  # {"1": {"actions": [...]}, ...}


@runtime_checkable
class DatasetAdapter(Protocol):
    """Per-dataset segmentation-eval adapter.

    Adapters live in ``long_audio.eval.segmentation.datasets.<name>``
    and are looked up via ``get_adapter(name)``.
    """

    name: str
    """Dataset name (``ego4d`` / ``egolife`` / ``sins``)."""

    default_manifest_path: Path
    """Default path to the annotated manifest for this dataset."""

    id_prefix: str
    """Prefix on ``run_index.json`` ``id`` field to strip when recovering
    a bare uid (e.g. ``"ego4d_"``). Empty string when the id IS the uid."""

    has_human_baseline: bool
    """True iff the dataset has ≥2 annotator passes and an inter-annotator
    baseline (pass1 vs pass2, both directions) is meaningful. Only Ego4D
    is True currently; EgoLife and SINS are single-annotator."""

    def metric_config(self) -> MetricConfig:
        """The dataset's canonical ``MetricConfig``."""
        ...

    def load_manifest(self, manifest_path: Path) -> dict:
        """Parse and return the manifest JSON."""
        ...

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        """All eval-set uids for this dataset (post-filter). Mirrors the
        inference-time dataset iteration order where possible."""
        ...

    def video_record(self, manifest: dict, uid: str) -> VideoRecord | None:
        """Extract one uid's ``VideoRecord`` from the manifest. Returns
        ``None`` if uid is absent — caller logs and skips."""
        ...
