# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""DatasetAdapter protocol for description-quality eval.

Each adapter conforms one dataset's manifest shape (annotator pass keys,
per-slice / per-summary field names, moments availability) so a single
generic CLI can score Ego4D (2 passes, moments, human-baseline supported)
and EgoLife (1 pass, no moments, no human baseline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from long_audio.eval.description.pipeline import GroundTruth


@runtime_checkable
class DatasetAdapter(Protocol):
    """Per-dataset description-eval adapter.

    Adapters live in ``long_audio.eval.description.datasets.<name>`` and
    are looked up via ``get_adapter(name)``.
    """

    name: str
    """Dataset name (``ego4d`` / ``egolife``)."""

    default_manifest_path: Path
    """Default path to the annotated manifest for this dataset."""

    has_human_baseline: bool
    """True iff the dataset has ≥2 annotator passes; enables the
    ``evaluate_human_baseline`` code path."""

    def load_manifest(self, manifest_path: Path) -> dict:
        """Parse and return the manifest JSON."""
        ...

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        """All eval-set uids (post-canonical-filter). Mirrors the
        inference-time dataset iteration order where possible."""
        ...

    def load_gt_for_uid(self, manifest: dict, uid: str) -> GroundTruth:
        """Extract per-uid GT (per-pass summary facts + moments) as a
        :class:`GroundTruth`. Datasets without moments return
        ``GroundTruth(passes={...}, moments=[])``."""
        ...

    def sc_lists_from_gt(self, manifest: dict, uid: str) -> list[dict]:
        """Per-pass, per-slice/summary facts lists for one uid — the
        shape :func:`evaluate_self_consistency_uid` expects
        (``{"key": ..., "facts": [...]}`` records, one per non-empty
        slice/summary, tagged with pass key). Used to compute the
        human-baseline SC score."""
        ...
