# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""EgoLife description-eval adapter.

Single annotator pass ("1"); per 5-min-window ``summaries`` (not
``slices`` like Ego4D). Moments field exists but is always empty for
EgoLife — Moments Recall is naturally 0/0 (no-op). Human baseline is
NOT supported (single-pass dataset) — CLI skips the code path.
"""

from __future__ import annotations

import json
from pathlib import Path

from long_audio.eval.description.pipeline import (
    GroundTruth,
    atoms_from_moments,
    atoms_from_pass,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


class EgoLifeAdapter:
    name = "egolife"
    default_manifest_path = (
        _REPO_ROOT / "datasets" / "egolife" / "manifest.json"
    )
    has_human_baseline = False

    def load_manifest(self, manifest_path: Path) -> dict:
        return json.loads(Path(manifest_path).read_text())

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        from long_audio.datasets.egolife.dataset import EgoLifeDataset
        ds = EgoLifeDataset.eval_set(manifest_path)
        return sorted({it.id.removeprefix("egolife_") for it in ds.iter_items()})

    def load_gt_for_uid(self, manifest: dict, uid: str) -> GroundTruth:
        v = next(x for x in manifest["videos"] if x["uid"] == uid)
        # EgoLife's per-pass field is ``summaries`` (5-min windows), not
        # ``slices`` (narration slices). Each summary carries a facts list.
        summaries = v["passes"]["1"]["summaries"]
        if summaries and "facts" not in summaries[0]:
            raise ValueError(
                f"EgoLife manifest is segmentation-only (summaries lack "
                f"`facts`). Description-quality eval requires the full "
                f"manifest; rehydrate via `python scripts/data/egolife/"
                f"build_manifest.py --public-manifest <path>` + "
                f"`translate_captions.py` + `annotate_manifest.py` to "
                f"populate `text_en`/`text_zh`/`facts`."
            )
        raw = [
            (fact, float(sm["start"]), float(sm["end"]))
            for sm in summaries
            for fact in sm.get("facts", []) or []
        ]
        passes = {"1": atoms_from_pass(raw)}
        # Moments is present but empty for EgoLife; honor whatever's there
        # so the code stays uniform.
        moments = atoms_from_moments(
            [
                (m["label"], float(m["start"]), float(m["end"]))
                for m in v.get("moments", []) or []
            ]
        )
        return GroundTruth(passes=passes, moments=moments)

    def sc_lists_from_gt(self, manifest: dict, uid: str) -> list[dict]:
        v = next(x for x in manifest["videos"] if x["uid"] == uid)
        out: list[dict] = []
        p1 = v.get("passes", {}).get("1", {})
        for si, sm in enumerate(p1.get("summaries", [])):
            facts = [f for f in (sm.get("facts") or []) if isinstance(f, str) and f]
            if not facts:
                continue
            out.append({
                "key": f"p1_summary{si}",
                "pass": "1",
                "summary_idx": si,
                "start": sm.get("start"),
                "end": sm.get("end"),
                "facts": facts,
            })
        return out
