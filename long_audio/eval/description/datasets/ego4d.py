"""Ego4D description-eval adapter.

Two annotator passes ("1" and "2"); per-slice narration facts. Has
moments GT (~24 moments per uid). Human baseline enabled.
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


class Ego4DAdapter:
    name = "ego4d"
    default_manifest_path = (
        _REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
    )
    has_human_baseline = True

    def load_manifest(self, manifest_path: Path) -> dict:
        return json.loads(Path(manifest_path).read_text())

    def all_eval_uids(self, manifest_path: Path) -> list[str]:
        from long_audio.datasets.ego4d.dataset import Ego4DDataset
        ds = Ego4DDataset.eval_set(manifest_path, split="test")
        return sorted({it.id.removeprefix("ego4d_") for it in ds.iter_items()})

    def load_gt_for_uid(self, manifest: dict, uid: str) -> GroundTruth:
        v = next(x for x in manifest["videos"] if x["uid"] == uid)
        passes: dict[str, list] = {}
        for pkey in ("1", "2"):
            slices = v["passes"][pkey]["slices"]
            if slices and "facts" not in slices[0]:
                raise ValueError(
                    f"Ego4D manifest is segmentation-only (slices lack `facts`). "
                    f"Description-quality eval requires the full annotated "
                    f"manifest; rehydrate via `python scripts/data/ego4d/"
                    f"build_manifest.py --public-manifest <path>` then run "
                    f"`annotate_manifest.py` to populate `facts`."
                )
            raw = [
                (fact, float(slc["start"]), float(slc["end"]))
                for slc in slices
                for fact in slc["facts"]
            ]
            passes[pkey] = atoms_from_pass(raw)
        moments = atoms_from_moments(
            [(m["label"], float(m["start"]), float(m["end"])) for m in v.get("moments", [])]
        )
        return GroundTruth(passes=passes, moments=moments)

    def sc_lists_from_gt(self, manifest: dict, uid: str) -> list[dict]:
        v = next(x for x in manifest["videos"] if x["uid"] == uid)
        out: list[dict] = []
        for pkey in ("1", "2"):
            if pkey not in v.get("passes", {}):
                continue
            for si, s in enumerate(v["passes"][pkey].get("slices", [])):
                facts = [f for f in (s.get("facts") or []) if isinstance(f, str) and f]
                if not facts:
                    continue
                out.append({
                    "key": f"p{pkey}_slice{si}",
                    "pass": pkey,
                    "slice_idx": si,
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "facts": facts,
                })
        return out
