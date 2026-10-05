# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Description-quality eval — full test-split, arbitrary cell set.

Supports Ego4D (2 passes + moments + human baseline) and EgoLife (1 pass,
no moments, no human baseline) via ``--dataset``. Dataset shape lives in
:mod:`long_audio.eval.description.datasets`. Metrics: Summary
Recall/Precision per pass, Moments Recall (Ego4D only), and optional
Self-consistency (BART-MNLI). Writes per-cell/per-uid AFG caches +
entailment traces plus a corpus-wide ``eval_descriptions.json``. Cells
are declared via ``--cells-config`` (JSON list of
``{name, path_template}``); ``{uid}`` is substituted at load time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from long_audio.eval.description import (
    AFGConfig,
    BartMNLI,
    DEFAULT_ENTAIL_THRESHOLD,
    MicroAccumulator,
    NLI_MODEL_ID,
    P_ENTAIL_THRESHOLD,
    evaluate_cell_uid,
    evaluate_human_baseline,
    evaluate_self_consistency_uid,
    micro_percents,
    pred_atoms_from_facts,
    run_afg,
)
from long_audio.eval.description.datasets import get_adapter


# ---- Cells registry (loaded from --cells-config JSON) ---------------------


def load_cells_config(path: Path) -> list[dict]:
    """Parse and validate the cells-config JSON.

    Schema: ``[{"name": str, "path_template": str}, ...]`` with ``{uid}``
    substring in each ``path_template``. Order is preserved (drives the
    output ordering).
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"cells-config must be a JSON list, got {type(data).__name__}")
    seen = set()
    for i, c in enumerate(data):
        if not isinstance(c, dict) or "name" not in c or "path_template" not in c:
            raise ValueError(
                f"cells-config[{i}] must have 'name' and 'path_template', got {c!r}"
            )
        if "{uid}" not in c["path_template"]:
            raise ValueError(
                f"cells-config[{i}].path_template must contain '{{uid}}', got {c['path_template']!r}"
            )
        if c["name"] in seen:
            raise ValueError(f"duplicate cell name: {c['name']!r}")
        seen.add(c["name"])
    return data


def cell_uid_dir(cell: dict, uid: str) -> Path:
    return Path(cell["path_template"].format(uid=uid))


# ---- I/O helpers (ego4d-shape-aware, path-agnostic) -----------------------


def load_predicted_segments(uid_dir: Path) -> list[dict]:
    """Flatten all chunk_*.json ``segments_abs`` for one cell/uid.

    Returns list of ``{chunk_idx, seg_idx_in_chunk, label, start, end, description}``.
    """
    segs = []
    for f in sorted(uid_dir.glob("chunk_*.json")):
        c = json.loads(f.read_text())
        chunk_idx = c["chunk_index"]
        for si, s in enumerate(c["segments_abs"]):
            if not s.get("description"):
                continue
            segs.append({
                "chunk_idx": chunk_idx,
                "seg_idx_in_chunk": si,
                "label": s["label"],
                "start": float(s["start"]),
                "end": float(s["end"]),
                "description": s["description"],
            })
    return segs


def _sc_lists_from_predicted_facts(pred_facts_doc: dict) -> list[dict]:
    """Per-segment facts lists from a ``predicted_facts.json`` document, in
    the shape ``evaluate_self_consistency_uid`` expects
    (``{"key": ..., "facts": [...]}``).

    Empty per-segment lists are skipped (SC score is undefined for n<2).
    """
    out: list[dict] = []
    for si, seg in enumerate(pred_facts_doc.get("segments", [])):
        facts = [f for f in (seg.get("facts") or []) if isinstance(f, str) and f]
        if not facts:
            continue
        out.append({
            "key": f"seg{si}",
            "seg_idx": si,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "facts": facts,
        })
    return out


def _sc_score_and_write(
    lists: list[dict],
    out_dir: Path,
    nli: BartMNLI,
    entail_threshold: float,
):
    """Score self-consistency for one uid, write ``self_consistency.json``,
    return the full ``UidResult`` (total + per_list) for corpus-level
    accumulation. Callers that only need totals use ``.total``; callers that
    need per-pass or per-list breakdown iterate ``.per_list``."""
    res = evaluate_self_consistency_uid(
        lists, nli, entail_threshold=entail_threshold,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "per_list": res.per_list,
        "total": res.total,
        **micro_percents(res.total),
    }
    (out_dir / "self_consistency.json").write_text(json.dumps(doc, indent=2))
    return res


# ---- Subcommand: eval ----------------------------------------------------


def cmd_eval(args) -> int:
    adapter = get_adapter(args.dataset)
    manifest_path = Path(args.manifest) if args.manifest else adapter.default_manifest_path
    manifest = adapter.load_manifest(manifest_path)
    cells = load_cells_config(Path(args.cells_config))
    cell_names = [c["name"] for c in cells]
    uids = args.uids if args.uids else adapter.all_eval_uids(manifest_path)
    print(f"[eval] dataset={args.dataset} cells={cell_names} uids={len(uids)}", flush=True)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cells_dir = out_root / "cells"
    hb_dir = out_root / "human_baseline"

    # -- Gather all descriptions for AFG --
    per_cell_uid_segs: dict[tuple[str, str], list[dict]] = {}
    all_descs: list[tuple[tuple, str]] = []
    for cell in cells:
        for uid in uids:
            uid_dir = cell_uid_dir(cell, uid)
            if not uid_dir.exists():
                print(f"[eval] WARN missing dir: {uid_dir}", flush=True)
                continue
            segs = load_predicted_segments(uid_dir)
            per_cell_uid_segs[(cell["name"], uid)] = segs
            for s in segs:
                key = (cell["name"], uid, s["chunk_idx"], s["seg_idx_in_chunk"])
                all_descs.append((key, s["description"]))
    print(f"[eval] collected {len(all_descs)} predicted-description-atoms across "
          f"{len(per_cell_uid_segs)} (cell,uid) pairs", flush=True)

    # -- Run AFG --
    afg_cfg = AFGConfig(
        model_id=args.afg_model_id,
        tensor_parallel_size=args.afg_tp,
        max_tokens=args.afg_max_tokens,
        seed=args.afg_seed,
    )
    afg_out = run_afg(all_descs, afg_cfg)

    # -- Write predicted_facts.json per (cell, uid) --
    for (cell_name, uid), segs in per_cell_uid_segs.items():
        target_dir = cells_dir / cell_name / uid
        target_dir.mkdir(parents=True, exist_ok=True)
        pred_facts_doc = {
            "afg_config": {
                "model_id": afg_cfg.model_id,
                "temperature": afg_cfg.temperature,
                "max_tokens": afg_cfg.max_tokens,
                "seed": afg_cfg.seed,
                "demo_source": "long_audio/datasets/ego4d/factscore/demons.json",
                "demo_retriever": "BM25 top-1",
            },
            "cell": cell_name,
            "uid": uid,
            "segments": [
                {**s, "facts": afg_out.get((cell_name, uid, s["chunk_idx"], s["seg_idx_in_chunk"]), [])}
                for s in segs
            ],
        }
        (target_dir / "predicted_facts.json").write_text(
            json.dumps(pred_facts_doc, indent=2)
        )
    print(f"[eval] wrote predicted_facts.json for {len(per_cell_uid_segs)} (cell,uid) pairs",
          flush=True)

    # -- Load BART-MNLI --
    print(f"[nli] loading {NLI_MODEL_ID} …", flush=True)
    nli = BartMNLI(batch_size=args.nli_batch_size)
    print(f"[nli] ready on {nli.device}", flush=True)

    # -- Per-cell per-uid metric compute --
    global_accs = {
        cell_name: {
            "summary_recall": MicroAccumulator(),
            "summary_precision": MicroAccumulator(),
            "moments_recall": MicroAccumulator(),
        }
        for cell_name in cell_names
    }
    global_counts = {
        cell_name: {"n_segments": 0, "n_pred_atoms": 0, "n_uids": 0}
        for cell_name in cell_names
    }
    # Per-cell corpus totals for self-consistency (redundant/facts).
    global_sc = {
        cell_name: {"n_facts": 0, "n_redundant": 0}
        for cell_name in cell_names
    }
    for (cell_name, uid), _segs in per_cell_uid_segs.items():
        pred_facts_doc = json.loads(
            (cells_dir / cell_name / uid / "predicted_facts.json").read_text()
        )
        pred_atoms = pred_atoms_from_facts(pred_facts_doc)
        global_counts[cell_name]["n_segments"] += len(pred_facts_doc["segments"])
        global_counts[cell_name]["n_pred_atoms"] += len(pred_atoms)
        global_counts[cell_name]["n_uids"] += 1
        gt = adapter.load_gt_for_uid(manifest, uid)
        counts, trace_rows = evaluate_cell_uid(
            pred_atoms=pred_atoms, gt=gt, nli=nli, uid=uid,
        )
        # Iterate whichever passes the dataset provides (Ego4D: '1','2'; EgoLife: '1').
        for pkey, pc in counts["passes"].items():
            global_accs[cell_name]["summary_recall"].add_from_raw(
                pc["summary_recall_covered"], pc["gt_n_atoms"])
            global_accs[cell_name]["summary_precision"].add_from_raw(
                pc["summary_precision_supported"], pc["n_pred_atoms_for_precision_denom"])
        mc = counts["moments"]
        global_accs[cell_name]["moments_recall"].add_from_raw(
            mc["moments_recall_covered"], mc["n_moments"])
        target_dir = cells_dir / cell_name / uid
        (target_dir / "summary_facts.json").write_text(json.dumps(counts, indent=2))
        with (target_dir / "entailment_trace.jsonl").open("w") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        # Self-consistency (redundancy) on the same per-segment facts lists.
        # Reuses the already-loaded BART-MNLI instance.
        if not args.skip_self_consistency:
            sc_lists = _sc_lists_from_predicted_facts(pred_facts_doc)
            sc_res = _sc_score_and_write(
                sc_lists, target_dir, nli, args.sc_entail_threshold,
            )
            for k in global_sc[cell_name]:
                global_sc[cell_name][k] += sc_res.total[k]
        p1 = counts["passes"].get("1", {"gt_n_atoms": 0, "summary_recall_covered": 0,
                                        "summary_precision_supported": 0})
        print(f"[metric] {cell_name}/{uid}: pred_atoms={counts['pred_n_atoms']} "
              f"p1_gt={p1['gt_n_atoms']} "
              f"p1_recall_cov={p1['summary_recall_covered']} "
              f"p1_prec_sup={p1['summary_precision_supported']} "
              f"moments_cov={mc['moments_recall_covered']}/{mc['n_moments']}",
              flush=True)

    # -- Human baseline per uid (Ego4D only — needs 2 annotator passes) --
    hb_acc = {
        "recall": MicroAccumulator(),
        "precision": MicroAccumulator(),
        "moments_recall": MicroAccumulator(),
    }
    hb_sc_total = {"n_facts": 0, "n_redundant": 0}
    # Per-pass breakdown for the human baseline SC — surfaces each
    # annotator's own atom count + redundancy rate so we can see if one
    # annotator is systematically more verbose / more redundant.
    hb_sc_per_pass: dict[str, dict[str, int]] = {
        "1": {"n_facts": 0, "n_redundant": 0},
        "2": {"n_facts": 0, "n_redundant": 0},
    }
    if adapter.has_human_baseline:
        for uid in uids:
            gt = adapter.load_gt_for_uid(manifest, uid)
            counts, trace_rows = evaluate_human_baseline(gt=gt, nli=nli, uid=uid)
            cb = counts["combined"]
            hb_acc["recall"].add_from_raw(cb["recall_covered"], cb["recall_denom"])
            hb_acc["precision"].add_from_raw(cb["precision_covered"], cb["precision_denom"])
            mc = counts["moments"]["combined"]
            hb_acc["moments_recall"].add_from_raw(
                mc["moments_recall_covered"], mc["n_moments"])
            target_dir = hb_dir / uid
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "summary_facts.json").write_text(json.dumps(counts, indent=2))
            with (target_dir / "entailment_trace.jsonl").open("w") as fh:
                for row in trace_rows:
                    fh.write(json.dumps(row) + "\n")
            if not args.skip_self_consistency:
                sc_lists = adapter.sc_lists_from_gt(manifest, uid)
                sc_res = _sc_score_and_write(
                    sc_lists, target_dir, nli, args.sc_entail_threshold,
                )
                for k in hb_sc_total:
                    hb_sc_total[k] += sc_res.total[k]
                # Per-pass roll-up: each sc_lists entry carries a "pass" tag
                # from ``sc_lists_from_gt`` (ego4d.py). Iterate ``per_list``
                # and accumulate into hb_sc_per_pass[pass].
                for entry in sc_res.per_list:
                    p = entry.get("pass")
                    if p in hb_sc_per_pass:
                        hb_sc_per_pass[p]["n_facts"] += entry["n_facts"]
                        hb_sc_per_pass[p]["n_redundant"] += entry["n_redundant"]
            print(f"[human-baseline] {uid}: recall={cb['recall_covered']}/{cb['recall_denom']} "
                  f"precision={cb['precision_covered']}/{cb['precision_denom']} "
                  f"moments_cov={mc['moments_recall_covered']}/{mc['n_moments']}",
                  flush=True)
    else:
        print(f"[human-baseline] SKIP — dataset {adapter.name!r} is single-annotator "
              "(no pass2, no human baseline)", flush=True)

    # -- Aggregate + config dump --
    def r(acc: MicroAccumulator) -> dict:
        return {"covered": acc.n_covered, "denom": acc.n_targets, "ratio": acc.ratio}
    agg = {
        "dataset": adapter.name,
        "cells": {
            cn: {
                **{k: r(v) for k, v in global_accs[cn].items()},
                "counts": global_counts[cn],
            }
            for cn in cell_names
        },
    }
    if adapter.has_human_baseline:
        agg["human_baseline"] = {
            "summary_recall": r(hb_acc["recall"]),
            "summary_precision": r(hb_acc["precision"]),
            "moments_recall": r(hb_acc["moments_recall"]),
            "n_uids": len(uids),
        }
    if not args.skip_self_consistency:
        # Attach self-consistency rollups to the same eval_descriptions.json cells
        # (+ human_baseline for datasets that have one) so downstream
        # readers don't need a separate file.
        for cn in cell_names:
            agg["cells"][cn]["self_consistency"] = {
                **global_sc[cn],
                **micro_percents(global_sc[cn]),
                "n_uids": global_counts[cn]["n_uids"],
            }
        if adapter.has_human_baseline:
            agg["human_baseline"]["self_consistency"] = {
                **hb_sc_total,
                **micro_percents(hb_sc_total),
                "per_pass": {
                    p: {**hb_sc_per_pass[p], **micro_percents(hb_sc_per_pass[p])}
                    for p in hb_sc_per_pass
                },
                "n_uids": len(uids),
            }
    (out_root / "eval_descriptions.json").write_text(json.dumps(agg, indent=2))
    (out_root / "config.json").write_text(json.dumps({
        "dataset": adapter.name,
        "uids": uids,
        "cells": cells,
        "manifest": str(manifest_path),
        "afg": {
            "model_id": afg_cfg.model_id, "tp": afg_cfg.tensor_parallel_size,
            "max_tokens": afg_cfg.max_tokens, "seed": afg_cfg.seed,
        },
        "nli": {"model_id": NLI_MODEL_ID, "threshold": P_ENTAIL_THRESHOLD,
                "batch_size": args.nli_batch_size},
        "self_consistency": {
            "enabled": not args.skip_self_consistency,
            "entail_threshold": args.sc_entail_threshold,
        },
    }, indent=2))
    print()
    print("=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    print(f"\nartifacts in: {out_root}")
    return 0


def _add_eval_args(sub):
    sub.add_argument("--dataset", required=True, choices=["ego4d", "egolife"],
                     help="Which dataset's shape to use for GT loading + "
                          "eval-set filter. SINS is not supported (no "
                          "description GT).")
    sub.add_argument("--manifest", default=None,
                     help="Override the dataset's default manifest path.")
    sub.add_argument("--cells-config", required=True,
                     help="JSON: [{name, path_template with {uid}}, ...]")
    sub.add_argument("--output-root", required=True)
    sub.add_argument("--uids", nargs="+", default=None,
                     help="restrict to a subset (debug); default = all test-split uids from --manifest")
    sub.add_argument("--afg-model-id", default="allenai/OLMo-2-1124-7B-SFT")
    sub.add_argument("--afg-tp", type=int, default=8)
    sub.add_argument("--afg-max-tokens", type=int, default=128)
    sub.add_argument("--afg-seed", type=int, default=42)
    sub.add_argument("--nli-batch-size", type=int, default=32)
    sub.add_argument("--skip-self-consistency", action="store_true",
                     help="Skip inline self-consistency scoring "
                          "(redundancy detection). Default is to score, "
                          "reusing the already-loaded BART-MNLI instance.")
    sub.add_argument("--sc-entail-threshold", type=float,
                     default=DEFAULT_ENTAIL_THRESHOLD,
                     help="Entail-prob threshold above which a fact is "
                          "flagged as redundant.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_eval = sub.add_parser("eval", help="run full-eval on all test-split uids")
    _add_eval_args(p_eval)
    args = parser.parse_args()
    if args.cmd == "eval":
        return cmd_eval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
