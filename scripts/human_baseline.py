# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Unified human-baseline calculator: segmentation + description sides.

For datasets with 2 annotator passes (currently Ego4D), computes the
inter-annotator baseline for segmentation (bF1, frame_acc, event_f1,
event_error_rate — CPU, direction-A + direction-B pooled) and
description (summary_recall/precision, moments_recall, self_consistency
via BART-MNLI — 1 GPU). Writes ``<output-root>/human_baseline/<dataset>/
human_baseline.json`` plus per-uid traces. Idempotent. Skip flags:
``--skip-segmentation``, ``--skip-description``, ``--skip-self-consistency``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.eval._json import NumpyEncoder as _NumpyEncoder
from long_audio.eval.description.datasets.ego4d import (
    Ego4DAdapter as Ego4DDescAdapter,
)
from long_audio.eval.description.nli import BartMNLI
from long_audio.eval.description.pipeline import evaluate_human_baseline
from long_audio.eval.description.self_consistency import (
    evaluate_self_consistency_uid,
    micro_percents,
)
from long_audio.eval.segmentation import (
    aggregate,
    evaluate_human_baseline_segmentation,
    get_adapter,
)


class MicroAccumulator:
    def __init__(self):
        self.n_covered = 0
        self.n_targets = 0

    def add_from_raw(self, covered: int, targets: int) -> None:
        self.n_covered += covered
        self.n_targets += targets

    @property
    def ratio(self) -> float:
        return self.n_covered / self.n_targets if self.n_targets else 0.0

    def as_dict(self) -> dict:
        return {"covered": self.n_covered, "denom": self.n_targets, "ratio": self.ratio}


# ---- Segmentation-side human baseline -----------------------------------


def compute_segmentation_human_baseline(
    *, adapter, manifest_path: Path, out_dir: Path,
    audio_duration_override_by_uid: dict[str, float] | None = None,
    max_uids: int | None = None,
) -> dict:
    """Compute segmentation HB. Writes ``segmentation.json``. Returns agg dict.

    ``audio_duration_override_by_uid`` (optional): use the actual inference-time
    audio duration for each uid (from a previous run's per-uid summary.json).
    If None, uses the manifest ``duration`` — appropriate for GT-only baselines
    where we don't want the score to depend on any specific inference run.
    """
    manifest = adapter.load_manifest(manifest_path)
    uids = adapter.all_eval_uids(manifest_path)
    if max_uids:
        uids = uids[:max_uids]
    cfg = adapter.metric_config()
    print(f"[hb-seg] {len(uids)} uids  boundary_tolerance_s={cfg.boundary_tolerance_s}",
          flush=True)

    per_datapoint: list[dict] = []
    n_skipped = 0
    for uid in uids:
        video = adapter.video_record(manifest, uid)
        if video is None:
            n_skipped += 1
            continue
        # Use manifest duration by default — human baseline is a GT-only property.
        duration = float(video.duration)
        if audio_duration_override_by_uid and uid in audio_duration_override_by_uid:
            duration = float(audio_duration_override_by_uid[uid])
        try:
            per_datapoint.extend(
                evaluate_human_baseline_segmentation(
                    uid=uid,
                    video={"uid": uid, "duration": video.duration, "passes": video.passes},
                    audio_duration_s=duration,
                    config=cfg,
                )
            )
        except ValueError as e:
            print(f"[hb-seg] uid={uid} skipped: {e}", flush=True)
            n_skipped += 1

    agg = aggregate(
        per_datapoint, model_name="human_baseline",
        boundary_tolerance_s=cfg.boundary_tolerance_s,
    )
    agg["n_uids_skipped"] = n_skipped
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segmentation.json").write_text(json.dumps(agg, indent=2, cls=_NumpyEncoder))
    return agg


# ---- Description-side human baseline ------------------------------------


def compute_description_human_baseline(
    *, manifest_path: Path, out_dir: Path,
    max_uids: int | None = None, nli_batch_size: int = 64,
    sc_entail_threshold: float = 0.9, skip_self_consistency: bool = False,
) -> dict:
    """Compute description HB (needs BART-MNLI on GPU). Writes ``description.json``."""
    adapter = Ego4DDescAdapter()
    if not adapter.has_human_baseline:
        raise SystemExit(f"{adapter.name} adapter reports has_human_baseline=False")
    manifest = json.loads(Path(manifest_path).read_text())
    # description adapter's all_eval_uids takes the parsed manifest dict
    # (unlike the segmentation adapter which takes a Path).
    uids = adapter.all_eval_uids(manifest_path)
    if max_uids:
        uids = uids[:max_uids]
    print(f"[hb-desc] {len(uids)} uids, loading BART-MNLI …", flush=True)
    nli = BartMNLI(batch_size=nli_batch_size)
    print(f"[hb-desc] NLI ready on {nli.device}", flush=True)

    hb_acc = {
        "summary_recall": MicroAccumulator(),
        "summary_precision": MicroAccumulator(),
        "moments_recall": MicroAccumulator(),
    }
    hb_sc_total = {"n_facts": 0, "n_redundant": 0}
    hb_sc_per_pass: dict[str, dict[str, int]] = {
        "1": {"n_facts": 0, "n_redundant": 0},
        "2": {"n_facts": 0, "n_redundant": 0},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_root = out_dir / "per_uid_description"
    detail_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, uid in enumerate(uids):
        gt = adapter.load_gt_for_uid(manifest, uid)
        counts, trace_rows = evaluate_human_baseline(gt=gt, nli=nli, uid=uid)
        cb = counts["combined"]
        hb_acc["summary_recall"].add_from_raw(cb["recall_covered"], cb["recall_denom"])
        hb_acc["summary_precision"].add_from_raw(cb["precision_covered"], cb["precision_denom"])
        mc = counts["moments"]["combined"]
        hb_acc["moments_recall"].add_from_raw(
            mc["moments_recall_covered"], mc["n_moments"])

        uid_dir = detail_root / uid
        uid_dir.mkdir(parents=True, exist_ok=True)
        (uid_dir / "summary_facts.json").write_text(json.dumps(counts, indent=2))
        with (uid_dir / "entailment_trace.jsonl").open("w") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")

        if not skip_self_consistency:
            sc_lists = adapter.sc_lists_from_gt(manifest, uid)
            sc_res = evaluate_self_consistency_uid(
                sc_lists, nli, entail_threshold=sc_entail_threshold,
            )
            (uid_dir / "self_consistency.json").write_text(json.dumps({
                "per_list": sc_res.per_list,
                "total": sc_res.total,
                **micro_percents(sc_res.total),
            }, indent=2))
            for k in hb_sc_total:
                hb_sc_total[k] += sc_res.total[k]
            for entry in sc_res.per_list:
                p_ = entry.get("pass")
                if p_ in hb_sc_per_pass:
                    hb_sc_per_pass[p_]["n_facts"] += entry["n_facts"]
                    hb_sc_per_pass[p_]["n_redundant"] += entry["n_redundant"]

        if (i + 1) % 10 == 0 or (i + 1) == len(uids):
            dt = time.time() - t0
            print(f"[hb-desc] {i + 1}/{len(uids)} uids  elapsed={dt:.0f}s "
                  f"summary_recall={hb_acc['summary_recall'].ratio:.3f} "
                  f"moments_recall={hb_acc['moments_recall'].ratio:.3f}",
                  flush=True)

    agg = {
        "n_uids": len(uids),
        "summary_recall": hb_acc["summary_recall"].as_dict(),
        "summary_precision": hb_acc["summary_precision"].as_dict(),
        "moments_recall": hb_acc["moments_recall"].as_dict(),
    }
    if not skip_self_consistency:
        agg["self_consistency"] = {
            **hb_sc_total,
            **micro_percents(hb_sc_total),
            "per_pass": {
                pk: {**hb_sc_per_pass[pk], **micro_percents(hb_sc_per_pass[pk])}
                for pk in hb_sc_per_pass
            },
            "n_uids": len(uids),
        }
    (out_dir / "description.json").write_text(json.dumps(agg, indent=2, cls=_NumpyEncoder))
    return agg


# ---- Top-level driver ---------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ego4d"], default="ego4d",
                   help="Only ego4d has 2 annotator passes (EgoLife/SINS are single-annotator).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Defaults to the dataset adapter's default_manifest_path.")
    p.add_argument("--output-root", type=Path, default=Path("./runs/hb"))
    p.add_argument("--skip-segmentation", action="store_true")
    p.add_argument("--skip-description", action="store_true")
    p.add_argument("--skip-self-consistency", action="store_true",
                   help="Description side only; SC is the slowest NLI part.")
    p.add_argument("--max-uids", type=int, default=None,
                   help="Cap uids (smoke).")
    p.add_argument("--nli-batch-size", type=int, default=64)
    p.add_argument("--sc-entail-threshold", type=float, default=0.9)
    args = p.parse_args()

    seg_adapter = get_adapter(args.dataset)
    manifest_path = args.manifest or seg_adapter.default_manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    out_dir = args.output_root / "human_baseline" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[human_baseline] dataset={args.dataset}  manifest={manifest_path}",
          flush=True)
    print(f"[human_baseline] out_dir={out_dir}", flush=True)

    combined: dict = {"dataset": args.dataset, "manifest": str(manifest_path)}

    if not args.skip_segmentation:
        print("\n=== SEGMENTATION human baseline ===", flush=True)
        seg_agg = compute_segmentation_human_baseline(
            adapter=seg_adapter, manifest_path=manifest_path,
            out_dir=out_dir, max_uids=args.max_uids,
        )
        combined["segmentation"] = seg_agg
        _print_seg(seg_agg)

    if not args.skip_description:
        print("\n=== DESCRIPTION human baseline ===", flush=True)
        desc_agg = compute_description_human_baseline(
            manifest_path=manifest_path, out_dir=out_dir,
            max_uids=args.max_uids, nli_batch_size=args.nli_batch_size,
            sc_entail_threshold=args.sc_entail_threshold,
            skip_self_consistency=args.skip_self_consistency,
        )
        combined["description"] = desc_agg
        _print_desc(desc_agg)

    (out_dir / "human_baseline.json").write_text(json.dumps(combined, indent=2, cls=_NumpyEncoder))
    print(f"\n[human_baseline] wrote {out_dir / 'human_baseline.json'}", flush=True)
    return 0


def _print_seg(agg: dict) -> None:
    m = agg["macro"]
    mi = agg["micro"]
    print(f"  n_videos={agg['n_videos']}  n_data_points={agg['n_data_points']}",
          flush=True)
    print(f"  MACRO  bF1={m['boundary_f1']:.3f}  frame_acc={m['frame_accuracy']:.3f}  "
          f"event_f1={m['event_f1']:.3f}  event_error_rate={m['event_error_rate']:.3f}",
          flush=True)
    print(f"  MICRO  bF1={mi['boundary_f1']:.3f}  frame_acc={mi['frame_accuracy']:.3f}  "
          f"event_f1={mi['event_f1']:.3f}", flush=True)


def _print_desc(agg: dict) -> None:
    print(f"  n_uids={agg['n_uids']}", flush=True)
    print(f"  summary_recall    = {agg['summary_recall']['ratio']:.3f} "
          f"({agg['summary_recall']['covered']}/{agg['summary_recall']['denom']})", flush=True)
    print(f"  summary_precision = {agg['summary_precision']['ratio']:.3f} "
          f"({agg['summary_precision']['covered']}/{agg['summary_precision']['denom']})", flush=True)
    print(f"  moments_recall    = {agg['moments_recall']['ratio']:.3f} "
          f"({agg['moments_recall']['covered']}/{agg['moments_recall']['denom']})", flush=True)
    if "self_consistency" in agg:
        sc = agg["self_consistency"]
        print(f"  redundancy (combined) = {sc['pct_redundant']:.3f} "
              f"({sc['n_redundant']}/{sc['n_facts']} atoms)", flush=True)
        for pk, pp in sc["per_pass"].items():
            print(f"    pass {pk}: {pp['pct_redundant']:.3f} "
                  f"({pp['n_redundant']}/{pp['n_facts']})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
