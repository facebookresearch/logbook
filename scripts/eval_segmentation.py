"""Unified segmentation eval CLI — Ego4D, EgoLife, or SINS.

Dispatches on ``--dataset {ego4d,egolife,sins}``:

  * ``ego4d``  — scores a run dir against per-pass Ego4D GT (2 passes,
    2 datapoints per uid). Reads ``<run_root>/<uid>/summary.json``
    + ``chunk_*.json``. Writes ``eval_segmentation.json``.
  * ``egolife``— scores a run dir against single-pass EgoLife GT (1
    datapoint per uid). Same on-disk shape.
  * ``sins``   — scores ``<run_dir>/summary.json`` against the SINS
    single-file manifest. Writes ``eval_scores`` back into
    ``summary.json`` (legacy behavior).

Usage:
    python scripts/eval_segmentation.py --dataset ego4d   runs/<UTC>/e2e_<model>/ego4d/
    python scripts/eval_segmentation.py --dataset egolife runs/<UTC>/e2e_<model>/egolife/
    python scripts/eval_segmentation.py --dataset sins    runs/<UTC>/e2e_<model>/sins/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.eval._json import NumpyEncoder as _NumpyEncoder
from long_audio.eval.segmentation import (
    aggregate,
    evaluate_human_baseline_segmentation,
    format_headline,
    format_pred_missing_line,
    get_adapter,
    score_video,
    stitch_chunk_predictions,
)


# ---- Per-video (Ego4D + EgoLife) driver --------------------------------


def _run_per_video(args, adapter) -> int:
    """Score a per-video run dir (Ego4D / EgoLife). Writes eval_segmentation.json."""
    run_root = Path(args.run_root)
    index_path = run_root / "run_index.json"
    if not index_path.exists():
        raise SystemExit(f"run_index.json not found at {index_path}")
    index = json.loads(index_path.read_text())

    manifest_path = Path(args.manifest) if args.manifest else adapter.default_manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found at {manifest_path}")
    manifest = adapter.load_manifest(manifest_path)

    cfg = adapter.metric_config()
    per_datapoint: list[dict] = []
    for vinfo in index["videos"]:
        uid = vinfo.get("uid") or vinfo["id"].removeprefix(adapter.id_prefix)
        video = adapter.video_record(manifest, uid)
        if video is None:
            print(f"[eval] uid {uid} not in manifest; skipping", flush=True)
            continue
        uid_dir = run_root / uid
        summary_path = uid_dir / "summary.json"
        if not summary_path.exists():
            print(f"[eval] {summary_path} missing; skipping", flush=True)
            continue
        summary = json.loads(summary_path.read_text())
        if "audio_duration_s" not in summary:
            raise SystemExit(
                f"{summary_path}: missing 'audio_duration_s'. Pre-Unit-11 "
                "inference output is no longer supported — re-run inference."
            )
        predictions = stitch_chunk_predictions(uid_dir)
        n_chunks = len(summary.get("chunk_files") or [])
        per_datapoint.extend(
            score_video(
                uid=uid,
                predictions=predictions,
                video={"uid": uid, "duration": video.duration, "passes": video.passes},
                audio_duration_s=float(summary["audio_duration_s"]),
                n_chunks=n_chunks,
                config=cfg,
            )
        )

    model_name = index.get("model") or index.get("describe_model", "unknown")
    out = aggregate(
        per_datapoint, model_name,
        boundary_tolerance_s=cfg.boundary_tolerance_s,
    )

    # Inter-annotator human baseline for datasets with ≥2 passes (Ego4D).
    # Reuses the same per-datapoint shape + aggregate() so metrics are
    # directly comparable to model cells. Direction A (pass1=GT, pass2=pred)
    # and B (swap) are pooled at the aggregate level (macro over datapoints,
    # micro over event counts) — matches summary-side human_baseline in
    # scripts/eval_descriptions.py.
    if getattr(adapter, "has_human_baseline", False):
        hb_datapoints: list[dict] = []
        for vinfo in index["videos"]:
            uid = vinfo.get("uid") or vinfo["id"].removeprefix(adapter.id_prefix)
            video = adapter.video_record(manifest, uid)
            if video is None:
                continue
            uid_dir = run_root / uid
            summary_path = uid_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            try:
                hb_datapoints.extend(
                    evaluate_human_baseline_segmentation(
                        uid=uid,
                        video={"uid": uid, "duration": video.duration, "passes": video.passes},
                        audio_duration_s=float(summary["audio_duration_s"]),
                        config=cfg,
                    )
                )
            except ValueError as e:
                # e.g. missing pass2 for a uid — log and skip
                print(f"[eval] uid={uid} human_baseline skipped: {e}", flush=True)
        if hb_datapoints:
            hb_agg = aggregate(
                hb_datapoints, model_name="human_baseline",
                boundary_tolerance_s=cfg.boundary_tolerance_s,
            )
            out["human_baseline"] = hb_agg
            print(f"[eval] human_baseline: {hb_agg['n_videos']} videos, "
                  f"{hb_agg['n_data_points']} datapoints (both directions pooled)",
                  flush=True)

    output_path = (
        Path(args.output) if args.output else (run_root / "eval_segmentation.json")
    )
    output_path.write_text(json.dumps(out, indent=2, cls=_NumpyEncoder))

    print(f"[eval] Wrote {output_path}", flush=True)
    print(
        f"  n_videos={out['n_videos']}  n_data_points={out['n_data_points']}",
        flush=True,
    )
    print(format_headline(out), flush=True)
    print(f"  Total chunks: {out['total_chunks']:,}", flush=True)
    print(format_pred_missing_line(out["pred_missing"]), flush=True)
    return 0


# ---- Single-file (SINS) driver ---------------------------------------


def _run_single_file(args, adapter) -> int:
    """Score a SINS run dir (single-file dataset). Writes eval_scores back
    into <run_dir>/summary.json."""
    from long_audio.eval.metrics import summarize

    run_dir = Path(args.run_root)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"summary.json not found in {run_dir}")
    summary = json.loads(summary_path.read_text())
    timeline_end = float(summary["audio_duration_s"])

    predictions = stitch_chunk_predictions(run_dir)

    manifest_path = Path(args.manifest) if args.manifest else adapter.default_manifest_path
    manifest = adapter.load_manifest(manifest_path)
    video = adapter.video_record(manifest, adapter.all_eval_uids(manifest_path)[0])
    if video is None:
        raise SystemExit(f"{adapter.name} adapter returned no video record")

    cfg = adapter.metric_config()
    # For SINS, GT is a single flat list — same shape as one pass's actions.
    gt_events = [
        {"label": a["event"], "start": a["start"], "end": a["end"]}
        for a in video.passes["1"]["actions"]
    ]
    scores = summarize(
        predictions, gt_events, config=cfg, duration_s=timeline_end,
    )

    # macro==micro for a single data point; report per-metric defaults uniformly.
    fla = scores["frame_level_accuracy"]
    pred_missing = {
        "macro_rate":    fla["pred_missing_rate"],
        "micro_rate":    fla["pred_missing_rate"],
        "total_regions": fla["n_pred_missing_regions"],
    }
    total_chunks = len(summary.get("chunk_files") or [])

    summary["eval_scores"] = scores
    summary["pred_missing"] = pred_missing
    summary_path.write_text(json.dumps(summary, indent=2, cls=_NumpyEncoder))

    print(
        f"[eval_run] Scored {len(predictions)} predicted segments vs "
        f"{len(gt_events)} ground-truth segments over "
        f"{timeline_end / 3600:.2f} h."
    )
    print(
        f"  DEFAULT  bF1={scores['boundary_f1']['f1']:.3f} (macro)  "
        f"frame={scores['frame_level_accuracy']['overall_accuracy']:.3f} (micro)  "
        f"evF1={scores['event_based_f1']['f1']:.3f} (macro)  "
        f"evER={scores['event_error_rate']['error_rate']:.3f} (macro)",
        flush=True,
    )
    print(f"  Total chunks: {total_chunks:,}", flush=True)
    print(
        f"  Pred-missing (eval gap-fill): "
        f"{pred_missing['macro_rate'] * 100:.1f}% macro | "
        f"{pred_missing['micro_rate'] * 100:.1f}% micro | "
        f"{pred_missing['total_regions']:,} regions",
        flush=True,
    )
    return 0


# ---- CLI ---------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=["ego4d", "egolife", "sins"],
        help="Which dataset's shape to use for GT loading + metric config.",
    )
    parser.add_argument(
        "run_root",
        help="For ego4d/egolife: dir with run_index.json + per-uid subdirs. "
             "For sins: dir with summary.json.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Override the dataset's default manifest path.",
    )
    parser.add_argument(
        "--output", default=None,
        help="ego4d/egolife: where to write eval_segmentation.json "
             "(default: <run_root>/eval_segmentation.json). Ignored for sins.",
    )
    args = parser.parse_args()

    adapter = get_adapter(args.dataset)
    if args.dataset == "sins":
        return _run_single_file(args, adapter)
    return _run_per_video(args, adapter)


if __name__ == "__main__":
    sys.exit(main())
