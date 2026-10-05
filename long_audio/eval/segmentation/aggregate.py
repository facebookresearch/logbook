# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Segmentation eval aggregation — the shared scoring core.

Dataset-agnostic; per-dataset shape (manifest layout, uid extraction,
pass structure) lives in :mod:`long_audio.eval.segmentation.datasets`.
Consumers:

  * :mod:`scripts.eval_segmentation` — score a run dir (any dataset).
  * :mod:`scripts.eval_finetuned` — score a fine-tuned LoRA checkpoint.
  * :mod:`long_audio.training.evaluate` — periodic in-training val eval.

All three produce byte-identical ``eval_segmentation.json``.

Scoring contract:
  * Each ``(video, pass)`` is a SEPARATE eval data point.
    Single-pass datasets (EgoLife, SINS) collapse to one data point per video.
  * ``MetricConfig`` (bF1 tolerance, event-exclude labels) is per-dataset —
    each adapter defines its own via :meth:`DatasetAdapter.metric_config`.
  * Macro = per-data-point mean; micro = pool tp/fp/fn corpus-wide.
  * The eval window is always ``[0, gt_end]`` where ``gt_end`` is the
    last GT event's end. Predictions are clipped to that window; trailing
    audio beyond GT coverage is excluded from scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from long_audio.eval.metrics import MetricConfig, summarize

#: Boundary-F1 timing tolerance for Ego4D / EgoLife (2.5 min = half the
#: 5-min slice quantization). New callers should read the tolerance from
#: the dataset adapter's ``metric_config()``.
EGO4D_BF1_TOLERANCE_S = 150.0


def actions_to_eval_events(actions: list[dict]) -> list[dict]:
    """Convert manifest ``actions`` ({start, end, event}) to eval-suite
    event records ({label, start, end})."""
    return [
        {"label": a["event"], "start": float(a["start"]), "end": float(a["end"])}
        for a in actions
    ]


def clip_predictions(predictions: list[dict], duration_s: float) -> list[dict]:
    """Clip predicted segments to ``[0, duration_s]`` (the GT timeline).

    Predictions can overflow the audio if the runner emitted a timeline at
    the nominal chunk-aligned end (``n_chunks * chunk_minutes * 60``) past
    the audio's actual length. Drop or trim those. Also used to clip GT
    when the audio is truncated below the manifest's ``duration`` field.
    """
    out: list[dict] = []
    for p in predictions:
        st = float(p["start"])
        en = float(p["end"])
        if st >= duration_s:
            continue
        en = min(en, duration_s)
        if en > st:
            out.append({**p, "start": st, "end": en})
    return out


def score_one_pass(
    predictions: list[dict],
    gt_actions: list[dict],
    duration_s: float,
    config: MetricConfig,
) -> dict:
    """Run the four-metric suite for one (video, pass) data point.

    ``duration_s`` is the GT timeline length; both predictions AND GT are
    clipped to it before scoring. Raises ``ValueError`` if GT becomes
    empty after clipping (audio truncated below the first GT action's
    start) — caller should treat as an unrecoverable data problem.
    """
    gt = actions_to_eval_events(gt_actions)
    gt = clip_predictions(gt, duration_s)
    if not gt:
        raise ValueError(
            f"GT actions empty after clipping to duration_s={duration_s} "
            f"(had {len(gt_actions)} pre-clip actions); audio window is "
            "shorter than the first GT action's start."
        )
    clipped = clip_predictions(predictions, duration_s)
    return summarize(clipped, gt, config=config, duration_s=duration_s)


def evaluate_human_baseline_segmentation(
    uid: str,
    video: dict,
    audio_duration_s: float,
    config: MetricConfig,
) -> list[dict]:
    """Inter-annotator human baseline for the segmentation metric suite.

    Requires exactly two GT passes ('1' and '2'). Scores both directions
    (A: pass1=GT, pass2=pred; B: pass2=GT, pass1=pred) and returns two
    datapoint dicts in the same shape as ``score_video`` so the existing
    ``aggregate()`` (macro over datapoints, micro over event counts) works
    unchanged. The two directions are pooled at the aggregate level;
    per-direction detail lives in the ``pass`` field (``1`` or ``2``,
    denoting which side was treated as GT — the "pred" side is the other
    pass).

    Only meaningful for multi-annotator datasets (Ego4D). Adapters that
    advertise ``has_human_baseline = False`` should not call this.
    """
    if set(video.get("passes", {}).keys()) != {"1", "2"}:
        raise ValueError(
            f"human baseline segmentation requires exactly two passes '1' "
            f"and '2', got {sorted(video.get('passes', {}).keys())}"
        )
    manifest_dur = float(video["duration"])
    duration_s = min(manifest_dur, float(audio_duration_s))
    p1_actions = video["passes"]["1"]["actions"]
    p2_actions = video["passes"]["2"]["actions"]
    if not p1_actions or not p2_actions:
        print(
            f"[eval] uid={uid} human_baseline skipped: one pass has empty "
            "GT actions in manifest",
            flush=True,
        )
        return []

    out: list[dict] = []
    for gt_key, pred_key, gt_actions, pred_actions in [
        ("1", "2", p1_actions, p2_actions),
        ("2", "1", p2_actions, p1_actions),
    ]:
        # Use the "pred_key" pass's actions AS predictions. Same shape as
        # model predictions (label/start/end) after actions_to_eval_events.
        pred_events = actions_to_eval_events(pred_actions)
        gt_end = max(float(a["end"]) for a in gt_actions)
        eval_duration = min(gt_end, duration_s)
        scores = score_one_pass(pred_events, gt_actions, eval_duration, config)
        out.append(
            {
                "uid": uid,
                "pass": gt_key,   # which pass was treated as GT this direction
                "duration_s": eval_duration,
                "n_chunks": 0,    # not applicable to inter-annotator baseline
                "n_pred": len(pred_events),
                "n_gt": len(gt_actions),
                "scores": scores,
            }
        )
    return out


def score_video(
    uid: str,
    predictions: list[dict],
    video: dict,
    audio_duration_s: float,
    n_chunks: int,
    config: MetricConfig,
) -> list[dict]:
    """Score every pass of one video; return one datapoint dict per pass.

    Uniform across datasets: iterates ``video["passes"]`` regardless of
    single- vs multi-annotator. Adapter conforms the manifest to this
    shape before calling. Single-pass datasets (EgoLife, SINS-as-video)
    produce exactly one datapoint per video.
    """
    manifest_dur = float(video["duration"])
    duration_s = min(manifest_dur, float(audio_duration_s))
    per_pass: list[dict] = []
    for pass_num, pass_data in video["passes"].items():
        if not pass_data.get("actions"):
            print(
                f"[eval] uid={uid} pass={pass_num} skipped: GT actions empty in manifest",
                flush=True,
            )
            continue
        gt_end = max(float(a["end"]) for a in pass_data["actions"])
        eval_duration = gt_end
        scores = score_one_pass(
            predictions, pass_data["actions"], eval_duration, config,
        )
        per_pass.append(
            {
                "uid": uid,
                "pass": pass_num,
                "duration_s": eval_duration,
                "n_chunks": n_chunks,
                "n_pred": len(predictions),
                "n_gt": len(pass_data["actions"]),
                "scores": scores,
            }
        )
    return per_pass


def _macro(values: list[float]) -> float | None:
    vs = [v for v in values if v is not None]
    return mean(vs) if vs else None


def build_pred_missing(per_datapoint: list[dict]) -> dict:
    """Aggregate frame-level pred_missing (eval gap-fill) across the corpus."""
    rates = [
        d["scores"]["frame_level_accuracy"]["pred_missing_rate"] for d in per_datapoint
    ]
    macro_rate = mean(rates) if rates else 0.0
    sum_frames = sum(
        d["scores"]["frame_level_accuracy"]["n_frames"] for d in per_datapoint
    )
    sum_missing = sum(
        d["scores"]["frame_level_accuracy"]["n_pred_missing_frames"]
        for d in per_datapoint
    )
    micro_rate = (sum_missing / sum_frames) if sum_frames else 0.0
    total_regions = sum(
        d["scores"]["frame_level_accuracy"]["n_pred_missing_regions"]
        for d in per_datapoint
    )
    return {
        "macro_rate": macro_rate,
        "micro_rate": micro_rate,
        "total_regions": total_regions,
    }


def aggregate(
    per_datapoint: list[dict],
    model_name: str,
    boundary_tolerance_s: float = EGO4D_BF1_TOLERANCE_S,
) -> dict:
    """Aggregate per-(video,pass) datapoints into the ``eval_segmentation.json`` dict.

    ``boundary_tolerance_s`` echoes into the output for provenance; pass
    the adapter's configured value (via ``metric_config().boundary_tolerance_s``)
    when calling.
    """
    macro = {
        "boundary_f1":        _macro([d["scores"]["boundary_f1"]["f1"]                    for d in per_datapoint]),
        "boundary_precision": _macro([d["scores"]["boundary_f1"]["precision"]             for d in per_datapoint]),
        "boundary_recall":    _macro([d["scores"]["boundary_f1"]["recall"]                for d in per_datapoint]),
        "frame_accuracy":     _macro([d["scores"]["frame_level_accuracy"]["overall_accuracy"] for d in per_datapoint]),
        "frame_macro_recall": _macro([d["scores"]["frame_level_accuracy"]["macro_recall"]     for d in per_datapoint]),
        "event_f1":           _macro([d["scores"]["event_based_f1"]["f1"]                 for d in per_datapoint]),
        "event_error_rate":   _macro([d["scores"]["event_error_rate"]["error_rate"]       for d in per_datapoint]),
    }

    tp_b = sum(d["scores"]["boundary_f1"]["tp_pred"] for d in per_datapoint)
    fp_b = sum(d["scores"]["boundary_f1"]["fp"] for d in per_datapoint)
    fn_b = sum(d["scores"]["boundary_f1"]["fn"] for d in per_datapoint)
    micro_prec = tp_b / max(1, tp_b + fp_b)
    micro_rec = tp_b / max(1, tp_b + fn_b)
    micro_bf1 = 2 * micro_prec * micro_rec / max(1e-9, micro_prec + micro_rec)
    tp_e = sum(d["scores"]["event_based_f1"]["tp"] for d in per_datapoint)
    fp_e = sum(d["scores"]["event_based_f1"]["fp"] for d in per_datapoint)
    fn_e = sum(d["scores"]["event_based_f1"]["fn"] for d in per_datapoint)
    micro_eprec = tp_e / max(1, tp_e + fp_e)
    micro_erec = tp_e / max(1, tp_e + fn_e)
    micro_ef1 = 2 * micro_eprec * micro_erec / max(1e-9, micro_eprec + micro_erec)
    n_frames = sum(d["scores"]["frame_level_accuracy"]["n_frames"] for d in per_datapoint)
    n_correct = sum(d["scores"]["frame_level_accuracy"]["n_correct"] for d in per_datapoint)
    micro_frame_acc = n_correct / max(1, n_frames)

    micro = {
        "boundary_precision": micro_prec,
        "boundary_recall":    micro_rec,
        "boundary_f1":        micro_bf1,
        "frame_accuracy":     micro_frame_acc,
        "event_precision":    micro_eprec,
        "event_recall":       micro_erec,
        "event_f1":           micro_ef1,
        "n_frames":           n_frames,
        "n_boundary_gt":      tp_b + fn_b,
        "n_boundary_pred":    tp_b + fp_b,
        "n_event_gt":         tp_e + fn_e,
        "n_event_pred":       tp_e + fp_e,
    }

    # Per-video chunks are scored once but appear in N passes — dedupe by uid.
    seen_uids: set[str] = set()
    total_chunks = 0
    for d in per_datapoint:
        if d["uid"] in seen_uids:
            continue
        seen_uids.add(d["uid"])
        total_chunks += int(d["n_chunks"])

    return {
        "model": model_name,
        "n_videos": len({d["uid"] for d in per_datapoint}),
        "n_data_points": len(per_datapoint),
        "total_chunks": total_chunks,
        "boundary_tolerance_s": boundary_tolerance_s,
        "macro": macro,
        "micro": micro,
        "pred_missing": build_pred_missing(per_datapoint),
        "per_datapoint": per_datapoint,
    }


def stitch_chunk_predictions(uid_run_dir: Path) -> list[dict]:
    """Concatenate raw per-chunk ``segments_abs`` from a run's per-uid dir.

    No gap-fill / coalesce here — that is the eval-side preprocessing
    contract inside ``metrics._prepare_eval``.
    """
    all_segs: list[dict] = []
    for cf in sorted(uid_run_dir.glob("chunk_*.json")):
        rec = json.loads(cf.read_text())
        all_segs.extend(rec.get("segments_abs") or [])
    return all_segs


def format_headline(agg: dict) -> str:
    """One-line default-per-metric headline (macro bF1/evF1/evER, micro frame)."""
    macro = agg["macro"]
    micro = agg["micro"]
    return (
        f"  DEFAULT  bF1={macro['boundary_f1']:.3f} (macro)  "
        f"frame={micro['frame_accuracy']:.3f} (micro)  "
        f"evF1={macro['event_f1']:.3f} (macro)  "
        f"evER={macro['event_error_rate']:.3f} (macro)"
    )


def format_pred_missing_line(pred_missing: dict) -> str:
    return (
        f"  Pred-missing (eval gap-fill): "
        f"{pred_missing['macro_rate'] * 100:.1f}% macro | "
        f"{pred_missing['micro_rate'] * 100:.1f}% micro | "
        f"{pred_missing['total_regions']:,} regions"
    )


def headline_metrics(agg: dict) -> dict:
    """Flat ``{name: value}`` of the default-per-metric headline numbers.

    Used by the training callback to log into Trainer logs (and thence
    wandb/tensorboard). Mirrors the CLI headline: bF1/evF1/evER are macro,
    frame accuracy is micro.
    """
    return {
        "boundary_f1": agg["macro"]["boundary_f1"],
        "frame_accuracy": agg["micro"]["frame_accuracy"],
        "event_f1": agg["macro"]["event_f1"],
        "event_error_rate": agg["macro"]["event_error_rate"],
    }
