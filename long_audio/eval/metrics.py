# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation metrics for long-audio activity segmentation + classification.

Conventions
-----------
Every metric takes ``predictions`` and ``ground_truth`` as a list of dicts:

    {"label": str, "start": float, "end": float}

where ``start`` and ``end`` are seconds. Within each list, events must be
non-overlapping (single-label timeline). Lists do not need to be sorted;
metrics sort internally where it matters.

GT vs pred: both are contiguous-by-contract
-------------------------------------------
By contract, BOTH ground truth and predictions reach the metric functions
as contiguous, gap-free, non-overlapping event streams covering
[0, duration_s]:

  * GT contiguity is enforced at data-prep time, not re-checked at eval.
  * Pred contiguity is MADE true at the start of every metric via
    :func:`_preprocess_pred` — a single standardized step that drops
    zero-length pred segments, fills intra-pred gaps with
    ``PRED_MISSING_LABEL``, and coalesces adjacent same-label runs.

Frame discretization
--------------------
Frame-level accuracy and event error rate both quantize the timeline
into fixed-size frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from long_audio.utils.events import coalesce_runs, fill_gaps

EventList = list[dict]


PRED_MISSING_LABEL = "__pred_missing__"


def _label_key(label: str) -> str:
    """Spelling-insensitive key: case, surrounding space, and word separators."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def _assert_no_label_variant_collision(
    predictions: EventList, ground_truth: EventList
) -> None:
    """Raise if pred and GT spell the same label differently."""
    gt_labels = {e["label"] for e in ground_truth}
    pred_labels = {e["label"] for e in predictions} - {PRED_MISSING_LABEL}
    collisions = sorted(
        (g, p)
        for g in gt_labels
        for p in pred_labels
        if g != p and _label_key(g) == _label_key(p)
    )
    if collisions:
        pairs = ", ".join(f"GT {g!r} vs pred {p!r}" for g, p in collisions)
        raise ValueError(
            "ground truth and predictions use different spellings of the same "
            f"label, which every metric scores as a total miss: {pairs}. "
            "Rebuild the dataset manifest with the current prep pipeline "
            "(scripts/data/<dataset>/build_manifest.py) so GT labels match "
            "the label set the model was prompted with."
        )


# --- Helpers ---------------------------------------------------------------


def _validate_events(events: EventList) -> None:
    if not isinstance(events, list):
        raise TypeError("events must be a list of dicts")
    for i, e in enumerate(events):
        for key in ("label", "start", "end"):
            if key not in e:
                raise ValueError(f"events[{i}] missing key '{key}'")
        if e["end"] < e["start"]:
            raise ValueError(f"events[{i}] has end < start: {e}")


def _validate_contiguous(events: EventList, duration_s: float) -> None:
    """Cheap surface contract check: events cover [0, duration_s] with no gaps."""
    if not events:
        raise ValueError(f"events list is empty but duration_s={duration_s}")
    if abs(events[0]["start"] - 0.0) > 1e-6:
        raise ValueError(f"events[0].start={events[0]['start']}, expected 0.0")
    if abs(events[-1]["end"] - duration_s) > 1e-6:
        raise ValueError(
            f"events[-1].end={events[-1]['end']}, " f"expected duration_s={duration_s}"
        )
    for i in range(len(events) - 1):
        if abs(events[i]["end"] - events[i + 1]["start"]) > 1e-6:
            raise ValueError(
                f"events not contiguous at index {i}: "
                f"events[{i}].end={events[i]['end']} != "
                f"events[{i+1}].start={events[i+1]['start']}"
            )


def _assert_coalesced(events: EventList, name: str) -> None:
    """Raise if two adjacent events share the same label."""
    for i in range(1, len(events)):
        if events[i - 1]["label"] == events[i]["label"]:
            raise ValueError(
                f"{name} not coalesced: events[{i-1}].label == events[{i}].label "
                f"== {events[i]['label']!r} (start={events[i]['start']})"
            )


def _prepare_eval(
    predictions: EventList,
    ground_truth: EventList,
    *,
    duration_s: float,
) -> tuple[EventList, EventList, int]:
    """Single entry-time preprocessing + validation for every metric.
    Returns (preprocessed_pred, ground_truth, n_zero_length_pred_dropped).
    """
    _validate_events(predictions)
    _validate_events(ground_truth)
    pred, n_zero = _preprocess_pred(predictions, duration_s)
    _validate_contiguous(pred, duration_s)
    _validate_contiguous(ground_truth, duration_s)
    _assert_coalesced(ground_truth, name="ground_truth")
    _assert_no_label_variant_collision(pred, ground_truth)
    return pred, ground_truth, n_zero


def _count_runs(frames: np.ndarray, label: str) -> int:
    """Return the number of maximal runs of ``label`` in ``frames``."""
    mask = frames == label
    if not bool(np.any(mask)):
        return 0
    # A "region" is a maximal run of True in the mask. Count rising edges
    # over a [False] ... [False] padded view.
    diffs = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    return int(np.sum(diffs == 1))


def _sort(events: EventList) -> EventList:
    return sorted(events, key=lambda e: (e["start"], e["end"]))


def _rasterize(
    events: EventList,
    duration_s: float,
    frame_size_s: float,
    *,
    default_label: str,
) -> np.ndarray:
    """Quantize an event list into an array of per-frame string labels."""
    n_frames = int(np.ceil(duration_s / frame_size_s))
    labels = np.full(n_frames, default_label, dtype=object)
    for e in events:
        f_start = int(np.floor(e["start"] / frame_size_s))
        f_end = int(np.ceil(e["end"] / frame_size_s))
        f_start = max(0, f_start)
        f_end = min(n_frames, f_end)
        if f_end > f_start:
            labels[f_start:f_end] = e["label"]
    return labels


def _drop_zero_length(events: EventList) -> tuple[EventList, int]:
    """Drop events with ``end <= start`` (degenerate point segments).
    Returns ``(kept, n_dropped)``.
    """
    kept = [e for e in events if e["end"] - e["start"] > 1e-9]
    return kept, len(events) - len(kept)


def _clamp_overlaps(events: EventList) -> EventList:
    """Earlier-wins clamp on a sorted event list.

    Models sometimes emit out-of-order or overlapping spans within a
    single chunk (e.g. Qwen2.5 in minute-mode sometimes returns a tail
    segment whose start jumps backward by several minutes). After
    sorting by ``(start, end)``, this pass clamps each event's end to
    the next event's start so the resulting stream is monotonically
    non-overlapping. Events that become zero-length post-clamp are
    dropped.
    """
    sorted_evs = sorted(events, key=lambda e: (e["start"], e["end"]))
    out: list[dict] = []
    for i, e in enumerate(sorted_evs):
        ne = dict(e)
        if i + 1 < len(sorted_evs) and ne["end"] > sorted_evs[i + 1]["start"]:
            ne["end"] = sorted_evs[i + 1]["start"]
        if ne["end"] - ne["start"] > 1e-9:
            out.append(ne)
    return out


def _preprocess_pred(
    events: EventList,
    duration_s: float,
    gap_fill_label: str = PRED_MISSING_LABEL,
) -> tuple[EventList, int]:
    """Standardized pred preprocessing: drop_zero_length -> clamp overlaps ->
    fill_gaps -> coalesce_runs.

    By contract, both GT and pred are contiguous when they reach the metric
    functions. GT is asserted at data-prep time (manifest_builder). Pred is
    MADE contiguous here by clamping overlaps to the next span's start,
    filling intra-pred gaps with PRED_MISSING_LABEL (a sentinel that does
    not match any real label, so missing frames count as wrong against any
    GT class), and coalescing adjacent same-label runs.

    Returns (preprocessed_events, n_zero_length_pred_segments_dropped).
    Callers should record n_zero_length in their return dict for auditing.
    """
    events, n_zero = _drop_zero_length(events)
    if duration_s <= 0:
        return events, n_zero
    events = _clamp_overlaps(events)
    events = fill_gaps(events, 0.0, duration_s, gap_label=gap_fill_label, key="label")
    events = coalesce_runs(events, key="label")
    return events, n_zero


def _boundary_times(events: EventList, duration_s: float | None = None) -> list[float]:
    """All unique event endpoints as a sorted boundary time list.

    For events ``[(s1,e1), (s2,e2), ...]`` returns
    ``sorted(set([s1, e1, s2, e2, ...]))``. This handles overlapping
    events correctly: each event contributes both its start and end as
    candidate boundaries (e.g. ``[(0,100), (50,150)]`` → ``[0, 50, 100,
    150]`` rather than collapsing the overlap to one transition).

    If ``duration_s`` is provided, the trivial timeline endpoints ``0``
    and ``duration_s`` are excluded — only *interior* transitions are
    returned. A gap-filled GT that covers ``[0, T]`` by construction
    would otherwise contribute two free boundaries every time (and any
    prediction stream covering [0, T] auto-matches them within tolerance,
    inflating the score on short-segment data points).
    """
    times: set[float] = set()
    for e in events:
        times.add(float(e["start"]))
        times.add(float(e["end"]))
    if duration_s is not None:
        times = {t for t in times if abs(t) > 1e-6 and abs(t - duration_s) > 1e-6}
    return sorted(times)


# --- 1. Boundary F1 --------------------------------------------------------


def boundary_f1(
    predictions: EventList,
    ground_truth: EventList,
    tolerance_s: float = 60.0,
    *,
    duration_s: float,
) -> dict:
    """Boundary F1 with a configurable timing tolerance, lenient matching.

    Returns dict with keys ``precision``, ``recall``, ``f1``,
    ``tp_pred``, ``tp_gt``, ``fp``, ``fn``, ``n_pred``, ``n_gt``,
    ``tolerance_s``.
    """
    predictions, ground_truth, _ = _prepare_eval(
        predictions,
        ground_truth,
        duration_s=duration_s,
    )

    pred_b = _boundary_times(predictions, duration_s=duration_s)
    gt_b = _boundary_times(ground_truth, duration_s=duration_s)

    n_pred = len(pred_b)
    n_gt = len(gt_b)

    # Precision side: each pred is TP if any GT is within tolerance.
    tp_pred = 0
    for t_p in pred_b:
        for t_g in gt_b:
            if abs(t_p - t_g) <= tolerance_s:
                tp_pred += 1
                break

    # Recall side: each GT is TP if any pred is within tolerance.
    tp_gt = 0
    for t_g in gt_b:
        for t_p in pred_b:
            if abs(t_p - t_g) <= tolerance_s:
                tp_gt += 1
                break

    fp = n_pred - tp_pred
    fn = n_gt - tp_gt
    precision = tp_pred / n_pred if n_pred else 0.0
    recall = tp_gt / n_gt if n_gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pred": tp_pred,
        "tp_gt": tp_gt,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "tolerance_s": tolerance_s,
    }


# --- 2. Frame-level accuracy ----------------------------------------------


def frame_level_accuracy(
    predictions: EventList,
    ground_truth: EventList,
    frame_size_s: float = 1.0,
    *,
    duration_s: float,
) -> dict:
    """Per-frame label accuracy + confusion matrix.

    Both prediction and GT are rasterized to ``frame_size_s`` frames over
    the same timeline, then compared frame-by-frame.

    Contract:
      * GT must fully cover ``[0, duration_s]`` with no gaps or overlaps.
        Enforced at DATA-PREP time — eval trusts the contract.
      * Prediction gaps are filled with ``PRED_MISSING_LABEL`` (via
        :func:`_preprocess_pred`) and count as wrong against any GT label.

    The prompt format used at inference is ``MM:SS`` (1-second
    resolution), so the default ``frame_size_s`` is ``1.0``. Coarser
    frame sizes (e.g. 10s) blur the comparison.

    Returns dict with:
        ``overall_accuracy``: correct frames / total frames (including
            pred-missing frames in the denominator).
        ``n_frames``, ``n_correct``, ``frame_size_s``.
        ``pred_missing_rate``: fraction of frames where the model emitted
            no event (the primary "how much time did the model fail on"
            number).
        ``n_pred_missing_frames``: integer count of frames filled with
            ``PRED_MISSING_LABEL`` — the numerator behind
            ``pred_missing_rate`` (useful when pooling across data points
            to compute a micro rate).
        ``n_pred_missing_regions``: count of distinct maximal runs of
            ``PRED_MISSING_LABEL`` — distinguishes "one big drop" from
            "many small drops" at the same rate.
        ``classes``: sorted list of label strings seen in either side;
            includes ``PRED_MISSING_LABEL`` iff pred had gaps.
        ``confusion_matrix``: (n_classes, n_classes) ``np.int64``;
            rows = GT, cols = pred.
        ``per_class_recall`` / ``per_class_precision``:
            dict {class: float}.
        ``per_class_n_gt`` / ``per_class_n_pred``: dict {class: int}.
        ``macro_recall``: mean of per_class_recall over classes with
            n_gt > 0. ``PRED_MISSING_LABEL`` never appears in GT, so it
            does not enter this average.
        ``macro_precision``: mean of per_class_precision over classes
            with n_pred > 0, *excluding* ``PRED_MISSING_LABEL``.
    """
    predictions, ground_truth, n_zero_length_pred_segments = _prepare_eval(
        predictions,
        ground_truth,
        duration_s=duration_s,
    )

    # Both sides are gap-free by contract (validated above). Pass a
    # uniquely-named sentinel as _rasterize's default_label so we can
    # assert post-rasterization that no frame fell back to it; if any
    # did, the post-preprocess contiguity contract was violated between
    # _prepare_eval and _rasterize.
    _RASTER_SENTINEL = "__rasterize_gap__"
    pred_frames = _rasterize(
        predictions,
        duration_s,
        frame_size_s,
        default_label=_RASTER_SENTINEL,
    )
    gt_frames = _rasterize(
        ground_truth,
        duration_s,
        frame_size_s,
        default_label=_RASTER_SENTINEL,
    )
    assert not (
        pred_frames == _RASTER_SENTINEL
    ).any(), "pred has gaps post-_preprocess_pred -- contract violation"
    assert not (
        gt_frames == _RASTER_SENTINEL
    ).any(), "gt has gaps post-_prepare_eval -- contract violation"

    n = len(gt_frames)
    correct = int(np.sum(pred_frames == gt_frames))
    n_pred_missing_frames = int(np.sum(pred_frames == PRED_MISSING_LABEL))
    pred_missing_rate = (n_pred_missing_frames / n) if n > 0 else 0.0
    n_pred_missing_regions = _count_runs(pred_frames, PRED_MISSING_LABEL)

    classes = sorted(set(gt_frames.tolist()) | set(pred_frames.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt_frames, pred_frames):
        cm[cls_to_idx[g], cls_to_idx[p]] += 1

    per_class_n_gt: dict[str, int] = {}
    per_class_n_pred: dict[str, int] = {}
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    for i, c in enumerate(classes):
        row_sum = int(cm[i, :].sum())
        col_sum = int(cm[:, i].sum())
        per_class_n_gt[c] = row_sum
        per_class_n_pred[c] = col_sum
        per_class_recall[c] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
        per_class_precision[c] = float(cm[i, i] / col_sum) if col_sum > 0 else 0.0

    # Macro precision over classes that pred actually emitted; do NOT count
    # the pred-missing sentinel.
    macro_precision_vals = [
        per_class_precision[c]
        for c in classes
        if per_class_n_pred[c] > 0 and c != PRED_MISSING_LABEL
    ]
    macro_recall_vals = [per_class_recall[c] for c in classes if per_class_n_gt[c] > 0]
    macro_precision = (
        float(np.mean(macro_precision_vals)) if macro_precision_vals else 0.0
    )
    macro_recall = float(np.mean(macro_recall_vals)) if macro_recall_vals else 0.0

    return {
        "overall_accuracy": correct / n if n > 0 else 0.0,
        "n_frames": n,
        "n_correct": correct,
        "pred_missing_rate": pred_missing_rate,
        "n_pred_missing_frames": n_pred_missing_frames,
        "n_pred_missing_regions": n_pred_missing_regions,
        "n_zero_length_pred_segments": n_zero_length_pred_segments,
        "frame_size_s": frame_size_s,
        "classes": classes,
        "confusion_matrix": cm,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_n_gt": per_class_n_gt,
        "per_class_n_pred": per_class_n_pred,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
    }


# --- 3. Event-based F1 (IoU threshold) ------------------------------------


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def event_based_f1(
    predictions: EventList,
    ground_truth: EventList,
    iou_threshold: float = 0.5,
    exclude_labels: Iterable[str] = (),
    *,
    duration_s: float,
) -> dict:
    """Event-based F1 with per-class IoU matching.

    Args:
        predictions, ground_truth: event lists.
        iou_threshold: minimum IoU for a match (default 0.5).
        exclude_labels: labels to ignore on both sides during event
            matching. Pass ``("other",)`` for ATUS-style taxonomies,
            ``("absence", "other")`` for SINS-style. Default is empty —
            every label scores.
        duration_s: timeline extent (required).

    Returns dict with ``precision``, ``recall``, ``f1``, ``per_class``,
    ``tp``, ``fp``, ``fn``, ``iou_threshold``.
    """
    predictions, ground_truth, _ = _prepare_eval(
        predictions,
        ground_truth,
        duration_s=duration_s,
    )
    exclude = set(exclude_labels)

    # Pred is already coalesced by _preprocess_pred; just filter excludes
    # and sort. GT gets coalesced here so that N contiguous minute-slices
    # of the same label score as one event.
    pred = [e for e in _sort(predictions) if e["label"] not in exclude]
    gt = coalesce_runs(
        [e for e in _sort(ground_truth) if e["label"] not in exclude],
        key="label",
    )

    # Build per-class GT index lists for fast filtering.
    gt_by_class: dict[str, list[int]] = {}
    for i, g in enumerate(gt):
        gt_by_class.setdefault(g["label"], []).append(i)
    matched: set[int] = set()

    # Compute all (pred_idx, gt_idx, iou) candidate matches with same label,
    # sort by IoU desc, greedily assign.
    candidates: list[tuple[float, int, int]] = []
    for p_idx, p in enumerate(pred):
        for g_idx in gt_by_class.get(p["label"], []):
            g = gt[g_idx]
            iou = _iou(p["start"], p["end"], g["start"], g["end"])
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))
    candidates.sort(key=lambda x: -x[0])

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for iou, p_idx, g_idx in candidates:
        if p_idx in matched_pred or g_idx in matched_gt:
            continue
        matched_pred.add(p_idx)
        matched_gt.add(g_idx)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-class breakdown
    per_class: dict[str, dict] = {}
    classes = sorted({p["label"] for p in pred} | {g["label"] for g in gt})
    pred_count_by_class: dict[str, int] = {}
    gt_count_by_class: dict[str, int] = {}
    tp_by_class: dict[str, int] = {}
    for c in classes:
        pred_count_by_class[c] = 0
        gt_count_by_class[c] = 0
        tp_by_class[c] = 0
    for p in pred:
        pred_count_by_class[p["label"]] += 1
    for g in gt:
        gt_count_by_class[g["label"]] += 1
    for p_idx in matched_pred:
        tp_by_class[pred[p_idx]["label"]] += 1
    for c in classes:
        n_pred_c = pred_count_by_class[c]
        n_gt_c = gt_count_by_class[c]
        tp_c = tp_by_class[c]
        p_c = tp_c / n_pred_c if n_pred_c > 0 else 0.0
        r_c = tp_c / n_gt_c if n_gt_c > 0 else 0.0
        f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        per_class[c] = {
            "precision": p_c,
            "recall": r_c,
            "f1": f_c,
            "tp": tp_c,
            "n_pred": n_pred_c,
            "n_gt": n_gt_c,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": per_class,
        "iou_threshold": iou_threshold,
    }


def _levenshtein_sdi(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int]:
    """Standard Levenshtein edit distance broken down into S, D, I."""
    from rapidfuzz.distance import Levenshtein

    n = len(reference)
    m = len(hypothesis)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)
    ops = Levenshtein.editops(reference, hypothesis)
    sub = del_ = ins = 0
    for op in ops:
        tag = op.tag
        if tag == "replace":
            sub += 1
        elif tag == "delete":
            del_ += 1
        elif tag == "insert":
            ins += 1
    return (sub, del_, ins)


# --- 4. Event Error Rate (event-level, WER) -------------------------------


def event_error_rate(
    predictions: EventList,
    ground_truth: EventList,
    *,
    duration_s: float,
) -> dict:
    """Event-level WER on the dedupe'd label sequences."""
    predictions, ground_truth, _ = _prepare_eval(
        predictions,
        ground_truth,
        duration_s=duration_s,
    )

    pred_events = predictions  # already coalesced by _preprocess_pred
    gt_events = (
        ground_truth  # already coalesced by data-prep (asserted in _prepare_eval)
    )

    pred_labels = [e["label"] for e in pred_events]
    gt_labels = [e["label"] for e in gt_events]
    n_reference = len(gt_labels)
    substitutions, deletions, insertions = _levenshtein_sdi(gt_labels, pred_labels)
    er = (substitutions + deletions + insertions) / max(n_reference, 1)

    return {
        "error_rate": er,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "n_reference": n_reference,
        "n_predicted": len(pred_labels),
    }


# --- Convenience: summarize() ---------------------------------------------


@dataclass
class MetricConfig:
    boundary_tolerance_s: float = 60.0
    frame_size_s: float = 1.0  # MM:SS prompt format → 1s resolution.
    event_iou_threshold: float = 0.5
    coalesce_runs: bool = True  # merge adjacent same-label runs before scoring
    # Passed to event_based_f1. Default is empty — every label scores.
    # Pass ("other",) for ATUS-style taxonomies, ("absence", "other") for
    # SINS-style.
    event_exclude_labels: tuple[str, ...] = ()


def summarize(
    predictions: EventList,
    ground_truth: EventList,
    config: MetricConfig | None = None,
    *,
    duration_s: float,
) -> dict:
    """Run all metrics and return them in a single dict, suitable for logging."""
    cfg = config or MetricConfig()
    return {
        "boundary_f1": boundary_f1(
            predictions,
            ground_truth,
            tolerance_s=cfg.boundary_tolerance_s,
            duration_s=duration_s,
        ),
        "frame_level_accuracy": frame_level_accuracy(
            predictions,
            ground_truth,
            frame_size_s=cfg.frame_size_s,
            duration_s=duration_s,
        ),
        "event_based_f1": event_based_f1(
            predictions,
            ground_truth,
            iou_threshold=cfg.event_iou_threshold,
            exclude_labels=cfg.event_exclude_labels,
            duration_s=duration_s,
        ),
        "event_error_rate": event_error_rate(
            predictions,
            ground_truth,
            duration_s=duration_s,
        ),
    }
