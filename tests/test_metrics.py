"""Tests for the eval metric suite.

Synthetic dry-run that confirms each metric responds sensibly to:
    - Perfect predictions
    - Time-shifted predictions
    - Label-shuffled predictions
    - Fully-random predictions
"""

from __future__ import annotations

import random
import unittest

from long_audio.eval.metrics import (
    boundary_f1,
    event_based_f1,
    event_error_rate,
    frame_level_accuracy,
)


def _gt_day() -> list[dict]:
    """A simple synthetic day of SINS-like events."""
    return [
        {"label": "absence", "start": 0.0, "end": 3600.0},
        {"label": "cooking", "start": 3600.0, "end": 5400.0},
        {"label": "eating", "start": 5400.0, "end": 6300.0},
        {"label": "absence", "start": 6300.0, "end": 9000.0},
        {"label": "watching_tv", "start": 9000.0, "end": 14400.0},
        {"label": "absence", "start": 14400.0, "end": 18000.0},
        {"label": "vacuumcleaner", "start": 18000.0, "end": 18600.0},
        {"label": "visit", "start": 18600.0, "end": 21600.0},
    ]


def _shift(events: list[dict], dt: float) -> list[dict]:
    return [{**e, "start": e["start"] + dt, "end": e["end"] + dt} for e in events]


def _shuffle_labels(events: list[dict], rng: random.Random) -> list[dict]:
    labels = [e["label"] for e in events]
    rng.shuffle(labels)
    return [{**e, "label": l} for e, l in zip(events, labels)]


def _random_events(duration_s: float, rng: random.Random, n: int = 10) -> list[dict]:
    pool = ["absence", "cooking", "eating", "watching_tv", "vacuumcleaner", "visit"]
    cuts = sorted(rng.uniform(0, duration_s) for _ in range(n - 1))
    times = [0.0] + cuts + [duration_s]
    out = []
    for i in range(n):
        out.append({"label": rng.choice(pool), "start": times[i], "end": times[i + 1]})
    return out


class TestPerfectPredictions(unittest.TestCase):
    def setUp(self):
        self.gt = _gt_day()

    def test_boundary_f1_is_1(self):
        m = boundary_f1(self.gt, self.gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertEqual(m["f1"], 1.0)

    def test_frame_accuracy_is_1(self):
        m = frame_level_accuracy(self.gt, self.gt, frame_size_s=10.0, duration_s=21600.0)
        self.assertEqual(m["overall_accuracy"], 1.0)
        self.assertEqual(m["pred_missing_rate"], 0.0)
        self.assertEqual(m["n_pred_missing_regions"], 0)

    def test_frame_confusion_matrix_is_diagonal(self):
        m = frame_level_accuracy(self.gt, self.gt, frame_size_s=10.0, duration_s=21600.0)
        cm = m["confusion_matrix"]
        # All off-diagonal cells are zero.
        diag_sum = int(sum(cm[i, i] for i in range(len(m["classes"]))))
        self.assertEqual(diag_sum, int(cm.sum()))
        # Every observed GT class has recall and precision == 1.0.
        for c, r in m["per_class_recall"].items():
            if m["per_class_n_gt"][c] > 0:
                self.assertEqual(r, 1.0, f"recall {c}={r}")
        for c, p in m["per_class_precision"].items():
            if m["per_class_n_pred"][c] > 0:
                self.assertEqual(p, 1.0, f"precision {c}={p}")
        self.assertEqual(m["macro_recall"], 1.0)
        self.assertEqual(m["macro_precision"], 1.0)

    def test_event_based_f1_is_1(self):
        m = event_based_f1(self.gt, self.gt, iou_threshold=0.5, duration_s=21600.0)
        self.assertEqual(m["f1"], 1.0)

    def test_event_error_rate_is_0(self):
        m = event_error_rate(self.gt, self.gt, duration_s=21600.0)
        self.assertEqual(m["error_rate"], 0.0)


class TestShiftedPredictions(unittest.TestCase):
    """Small time shifts: boundary F1 forgives within tolerance, frame accuracy degrades smoothly."""

    def setUp(self):
        self.gt = _gt_day()

    def test_small_shift_within_tolerance_still_perfect_boundaries(self):
        # 30s shift puts every interior pred boundary within the 60s
        # tolerance of its GT counterpart. The leading PRED_MISSING
        # gap-fill (covering [0, 30]) adds one spurious pred boundary at
        # 30 with no matching GT → precision drops slightly below 1.0
        # but recall stays at 1.0.
        pred = _shift(self.gt, dt=30.0)  # 30s shift, tolerance 60s
        m = boundary_f1(pred, self.gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertGreater(m["f1"], 0.9)

    def test_large_shift_outside_tolerance_misses_boundaries(self):
        pred = _shift(self.gt, dt=120.0)  # 120s shift, tolerance 60s
        m = boundary_f1(pred, self.gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertLess(m["f1"], 1.0)

    def test_shift_reduces_frame_accuracy(self):
        m_pf = frame_level_accuracy(self.gt, self.gt, frame_size_s=10.0, duration_s=21600.0)
        pred = _shift(self.gt, dt=300.0)
        m_sh = frame_level_accuracy(pred, self.gt, frame_size_s=10.0, duration_s=21600.0)
        self.assertLess(m_sh["overall_accuracy"], m_pf["overall_accuracy"])

    def test_event_iou_degrades_with_shift(self):
        # Shift small enough that classes still align, but IoU may drop below 0.5
        pred = _shift(self.gt, dt=300.0)  # 5 min shift
        m = event_based_f1(pred, self.gt, iou_threshold=0.5, duration_s=21600.0)
        # Some events should still hit (longer ones), others miss.
        self.assertLess(m["f1"], 1.0)


class TestShuffledLabels(unittest.TestCase):
    """Perfect boundaries, scrambled labels. boundary_f1 was originally
    purely label-agnostic, but now coalesces adjacent same-label runs on
    both pred and GT before deriving boundary times (to handle minute-
    sliced predictions correctly). That makes shuffled-label predictions
    sometimes drop below F1=1.0 when the shuffle creates new adjacencies."""

    def setUp(self):
        self.gt = _gt_day()
        self.pred = _shuffle_labels(self.gt, random.Random(42))

    def test_boundary_f1_label_sensitive_via_coalesce(self):
        # With label-aware coalescing, shuffling can merge interior
        # boundaries where two adjacent shuffled labels happen to match
        # → F1 may drop below 1.0 but should still be a sensible score.
        m = boundary_f1(self.pred, self.gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertLessEqual(m["f1"], 1.0)
        self.assertGreater(m["f1"], 0.0)

    def test_event_based_f1_drops(self):
        m = event_based_f1(self.pred, self.gt, iou_threshold=0.5, duration_s=21600.0)
        self.assertLess(m["f1"], 1.0)

    def test_frame_confusion_macro_drops(self):
        m = frame_level_accuracy(self.pred, self.gt, frame_size_s=10.0, duration_s=21600.0)
        # Shuffled labels rasterize to per-frame mismatches, so macro
        # recall must drop below 1.0.
        self.assertLess(m["macro_recall"], 1.0)


class TestRandomPredictions(unittest.TestCase):
    def setUp(self):
        self.gt = _gt_day()
        rng = random.Random(0)
        self.pred = _random_events(21600.0, rng, n=12)

    def test_random_boundary_f1_low(self):
        m = boundary_f1(self.pred, self.gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertLess(m["f1"], 0.7)

    def test_random_event_error_rate_positive(self):
        m = event_error_rate(self.pred, self.gt, duration_s=21600.0)
        self.assertGreater(m["error_rate"], 0.0)

    def test_random_event_based_f1_low(self):
        m = event_based_f1(self.pred, self.gt, iou_threshold=0.5, duration_s=21600.0)
        self.assertLess(m["f1"], 0.5)


class TestEdgeCases(unittest.TestCase):
    def test_empty_predictions(self):
        # Standardized pred preprocessing fills empty pred with one
        # PRED_MISSING_LABEL span covering [0, duration_s]. That stream
        # has no INTERIOR boundaries (only 0 and duration). With
        # duration_s passed, those trivial endpoints are excluded → 0
        # pred boundaries vs. many GT boundaries → recall=0, f1=0.
        gt = _gt_day()
        m = boundary_f1([], gt, tolerance_s=60.0, duration_s=21600.0)
        self.assertEqual(m["n_pred"], 0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)

    def test_empty_ground_truth_rejected(self):
        # GT must be contiguous over [0, duration_s] by contract
        # (enforced at data-prep time); empty GT is a contract violation
        # and is now rejected at the eval entry point by _prepare_eval's
        # contiguity check.
        pred = [{"label": "cooking", "start": 0, "end": 100}]
        with self.assertRaises(ValueError):
            event_based_f1(pred, [], duration_s=100.0)

    def test_validate_bad_event(self):
        with self.assertRaises(ValueError):
            boundary_f1(
                [{"label": "x", "start": 10, "end": 5}],
                [{"label": "x", "start": 0, "end": 10}],
                duration_s=10.0,
            )


class TestFrameLevelStrictGT(unittest.TestCase):
    """Pin the pred-missing-sentinel contract.

    Note: GT contiguity is asserted at DATA-PREP time (manifest_builder),
    not in eval. Tests that previously exercised the eval-side assert have
    moved to TestAssertGTContiguous (below) and to test_manifest_builder.
    """

    def setUp(self):
        self.gt = _gt_day()

    def test_empty_pred_all_counted_as_missing(self):
        m = frame_level_accuracy([], self.gt, frame_size_s=10.0, duration_s=21600.0)
        self.assertEqual(m["n_correct"], 0)
        self.assertEqual(m["overall_accuracy"], 0.0)
        self.assertEqual(m["pred_missing_rate"], 1.0)
        self.assertEqual(m["n_pred_missing_regions"], 1)
        # The sentinel shows up in the confusion matrix as a column.
        from long_audio.eval.metrics import PRED_MISSING_LABEL
        self.assertIn(PRED_MISSING_LABEL, m["classes"])
        # The sentinel does NOT appear in macro precision (would be 0
        # and bias the average).
        self.assertEqual(m["macro_precision"], 0.0)

    def test_partial_pred_gap_counted_as_missing(self):
        # Pred only covers the first half of the GT timeline.
        pred = [
            {"label": "absence", "start": 0.0, "end": 3600.0},
            {"label": "cooking", "start": 3600.0, "end": 5400.0},
            # Gap from 5400 onwards.
        ]
        m = frame_level_accuracy(pred, self.gt, frame_size_s=10.0, duration_s=21600.0)
        self.assertGreater(m["pred_missing_rate"], 0.0)
        # Pred-missing frames count as wrong, so overall < 1.
        self.assertLess(m["overall_accuracy"], 1.0)
        # One contiguous trailing gap.
        self.assertEqual(m["n_pred_missing_regions"], 1)

    def test_pred_missing_rate_matches_fraction_of_failed_time(self):
        # Single trailing 600s gap on a 21600s timeline → rate = 600/21600.
        pred = [{"label": "absence", "start": 0.0, "end": 21000.0}]
        m = frame_level_accuracy(
            pred, self.gt, frame_size_s=10.0, duration_s=21600.0
        )
        self.assertAlmostEqual(m["pred_missing_rate"], 600.0 / 21600.0)
        self.assertEqual(m["n_pred_missing_regions"], 1)

    def test_gap_regions_distinguish_one_big_from_many_small(self):
        # n_pred_missing_regions distinguishes "1 big drop" from
        # "many small drops" even when the total missing time is similar.
        pred_one_big = [{"label": "absence", "start": 0.0, "end": 21000.0}]
        pred_many_small = []
        t = 0.0
        for _ in range(5):
            pred_many_small.append(
                {"label": "absence", "start": t, "end": t + 4000.0}
            )
            t += 4000.0 + 60.0
        m_big = frame_level_accuracy(
            pred_one_big, self.gt, frame_size_s=10.0, duration_s=21600.0
        )
        m_small = frame_level_accuracy(
            pred_many_small, self.gt, frame_size_s=10.0, duration_s=21600.0
        )
        self.assertEqual(m_big["n_pred_missing_regions"], 1)
        self.assertEqual(m_small["n_pred_missing_regions"], 5)


class TestEventErrorRateStrictGT(unittest.TestCase):
    """Event-error-rate behavior given a contract-conforming GT.

    GT contiguity is asserted at data-prep time, not in eval. The
    edit-distance metric trusts the contract and coalesces adjacent
    same-label GT runs (legitimate annotator concatenation) for the
    comparison.
    """

    def setUp(self):
        self.gt = _gt_day()

    def test_empty_pred_is_single_pred_missing_span(self):
        # No predicted events → standardized preprocessing fills the
        # entire timeline with one PRED_MISSING_LABEL span. That is
        # length-1 vs the n_gt GT label sequence → exactly one of the
        # GT events is substituted (PRED_MISSING) and the remaining
        # n_gt-1 are deletions. S+D+I = n_gt; ER = n_gt / n_gt = 1.0.
        m = event_error_rate([], self.gt, duration_s=21600.0)
        n_gt = m["n_reference"]
        self.assertGreater(n_gt, 0)
        self.assertEqual(m["substitutions"] + m["deletions"], n_gt)
        self.assertEqual(m["insertions"], 0)
        self.assertEqual(m["n_predicted"], 1)
        self.assertEqual(m["error_rate"], 1.0)

    def test_substitution_only(self):
        # GT labels: [a, b, c]; pred labels: [a, X, c] → S=1, D=I=0,
        # N=3, ER=1/3.
        gt = [
            {"label": "a", "start": 0.0,   "end": 100.0},
            {"label": "b", "start": 100.0, "end": 200.0},
            {"label": "c", "start": 200.0, "end": 300.0},
        ]
        pred = [
            {"label": "a", "start": 0.0,   "end": 100.0},
            {"label": "X", "start": 100.0, "end": 200.0},
            {"label": "c", "start": 200.0, "end": 300.0},
        ]
        m = event_error_rate(pred, gt, duration_s=300.0)
        self.assertEqual(m["n_reference"], 3)
        self.assertEqual(m["n_predicted"], 3)
        self.assertEqual(m["substitutions"], 1)
        self.assertEqual(m["deletions"], 0)
        self.assertEqual(m["insertions"], 0)
        self.assertAlmostEqual(m["error_rate"], 1.0 / 3.0)

    def test_substitution_when_pred_skips_a_gt_event(self):
        # GT [a, b, c, d]; pred originally [a, b, d] with a gap where 'c'
        # was. Standardized preprocessing fills that gap with
        # PRED_MISSING_LABEL → pred becomes [a, b, PRED_MISSING, d].
        # Optimal Levenshtein vs GT [a, b, c, d]: substitute 'c' with
        # PRED_MISSING. S=1, D=0, I=0, N=4, ER=1/4.
        gt = [
            {"label": "a", "start": 0.0,   "end": 50.0},
            {"label": "b", "start": 50.0,  "end": 100.0},
            {"label": "c", "start": 100.0, "end": 150.0},
            {"label": "d", "start": 150.0, "end": 200.0},
        ]
        # Pred has a gap where 'c' should be (100..150), then resumes d.
        pred = [
            {"label": "a", "start": 0.0,   "end": 50.0},
            {"label": "b", "start": 50.0,  "end": 100.0},
            {"label": "d", "start": 150.0, "end": 200.0},
        ]
        m = event_error_rate(pred, gt, duration_s=200.0)
        self.assertEqual(m["n_reference"], 4)
        self.assertEqual(m["n_predicted"], 4)
        self.assertEqual(m["substitutions"], 1)
        self.assertEqual(m["deletions"], 0)
        self.assertEqual(m["insertions"], 0)
        self.assertAlmostEqual(m["error_rate"], 1.0 / 4.0)

    def test_insertion_when_pred_has_extra_event(self):
        # GT [a, b]; pred [a, X, b] — inserted 'X' between them.
        gt = [
            {"label": "a", "start": 0.0,   "end": 100.0},
            {"label": "b", "start": 100.0, "end": 200.0},
        ]
        pred = [
            {"label": "a", "start": 0.0,   "end": 50.0},
            {"label": "X", "start": 50.0,  "end": 100.0},
            {"label": "b", "start": 100.0, "end": 200.0},
        ]
        m = event_error_rate(pred, gt, duration_s=200.0)
        self.assertEqual(m["n_reference"], 2)
        self.assertEqual(m["n_predicted"], 3)
        self.assertEqual(m["substitutions"], 0)
        self.assertEqual(m["deletions"], 0)
        self.assertEqual(m["insertions"], 1)
        self.assertAlmostEqual(m["error_rate"], 1.0 / 2.0)


class TestBoundaryF1Lenient(unittest.TestCase):
    """Pin the lenient-matching contract: precision and recall are scored
    independently — no one-to-one match constraint."""

    def test_one_pred_satisfies_two_close_gts(self):
        # Two interior GT transitions 30s apart (100s, 130s), well within
        # tolerance. Trivial timeline endpoints (0 and 300) are excluded
        # via duration_s. Pred has interior transition at 115s.
        # Strict 1:1 matching would credit only one of the interior GTs;
        # lenient credits both via the single nearby pred boundary.
        gt = [
            {"label": "a", "start": 0, "end": 100},
            {"label": "b", "start": 100, "end": 130},
            {"label": "c", "start": 130, "end": 300},
        ]
        pred = [
            {"label": "a", "start": 0, "end": 115},
            {"label": "c", "start": 115, "end": 300},
        ]
        m = boundary_f1(pred, gt, tolerance_s=60.0, duration_s=300.0)
        # Interior boundary set (trivial 0, 300 excluded).
        # GT  : {100, 130} → 2 boundaries.
        # Pred: {115}      → 1 boundary.
        self.assertEqual(m["n_gt"], 2)
        self.assertEqual(m["n_pred"], 1)
        # 115↔100 within tolerance → tp_pred=1, precision=1.0.
        self.assertEqual(m["tp_pred"], 1)
        self.assertEqual(m["precision"], 1.0)
        # Both GT interior boundaries (100, 130) within tolerance of pred 115
        # → tp_gt=2, recall=1.0 (lenient: same pred can satisfy both GTs).
        self.assertEqual(m["tp_gt"], 2)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_zero_length_pred_does_not_inflate_precision(self):
        # Pred sandwiches a zero-length segment at t=100. Without
        # cleanup that endpoint would appear twice in the boundary set
        # (start and end of the zero-length seg are equal) — the dedup
        # via set() would catch it anyway, but the cleanup also keeps
        # n_pred honest by dropping the zero-length segment so it
        # doesn't manufacture phantom boundaries.
        gt = [
            {"label": "cooking", "start": 0,   "end": 100},
            {"label": "eating",  "start": 100, "end": 300},
        ]
        pred = [
            {"label": "cooking", "start": 0,   "end": 100},
            {"label": "bogus",   "start": 100, "end": 100},  # zero-length
            {"label": "eating",  "start": 100, "end": 300},
        ]
        m = boundary_f1(pred, gt, tolerance_s=60.0, duration_s=300.0)
        # Interior boundary set (trivial 0, 300 excluded) = {100} for both sides.
        self.assertEqual(m["n_pred"], 1)
        self.assertEqual(m["n_gt"], 1)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_two_preds_match_one_gt(self):
        # One interior GT transition at 100; pred has two interior
        # transitions (80, 130) straddling it within tolerance.
        # Trivial timeline endpoints (0, 300) excluded via duration_s.
        gt = [
            {"label": "a", "start": 0, "end": 100},
            {"label": "b", "start": 100, "end": 300},
        ]
        pred = [
            {"label": "a", "start": 0, "end": 80},
            {"label": "b", "start": 80, "end": 130},
            {"label": "c", "start": 130, "end": 300},
        ]
        m = boundary_f1(pred, gt, tolerance_s=60.0, duration_s=300.0)
        # GT  : {100}     → 1 boundary.
        # Pred: {80, 130} → 2 boundaries.
        self.assertEqual(m["n_gt"], 1)
        self.assertEqual(m["n_pred"], 2)
        # Both preds within tolerance of GT 100
        # (80↔100, 130↔100) → tp_pred=2, precision=1.0.
        self.assertEqual(m["tp_pred"], 2)
        self.assertEqual(m["precision"], 1.0)
        # GT 100 within tolerance of pred 80 → tp_gt=1, recall=1.0.
        self.assertEqual(m["tp_gt"], 1)
        self.assertEqual(m["recall"], 1.0)


class TestAdjacencyMerging(unittest.TestCase):
    """One 600 s span and ten contiguous 60 s minute-slices of the same label
    must score identically across metrics that are sensitive to segment count
    (boundary_f1, event_based_f1, event_error_rate). Frame-level accuracy is
    not affected by segment count and is asserted to also remain identical.
    """

    def _gt(self) -> list[dict]:
        return [{"label": "productive", "start": 0.0, "end": 600.0}]

    def _pred_merged(self) -> list[dict]:
        return [{"label": "productive", "start": 0.0, "end": 600.0}]

    def _pred_minute_slices(self) -> list[dict]:
        return [
            {"label": "productive", "start": float(i * 60), "end": float((i + 1) * 60)}
            for i in range(10)
        ]

    def test_boundary_f1_identical(self):
        gt = self._gt()
        m_merged = boundary_f1(self._pred_merged(), gt, tolerance_s=60.0, duration_s=600.0)
        m_slices = boundary_f1(self._pred_minute_slices(), gt, tolerance_s=60.0, duration_s=600.0)
        self.assertEqual(m_merged["f1"], m_slices["f1"])
        self.assertEqual(m_merged["n_pred"], m_slices["n_pred"])
        self.assertEqual(m_merged["tp_pred"], m_slices["tp_pred"])
        self.assertEqual(m_merged["fp"], m_slices["fp"])

    def test_event_based_f1_identical(self):
        gt = self._gt()
        m_merged = event_based_f1(self._pred_merged(), gt, iou_threshold=0.5, duration_s=600.0)
        m_slices = event_based_f1(self._pred_minute_slices(), gt, iou_threshold=0.5, duration_s=600.0)
        self.assertEqual(m_merged["f1"], m_slices["f1"])
        self.assertEqual(m_merged["tp"], m_slices["tp"])
        self.assertEqual(m_merged["fp"], m_slices["fp"])
        self.assertEqual(m_merged["fn"], m_slices["fn"])
        # Both should be a perfect match (one event, IoU 1.0 with the GT).
        self.assertEqual(m_merged["f1"], 1.0)
        self.assertEqual(m_slices["f1"], 1.0)

    def test_event_error_rate_identical(self):
        # event_error_rate coalesces both sides internally; this is an
        # invariance regression test for that behavior.
        gt = self._gt()
        m_merged = event_error_rate(self._pred_merged(), gt, duration_s=600.0)
        m_slices = event_error_rate(self._pred_minute_slices(), gt, duration_s=600.0)
        self.assertEqual(m_merged["error_rate"], m_slices["error_rate"])
        self.assertEqual(m_merged["error_rate"], 0.0)

    def test_frame_level_accuracy_identical(self):
        # Not affected by segment count, but assert anyway as a sanity check.
        gt = self._gt()
        m_merged = frame_level_accuracy(self._pred_merged(), gt, duration_s=600.0)
        m_slices = frame_level_accuracy(self._pred_minute_slices(), gt, duration_s=600.0)
        self.assertEqual(m_merged["overall_accuracy"], m_slices["overall_accuracy"])
        self.assertEqual(m_merged["overall_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
