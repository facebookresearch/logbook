"""Unit tests for Stage 2 helpers inlined in
``scripts/data/ego4d/build_manifest.py``.

Covers `compute_lead_shift`, `_shift_slices`, `_dedupe_summaries`, the
interior-gap detector `_interior_gap`, and the deterministic split
hash `_assign_split`. Tests for Stage 3 helpers (`_apply_midpoint_*`,
`_build_actions`, OLMo parsers) live in `test_annotate_manifest.py`.
"""

from __future__ import annotations

import unittest

from scripts.data.ego4d.build_manifest import (
    _assign_split,
    _dedupe_summaries,
    _interior_gap,
    _shift_slices,
    compute_lead_shift,
)


class TestComputeLeadShift(unittest.TestCase):
    def test_empty_entry(self):
        self.assertEqual(compute_lead_shift(None), (0.0, 0.0))
        self.assertEqual(compute_lead_shift({}), (0.0, 0.0))

    def test_no_summaries(self):
        # Both pass keys + summaries arrays must exist (callers enforce this
        # before calling compute_lead_shift; the function dies noisily on
        # missing keys per the no-protective-coding rule).
        entry = {
            "narration_pass_1": {"summaries": []},
            "narration_pass_2": {"summaries": []},
        }
        self.assertEqual(compute_lead_shift(entry), (0.0, 0.0))

    def test_snap_to_zero_below_threshold(self):
        entry = {
            "narration_pass_1": {"summaries": [{"start_sec": 0.05, "end_sec": 300}]},
            "narration_pass_2": {"summaries": [{"start_sec": 0.07, "end_sec": 300}]},
        }
        self.assertEqual(compute_lead_shift(entry), (0.0, 0.0))

    def test_shifts_at_or_above_threshold(self):
        entry = {
            "narration_pass_1": {"summaries": [{"start_sec": 5.0, "end_sec": 300}]},
            "narration_pass_2": {"summaries": [{"start_sec": 7.0, "end_sec": 300}]},
        }
        self.assertEqual(compute_lead_shift(entry), (5.0, 5.0))


class TestShiftSlices(unittest.TestCase):
    def test_zero_shift_is_passthrough(self):
        slices = [{"start_sec": 0.0, "end_sec": 60.0, "activity": "a"}]
        self.assertEqual(_shift_slices(slices, 0.0), slices)

    def test_subtracts_shift(self):
        slices = [
            {"start_sec": 10.0, "end_sec": 70.0, "activity": "a"},
            {"start_sec": 70.0, "end_sec": 130.0, "activity": "b"},
        ]
        out = _shift_slices(slices, 10.0)
        self.assertEqual(out[0]["start_sec"], 0.0)
        self.assertEqual(out[0]["end_sec"], 60.0)
        self.assertEqual(out[1]["start_sec"], 60.0)
        self.assertEqual(out[1]["end_sec"], 120.0)

    def test_drops_pre_shift_slices(self):
        slices = [
            {"start_sec": 0.0, "end_sec": 5.0, "activity": "drop"},
            {"start_sec": 10.0, "end_sec": 70.0, "activity": "keep"},
        ]
        out = _shift_slices(slices, 10.0, uid="x")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["activity"], "keep")

    def test_clamps_partial_leading_to_zero(self):
        slices = [{"start_sec": 5.0, "end_sec": 70.0, "activity": "a"}]
        out = _shift_slices(slices, 10.0)
        self.assertEqual(out[0]["start_sec"], 0.0)
        self.assertEqual(out[0]["end_sec"], 60.0)

    def test_raises_if_all_dropped(self):
        slices = [{"start_sec": 0.0, "end_sec": 5.0}]
        with self.assertRaises(ValueError):
            _shift_slices(slices, 100.0, uid="x")


class TestDedupeSummaries(unittest.TestCase):
    def test_empty(self):
        kept, dropped = _dedupe_summaries([])
        self.assertEqual((kept, dropped), ([], 0))

    def test_no_duplicates(self):
        recs = [
            {"start_sec": 0.0, "end_sec": 300.0, "activity": "a"},
            {"start_sec": 300.0, "end_sec": 600.0, "activity": "b"},
        ]
        kept, dropped = _dedupe_summaries(recs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)

    def test_drops_contained_duplicate(self):
        recs = [
            {"start_sec": 0.0, "end_sec": 300.0, "activity": "a"},
            {"start_sec": 100.0, "end_sec": 200.0, "activity": "b"},
        ]
        kept, dropped = _dedupe_summaries(recs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_drops_degenerate(self):
        recs = [
            {"start_sec": 0.0, "end_sec": 300.0, "activity": "a"},
            {"start_sec": 1365.0, "end_sec": 776.6, "activity": "bad"},
        ]
        kept, dropped = _dedupe_summaries(recs)
        self.assertEqual([k["activity"] for k in kept], ["a"])
        self.assertEqual(dropped, 1)

    def test_preserves_source_order(self):
        recs = [
            {"start_sec": 100.0, "end_sec": 200.0, "activity": "x"},
            {"start_sec": 0.0, "end_sec": 300.0, "activity": "y"},
            {"start_sec": 400.0, "end_sec": 500.0, "activity": "z"},
        ]
        kept, _ = _dedupe_summaries(recs)
        self.assertEqual([k["activity"] for k in kept], ["y", "z"])


class TestInteriorGap(unittest.TestCase):
    def test_single_slice_no_gap(self):
        slices = [{"start_sec": 0.0, "end_sec": 100.0}]
        self.assertEqual(_interior_gap(slices), (False, ""))

    def test_continuous_no_gap(self):
        slices = [
            {"start_sec": 0.0, "end_sec": 300.0},
            {"start_sec": 300.0, "end_sec": 600.0},
        ]
        self.assertEqual(_interior_gap(slices), (False, ""))

    def test_sub_drift_tolerance_no_gap(self):
        # Sub-10ms drift is float noise, not a real gap.
        slices = [
            {"start_sec": 0.0, "end_sec": 300.0},
            {"start_sec": 300.005, "end_sec": 600.0},
        ]
        self.assertEqual(_interior_gap(slices), (False, ""))

    def test_real_gap_detected(self):
        slices = [
            {"start_sec": 0.0, "end_sec": 300.0},
            {"start_sec": 600.0, "end_sec": 900.0},  # 300s interior gap
        ]
        has_gap, reason = _interior_gap(slices)
        self.assertTrue(has_gap)
        self.assertIn("interior gap", reason)

    def test_overlapping_slices_no_gap(self):
        # Earlier slice extends past next start — cursor advances correctly.
        slices = [
            {"start_sec": 0.0, "end_sec": 350.0},
            {"start_sec": 300.0, "end_sec": 600.0},
        ]
        self.assertEqual(_interior_gap(slices), (False, ""))


class TestAssignSplit(unittest.TestCase):
    def test_deterministic(self):
        # Same input → same split, every run.
        for _ in range(5):
            self.assertEqual(_assign_split("cmu", "pid_42"),
                             _assign_split("cmu", "pid_42"))

    def test_buckets_in_60_20_20(self):
        # Distribution check: across many synthetic participants in one
        # source, expect roughly 60/20/20.
        from collections import Counter
        c = Counter(_assign_split("cmu", f"pid_{i}") for i in range(10000))
        self.assertEqual(c["train"] + c["val"] + c["test"], 10000)
        # Loose bounds — sha256 should be uniform.
        self.assertGreater(c["train"], 5500)
        self.assertLess(c["train"], 6500)
        self.assertGreater(c["val"], 1500)
        self.assertLess(c["val"], 2500)
        self.assertGreater(c["test"], 1500)
        self.assertLess(c["test"], 2500)

    def test_different_sources_get_independent_splits(self):
        # Same pid, different sources can land in different splits.
        # (Not guaranteed for every pid, but the hash includes the source.)
        n_diff = sum(
            1 for i in range(100)
            if _assign_split("cmu", f"pid_{i}") != _assign_split("unict", f"pid_{i}")
        )
        # At least some should differ; if 0, the hash isn't using source.
        self.assertGreater(n_diff, 20)


if __name__ == "__main__":
    unittest.main()
