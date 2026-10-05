# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for long_audio/utils/events.py.

Covers the three primitives that replaced ~5 copies of merge / gap-fill
/ overlap-resolution code spread across the package.
"""

from __future__ import annotations

import unittest

from long_audio.utils.events import (
    coalesce_runs,
    fill_gaps,
    midpoint_boundaries,
    resolve_overlaps,
)


class TestCoalesceRuns(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(coalesce_runs([]), [])

    def test_single(self):
        events = [{"label": "a", "start": 0.0, "end": 1.0}]
        self.assertEqual(
            coalesce_runs(events),
            [{"label": "a", "start": 0.0, "end": 1.0}],
        )

    def test_merges_adjacent_same_label(self):
        events = [
            {"label": "a", "start": 0.0, "end": 1.0},
            {"label": "a", "start": 1.0, "end": 2.0},
            {"label": "a", "start": 2.0, "end": 3.0},
        ]
        out = coalesce_runs(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 3.0)

    def test_keeps_different_labels(self):
        events = [
            {"label": "a", "start": 0.0, "end": 1.0},
            {"label": "b", "start": 1.0, "end": 2.0},
            {"label": "a", "start": 2.0, "end": 3.0},
        ]
        out = coalesce_runs(events)
        self.assertEqual(len(out), 3)

    def test_tolerance(self):
        # Tiny float dust within tol_s should still merge.
        events = [
            {"label": "a", "start": 0.0, "end": 1.0},
            {"label": "a", "start": 1.0 + 1e-9, "end": 2.0},
        ]
        out = coalesce_runs(events, tol_s=1e-6)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["end"], 2.0)

    def test_does_not_modify_input(self):
        events = [{"label": "a", "start": 0.0, "end": 1.0}]
        before = list(events)
        coalesce_runs(events)
        self.assertEqual(events, before)

    def test_custom_key(self):
        events = [
            {"event": "x", "start": 0.0, "end": 1.0},
            {"event": "x", "start": 1.0, "end": 2.0},
        ]
        out = coalesce_runs(events, key="event")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["event"], "x")


class TestFillGaps(unittest.TestCase):
    def test_empty(self):
        out = fill_gaps([], 0.0, 10.0, "GAP")
        self.assertEqual(
            out,
            [{"label": "GAP", "start": 0.0, "end": 10.0}],
        )

    def test_leading_gap(self):
        events = [{"label": "a", "start": 5.0, "end": 10.0}]
        out = fill_gaps(events, 0.0, 10.0, "GAP")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "GAP")
        self.assertEqual(out[0]["end"], 5.0)
        self.assertEqual(out[1]["label"], "a")

    def test_trailing_gap(self):
        events = [{"label": "a", "start": 0.0, "end": 5.0}]
        out = fill_gaps(events, 0.0, 10.0, "GAP")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1]["label"], "GAP")
        self.assertEqual(out[-1]["start"], 5.0)
        self.assertEqual(out[-1]["end"], 10.0)

    def test_middle_gap(self):
        events = [
            {"label": "a", "start": 0.0, "end": 3.0},
            {"label": "b", "start": 7.0, "end": 10.0},
        ]
        out = fill_gaps(events, 0.0, 10.0, "GAP")
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1]["label"], "GAP")
        self.assertEqual(out[1]["start"], 3.0)
        self.assertEqual(out[1]["end"], 7.0)

    def test_no_gap(self):
        events = [
            {"label": "a", "start": 0.0, "end": 5.0},
            {"label": "b", "start": 5.0, "end": 10.0},
        ]
        out = fill_gaps(events, 0.0, 10.0, "GAP")
        self.assertEqual(len(out), 2)

    def test_custom_key(self):
        events = [{"event": "x", "start": 0.0, "end": 5.0}]
        out = fill_gaps(events, 0.0, 10.0, "GAP", key="event")
        self.assertEqual(out[-1]["event"], "GAP")


class TestResolveOverlaps(unittest.TestCase):
    @staticmethod
    def _priority(label: str) -> int:
        if label == "absence":
            return 0
        if label == "other":
            return 1
        return 2

    def test_empty_with_gap_label(self):
        out = resolve_overlaps(
            [], self._priority,
            min_duration_s=0.0, gap_label="GAP", duration_s=10.0,
        )
        self.assertEqual(out, [{"label": "GAP", "start": 0.0, "end": 10.0}])

    def test_overlap_priority(self):
        # 'cooking' (specific) overlaps 'other' (low). Specific wins.
        events = [
            {"label": "other", "start": 0.0, "end": 10.0},
            {"label": "cooking", "start": 3.0, "end": 7.0},
        ]
        out = resolve_overlaps(
            events, self._priority,
            min_duration_s=0.0, gap_label=None, duration_s=10.0,
        )
        labels = [e["label"] for e in out]
        self.assertEqual(labels, ["other", "cooking", "other"])

    def test_gap_label_fills_uncovered_regions(self):
        events = [{"label": "cooking", "start": 3.0, "end": 7.0}]
        out = resolve_overlaps(
            events, self._priority,
            min_duration_s=0.0, gap_label="GAP", duration_s=10.0,
        )
        self.assertEqual(out[0]["label"], "GAP")
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 3.0)
        self.assertEqual(out[-1]["label"], "GAP")
        self.assertEqual(out[-1]["start"], 7.0)
        self.assertEqual(out[-1]["end"], 10.0)

    def test_sliver_absorbed(self):
        # 'cooking' for 0.5s should get absorbed into previous span when
        # min_duration_s=1.0.
        events = [
            {"label": "other", "start": 0.0, "end": 3.0},
            {"label": "cooking", "start": 3.0, "end": 3.5},
            {"label": "other", "start": 3.5, "end": 10.0},
        ]
        out = resolve_overlaps(
            events, self._priority,
            min_duration_s=1.0, gap_label=None, duration_s=10.0,
        )
        # Sliver absorbed; remaining is one merged 'other'.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "other")

    def test_merges_adjacent_same_label_post_arbitration(self):
        events = [
            {"label": "cooking", "start": 0.0, "end": 5.0},
            {"label": "cooking", "start": 5.0, "end": 10.0},
        ]
        out = resolve_overlaps(
            events, self._priority,
            min_duration_s=0.0, gap_label=None, duration_s=10.0,
        )
        self.assertEqual(len(out), 1)


class TestMidpointBoundaries(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(
            midpoint_boundaries([], 100.0, carry={"event": "event"}), []
        )

    def test_two_adjacent(self):
        slices = [
            {"start": 0.0, "end": 60.0, "event": "a"},
            {"start": 60.0, "end": 120.0, "event": "b"},
        ]
        out = midpoint_boundaries(slices, 120.0, carry={"event": "event"})
        self.assertEqual([e["start"] for e in out], [0.0, 60.0])
        self.assertEqual([e["end"] for e in out], [60.0, 120.0])
        self.assertEqual([e["event"] for e in out], ["a", "b"])

    def test_overlap_midpoint(self):
        slices = [
            {"start": 0.0, "end": 60.0, "event": "a"},
            {"start": 40.0, "end": 100.0, "event": "b"},
        ]
        out = midpoint_boundaries(slices, 100.0, carry={"event": "event"})
        self.assertAlmostEqual(out[0]["end"], 50.0)
        self.assertAlmostEqual(out[1]["start"], 50.0)

    def test_sorts_input(self):
        # Unsorted input is sorted internally by (start, -end).
        slices = [
            {"start": 60.0, "end": 120.0, "event": "b"},
            {"start": 0.0, "end": 60.0, "event": "a"},
        ]
        out = midpoint_boundaries(slices, 120.0, carry={"event": "event"})
        self.assertEqual([e["event"] for e in out], ["a", "b"])

    def test_carry_renames_keys(self):
        # output_key -> source_key mapping (the training gold path).
        slices = [
            {"start": 0.0, "end": 300.0, "event": "productive", "summary": "s0"},
            {"start": 270.0, "end": 570.0, "event": "leisure", "summary": "s1"},
        ]
        out = midpoint_boundaries(
            slices, 570.0, carry={"event": "event", "description": "summary"}
        )
        self.assertEqual([e["description"] for e in out], ["s0", "s1"])
        self.assertEqual([e["event"] for e in out], ["productive", "leisure"])
        self.assertAlmostEqual(out[0]["end"], 285.0)  # midpoint(300, 270)

    def test_snap_edges_true_forces_exact_bounds(self):
        # First start snapped to 0.0, last end to duration, even when the
        # clamped values would differ.
        slices = [{"start": 5.0, "end": 90.0, "event": "a"}]
        out = midpoint_boundaries(
            slices, 100.0, carry={"event": "event"}, snap_edges=True
        )
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 100.0)

    def test_snap_edges_false_keeps_clamped_bounds(self):
        # Clamp only: positive first-start is preserved (annotate path, where
        # the caller does its own snap before coalescing).
        slices = [{"start": 5.0, "end": 90.0, "event": "a"}]
        out = midpoint_boundaries(
            slices, 100.0, carry={"event": "event"}, snap_edges=False
        )
        self.assertEqual(out[0]["start"], 5.0)
        self.assertEqual(out[0]["end"], 90.0)

    def test_clamps_negative_first_and_overlong_last(self):
        slices = [{"start": -5.0, "end": 200.0, "event": "a"}]
        out = midpoint_boundaries(
            slices, 100.0, carry={"event": "event"}, snap_edges=False
        )
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 100.0)

    def test_degenerate_span_raises(self):
        # A slice fully contained in another can midpoint to end <= start.
        slices = [
            {"start": 0.0, "end": 100.0, "event": "a"},
            {"start": 40.0, "end": 50.0, "event": "b"},
            {"start": 0.0, "end": 100.0, "event": "c"},
        ]
        with self.assertRaises(ValueError):
            midpoint_boundaries(slices, 100.0, carry={"event": "event"})


if __name__ == "__main__":
    unittest.main()
