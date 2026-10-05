# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the pure helpers in
``scripts/data/ego4d/annotate_manifest.py``.

Covers parsers (parse_class, parse_cleaned, parse_facts) and the
timeline-assembly helper _apply_midpoint_boundaries. OLMo orchestration
(_enrich_pass, annotate_manifest) requires a vLLM + GPU and is covered
end-to-end via the SLURM job, not here.
"""

from __future__ import annotations

import unittest

from scripts.data.ego4d.annotate_manifest import (
    _apply_midpoint_boundaries,
    load_atus,
    parse_class,
    parse_cleaned,
    parse_facts,
)


class TestParseClass(unittest.TestCase):
    def setUp(self):
        self.valid = list(load_atus().keys())

    def test_exact(self):
        self.assertEqual(parse_class("food", self.valid), ("food", True))

    def test_strip_trailing_period(self):
        self.assertEqual(parse_class("food.", self.valid), ("food", True))

    def test_strip_quotes(self):
        self.assertEqual(parse_class('"productive"', self.valid),
                         ("productive", True))

    def test_case_insensitive(self):
        self.assertEqual(parse_class("Productive", self.valid),
                         ("productive", True))

    def test_extract_from_preamble(self):
        self.assertEqual(parse_class("Class label: leisure", self.valid),
                         ("leisure", True))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            parse_class("definitely_not_a_class", self.valid)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_class("", self.valid)
        with self.assertRaises(ValueError):
            parse_class("   ", self.valid)


class TestParseCleaned(unittest.TestCase):
    def test_strips_output_prefix(self):
        self.assertEqual(parse_cleaned("Output: hello world", "fb"), "hello world")

    def test_first_nonempty_line(self):
        self.assertEqual(parse_cleaned("\n\nfirst line\nsecond", "fb"), "first line")

    def test_strips_balanced_quotes(self):
        self.assertEqual(parse_cleaned('"quoted"', "fb"), "quoted")

    def test_falls_back_on_empty(self):
        self.assertEqual(parse_cleaned("", "fb"), "fb")
        self.assertEqual(parse_cleaned("\n\n", "fb"), "fb")


class TestParseFacts(unittest.TestCase):
    def test_dash_bullets(self):
        out = parse_facts("- one\n- two\n- three")
        self.assertEqual(out, ["one.", "two.", "three."])

    def test_skips_non_bullets(self):
        out = parse_facts("preamble line\n- a\nrandom\n- b")
        self.assertEqual(out, ["a.", "b."])

    def test_already_terminal_period_preserved(self):
        self.assertEqual(parse_facts("- already a fact."), ["already a fact."])

    def test_empty(self):
        self.assertEqual(parse_facts(""), [])


class TestMidpointBoundaries(unittest.TestCase):
    def test_two_adjacent(self):
        slices = [
            {"start": 0.0, "end": 60.0, "event": "a"},
            {"start": 60.0, "end": 120.0, "event": "b"},
        ]
        out = _apply_midpoint_boundaries(slices, 120.0)
        self.assertEqual([e["start"] for e in out], [0.0, 60.0])
        self.assertEqual([e["end"] for e in out], [60.0, 120.0])
        self.assertEqual([e["event"] for e in out], ["a", "b"])

    def test_overlap_midpoint(self):
        slices = [
            {"start": 0.0, "end": 60.0, "event": "a"},
            {"start": 40.0, "end": 100.0, "event": "b"},
        ]
        out = _apply_midpoint_boundaries(slices, 100.0)
        self.assertAlmostEqual(out[0]["end"], 50.0)
        self.assertAlmostEqual(out[1]["start"], 50.0)

    def test_first_clipped_to_zero(self):
        slices = [{"start": -5.0, "end": 60.0, "event": "a"}]
        out = _apply_midpoint_boundaries(slices, 100.0)
        self.assertEqual(out[0]["start"], 0.0)

    def test_last_clipped_to_duration(self):
        slices = [{"start": 0.0, "end": 200.0, "event": "a"}]
        out = _apply_midpoint_boundaries(slices, 100.0)
        self.assertEqual(out[0]["end"], 100.0)

    def test_empty(self):
        self.assertEqual(_apply_midpoint_boundaries([], 100.0), [])


if __name__ == "__main__":
    unittest.main()
