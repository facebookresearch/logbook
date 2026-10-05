"""Tests for prompt rendering, schema parsing, and round-trip behavior."""

from __future__ import annotations

import json
import unittest

from long_audio.datasets.sins.schema import SINS_LABEL_HINTS, SINS_LABELS, sins_schema
from long_audio.inference.prompt import (
    ChunkContext,
    render_prompt,
)
from long_audio.inference.schema import (
    make_segmentation_schema,
    parse_segments,
    parse_timestamp,
)

class TestSchema(unittest.TestCase):
    def test_schema_is_valid_json(self):
        s = json.dumps(sins_schema())
        self.assertGreater(len(s), 0)

    def test_label_set_is_sins(self):
        # Fine-grained activity set (no rooms, underscore-normalized).
        self.assertIn("cooking", SINS_LABELS)
        self.assertIn("absence", SINS_LABELS)
        self.assertIn("vacuumcleaner", SINS_LABELS)
        self.assertIn("sleeping", SINS_LABELS)
        self.assertIn("watching_tv", SINS_LABELS)
        self.assertIn("getting_dry", SINS_LABELS)
        self.assertEqual(len(SINS_LABELS), 17)
        # No DCASE-style merged class survives.
        self.assertNotIn("social_activity", SINS_LABELS)
        self.assertNotIn("vacuum_cleaner", SINS_LABELS)

    def test_make_schema_with_custom_labels(self):
        custom = make_segmentation_schema(labels=("foo", "bar"))
        enum = custom["schema"]["properties"]["segments"]["items"]["properties"]["event"]["enum"]
        self.assertEqual(set(enum), {"foo", "bar"})

    def test_minute_schema_uses_integer_pattern(self):
        sec = make_segmentation_schema(labels=SINS_LABELS)  # default = second
        minu = make_segmentation_schema(labels=SINS_LABELS, time_unit="minute")
        sec_pat = sec["schema"]["properties"]["segments"]["items"]["properties"]["start"]["pattern"]
        min_pat = minu["schema"]["properties"]["segments"]["items"]["properties"]["start"]["pattern"]
        self.assertIn(":", sec_pat)        # MM:SS pattern has a colon
        self.assertNotIn(":", min_pat)     # minute pattern is digit-only
        self.assertEqual(min_pat, r"^\d{1,2}$")


class TestTimestampParsing(unittest.TestCase):
    def test_mmss(self):
        self.assertEqual(parse_timestamp("00:00"), 0)
        self.assertEqual(parse_timestamp("01:30"), 90)
        self.assertEqual(parse_timestamp("59:59"), 3599)

    def test_hmmss(self):
        self.assertEqual(parse_timestamp("1:00:00"), 3600)
        self.assertEqual(parse_timestamp("10:30:45"), 37845)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_timestamp("nope")
        with self.assertRaises(ValueError):
            parse_timestamp("60:60")  # invalid MM:SS

    def test_minute_mode(self):
        # Bare integer minutes; multiplied by 60 → seconds.
        self.assertEqual(parse_timestamp("0", time_unit="minute"), 0.0)
        self.assertEqual(parse_timestamp("1", time_unit="minute"), 60.0)
        self.assertEqual(parse_timestamp("10", time_unit="minute"), 600.0)
        # MM:SS rejected in minute mode.
        with self.assertRaises(ValueError):
            parse_timestamp("01:30", time_unit="minute")
        # Garbage rejected.
        with self.assertRaises(ValueError):
            parse_timestamp("abc", time_unit="minute")

    def test_minute_mode_accepts_raw_int(self):
        # Lenient-mode models may emit unquoted integers (the schema forces
        # strings but freeform decoding has no enforcement). Must not crash.
        self.assertEqual(parse_timestamp(0, time_unit="minute"), 0.0)
        self.assertEqual(parse_timestamp(5, time_unit="minute"), 300.0)
        self.assertEqual(parse_timestamp(10, time_unit="minute"), 600.0)
        # And floats (fractional minutes).
        self.assertEqual(parse_timestamp(1.5, time_unit="minute"), 90.0)
        # Negatives rejected.
        with self.assertRaises(ValueError):
            parse_timestamp(-1, time_unit="minute")
        # bool not silently coerced (True == 1 would be a footgun).
        with self.assertRaises(ValueError):
            parse_timestamp(True, time_unit="minute")

    def test_minute_parse_segments_handles_integer_timestamps(self):
        # Regression for the freeform+minute zero-readable-% bug:
        # freeform-mode JSON with integer start/end shouldn't drop everything.
        raw = {"segments": [
            {"start": 0, "end": 5, "event": "cooking"},
            {"start": 5, "end": 10, "event": "eating"},
        ]}
        events = parse_segments(raw, SINS_LABELS, chunk_start_s=0.0, time_unit="minute")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"label": "cooking", "start": 0.0, "end": 300.0})
        self.assertEqual(events[1], {"label": "eating",  "start": 300.0, "end": 600.0})

    def test_minute_parse_segments_roundtrip(self):
        # Minute-grain JSON dict → eval-suite event list with absolute seconds.
        raw = {
            "segments": [
                {"start": "0", "end": "5", "event": "cooking"},
                {"start": "5", "end": "10", "event": "eating"},
            ]
        }
        events = parse_segments(raw, SINS_LABELS, chunk_start_s=3600.0, time_unit="minute")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"label": "cooking", "start": 3600.0, "end": 3900.0})
        self.assertEqual(events[1], {"label": "eating",  "start": 3900.0, "end": 4200.0})


class TestParseSegments(unittest.TestCase):
    def test_basic_parse_with_offset(self):
        raw = {
            "segments": [
                {"start": "00:00", "end": "30:00", "event": "cooking"},
                {"start": "30:00", "end": "45:00", "event": "eating"},
            ]
        }
        events = parse_segments(raw, SINS_LABELS, chunk_start_s=3600.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start"], 3600.0)
        self.assertEqual(events[0]["end"], 5400.0)
        self.assertEqual(events[0]["label"], "cooking")
        self.assertEqual(events[1]["start"], 5400.0)
        self.assertEqual(events[1]["end"], 6300.0)

    def test_drops_unknown_labels(self):
        raw = {
            "segments": [
                {"start": "00:00", "end": "10:00", "event": "cooking"},
                {"start": "10:00", "end": "20:00", "event": "skydiving"},
            ]
        }
        events = parse_segments(raw, SINS_LABELS)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "cooking")

    def test_drops_inverted_intervals(self):
        raw = {
            "segments": [
                {"start": "20:00", "end": "10:00", "event": "cooking"},
            ]
        }
        events = parse_segments(raw, SINS_LABELS)
        self.assertEqual(len(events), 0)

    def test_handles_garbage_gracefully(self):
        self.assertEqual(parse_segments({}, SINS_LABELS), [])
        self.assertEqual(parse_segments({"segments": [{"bad": "shape"}]}, SINS_LABELS), [])
        self.assertEqual(parse_segments({"segments": [{"start": "??", "end": "00:10", "event": "cooking"}]}, SINS_LABELS), [])


class TestPromptRender(unittest.TestCase):
    def setUp(self):
        self.chunk = ChunkContext(
            chunk_index=2,
            n_chunks=8,
            chunk_start_wallclock_s=7200.0,
            chunk_end_wallclock_s=10800.0,
        )

    def test_contains_chunk_metadata(self):
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS)
        # Chunk index/count intentionally omitted from the prompt so the
        # model treats each chunk independently — but wall-clock + duration
        # of the audio MUST be in the header.
        self.assertNotIn("chunk 3 of 8", prompt)
        self.assertIn("The recording is", prompt)
        self.assertIn("3600s", prompt)
        self.assertIn("cooking", prompt)  # label hint present

    def test_contains_label_set(self):
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS)
        for lbl in SINS_LABELS:
            self.assertIn(lbl, prompt)

    def test_context_none_omits_prior_block(self):
        # Default context_mode == "none" → no "Prior context" header.
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS)
        self.assertNotIn("Prior context", prompt)

    def test_context_prev_with_label_appears(self):
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS,
                               context_mode="prev", prev_label="cooking")
        self.assertIn("Prior context", prompt)
        self.assertIn("Immediately before this chunk", prompt)
        # Critical: just the label name. No timestamps, no totals, no
        # multi-event "Recent activity" list, no leaked stats.
        self.assertNotIn("Cumulative", prompt)
        self.assertNotIn("Recent activity", prompt)
        self.assertNotIn("Story so far", prompt)
        self.assertNotIn("still ongoing", prompt)

    def test_context_prev_without_label_is_silent(self):
        # First chunk: no prior label yet → block should be empty even
        # when context_mode is 'prev'.
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS,
                               context_mode="prev", prev_label=None)
        self.assertNotIn("Prior context", prompt)

    def test_second_mode_mentions_mmss(self):
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS)  # default time_unit="second"
        self.assertIn("MM:SS", prompt)

    def test_minute_mode_mentions_integer_minutes(self):
        # Chunk is 3600s = 60 min in the test fixture.
        prompt = render_prompt(self.chunk, SINS_LABELS, label_hints=SINS_LABEL_HINTS,
                               time_unit="minute")
        # The prompt must instruct integer-minute boundaries and the
        # 0..N range derived from chunk_duration.
        self.assertIn("integer minute", prompt)
        self.assertIn("whole-minute", prompt)
        self.assertIn("0 to 60", prompt)
        # And not the MM:SS phrasing.
        self.assertNotIn("MM:SS", prompt)


if __name__ == "__main__":
    unittest.main()
