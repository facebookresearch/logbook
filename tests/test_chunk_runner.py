# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the chunk runner orchestration (no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from long_audio.datasets.sins.schema import SINS_LABELS
from long_audio.inference.chunk_runner import run_inference
from long_audio.utils.events import coalesce_runs
from long_audio.inference.models.fake import FakeAdapter


SR = 16000


def _make_test_audio(path: Path, duration_s: float = 1500.0):
    n = int(SR * duration_s)
    # Low-amplitude noise so the file is not pathologically empty.
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(n).astype(np.float32) * 1e-3)
    soundfile.write(str(path), audio, SR, format="FLAC")


class TestAbsenceFilling(unittest.TestCase):
    def test_coalesce_runs_merges_adjacent_same_label(self):
        segs = [
            {"label": "absence", "start": 0.0, "end": 10.0},
            {"label": "absence", "start": 10.0, "end": 20.0},
            {"label": "cooking", "start": 20.0, "end": 30.0},
        ]
        out = coalesce_runs(segs, key="label")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], {"label": "absence", "start": 0.0, "end": 20.0})


class TestRunInferenceWithFakeAdapter(unittest.TestCase):
    def test_end_to_end_with_canned_segments(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=120.0)  # 2 min total

            # 30s chunks → 4 chunks.
            responses = [
                {"segments": [{"start": "00:00", "end": "00:30", "event": "cooking"}]},
                {"segments": [{"start": "00:00", "end": "00:30", "event": "eating"}]},
                {"segments": [{"start": "00:00", "end": "00:15", "event": "watching_tv"}]},  # leaves a gap
                {"segments": []},  # all absence
            ]
            adapter = FakeAdapter(responses=responses, latency_s=0.0)
            out_dir = td / "out"
            summary = run_inference(
                audio_path, adapter, out_dir, SINS_LABELS,
                chunk_minutes=0.5, decoder="structured",
                dataset_name="test", t0_iso="2017-01-30T04:38:36.435Z",
            )
            self.assertEqual(summary.n_chunks, 4)
            self.assertEqual(summary.n_chunks_completed, 4)
            # summary.json must NOT carry stitched_segments anymore (raw
            # predictions only — DECISION.md §9 N2).
            self.assertFalse(hasattr(summary, "stitched_segments"))
            # Per-chunk files exist; raw labels survive verbatim per chunk.
            self.assertEqual(len(list(out_dir.glob("chunk_*.json"))), 4)
            self.assertTrue((out_dir / "summary.json").exists())
            all_labels: set[str] = set()
            for cf in sorted(out_dir.glob("chunk_*.json")):
                rec = json.loads(cf.read_text())
                for seg in rec["segments_abs"]:
                    all_labels.add(seg["label"])
            self.assertIn("cooking", all_labels)
            self.assertIn("eating", all_labels)
            self.assertIn("watching_tv", all_labels)
            # Absence/other is no longer pre-filled at chunk granularity;
            # only labels actually emitted by the (Fake) model show up.

    def test_max_chunks_cap(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=300.0)
            adapter = FakeAdapter(responses=[{"segments": []}])
            summary = run_inference(
                audio_path, adapter, td / "out", SINS_LABELS,
                chunk_minutes=0.5, max_chunks=2,
            )
            self.assertEqual(summary.n_chunks, 2)
            self.assertEqual(summary.n_chunks_completed, 2)

    def test_structured_request_against_unstructured_adapter_raises(self):
        # Per the no-fallback rule: asking for structured decoding from
        # an adapter that doesn't support it raises loudly. Callers who
        # want free-text mode pass `decoder="freeform"` explicitly.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=60.0)
            adapter = FakeAdapter(
                responses=[{"segments": []}], structured=False,
            )
            with self.assertRaises(RuntimeError):
                run_inference(
                    audio_path, adapter, td / "out", SINS_LABELS,
                    chunk_minutes=0.5, decoder="structured",
                )

    def test_unparseable_output_produces_empty_chunks(self):
        # Free-text reply that json_repair can't extract anything from
        # yields zero segments — no chunk-wide-guess fallback.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=60.0)
            adapter = FakeAdapter(
                responses=["I don't know how to format JSON sorry"],
                structured=False,
            )
            out_dir = td / "out"
            run_inference(
                audio_path, adapter, out_dir, SINS_LABELS,
                chunk_minutes=0.5, decoder="freeform",
            )
            chunk0 = json.loads((out_dir / "chunk_00000.json").read_text())
            self.assertEqual(chunk0["segments_abs"], [])
            # parse_status field no longer exists on ChunkRecord.
            self.assertNotIn("parse_status", chunk0)


if __name__ == "__main__":
    unittest.main()
