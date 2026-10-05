# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cascade pipeline (Stage A + Stage B, no GPU / no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from long_audio.datasets.sins.schema import SINS_LABELS
from long_audio.inference.cascade import (
    _window_descriptions,
    run_cascade_segmentation,
)
from long_audio.inference.describe import (
    DESCRIBE_PROMPT,
    describe_audio_chunks,
    load_descriptions,
)
from long_audio.inference.models.fake import FakeAdapter, FakeTextLLMAdapter
from long_audio.inference.prompt import (
    ChunkContext,
    render_prompt_from_descriptions,
)


SR = 16000


def _make_test_audio(path: Path, duration_s: float):
    n = int(SR * duration_s)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(n).astype(np.float32) * 1e-3)
    soundfile.write(str(path), audio, SR, format="FLAC")


class TestDescribeStage(unittest.TestCase):
    def test_writes_one_jsonl_line_per_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=35.0)  # 4 ten-sec chunks

            captions = [
                "Someone talking quietly.",
                "Distant kitchen sounds.",
                "TV in the background.",
                "Quiet, no clear activity.",
            ]
            adapter = FakeAdapter(responses=captions, structured=False, name="fake-cap")
            summary = describe_audio_chunks(
                audio_path, adapter, td / "out",
                chunk_seconds=10.0,
            )
            self.assertEqual(summary.n_chunks, 4)
            self.assertEqual(summary.n_chunks_completed, 4)
            self.assertEqual(summary.prompt, DESCRIBE_PROMPT)

            descs = load_descriptions(summary.descriptions_path)
            self.assertEqual(len(descs), 4)
            self.assertEqual(descs[0]["chunk_idx"], 0)
            self.assertEqual(descs[0]["start_s"], 0.0)
            self.assertEqual(descs[0]["end_s"], 10.0)
            self.assertEqual(descs[0]["description"], "Someone talking quietly.")
            # Last chunk is shorter (35 s total, 10 s chunks → 5 s tail).
            self.assertEqual(descs[3]["start_s"], 30.0)
            self.assertEqual(descs[3]["end_s"], 35.0)

    def test_describe_prompt_is_locked(self):
        # Single source of truth — the cascade design depends on this prompt
        # matching AF3's Meta-internal AudioCaps default.
        self.assertEqual(DESCRIBE_PROMPT, "Describe the audio clip in one sentence.")

    def test_batch_size_partitions_calls(self):
        # Verify describe.py groups chunks into batches of batch_size and
        # calls generate_batch per group (no chunk skipped, no chunk doubled).
        seen_batch_sizes: list[int] = []

        def cb(audio, sr, prompt, **kw):
            return f"caption from a 1-call cb shouldn't fire"  # unused (batch path uses generate_batch)

        class CountingAdapter(FakeAdapter):
            def generate_batch(self_inner, audios, sample_rate, prompts, **kw):
                seen_batch_sizes.append(len(audios))
                return super().generate_batch(
                    audios, sample_rate, prompts, **kw,
                )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio_path = td / "tiny.flac"
            _make_test_audio(audio_path, duration_s=55.0)  # 6 ten-sec chunks

            adapter = CountingAdapter(
                responses=["a", "b", "c", "d", "e", "f"],
                structured=False, name="count",
            )
            summary = describe_audio_chunks(
                audio_path, adapter, td / "out",
                chunk_seconds=10.0,
                batch_size=4,  # 6 chunks → batches of [4, 2]
            )
            self.assertEqual(summary.n_chunks_completed, 6)
            self.assertEqual(seen_batch_sizes, [4, 2])
            descs = load_descriptions(summary.descriptions_path)
            self.assertEqual([d["description"] for d in descs],
                             ["a", "b", "c", "d", "e", "f"])
            self.assertEqual([d["chunk_idx"] for d in descs],
                             [0, 1, 2, 3, 4, 5])


class TestWindowing(unittest.TestCase):
    def test_groups_by_window_index(self):
        descs = [
            {"chunk_idx": 0, "start_s": 0.0, "end_s": 10.0, "description": "a"},
            {"chunk_idx": 1, "start_s": 10.0, "end_s": 20.0, "description": "b"},
            {"chunk_idx": 60, "start_s": 600.0, "end_s": 610.0, "description": "c"},
        ]
        # 10-min (600 s) windows: indices 0..59 → window 0, 60+ → window 1.
        windows = _window_descriptions(descs, window_seconds=600.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual([d["description"] for d in windows[0]], ["a", "b"])
        self.assertEqual([d["description"] for d in windows[1]], ["c"])


class TestPromptRendering(unittest.TestCase):
    def test_descriptions_appear_with_chunk_relative_times(self):
        ctx = ChunkContext(
            chunk_index=0, n_chunks=1,
            chunk_start_wallclock_s=600.0,  # 10:00 absolute
            chunk_end_wallclock_s=1200.0,   # 20:00 absolute
        )
        descs = [
            {"start_s": 600.0, "end_s": 610.0, "description": "kitchen sounds"},
            {"start_s": 610.0, "end_s": 620.0, "description": "running water"},
        ]
        prompt = render_prompt_from_descriptions(
            ctx, descs, SINS_LABELS, time_unit="second",
        )
        # Absolute times mapped to chunk-relative.
        self.assertIn("[00:00-00:10] kitchen sounds", prompt)
        self.assertIn("[00:10-00:20] running water", prompt)
        # Cascade system instruction (text input, not audio).
        self.assertIn("timestamped descriptions", prompt)
        # Label set is included.
        self.assertIn("cooking", prompt)


class TestCascadeStageB(unittest.TestCase):
    def test_end_to_end_with_fake_text_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # 70 descriptions × 10 s = 700 s timeline. With 10-min windows
            # that's 2 windows: 0..59 → win 0 (600s), 60..69 → win 1 (100s).
            descs = [
                {"chunk_idx": i, "start_s": float(i * 10), "end_s": float((i + 1) * 10),
                 "description": "cooking sounds in the kitchen."}
                for i in range(70)
            ]
            jsonl = td / "fake.descriptions.jsonl"
            with jsonl.open("w") as fp:
                for d in descs:
                    fp.write(json.dumps(d) + "\n")

            text_adapter = FakeTextLLMAdapter(responses=[
                {"segments": [{"start": "00:00", "end": "10:00", "event": "cooking"}]},
                {"segments": [{"start": "00:00", "end": "01:40", "event": "cooking"}]},
            ])
            summary = run_cascade_segmentation(
                jsonl, text_adapter, td / "out", SINS_LABELS,
                window_minutes=10.0,
                audio_path=td / "fake.flac",
                audio_duration_s=700.0,
            )
            self.assertEqual(summary.n_chunks, 2)
            self.assertEqual(summary.n_chunks_completed, 2)
            self.assertEqual(summary.n_segments_total, 2)
            # summary.json drops in eval scripts (parity with chunk_runner).
            self.assertTrue((td / "out" / "summary.json").exists())
            self.assertEqual(len(list((td / "out").glob("chunk_*.json"))), 2)
            # Absolute coords: window 1's segment starts at 600 s, ends at 700.
            chunk1 = json.loads((td / "out" / "chunk_00001.json").read_text())
            self.assertEqual(chunk1["segments_abs"][0]["label"], "cooking")
            self.assertEqual(chunk1["segments_abs"][0]["start"], 600.0)
            self.assertEqual(chunk1["segments_abs"][0]["end"], 700.0)

    def test_structured_against_unstructured_text_adapter_raises(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            jsonl = td / "fake.descriptions.jsonl"
            jsonl.write_text(
                json.dumps({"chunk_idx": 0, "start_s": 0.0, "end_s": 10.0,
                            "description": "a"}) + "\n"
            )
            text_adapter = FakeTextLLMAdapter(
                responses=[{"segments": []}], structured=False,
            )
            with self.assertRaises(RuntimeError):
                run_cascade_segmentation(
                    jsonl, text_adapter, td / "out", SINS_LABELS,
                    window_minutes=10.0,
                )


if __name__ == "__main__":
    unittest.main()
