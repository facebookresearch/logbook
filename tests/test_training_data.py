# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Ego4D SFT training-data builder (long_audio.training.data).

Core tests use a hermetic synthetic manifest (no audio files, no network) so
segment counts are exactly predictable. A final group runs against the real
``datasets/ego4d/annotated_manifest.json`` when present, to sanity-check the
count-vs-chunk-length trend on real sliding-window slices.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_audio.datasets.ego4d.schema import ATUS_LABELS
from long_audio.training.data import (
    Ego4DSFTDataset,
    disjoint_slice_segments,
    quantize_and_format,
    split_counts,
    target_to_eval_events,
)

_REAL_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "datasets" / "ego4d" / "annotated_manifest.json"
)


def _slice(start, end, event, summary):
    return {
        "start": start, "end": end, "event": event,
        "summary": summary, "facts": [], "raw_summary": "", "annotation_uid": "x",
    }


def _make_video(uid, split, duration, events):
    """Build a synthetic video with disjoint 5-min slices, one per event.

    Slices tile ``[0, duration]`` exactly (no sliding-window overlap) so the
    midpoint boundaries land on the 5-min grid and segment counts are exact.
    """
    n = len(events)
    block = duration / n
    slices = [
        _slice(i * block, (i + 1) * block, ev, f"desc {ev} {i}")
        for i, ev in enumerate(events)
    ]
    # actions = coalesced timeline (events are distinct here so 1:1 with slices).
    actions = [{"start": s["start"], "end": s["end"], "event": s["event"]} for s in slices]
    pass_body = {"slices": slices, "actions": actions, "facts": []}
    return {
        "uid": uid,
        "audio_path": f"/nonexistent/{uid}.flac",
        "audio_offset_s": 0.0,
        "duration": float(duration),
        "sample_rate": 16000,
        "scenarios": ["synthetic"],
        "fb_participant_id": uid,
        "video_source": "synthetic",
        "split": split,
        "passes": {"1": pass_body, "2": json.loads(json.dumps(pass_body))},
        "moments": [],
    }


def _write_manifest(videos):
    d = tempfile.mkdtemp()
    path = Path(d) / "annotated_manifest.json"
    path.write_text(json.dumps({"dataset": "ego4d", "videos": videos}))
    return path


# Four distinct ATUS events so no same-label coalescing muddies the counts.
_EV4 = ["productive", "food", "leisure", "purchasing"]


class TestDisjointSliceSegments(unittest.TestCase):
    def test_contiguous_and_carries_description(self):
        # Overlapping sliding-window slices (like real Ego4D).
        slices = [
            _slice(0.0, 300.0, "productive", "s0"),
            _slice(270.0, 570.0, "leisure", "s1"),
            _slice(540.0, 840.0, "leisure", "s2"),
        ]
        segs = disjoint_slice_segments(slices, duration=840.0)
        self.assertEqual(len(segs), 3)  # one per slice, no coalescing
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[-1]["end"], 840.0)
        # contiguous, non-overlapping
        for a, b in zip(segs, segs[1:]):
            self.assertAlmostEqual(a["end"], b["start"])
        # description == slice summary
        self.assertEqual([s["description"] for s in segs], ["s0", "s1", "s2"])
        # midpoint boundary between slice0 (end 300) and slice1 (start 270) = 285
        self.assertAlmostEqual(segs[0]["end"], 285.0)

    def test_empty(self):
        self.assertEqual(disjoint_slice_segments([], 100.0), [])


class TestQuantizeAndFormat(unittest.TestCase):
    def test_minute_grid_basic(self):
        rel = [
            {"start": 0.0, "end": 300.0, "event": "food", "description": "d0"},
            {"start": 300.0, "end": 600.0, "event": "leisure", "description": "d1"},
        ]
        out = quantize_and_format(rel, 600.0, "minute", with_description=True)
        self.assertEqual([(s["start"], s["end"]) for s in out], [("0", "5"), ("5", "10")])
        self.assertEqual(out[0]["description"], "d0")

    def test_second_grid_mmss(self):
        # Two segments so 285 is an *internal* boundary (04:45), not snapped
        # to chunk end; the final segment extends to chunk end (10:00).
        rel = [
            {"start": 0.0, "end": 285.0, "event": "food", "description": "d0"},
            {"start": 285.0, "end": 600.0, "event": "leisure", "description": "d1"},
        ]
        out = quantize_and_format(rel, 600.0, "second", with_description=True)
        self.assertEqual((out[0]["start"], out[0]["end"]), ("00:00", "04:45"))
        self.assertEqual((out[1]["start"], out[1]["end"]), ("04:45", "10:00"))

    def test_collapsed_segment_dropped_but_coverage_kept(self):
        # A sub-minute sliver that rounds to zero length must not create a gap.
        rel = [
            {"start": 0.0, "end": 290.0, "event": "food", "description": "d0"},
            {"start": 290.0, "end": 300.0, "event": "leisure", "description": "d1"},
        ]
        out = quantize_and_format(rel, 300.0, "minute", with_description=True)
        # 290/60=4.83->5, 300/60=5 => second seg [5,5] drops; first covers [0,5]
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["start"], out[0]["end"]), ("0", "5"))

    def test_without_description(self):
        rel = [{"start": 0.0, "end": 300.0, "event": "food", "description": "d"}]
        out = quantize_and_format(rel, 300.0, "minute", with_description=False)
        self.assertNotIn("description", out[0])


class TestEgo4DSFTDataset(unittest.TestCase):
    def setUp(self):
        # One 20-min video (4 distinct 5-min blocks) in each split.
        self.videos = [
            _make_video("v_train", "train", 1200, _EV4),
            _make_video("v_val", "val", 1200, _EV4),
            _make_video("v_test", "test", 1200, _EV4),
        ]
        self.manifest = _write_manifest(self.videos)

    def _ds(self, chunk_minutes, split="train", time_unit="minute", **kw):
        return Ego4DSFTDataset(
            self.manifest, split=split, chunk_minutes=chunk_minutes,
            time_unit=time_unit, load_audio=False, **kw,
        )

    def test_segment_counts_scale_with_chunk_length(self):
        # 20-min video tiled into 4 distinct 5-min blocks.
        self.assertEqual([e.n_segments for e in self._ds(5).iter_examples()],
                         [1, 1, 1, 1])   # 4 chunks x 1 segment (classification)
        self.assertEqual([e.n_segments for e in self._ds(10).iter_examples()],
                         [2, 2])          # 2 chunks x 2 segments
        self.assertEqual([e.n_segments for e in self._ds(20).iter_examples()],
                         [4])             # 1 chunk x 4 segments

    def test_10min_two_segments_20min_four(self):
        self.assertEqual(self._ds(10)[0].n_segments, 2)
        self.assertEqual(self._ds(20)[0].n_segments, 4)

    def test_target_format_matches_schema_and_inference_parser(self):
        ex = self._ds(20)[0]
        obj = ex.target_obj
        self.assertEqual(list(obj.keys()), ["segments"])
        for seg in obj["segments"]:
            self.assertEqual(set(seg.keys()), {"start", "end", "event", "description"})
            self.assertIsInstance(seg["start"], str)
            self.assertIn(seg["event"], ATUS_LABELS)
        # Round-trip through the *inference* parser => byte-compatible gold.
        events = target_to_eval_events(ex)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["label"], "productive")
        self.assertEqual(events[0]["start"], 0.0)
        self.assertEqual(events[-1]["end"], 1200.0)
        self.assertTrue(all("description" in e for e in events))

    def test_target_covers_chunk_contiguously(self):
        ex = self._ds(20)[0]
        events = target_to_eval_events(ex)
        self.assertEqual(events[0]["start"], 0.0)
        for a, b in zip(events, events[1:]):
            self.assertEqual(a["end"], b["start"])  # no gaps / overlaps

    def test_prompt_uses_ego4d_labels_and_description(self):
        ex = self._ds(10)[0]
        self.assertIn("segments", ex.prompt)
        self.assertIn("description", ex.prompt)      # with_description bullet
        for lbl in ATUS_LABELS:
            self.assertIn(lbl, ex.prompt)

    def test_split_filtering_and_counts(self):
        self.assertEqual(len(self._ds(20, split="train")._ds.videos), 1)
        self.assertEqual(len(self._ds(20, split="val")._ds.videos), 1)
        self.assertEqual(len(self._ds(20, split="test")._ds.videos), 1)
        counts = split_counts(self.manifest)
        self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})
        # A train dataset must not surface val/test uids.
        uids = {e.uid for e in self._ds(20, split="train").iter_examples()}
        self.assertEqual(uids, {"v_train"})

    def test_chunk_relative_timestamps(self):
        # Second chunk of a 10-min split must restart timestamps at 0.
        ds = self._ds(10)
        ex1 = ds[1]
        self.assertEqual(ex1.chunk_start_s, 600.0)
        events = target_to_eval_events(ex1)
        self.assertEqual(events[0]["start"], 0.0)  # chunk-relative, not 600
        self.assertEqual(events[-1]["end"], 600.0)

    def test_without_description_flag(self):
        ex = self._ds(10, with_description=False)[0]
        for seg in ex.target_obj["segments"]:
            self.assertNotIn("description", seg)
        self.assertNotIn("description", ex.prompt)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant\n":
            return [777]
        return [abs(hash(t)) % 1000 + 1 for t in text.split()]


class _FakeEosTokenizer(_FakeTokenizer):
    pad_token_id = 0
    eos_token_id = 0


class _FakeProcessor:
    """Minimal stand-in for Qwen3OmniMoeProcessor to test label masking.

    Whitespace-tokenizes text into integer ids; the prompt-only render is a
    strict prefix of the full render, mirroring the real processor's behaviour
    so the collator's prefix-masking is exercised without the 30B model.
    """

    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, conv, add_generation_prompt=False, tokenize=False):
        prefix = "U U U <|im_start|>assistant"
        if len(conv) == 2 and conv[-1]["role"] == "assistant":
            return prefix + " " + conv[-1]["content"][0]["text"]
        return prefix

    def __call__(self, text, audio, sampling_rate, return_tensors, padding):
        import torch

        vocab = {"<|im_start|>assistant": 777}
        ids = []
        for s in text:
            row = []
            for tok in s.split():
                if tok not in vocab:
                    vocab[tok] = len(vocab) + 1
                row.append(vocab[tok])
            ids.append(row)
        maxlen = max(len(x) for x in ids)
        att = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]
        ids = [x + [0] * (maxlen - len(x)) for x in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}


class _FakeEosProcessor(_FakeProcessor):
    tokenizer = _FakeEosTokenizer()

    def apply_chat_template(self, conv, add_generation_prompt=False, tokenize=False):
        prefix = "U U U <|im_start|>assistant"
        if len(conv) == 2 and conv[-1]["role"] == "assistant":
            return prefix + " " + conv[-1]["content"][0]["text"] + " <eos>"
        return prefix



class TestCollatorLabelMasking(unittest.TestCase):
    def test_prefix_masked_assistant_kept(self):
        import numpy as np

        from long_audio.training.data import Qwen3OmniSFTCollator, SFTExample

        ex = SFTExample(
            uid="v", chunk_index=0, n_chunks=1, split="train",
            chunk_start_s=0.0, chunk_end_s=600.0,
            prompt="segment this",
            target_text='{"segments": []}',
            target_obj={"segments": []},
            n_segments=0, gold_segments=[], time_unit="minute",
            audio=np.zeros(16000, dtype="float32"),
        )
        batch = Qwen3OmniSFTCollator(_FakeProcessor())([ex])
        labels = batch["labels"][0]
        # First 4 tokens ("U U U |||") are the prompt prefix -> masked.
        self.assertTrue((labels[:4] == -100).all())
        # At least one assistant token survives (not all -100).
        self.assertTrue((labels[4:] != -100).any())

    def test_pad_equals_eos_fails_fast(self):
        from long_audio.training.data import Qwen3OmniSFTCollator

        with self.assertRaisesRegex(RuntimeError, "pad_token_id .* equals eos_token_id"):
            Qwen3OmniSFTCollator(_FakeEosProcessor())

    def test_missing_audio_raises(self):
        from long_audio.training.data import Qwen3OmniSFTCollator, SFTExample

        ex = SFTExample(
            uid="v", chunk_index=0, n_chunks=1, split="train",
            chunk_start_s=0.0, chunk_end_s=600.0, prompt="p",
            target_text="{}", target_obj={}, n_segments=0, gold_segments=[],
            time_unit="minute", audio=None,
        )
        with self.assertRaises(ValueError):
            Qwen3OmniSFTCollator(_FakeProcessor())([ex])


@unittest.skipUnless(_REAL_MANIFEST.exists(), "real annotated_manifest.json absent")
class TestRealManifest(unittest.TestCase):
    def test_segment_count_trend(self):
        means = {}
        for cm in (5, 10, 20):
            ds = Ego4DSFTDataset(
                _REAL_MANIFEST, split="train", chunk_minutes=cm,
                time_unit="minute", load_audio=False,
            )
            n = min(300, len(ds))
            counts = [ds[i].n_segments for i in range(n)]
            means[cm] = sum(counts) / len(counts)
        # Longer chunks must teach more segments (the whole point).
        self.assertLess(means[5], means[10])
        self.assertLess(means[10], means[20])
        self.assertGreaterEqual(means[10], 1.8)   # ~2 segments for 10-min
        self.assertGreaterEqual(means[20], 3.0)   # ~4 segments for 20-min

    def test_gold_roundtrips_through_inference_parser(self):
        ds = Ego4DSFTDataset(
            _REAL_MANIFEST, split="train", chunk_minutes=10,
            time_unit="minute", load_audio=False,
        )
        ex = ds[0]
        events = target_to_eval_events(ex)
        self.assertTrue(events)
        for e in events:
            self.assertEqual(set(e.keys()) >= {"label", "start", "end"}, True)
            self.assertIn(e["label"], ATUS_LABELS)


if __name__ == "__main__":
    unittest.main()
