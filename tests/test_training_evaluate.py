# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the generation-based Ego4D eval driver + callback + HF adapter.

The driver (:func:`run_ego4d_eval`) and callback are exercised END-TO-END on CPU
with ``FakeAdapter`` (it drives the real ``chunk_runner`` + shared aggregation);
only the real Qwen3-Omni ``generate`` is GPU-only. The HF adapter's generate
plumbing (chat template -> processor -> generate -> slice/decode) is tested with
tiny fakes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from long_audio.inference.models.fake import FakeAdapter
from long_audio.training.evaluate import Ego4DEvalCallback, run_ego4d_eval


def _write_flac(path: Path, duration_s: float, sr: int = 16000) -> None:
    n = int(round(duration_s * sr))
    rng = np.random.default_rng(0)
    soundfile.write(str(path), rng.normal(0, 0.01, n).astype(np.float32), sr,
                    format="FLAC", subtype="PCM_16")


def _pass(actions):
    return {"actions": actions, "slices": [], "facts": []}


def _make_manifest(tmp: Path, n_videos: int = 2, duration: float = 60.0) -> Path:
    acts = [
        {"start": 0.0, "end": 30.0, "event": "productive"},
        {"start": 30.0, "end": 60.0, "event": "leisure"},
    ]
    videos = []
    for i in range(n_videos):
        uid = f"uid{i}"
        _write_flac(tmp / f"{uid}.flac", duration)
        videos.append({
            "uid": uid,
            "audio_path": str(tmp / f"{uid}.flac"),
            "audio_offset_s": 0.0,
            "duration": duration,
            "sample_rate": 16000,
            "scenarios": ["x"],
            "fb_participant_id": uid,
            "video_source": "syn",
            "split": "val",
            "passes": {"1": _pass(acts), "2": _pass(acts)},
            "moments": [],
        })
    path = tmp / "manifest.json"
    path.write_text(json.dumps({"dataset": "ego4d", "videos": videos}))
    return path


# FakeAdapter responses (MM:SS, 30 s chunks): productive then leisure, cycled.
_RESPONSES = [
    {"segments": [{"start": "00:00", "end": "00:30", "event": "productive"}]},
    {"segments": [{"start": "00:00", "end": "00:30", "event": "leisure"}]},
]


class TestRunEgo4DEval(unittest.TestCase):
    def test_end_to_end_with_fake_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            manifest = _make_manifest(tmp, n_videos=2)
            adapter = FakeAdapter(responses=_RESPONSES, structured=True)
            agg = run_ego4d_eval(
                adapter, manifest, "val",
                chunk_minutes=0.5, time_unit="second", with_description=False,
                model_name="fake",
            )
            # Same format as scripts/eval_run_ego4d.py.
            self.assertEqual(
                set(agg.keys()),
                {"model", "n_videos", "n_data_points", "total_chunks",
                 "boundary_tolerance_s", "macro", "micro", "pred_missing",
                 "per_datapoint"},
            )
            self.assertEqual(agg["n_videos"], 2)
            self.assertEqual(agg["n_data_points"], 4)  # 2 videos x 2 passes
            # Predictions exactly match GT -> perfect frame accuracy.
            self.assertAlmostEqual(agg["micro"]["frame_accuracy"], 1.0)
            self.assertIsNotNone(agg["macro"]["boundary_f1"])

    def test_max_videos_caps_the_set(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            manifest = _make_manifest(tmp, n_videos=3)
            adapter = FakeAdapter(responses=_RESPONSES, structured=True)
            agg = run_ego4d_eval(
                adapter, manifest, "val", chunk_minutes=0.5, time_unit="second",
                with_description=False, max_videos=1, model_name="fake",
            )
            self.assertEqual(agg["n_videos"], 1)


class _FakeTrainer:
    def __init__(self):
        self.logged: dict = {}

    def log(self, metrics):
        self.logged.update(metrics)


class TestEgo4DEvalCallback(unittest.TestCase):
    def test_on_evaluate_logs_headline_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            manifest = _make_manifest(tmp, n_videos=2)
            cb = Ego4DEvalCallback(
                FakeAdapter(responses=_RESPONSES, structured=True),
                manifest, "val",
                chunk_minutes=0.5, time_unit="second", with_description=False,
                max_videos=2, model_name="fake",
            )
            trainer = _FakeTrainer()
            cb.bind_trainer(trainer)
            # model=None -> the eval/use_cache toggles are safely skipped.
            cb.on_evaluate(args=None, state=None, control="CTRL", model=None)
            self.assertEqual(
                set(trainer.logged.keys()),
                {"eval_seg/boundary_f1", "eval_seg/frame_accuracy",
                 "eval_seg/event_f1", "eval_seg/event_error_rate"},
            )
            self.assertAlmostEqual(trainer.logged["eval_seg/frame_accuracy"], 1.0)


class _FakeProcessor:
    def apply_chat_template(self, conv, add_generation_prompt=False, tokenize=False):
        return "PROMPT"

    def __call__(self, text, audio, sampling_rate, return_tensors):
        import torch
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ['{"segments": [{"start": "0", "end": "1", "event": "leisure"}]}']


class _FakeModel:
    def parameters(self):
        return iter([])  # -> adapter._device() returns None, skips .to()

    def generate(self, **kwargs):
        import torch
        # Prompt was 3 tokens; return prompt + 4 generated tokens.
        return torch.tensor([[1, 2, 3, 9, 9, 9, 9]])


class TestQwen3OmniHFAdapter(unittest.TestCase):
    def test_generate_slices_and_decodes(self):
        from long_audio.training.infer import Qwen3OmniHFAdapter

        adapter = Qwen3OmniHFAdapter(_FakeModel(), _FakeProcessor())
        self.assertFalse(adapter.supports_structured())
        out = adapter.generate(
            np.zeros(16000, dtype="float32"), 16000, "seg this", max_new_tokens=8
        )
        self.assertIn("segments", out.raw_text)
        self.assertIsNone(out.raw_json)
        self.assertIsNotNone(out.latency_s)


if __name__ == "__main__":
    unittest.main()
