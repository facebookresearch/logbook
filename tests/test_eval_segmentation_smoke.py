"""Smoke test for scripts/eval_run_ego4d.py against a synthetic manifest.

Runs the full inference + eval cycle end-to-end using FakeAdapter for
the inference side and a tiny hand-built Ego4D-shape manifest for the
GT side. Verifies that eval_run_ego4d.py correctly stitches per-chunk
raw predictions and produces an eval_segmentation.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from long_audio.datasets.ego4d.schema import ATUS_LABELS
from long_audio.inference.chunk_runner import run_inference
from long_audio.inference.models.fake import FakeAdapter


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_test_audio(path: Path, duration_s: float, sr: int = 16000) -> None:
    n = int(round(duration_s * sr))
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.01, n).astype(np.float32)
    soundfile.write(str(path), audio, sr, format="FLAC", subtype="PCM_16")


class TestEvalRunEgo4DSmoke(unittest.TestCase):
    def test_end_to_end_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio_path = tmp / "uid_smoke.flac"
            _make_test_audio(audio_path, duration_s=60.0)

            # Two chunks of 30 s; emit one work and one leisure-class segment.
            adapter = FakeAdapter(
                responses=[
                    {"segments": [
                        {"start": "00:00", "end": "00:30", "event": "productive"},
                    ]},
                    {"segments": [
                        {"start": "00:00", "end": "00:30", "event": "leisure"},
                    ]},
                ],
                structured=True,
                latency_s=0.0,
            )
            run_dir = tmp / "run" / "uid_smoke"
            summary = run_inference(
                audio_path, adapter, run_dir, ATUS_LABELS,
                chunk_minutes=0.5, decoder="structured",
                dataset_name="Ego4D", t0_iso="1970-01-01T00:00:00Z",
            )
            # Build the run_index.json that eval_run_ego4d expects.
            run_root = run_dir.parent
            (run_root / "run_index.json").write_text(json.dumps({
                "model": "fake",
                "dataset": "ego4d",
                "n_videos": 1,
                "videos": [{
                    "uid": "uid_smoke",
                    "duration": 60.0,
                    "n_chunks": summary.n_chunks,
                    "n_chunks_completed": summary.n_chunks_completed,
                    "n_segments": summary.n_segments_total,
                    "rt_factor": summary.rt_factor,
                    "wall_s": summary.total_inference_s,
                    "summary_path": str(run_dir / "summary.json"),
                }],
            }))

            # Build a hand-crafted Ego4D-shape manifest with one video and
            # pass-1 + pass-2 ground truth that mostly agrees with the
            # FakeAdapter's predictions.
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps({
                "videos": [{
                    "uid": "uid_smoke",
                    "duration": 60.0,
                    "audio_path": str(audio_path),
                    "sample_rate": 16000,
                    "passes": {
                        "1": {"actions": [
                            {"start": 0.0, "end": 30.0, "event": "productive"},
                            {"start": 30.0, "end": 60.0, "event": "leisure"},
                        ]},
                        "2": {"actions": [
                            {"start": 0.0, "end": 30.0, "event": "productive"},
                            {"start": 30.0, "end": 60.0, "event": "leisure"},
                        ]},
                    },
                }],
            }))

            out_path = tmp / "eval_segmentation.json"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_segmentation.py"),
                "--dataset", "ego4d",
                str(run_root),
                "--manifest", str(manifest_path),
                "--output", str(out_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                res.returncode, 0,
                f"eval_run_ego4d failed: STDOUT={res.stdout}\nSTDERR={res.stderr}",
            )
            self.assertTrue(out_path.exists())
            agg = json.loads(out_path.read_text())
            self.assertEqual(agg["n_videos"], 1)
            self.assertEqual(agg["n_data_points"], 2)  # pass-1 and pass-2
            # Perfect agreement → frame accuracy should be 1.0.
            self.assertAlmostEqual(agg["micro"]["frame_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
