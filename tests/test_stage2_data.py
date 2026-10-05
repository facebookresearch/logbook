"""Tests for Stage-2 text-only SFT data built from fixed Stage-A captions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
from long_audio.inference.prompt import ChunkContext, render_prompt_from_descriptions
from long_audio.training.stage2_data import Stage2SFTDataset, target_to_eval_events


def _slice(start, end, event, summary):
    return {
        "start": start, "end": end, "event": event,
        "summary": summary, "facts": [], "raw_summary": "", "annotation_uid": "x",
    }


def _make_video(uid, split, duration, events):
    n = len(events)
    block = duration / n
    slices = [
        _slice(i * block, (i + 1) * block, ev, f"desc {ev} {i}")
        for i, ev in enumerate(events)
    ]
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


def _write_manifest(root: Path, videos):
    path = root / "annotated_manifest.json"
    path.write_text(json.dumps({"dataset": "ego4d", "videos": videos}))
    return path


def _write_descriptions(root: Path, uid: str, *, model: str, n: int, step_s: float = 10.0, split: str | None = None):
    out_dir = root / "ego4d"
    if split is not None:
        out_dir = out_dir / split
    out_dir = out_dir / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}.descriptions.jsonl"
    with path.open("w") as fp:
        for i in range(n):
            fp.write(json.dumps({
                "chunk_idx": i,
                "start_s": i * step_s,
                "end_s": (i + 1) * step_s,
                "description": f"caption {i}",
                "latency_s": 0.01,
                "metadata": {"model_id": "fake-captioner"},
            }) + "\n")
    return path


_EV4 = ["productive", "food", "leisure", "purchasing"]


class TestStage2SFTDataset(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.model = "af-next-captioner"
        self.videos = [
            _make_video("v_train", "train", 1200.0, _EV4),
            _make_video("v_val", "val", 1200.0, _EV4),
        ]
        self.manifest = _write_manifest(self.root, self.videos)
        self.captions_root = self.root / "runs" / "cascA_inference" / self.model
        _write_descriptions(self.captions_root, "v_train", model=self.model, n=120, split="train")

    def tearDown(self):
        self.td.cleanup()

    def _ds(self, **kw):
        return Stage2SFTDataset(
            self.manifest,
            self.captions_root,
            describe_model=self.model,
            split="train",
            window_minutes=10.0,
            time_unit="minute",
            **kw,
        )

    def test_builds_one_example_per_stage_b_window(self):
        ds = self._ds()
        self.assertEqual(len(ds), 2)
        ex0, ex1 = ds[0], ds[1]
        self.assertEqual(ex0.uid, "v_train")
        self.assertEqual(ex0.window_index, 0)
        self.assertEqual(ex0.n_windows, 2)
        self.assertEqual((ex0.window_start_s, ex0.window_end_s), (0.0, 600.0))
        self.assertEqual((ex1.window_start_s, ex1.window_end_s), (600.0, 1200.0))
        self.assertEqual(len(ex0.descriptions), 60)
        self.assertEqual(len(ex1.descriptions), 60)

    def test_target_matches_gold_and_inference_parser(self):
        ex = self._ds()[0]
        self.assertEqual(list(ex.target_obj.keys()), ["segments"])
        self.assertEqual(
            [(s["start"], s["end"], s["event"]) for s in ex.target_obj["segments"]],
            [("0", "5", "productive"), ("5", "10", "food")],
        )
        events = target_to_eval_events(ex)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["label"], "productive")
        self.assertEqual(events[0]["start"], 0.0)
        self.assertEqual(events[-1]["end"], 600.0)
        self.assertTrue(all("description" in e for e in events))

    def test_prompt_is_byte_compatible_with_cascade_renderer(self):
        ex = self._ds()[0]
        ctx = ChunkContext(
            chunk_index=0,
            n_chunks=2,
            chunk_start_wallclock_s=0.0,
            chunk_end_wallclock_s=600.0,
            t0_iso="1970-01-01T00:00:00Z",
            dataset_name="ego4d",
        )
        expected = render_prompt_from_descriptions(
            ctx,
            ex.descriptions,
            ATUS_LABELS,
            label_hints=ATUS_HINTS,
            context_mode="none",
            time_unit="minute",
            with_description=True,
        )
        self.assertEqual(ex.prompt, expected)
        self.assertIn("# Audio descriptions", ex.prompt)
        self.assertIn("[0-0] caption 0", ex.prompt)

    def test_without_description_flag(self):
        ex = self._ds(with_description=False)[0]
        for seg in ex.target_obj["segments"]:
            self.assertNotIn("description", seg)
        self.assertNotIn('"description"', ex.target_text)
        self.assertNotIn('"description": 1-2 sentences', ex.prompt)

    def test_text_only_conversation_shape(self):
        ex = self._ds()[0]
        conv = ex.conversation()
        self.assertEqual([m["role"] for m in conv], ["user", "assistant"])
        self.assertEqual(conv[0]["content"][0]["type"], "text")
        self.assertEqual(conv[0]["content"][0]["text"], ex.prompt)
        self.assertEqual(conv[1]["content"][0]["text"], ex.target_text)

    def test_missing_captions_error_or_skip(self):
        with self.assertRaises(FileNotFoundError):
            Stage2SFTDataset(
                self.manifest,
                self.captions_root,
                describe_model=self.model,
                split="val",
            )
        ds = Stage2SFTDataset(
            self.manifest,
            self.captions_root,
            describe_model=self.model,
            split="val",
            skip_missing_captions=True,
        )
        self.assertEqual(len(ds), 0)

    def test_captions_root_can_point_at_split_dir(self):
        ds = Stage2SFTDataset(
            self.manifest,
            self.captions_root / "ego4d" / "train",
            describe_model=self.model,
            split="train",
            window_minutes=10.0,
        )
        self.assertEqual(len(ds), 2)

if __name__ == "__main__":
    unittest.main()
