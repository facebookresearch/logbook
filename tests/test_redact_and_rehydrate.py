# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Round-trip tests for the public-manifest pipeline:

- ``to_repo_relative`` / ``resolve_audio_path`` helpers.
- ``redact_manifest.py`` for ego4d + egolife — field whitelist, path
  stripping, strict-abs error.
- ``build_manifest.py --public-manifest`` for ego4d — rehydrate from
  public + synthetic narration.json, verify raw_summary join (including
  the duplicate-annotation_uid collision case).
- Description-eval adapters raise cleanly on segmentation-only
  manifests.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_audio.utils.paths import (
    REPO_ROOT,
    resolve_audio_path,
    to_repo_relative,
)
from scripts.data.ego4d.build_manifest import (
    MissingAnnotationError,
    MissingVideoError,
    build_manifest_from_public,
)
from scripts.data.ego4d.redact_manifest import (
    _relativize_audio_path as ego4d_relativize,
    redact as ego4d_redact,
)
from scripts.data.egolife.redact_manifest import (
    _relativize_audio_path as egolife_relativize,
    redact as egolife_redact,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers(unittest.TestCase):
    def test_relative_under_repo(self):
        abs_path = REPO_ROOT / "datasets" / "ego4d" / "audio" / "x.flac"
        self.assertEqual(
            to_repo_relative(abs_path),
            "datasets/ego4d/audio/x.flac",
        )

    def test_already_relative_passthrough(self):
        self.assertEqual(
            to_repo_relative("datasets/egolife/audio/y.flac"),
            "datasets/egolife/audio/y.flac",
        )

    def test_outside_repo_fallback_to_absolute(self):
        # Does not raise; returns the absolute path (and prints WARN).
        out = to_repo_relative("/mnt/scratch/foo.flac", "[test]")
        self.assertEqual(out, "/mnt/scratch/foo.flac")

    def test_resolve_absolute_passthrough(self):
        p = resolve_audio_path("/abs/path.flac")
        self.assertEqual(p, Path("/abs/path.flac"))

    def test_resolve_relative_prepends_repo(self):
        p = resolve_audio_path("datasets/ego4d/audio/x.flac")
        self.assertTrue(p.is_absolute())
        self.assertEqual(p, REPO_ROOT / "datasets/ego4d/audio/x.flac")


# ---------------------------------------------------------------------------
# Redactor error semantics
# ---------------------------------------------------------------------------

class TestRedactorRelativize(unittest.TestCase):
    def test_ego4d_relativize_under_repo(self):
        s = ego4d_relativize(str(REPO_ROOT / "datasets" / "ego4d" / "audio" / "x.flac"))
        self.assertEqual(s, "datasets/ego4d/audio/x.flac")

    def test_ego4d_relativize_absolute_outside_raises(self):
        with self.assertRaises(ValueError) as cm:
            ego4d_relativize("/elsewhere/x.flac")
        self.assertIn("not under REPO_ROOT", str(cm.exception))

    def test_egolife_relativize_absolute_outside_raises(self):
        with self.assertRaises(ValueError):
            egolife_relativize("/elsewhere/y.flac")

    def test_relativize_already_relative_passthrough(self):
        self.assertEqual(
            ego4d_relativize("datasets/ego4d/audio/x.flac"),
            "datasets/ego4d/audio/x.flac",
        )


# ---------------------------------------------------------------------------
# Synthetic full manifests → redactor whitelist check
# ---------------------------------------------------------------------------

def _tiny_ego4d_full() -> dict:
    """One-video Ego4D annotated-manifest fixture."""
    return {
        "dataset": "Ego4D",
        "sample_rate": 16000,
        "n_videos": 1,
        "annotated_at": "2026-01-01T00:00:00+00:00",
        "olmo_model_id": "test/olmo",
        "split_config": {"ratios": {"train": 60, "val": 20, "test": 20}},
        "videos": [
            {
                "uid": "u1",
                "audio_path": str(REPO_ROOT / "datasets" / "ego4d" / "audio" / "u1.flac"),
                "audio_offset_s": 0.5,
                "duration": 600.0,
                "sample_rate": 16000,
                "scenarios": ["cooking"],
                "fb_participant_id": "pid1",
                "video_source": "iiith",
                "split": "test",
                "passes": {
                    "1": {
                        "slices": [
                            {
                                "start": 0.0, "end": 300.0,
                                "annotation_uid": "a1",
                                "raw_summary": "LICENSED TEXT 1",
                                "summary": "cleaned",
                                "facts": ["fact1"],
                                "event": "productive",
                                "raw_classify": "productive",
                            },
                        ],
                        "actions": [{"start": 0.0, "end": 300.0, "event": "productive"}],
                        "facts": [{"start": 0.0, "end": 300.0, "facts": ["fact1"]}],
                    },
                    "2": {
                        "slices": [
                            {
                                "start": 0.0, "end": 300.0,
                                "annotation_uid": "a2",
                                "raw_summary": "LICENSED TEXT 2",
                                "summary": "cleaned2",
                                "facts": ["fact2"],
                                "event": "leisure",
                                "raw_classify": "leisure",
                            },
                        ],
                        "actions": [{"start": 0.0, "end": 300.0, "event": "leisure"}],
                        "facts": [{"start": 0.0, "end": 300.0, "facts": ["fact2"]}],
                    },
                },
                "moments": [{"start": 10.0, "end": 20.0, "label": "cutting"}],
            }
        ],
    }


def _tiny_egolife_full() -> dict:
    return {
        "dataset": "egolife",
        "version": "v1",
        "date": "2026-01-01T00:00:00+00:00",
        "sample_rate": 16000,
        "channels": 1,
        "summary_window_s": 300.0,
        "n_videos": 1,
        "videos": [
            {
                "uid": "A1_JAKE_DAY1_S01",
                "audio_path": str(REPO_ROOT / "datasets" / "egolife" / "audio" / "A1_JAKE" / "DAY1" / "S01.flac"),
                "sample_rate": 16000,
                "duration": 600.0,
                "audio_offset_s": 0.0,
                "wall_clock_start": "11:00:00.00",
                "wall_clock_end": "11:10:00.00",
                "scenarios": ["egolife_home"],
                "participant": "A1_JAKE",
                "day": "DAY1",
                "session_idx": 1,
                "clip_offsets": [],
                "passes": {
                    "1": {
                        "actions": [{"start": 0.0, "end": 600.0, "event": "leisure"}],
                        "summaries": [
                            {
                                "start": 0.0, "end": 300.0,
                                "text_en": "LICENSED EN 1", "text_zh": "LICENSED ZH 1",
                                "facts": ["f1"],
                            },
                            {
                                "start": 300.0, "end": 600.0,
                                "text_en": "LICENSED EN 2", "text_zh": "LICENSED ZH 2",
                                "facts": ["f2"],
                            },
                        ],
                    },
                },
                "moments": [],
            }
        ],
    }


_LICENSED_EGO4D_FIELDS = {"raw_summary", "summary", "facts", "event", "raw_classify"}
_LICENSED_EGOLIFE_FIELDS = {"text_en", "text_zh", "facts"}


class TestRedactorEgo4D(unittest.TestCase):
    def test_whitelist_shape_and_strips_licensed(self):
        with tempfile.TemporaryDirectory() as td:
            in_p = Path(td) / "full.json"
            out_p = Path(td) / "public.json"
            in_p.write_text(json.dumps(_tiny_ego4d_full()))
            public = ego4d_redact(in_p, out_p)

        v = public["videos"][0]
        self.assertEqual(
            set(v.keys()),
            {"uid", "audio_path", "audio_offset_s", "duration", "sample_rate",
             "scenarios", "fb_participant_id", "video_source", "split", "passes"},
        )
        self.assertNotIn("moments", v)
        self.assertEqual(v["audio_path"], "datasets/ego4d/audio/u1.flac")
        for pkey in ("1", "2"):
            self.assertEqual(set(v["passes"][pkey].keys()), {"slices", "actions"})
            self.assertNotIn("facts", v["passes"][pkey])
            for s in v["passes"][pkey]["slices"]:
                self.assertEqual(set(s.keys()), {"start", "end", "annotation_uid"})
                for forbidden in _LICENSED_EGO4D_FIELDS:
                    self.assertNotIn(forbidden, s)

    def test_absolute_outside_repo_raises(self):
        full = _tiny_ego4d_full()
        full["videos"][0]["audio_path"] = "/elsewhere/u1.flac"
        with tempfile.TemporaryDirectory() as td:
            in_p = Path(td) / "full.json"
            out_p = Path(td) / "public.json"
            in_p.write_text(json.dumps(full))
            with self.assertRaises(ValueError):
                ego4d_redact(in_p, out_p)


class TestRedactorEgoLife(unittest.TestCase):
    def test_whitelist_shape_and_strips_licensed(self):
        with tempfile.TemporaryDirectory() as td:
            in_p = Path(td) / "full.json"
            out_p = Path(td) / "public.json"
            in_p.write_text(json.dumps(_tiny_egolife_full()))
            public = egolife_redact(in_p, out_p)

        v = public["videos"][0]
        self.assertNotIn("moments", v)
        self.assertEqual(v["audio_path"], "datasets/egolife/audio/A1_JAKE/DAY1/S01.flac")
        self.assertEqual(set(v["passes"]["1"].keys()), {"actions", "summaries"})
        for sm in v["passes"]["1"]["summaries"]:
            self.assertEqual(set(sm.keys()), {"start", "end"})
            for forbidden in _LICENSED_EGOLIFE_FIELDS:
                self.assertNotIn(forbidden, sm)


# ---------------------------------------------------------------------------
# Ego4D rehydrate round-trip
# ---------------------------------------------------------------------------

def _tiny_narration() -> dict:
    """Narration.json fixture matching the slice annotation_uids above."""
    return {
        "u1": {
            "narration_pass_1": {
                "summaries": [
                    {"annotation_uid": "a1", "start_sec": 0.5, "end_sec": 300.5,
                     "summary_text": "LICENSED TEXT 1"},
                ],
            },
            "narration_pass_2": {
                "summaries": [
                    {"annotation_uid": "a2", "start_sec": 0.5, "end_sec": 300.5,
                     "summary_text": "LICENSED TEXT 2"},
                ],
            },
        },
    }


def _tiny_narration_with_duplicate() -> dict:
    """Narration.json fixture where annotation_uid=a1 maps to two
    distinct summary_texts within pass 1. The one with start_sec=0.5
    corresponds to the public slice at start=0.0 (public_start +
    audio_offset_s==0.5)."""
    base = _tiny_narration()
    base["u1"]["narration_pass_1"]["summaries"].insert(
        0,
        {"annotation_uid": "a1", "start_sec": 1000.0, "end_sec": 1300.0,
         "summary_text": "WRONG DUPLICATE"},
    )
    return base


class TestRehydrateEgo4D(unittest.TestCase):
    def test_roundtrip_no_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            narr_p = Path(td) / "narration.json"
            reh_p = Path(td) / "rehydrated.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            narr_p.write_text(json.dumps(_tiny_narration()))
            ego4d_redact(full_p, pub_p)

            build_manifest_from_public(
                public_manifest_path=pub_p,
                narration_path=narr_p,
                moments_paths=[],
                audio_root=REPO_ROOT / "datasets" / "ego4d" / "audio",
                output_path=reh_p,
            )
            reh = json.loads(reh_p.read_text())

        v = reh["videos"][0]
        self.assertEqual(v["uid"], "u1")
        self.assertEqual(v["audio_path"], "datasets/ego4d/audio/u1.flac")
        self.assertEqual(
            v["passes"]["1"]["slices"][0]["raw_summary"], "LICENSED TEXT 1"
        )
        self.assertEqual(
            v["passes"]["2"]["slices"][0]["raw_summary"], "LICENSED TEXT 2"
        )
        # Actions preserved verbatim from public manifest.
        self.assertEqual(
            v["passes"]["1"]["actions"],
            [{"start": 0.0, "end": 300.0, "event": "productive"}],
        )

    def test_duplicate_annotation_uid_resolved_by_start(self):
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            narr_p = Path(td) / "narration.json"
            reh_p = Path(td) / "rehydrated.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            narr_p.write_text(json.dumps(_tiny_narration_with_duplicate()))
            ego4d_redact(full_p, pub_p)
            build_manifest_from_public(
                public_manifest_path=pub_p,
                narration_path=narr_p,
                moments_paths=[],
                audio_root=REPO_ROOT / "datasets" / "ego4d" / "audio",
                output_path=reh_p,
            )
            reh = json.loads(reh_p.read_text())
        # Must pick the start=0.5 candidate (expected narr_start =
        # public_start + audio_offset_s = 0.0 + 0.5), not the start=1000.0
        # duplicate.
        self.assertEqual(
            reh["videos"][0]["passes"]["1"]["slices"][0]["raw_summary"],
            "LICENSED TEXT 1",
        )

    def test_missing_video_raises(self):
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            narr_p = Path(td) / "narration.json"
            reh_p = Path(td) / "rehydrated.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            narr_p.write_text(json.dumps({}))  # empty narration
            ego4d_redact(full_p, pub_p)
            with self.assertRaises(MissingVideoError):
                build_manifest_from_public(
                    public_manifest_path=pub_p,
                    narration_path=narr_p,
                    moments_paths=[],
                    audio_root=REPO_ROOT / "datasets" / "ego4d" / "audio",
                    output_path=reh_p,
                )

    def test_missing_annotation_raises(self):
        narr = _tiny_narration()
        # Delete annotation_uid=a2 from pass_2 → public slice can't join.
        narr["u1"]["narration_pass_2"]["summaries"] = []
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            narr_p = Path(td) / "narration.json"
            reh_p = Path(td) / "rehydrated.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            narr_p.write_text(json.dumps(narr))
            ego4d_redact(full_p, pub_p)
            with self.assertRaises(MissingAnnotationError):
                build_manifest_from_public(
                    public_manifest_path=pub_p,
                    narration_path=narr_p,
                    moments_paths=[],
                    audio_root=REPO_ROOT / "datasets" / "ego4d" / "audio",
                    output_path=reh_p,
                )


# ---------------------------------------------------------------------------
# Description-eval adapters raise on segmentation-only manifests
# ---------------------------------------------------------------------------

class TestDescriptionEvalRaisesOnPublic(unittest.TestCase):
    def test_ego4d_raises(self):
        try:
            from long_audio.eval.description.datasets.ego4d import Ego4DAdapter
        except ModuleNotFoundError:
            self.skipTest("numpy / sklearn not installed in this env")
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            ego4d_redact(full_p, pub_p)
            pub = json.loads(pub_p.read_text())
        with self.assertRaises(ValueError) as cm:
            Ego4DAdapter().load_gt_for_uid(pub, "u1")
        self.assertIn("segmentation-only", str(cm.exception))

    def test_egolife_raises(self):
        try:
            from long_audio.eval.description.datasets.egolife import EgoLifeAdapter
        except ModuleNotFoundError:
            self.skipTest("numpy / sklearn not installed in this env")
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            full_p.write_text(json.dumps(_tiny_egolife_full()))
            egolife_redact(full_p, pub_p)
            pub = json.loads(pub_p.read_text())
        with self.assertRaises(ValueError) as cm:
            EgoLifeAdapter().load_gt_for_uid(pub, "A1_JAKE_DAY1_S01")
        self.assertIn("segmentation-only", str(cm.exception))


# ---------------------------------------------------------------------------
# Loader resolves repo-relative audio_path
# ---------------------------------------------------------------------------

class TestLoaderResolvesRelativePath(unittest.TestCase):
    def test_ego4d_dataset_resolves_relative_audio_path(self):
        try:
            from long_audio.datasets.ego4d.dataset import Ego4DDataset
        except ModuleNotFoundError:
            self.skipTest("optional loader deps missing")
        with tempfile.TemporaryDirectory() as td:
            full_p = Path(td) / "full.json"
            pub_p = Path(td) / "public.json"
            full_p.write_text(json.dumps(_tiny_ego4d_full()))
            ego4d_redact(full_p, pub_p)
            ds = Ego4DDataset(manifest_path=pub_p)
            item = next(ds.iter_items())
            self.assertTrue(item.audio_path.is_absolute())
            self.assertEqual(
                item.audio_path,
                REPO_ROOT / "datasets" / "ego4d" / "audio" / "u1.flac",
            )


if __name__ == "__main__":
    unittest.main()
