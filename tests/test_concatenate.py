# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the wall-clock-aligned mono FLAC concatenator."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import soundfile

from scripts.data.sins.build_flacs import concatenate_wavs


SR = 16000


def _write_wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    soundfile.write(str(path), data, sr, subtype="PCM_16")


def _make_clip(
    td: Path, name: str, n_seconds: float, fill: int, n_ch: int = 1, sr: int = SR
) -> Path:
    if n_ch == 1:
        data = np.full(int(n_seconds * sr), fill, dtype=np.int16)
    else:
        data = np.full((int(n_seconds * sr), n_ch), fill, dtype=np.int16)
    p = td / name
    _write_wav(p, data, sr=sr)
    return p


class TestBackToBackConcatenation(unittest.TestCase):
    """Tight-packed clips (no gaps) → output equals concatenation."""

    def test_no_gaps_no_padding(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            clips_data = []
            clips = []
            for i in range(3):
                p = _make_clip(td, f"c{i}.wav", n_seconds=1.0, fill=100 * (i + 1))
                clips_data.append(np.full((SR, 1), 100 * (i + 1), dtype=np.int16))
                clips.append({"audio_path": str(p), "timestamp": float(i)})

            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=False)
            self.assertEqual(duration, 3.0)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            expected = np.concatenate(clips_data, axis=0)
            np.testing.assert_array_equal(roundtrip, expected)


class TestGapSilenceInsertion(unittest.TestCase):
    """The headline behavior: gaps between clips become silence in the output."""

    def test_one_gap_becomes_silence(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p0 = _make_clip(td, "c0.wav", 1.0, 100)
            p1 = _make_clip(td, "c1.wav", 1.0, 200)
            clips = [
                {"audio_path": str(p0), "timestamp": 0.0},
                {"audio_path": str(p1), "timestamp": 2.0},
            ]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=True)
            self.assertEqual(duration, 3.0)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            self.assertEqual(data.shape, (3 * SR, 1))
            np.testing.assert_array_equal(data[: SR], 100)
            np.testing.assert_array_equal(data[SR : 2 * SR], 0)
            np.testing.assert_array_equal(data[2 * SR :], 200)

    def test_leading_silence_pads_to_t0(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = _make_clip(td, "c.wav", 1.0, 100)
            clips = [{"audio_path": str(p), "timestamp": 2.0}]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=True)
            self.assertEqual(duration, 3.0)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            self.assertEqual(data.shape, (3 * SR, 1))
            np.testing.assert_array_equal(data[: 2 * SR], 0)
            np.testing.assert_array_equal(data[2 * SR :], 100)

    def test_pad_to_t0_false_starts_at_first_clip(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = _make_clip(td, "c.wav", 1.0, 100)
            clips = [{"audio_path": str(p), "timestamp": 2.0}]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=False)
            self.assertEqual(duration, 1.0)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            self.assertEqual(data.shape, (SR, 1))
            np.testing.assert_array_equal(data, 100)


class TestWallClockProperty(unittest.TestCase):
    """Sample N must == wall-clock time N / SR (when pad_to_t0=True)."""

    def test_sample_index_matches_wallclock_time(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            schedule = [
                (0.5, 1.0, 11),
                (3.25, 0.5, 22),
                (7.0, 2.0, 33),
            ]
            clips = []
            for i, (t_start, dur, fill) in enumerate(schedule):
                p = _make_clip(td, f"c{i}.wav", dur, fill)
                clips.append({"audio_path": str(p), "timestamp": t_start})

            out = td / "out.flac"
            concatenate_wavs(clips, out, pad_to_t0=True)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)

            for t_start, dur, fill in schedule:
                s = int(round(t_start * SR))
                e = s + int(round(dur * SR))
                np.testing.assert_array_equal(
                    data[s:e],
                    fill,
                    err_msg=f"clip @ t={t_start}s not at sample {s}",
                )

    def test_overlap_first_writer_wins(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p0 = _make_clip(td, "c0.wav", 2.0, 100)
            p1 = _make_clip(td, "c1.wav", 2.0, 200)
            clips = [
                {"audio_path": str(p0), "timestamp": 0.0},
                {"audio_path": str(p1), "timestamp": 1.0},
            ]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=True)
            self.assertEqual(duration, 3.0)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            np.testing.assert_array_equal(data[: SR], 100)
            np.testing.assert_array_equal(data[SR : 2 * SR], 100)
            np.testing.assert_array_equal(data[2 * SR : 3 * SR], 200)

    def test_drops_clips_entirely_before_t0(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p0 = _make_clip(td, "c0.wav", 1.0, 100)
            p1 = _make_clip(td, "c1.wav", 1.0, 200)
            clips = [
                {"audio_path": str(p0), "timestamp": -2.0},
                {"audio_path": str(p1), "timestamp": 0.0},
            ]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=True)
            self.assertEqual(duration, 1.0)
            data, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            np.testing.assert_array_equal(data, 200)

    def test_partial_pre_t0_clip_gets_prefix_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = np.zeros(SR, dtype=np.int16)
            data[: SR // 2] = 100
            data[SR // 2 :] = 200
            p = td / "c.wav"
            _write_wav(p, data)
            clips = [{"audio_path": str(p), "timestamp": -0.5}]
            out = td / "out.flac"
            duration = concatenate_wavs(clips, out, pad_to_t0=True)
            self.assertEqual(duration, 0.5)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            self.assertEqual(roundtrip.shape, (SR // 2, 1))
            np.testing.assert_array_equal(roundtrip, 200)


class TestMonoDownmix(unittest.TestCase):
    """Multi-channel input is mean-downmixed to mono on the fly."""

    def test_4ch_input_downmixes_to_mono(self):
        # 4-channel input with distinct constants per channel; output is
        # mono with the per-sample mean.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = np.zeros((SR, 4), dtype=np.int16)
            data[:, 0] = 100
            data[:, 1] = 200
            data[:, 2] = 300
            data[:, 3] = 400
            p = td / "c.wav"
            _write_wav(p, data)
            clips = [{"audio_path": str(p), "timestamp": 0.0}]
            out = td / "out.flac"
            concatenate_wavs(clips, out, pad_to_t0=False)
            info = soundfile.info(str(out))
            self.assertEqual(info.channels, 1)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            self.assertEqual(roundtrip.shape, (SR, 1))
            # Mean of (100, 200, 300, 400) = 250.
            np.testing.assert_allclose(roundtrip, 250, atol=1)

    def test_mono_input_passes_through(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = np.full(SR, 500, dtype=np.int16)
            p = td / "c.wav"
            _write_wav(p, data)
            clips = [{"audio_path": str(p), "timestamp": 0.0}]
            out = td / "out.flac"
            concatenate_wavs(clips, out, pad_to_t0=False)
            info = soundfile.info(str(out))
            self.assertEqual(info.channels, 1)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            np.testing.assert_array_equal(roundtrip, 500)


class TestSampleRateResample(unittest.TestCase):
    """target_sample_rate mismatch → warn + resample (don't raise)."""

    def test_resamples_from_8khz_to_16khz(self):
        # 1s @ 8 kHz of constant 100s → after resample to 16 kHz, should be
        # ~1s of constant 100 (low-pass filter has tiny edge effects but
        # interior should be ~100).
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = np.full(8000, 100, dtype=np.int16)
            p = td / "c.wav"
            _write_wav(p, data, sr=8000)
            clips = [{"audio_path": str(p), "timestamp": 0.0}]
            out = td / "out.flac"
            buf = io.StringIO()
            with redirect_stdout(buf):
                concatenate_wavs(
                    clips, out, pad_to_t0=False, target_sample_rate=16000,
                )
            info = soundfile.info(str(out))
            self.assertEqual(info.samplerate, 16000)
            self.assertAlmostEqual(info.duration, 1.0, places=2)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=True)
            # Interior should still be ~100 (allow ±5 for filter ripple/quantization).
            mid = roundtrip[1000:-1000]
            self.assertTrue(np.all(np.abs(mid - 100) < 5))
            self.assertIn("[WARN]", buf.getvalue())
            self.assertIn("8000", buf.getvalue())

    def test_warn_emitted_once_per_distinct_source_rate(self):
        # Three 8 kHz clips → only one warning line about "8000".
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            clips = []
            for i in range(3):
                p = _make_clip(td, f"c{i}.wav", 0.5, 100, n_ch=1, sr=8000)
                clips.append({"audio_path": str(p), "timestamp": float(i)})
            out = td / "out.flac"
            buf = io.StringIO()
            with redirect_stdout(buf):
                concatenate_wavs(
                    clips, out, pad_to_t0=False, target_sample_rate=16000,
                )
            warn_lines = [
                l for l in buf.getvalue().splitlines() if "[WARN]" in l and "8000" in l
            ]
            self.assertEqual(len(warn_lines), 1)


class TestStreamingAndEdgeCases(unittest.TestCase):
    def test_small_chunk_frames_doesnt_lose_frames(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rng = np.random.default_rng(42)
            data = rng.integers(-1000, 1000, size=SR, dtype=np.int16)
            p = td / "clip.wav"
            _write_wav(p, data)
            clips = [{"audio_path": str(p), "timestamp": 0.0}]
            out = td / "out.flac"
            concatenate_wavs(clips, out, chunk_frames=137, pad_to_t0=False)
            roundtrip, _ = soundfile.read(str(out), dtype="int16", always_2d=False)
            np.testing.assert_array_equal(roundtrip, data)

    def test_empty_clips_returns_zero_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.flac"
            duration = concatenate_wavs([], out)
            self.assertEqual(duration, 0.0)
            self.assertFalse(out.exists())

    def test_format_forced_to_flac_even_with_wav_extension(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = _make_clip(td, "c.wav", 1.0, 100)
            clips = [{"audio_path": str(p), "timestamp": 0.0}]
            out_misnamed = td / "out.wav"
            concatenate_wavs(clips, out_misnamed, pad_to_t0=False)
            self.assertEqual(soundfile.info(str(out_misnamed)).format, "FLAC")


if __name__ == "__main__":
    unittest.main()
