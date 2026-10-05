# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""MS-CLAP-ClapCap audio captioner adapter for cascade Stage A.

Wraps ``msclap.CLAP(version='clapcap')`` — Microsoft's audio captioning
variant of CLAP. Architecture (from CLAPWrapper.generate_caption):

    audio → MS-CLAP-2023 audio encoder → prefix vector
          → clap_project Linear → GPT-2 prefix-tuned decoder
          → beam-search → free-form English caption

INSTALL (already in las-enclap):
    envs/las-enclap/bin/pip install --no-deps msclap

WEIGHTS:
    Auto-downloaded by msclap on first CLAP(version='clapcap') call
    (~1.2 GB to ~/.cache/clap/). No external setup needed.

API note: msclap.generate_caption() takes a list of FILE PATHS, not
numpy arrays. We write per-chunk audio to a tempfile, call it, and
clean up. Per-call overhead is ~5 ms (tempfile + soundfile.write) —
negligible against the beam-search decode latency (~hundreds of ms).
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


DEFAULT_MODEL_ID = "microsoft/msclap-clapcap"


def _install_torchaudio_soundfile_shim() -> None:
    """msclap's ``CLAPWrapper.read_audio`` calls ``torchaudio.load(path)``.
    torchaudio >= 2.11 hard-routes that through TorchCodec, which on PyPI
    only ships cu13 wheels — our env is cu129 so the load fails with
    ``libnvrtc.so.13: cannot open shared object file``. Patch
    ``CLAPWrapper.read_audio`` to use soundfile + torchaudio.Resample
    directly, bypassing TorchCodec. Idempotent.
    """
    from msclap.CLAPWrapper import CLAPWrapper

    if getattr(CLAPWrapper.read_audio, "_soundfile_shim", False):
        return

    import torch
    import soundfile as sf
    import torchaudio.transforms as T

    def read_audio(self, audio_path, resample=True):
        # numpy → torch, mono channel-first to match torchaudio.load's
        # (channels, time) convention.
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 1:
            audio_time_series = torch.from_numpy(audio).unsqueeze(0)
        else:
            audio_time_series = torch.from_numpy(audio.T)
        resample_rate = self.args.sampling_rate
        if resample and resample_rate != sample_rate:
            audio_time_series = T.Resample(sample_rate, resample_rate)(audio_time_series)
        return audio_time_series, resample_rate

    read_audio._soundfile_shim = True
    CLAPWrapper.read_audio = read_audio


class MSClapCapAdapter(ModelAdapter):
    """MS-CLAP ``clapcap`` audio captioner (free-form English caption).

    Args:
        beam_size: beam search width. msclap default = 5.
        entry_length: max output tokens. msclap default = 67.
        temperature: sampling temperature. msclap default = 1.0.
            For deterministic-ish output use 0.01 (matches the
            example in ``examples/audio_captioning.py``).
        model_id: cosmetic — recorded in output metadata.
        device: ``cuda`` or ``cpu``.
    """

    name = "msclap-cap"

    def __init__(
        self,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 0.01,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
    ):
        self.beam_size = beam_size
        self.entry_length = entry_length
        self.temperature = temperature
        self.model_id = model_id
        self.device = device
        self._model = None
        self._torch = None
        self._sf = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import soundfile as sf
        from msclap import CLAP

        self._torch = torch
        self._sf = sf
        _install_torchaudio_soundfile_shim()
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self._model = CLAP(version="clapcap", use_cuda=use_cuda)
        self.device = "cuda" if use_cuda else "cpu"

    def unload(self) -> None:
        self._model = None
        try:
            self._torch.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            pass

    def supports_structured(self) -> bool:
        # GPT-2 beam-search caption — no JSON-schema constraint.
        return False

    def _generate_one(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        """Write audio to a tempfile, generate caption, return (text, wall_s)."""
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}")
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            # msclap.preprocess_audio resamples internally to its target
            # sr (44.1 kHz for the 2023/clapcap encoders), so we can hand
            # off whatever sample rate we have.
            self._sf.write(tmp_path, audio, sample_rate)
            captions = self._model.generate_caption(
                [tmp_path],
                resample=True,
                beam_size=self.beam_size,
                entry_length=self.entry_length,
                temperature=self.temperature,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return (captions[0] if captions else ""), time.time() - t0

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "freeform",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Call load() first.")
        # MS-CLAP-clapcap ignores the prompt — GPT-2 decoder is
        # unconditional given the audio prefix.
        caption, wall = self._generate_one(audio, sample_rate)
        return ModelOutput(
            raw_text=caption,
            raw_json=None,
            latency_s=wall,
            metadata={
                "model_id": self.model_id,
                "decoder": "beam_search",
                "structured_supported": False,
                "beam_size": self.beam_size,
                "entry_length": self.entry_length,
                "batch_size": 1,
                "batch_wall_s": wall,
            },
        )

    def generate_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int,
        prompts: list[str],
        *,
        schema: dict | None = None,
        decoder: Decoder = "freeform",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[ModelOutput]:
        # msclap.generate_caption takes a list of file paths but loops
        # _generate_beam per item internally, so no true batching benefit.
        # Loop in Python for symmetry with other captioner adapters.
        return [
            self.generate(
                a, sample_rate, p,
                schema=schema, decoder=decoder,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            for a, p in zip(audios, prompts)
        ]
