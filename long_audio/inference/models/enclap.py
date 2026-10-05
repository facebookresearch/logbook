"""EnCLAP-large adapter for cascade Stage A (free-form audio captioning).

Wraps the ``EnClap`` class from the EnCLAP repo
(github.com/jaeyeonkim99/EnCLAP, ICASSP 2024). Architecture:

    wav → torchaudio
       ├→ EnCodec @ 24 kHz, 12 kbps  → discrete codebook tokens
       └→ resample 48 kHz → LAION-CLAP → 512-d audio embedding
    (encodec_frames + clap_embedding) → EnClapBart.generate()
                                      → BART beam=4, max_length=50
                                      → caption text

Per-chunk inference: ~200 ms on H100 (no native batch — loop sequentially).
Audio length cap: BART max_position_embeddings − 3 ≈ 1021 EnCodec frames
@ 75 fps ⇒ ~13.6 s. Our 10-s chunks fit comfortably.

INSTALL (one-shot, additive to las-all):
    envs/las-all/bin/pip install --no-deps laion-clap encodec
    # (also needs librosa, transformers, torchaudio — already in las-all)

WEIGHTS — download out-of-band (the EnCLAP repo doesn't ship them):
    1. EnCLAP-large BART checkpoint
         https://drive.google.com/drive/folders/1JOcKyNOlKud0PY93ETGDlUJnWhmSC35m
       Pick ``both/large/`` (trained on AudioCaps + Clotho, used by the
       official gradio demo) or ``audiocaps/large/``; unzip locally.
       Point ENCLAP_CKPT_DIR at the resulting directory.

    2. LAION-CLAP audio encoder checkpoint
         https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt
       Point LAION_CLAP_CKPT at the downloaded ``.pt``.

REPO: clone github.com/jaeyeonkim99/EnCLAP and point ENCLAP_REPO_DIR at
      the working tree; we inject it at load() time.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


DEFAULT_MODEL_ID = "enclap-large"


class EnClapLargeAdapter(ModelAdapter):
    """EnCLAP-large free-form audio captioner.

    Args:
        repo_dir: path to the cloned EnCLAP repo. Falls back to
            ``$ENCLAP_REPO_DIR``.
        ckpt_dir: directory holding ``config.json`` + ``pytorch_model.bin``
            for the EnCLAP-BART checkpoint. Falls back to
            ``$ENCLAP_CKPT_DIR``.
        clap_ckpt_path: path to the LAION-CLAP ``.pt`` checkpoint. Falls
            back to ``$LAION_CLAP_CKPT``.
        model_id: cosmetic — recorded in output metadata.
        device: ``cuda`` or ``cpu``.
    """

    name = "enclap"

    def __init__(
        self,
        repo_dir: str | None = None,
        ckpt_dir: str | None = None,
        clap_ckpt_path: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
    ):
        self.repo_dir = repo_dir or os.environ.get("ENCLAP_REPO_DIR")
        self.ckpt_dir = ckpt_dir or os.environ.get("ENCLAP_CKPT_DIR")
        self.clap_ckpt_path = clap_ckpt_path or os.environ.get("LAION_CLAP_CKPT")
        self.model_id = model_id
        self.device = device
        self._enclap = None
        self._torch = None

    def load(self) -> None:
        if self._enclap is not None:
            return
        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"EnCLAP repo not found (repo_dir={self.repo_dir!r}). "
                "Clone github.com/jaeyeonkim99/EnCLAP and set ENCLAP_REPO_DIR."
            )
        if not self.ckpt_dir or not os.path.isdir(self.ckpt_dir):
            raise FileNotFoundError(
                f"EnCLAP-BART checkpoint dir not found (ckpt_dir={self.ckpt_dir!r}). "
                "Download from https://drive.google.com/drive/folders/"
                "1JOcKyNOlKud0PY93ETGDlUJnWhmSC35m (pick both/large/ or "
                "audiocaps/large/), unzip, and set ENCLAP_CKPT_DIR."
            )
        if not self.clap_ckpt_path or not os.path.isfile(self.clap_ckpt_path):
            raise FileNotFoundError(
                f"LAION-CLAP checkpoint not found (clap_ckpt_path={self.clap_ckpt_path!r}). "
                "Download 630k-audioset-fusion-best.pt from "
                "https://huggingface.co/lukewys/laion_clap and set LAION_CLAP_CKPT."
            )

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        import torch
        # EnCLAP repo's top-level inference.py exports EnClap.
        from inference import EnClap  # type: ignore

        self._torch = torch
        device = self.device if (
            self.device != "cuda" or torch.cuda.is_available()
        ) else "cpu"
        self._enclap = EnClap(
            ckpt_path=self.ckpt_dir,
            clap_ckpt_path=self.clap_ckpt_path,
            device=device,
        )
        self.device = device

    def unload(self) -> None:
        self._enclap = None
        try:
            self._torch.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            pass

    def supports_structured(self) -> bool:
        # BART decoder is free-form text; no JSON-schema constraint.
        return False

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
        if self._enclap is None:
            raise RuntimeError("Call load() first.")
        torch = self._torch
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}")
        # EnCLAP ignores the prompt — its BART decoder is unconditional
        # given (encodec_frames + clap_embedding).
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        t0 = time.time()
        captions = self._enclap.infer_from_audio(audio_tensor, sample_rate)
        wall = time.time() - t0
        caption = captions[0] if captions else ""
        return ModelOutput(
            raw_text=caption,
            raw_json=None,
            latency_s=wall,
            metadata={
                "model_id": self.model_id,
                "decoder": "beam_search",
                "structured_supported": False,
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
        # EnCLAP.infer_from_audio is single-clip; no benefit to fake-batching
        # since EnCodec + LAION-CLAP + BART each launch their own GPU kernels.
        return [
            self.generate(
                a, sample_rate, p,
                schema=schema, decoder=decoder,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            for a, p in zip(audios, prompts)
        ]
