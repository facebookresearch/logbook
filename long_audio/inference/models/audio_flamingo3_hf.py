# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter for nvidia/audio-flamingo-3-hf (transformers-integrated AF3).

Different from ``audio_flamingo3.py`` (the chat variant which requires the
local ``llava`` package). The ``-hf`` variant is registered in transformers
as ``AudioFlamingo3ForConditionalGeneration`` / ``AudioFlamingo3Processor``
and runs without any custom code.

CONSTRAINTS:
- ``AudioFlamingo3Processor`` natively handles up to 600 s (10 min) by
  internally chunking through the Whisper encoder. We pass full chunk
  audio and let the processor split. (An earlier draft of this adapter
  clipped to 30 s based on a misread of the Whisper feature extractor
  limit; that was wrong.)
- AF3 in our tests returns placeholder text on structured-output prompts,
  so we use a free-text single-label prompt. The chunk runner runs the
  raw reply through ``json_repair``; non-JSON output yields zero
  segments (no chunk-wide-guess fallback).
"""

from __future__ import annotations

import json as _json
import os
import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


# HuggingFace repo ID — pass --model-id <local-path> to use a local symlink.
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"

class AudioFlamingo3HFAdapter(ModelAdapter):
    """nvidia/audio-flamingo-3-hf via the transformers-integrated path.

    Args:
        model_id: HF repo id or local path.
        device: cuda device pin. ``device_map="auto"`` causes a cross-device
            tensor error during ``generate``, so we pin to a single GPU.
    """

    name = "af3-hf"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            AudioFlamingo3ForConditionalGeneration,
            AudioFlamingo3Processor,
        )

        self._torch = torch
        self._processor = AudioFlamingo3Processor.from_pretrained(self.model_id)
        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass

    def supports_structured(self) -> bool:
        # AF3 returns literal placeholder text on JSON-schema prompts in
        # our SINS tests; structured decoding is effectively unavailable.
        return False

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        return self.generate_batch(
            [audio], sample_rate, [prompt],
            schema=schema, decoder=decoder,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )[0]

    def generate_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int,
        prompts: list[str],
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[ModelOutput]:
        if self._model is None:
            raise RuntimeError("Call load() first.")
        if len(audios) != len(prompts):
            raise ValueError(
                f"generate_batch: len(audios)={len(audios)} != len(prompts)={len(prompts)}"
            )
        prepared: list[np.ndarray] = []
        for a in audios:
            if a.ndim != 1:
                raise ValueError(f"Expected mono audio, got shape {a.shape}")
            prepared.append(a.astype(np.float32, copy=False))

        # No system message — the user prompt (rendered by render_prompt) is
        # fully self-describing for both E2E segmentation and cascade-A
        # captioning. The previous DEFAULT_SYSTEM ("answer with one short
        # label") was a stale captioning-era constant that actively
        # contradicted the segmentation user prompt.
        texts: list[str] = []
        for p in prompts:
            messages = [
                {"role": "user",
                 "content": [
                     {"type": "audio", "audio": "placeholder"},
                     {"type": "text", "text": p},
                 ]},
            ]
            texts.append(self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        # IMPORTANT: do NOT pass a top-level `padding=...` here. The
        # AudioFlamingo3Processor's _defaults are already correct:
        #   text_kwargs  = {"padding": True}            # pad text to longest-in-batch
        #   audio_kwargs = {"padding": "max_length"}    # pad audio to 30s, set
        #                                                  input_features_mask to
        #                                                  reflect the TRUE per-item
        #                                                  length so the encoder /
        #                                                  multi_modal_projector
        #                                                  scatter only real audio
        #                                                  tokens into the LLM.
        # Passing a top-level `padding=True` overrides both — turning the
        # audio side into "pad to longest in batch", which (a) crashes the
        # encoder when batched short clips don't reach 1500 post-conv
        # frames, and (b) even if we work around (a) by pre-padding to
        # 30s ourselves, the processor's mask comes back all-1s — the
        # model then treats the trailing silence as real audio.
        inputs = self._processor(
            text=texts,
            audio=prepared,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device).to(self._model.dtype)

        t0 = time.time()
        with self._torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-3),
            )
        wall = time.time() - t0
        per_call_latency = wall / max(1, len(audios))

        in_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[:, in_len:]
        raw_texts = self._processor.batch_decode(
            new_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: list[ModelOutput] = []
        for i, raw_text in enumerate(raw_texts):
            results.append(ModelOutput(
                raw_text=raw_text,
                raw_json=None,
                latency_s=per_call_latency,
                metadata={
                    "model_id": self.model_id,
                    "decoder": "freetext_simple",
                    "structured_supported": False,
                    "audio_seconds": audios[i].shape[0] / sample_rate,
                    "completion_tokens": int((new_ids[i] != self._processor.tokenizer.pad_token_id).sum()),
                    "batch_size": len(audios),
                    "batch_wall_s": wall,
                },
            ))
        return results
