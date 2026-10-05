# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter for Qwen/Qwen2.5-Omni-7B via mainstream vLLM 0.21+.

The official transformers/vLLM integration: ``Qwen2_5OmniForConditionalGeneration``
is supported natively (no ``vllm-omni`` fork needed). The model's
thinker-only variant is auto-selected by vLLM for text-out generation.

INSTALL:
    See ``$REPO/envs/las-gpu128`` for the working torch 2.11+cu128 / vllm
    0.21.0 stack on AWS H100 (driver 12.8 cluster).

PROMPT FORMAT:
    Don't hand-roll. ``AutoProcessor.apply_chat_template`` renders the
    correct ``<|audio_bos|><|AUDIO|><|audio_eos|>`` placeholders so the
    audio aligns with the right token positions.

STRUCTURED OUTPUT:
    ``StructuredOutputsParams(json=schema)`` enforces a JSON schema at
    decode time (xgrammar backend in vLLM 0.21).
"""

from __future__ import annotations

import json as _json
import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


# HuggingFace repo ID — pass --model-id <local-path> to use a local symlink.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

DEFAULT_SYSTEM = (
    "You are an expert audio analyst for wearable-device recordings of a "
    "single domestic resident. Produce a temporal segmentation of human "
    "activities."
)


class Qwen2_5OmniVLLMAdapter(ModelAdapter):
    """Qwen2.5-Omni-7B via mainstream vLLM 0.21+.

    Args:
        model_id: HF repo id or local path. Defaults to local
            ``~/storage/models/Qwen2.5-Omni-7B``.
        gpu_memory_utilization: vLLM target.
        max_model_len: total token budget per request.
        tensor_parallel_size: 1 fits 7B on a single H100 comfortably.
        enforce_eager: skip torch.compile / CUDA graphs (debug fallback).
    """

    name = "qwen2.5-omni"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 1,
        enforce_eager: bool = False,
    ):
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self._llm = None
        self._processor = None
        self._sampling_params_cls = None
        self._structured_outputs_cls = None

    def load(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._sampling_params_cls = SamplingParams
        self._structured_outputs_cls = StructuredOutputsParams

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            dtype="bfloat16",
        )
        if self.enforce_eager:
            kwargs["enforce_eager"] = True
        self._llm = LLM(**kwargs)

    def unload(self) -> None:
        self._llm = None
        self._processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def supports_structured(self) -> bool:
        return True

    def _render_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": DEFAULT_SYSTEM}]},
            {"role": "user",
             "content": [{"type": "audio", "audio": "placeholder"},
                         {"type": "text", "text": user_text}]},
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
        if self._llm is None:
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

        sp_kwargs: dict[str, Any] = dict(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if decoder == "structured" and schema is not None:
            sp_kwargs["structured_outputs"] = self._structured_outputs_cls(
                json=schema["schema"]
            )
        sampling_params = self._sampling_params_cls(**sp_kwargs)

        # vLLM batches natively. ``max_num_seqs`` on the LLM constructor
        # caps concurrent on-GPU sequences — bump it to ≥ batch_size for
        # Stage A captioning, otherwise vLLM serializes inside.
        requests = [
            {
                "prompt": self._render_prompt(p),
                "multi_modal_data": {"audio": (a, sample_rate)},
            }
            for a, p in zip(prepared, prompts)
        ]

        t0 = time.time()
        outputs = self._llm.generate(requests, sampling_params=sampling_params)
        wall = time.time() - t0
        per_call_latency = wall / max(1, len(outputs))

        results: list[ModelOutput] = []
        for out in outputs:
            gen = out.outputs[0]
            raw_text = gen.text
            raw_json: dict | None = None
            if decoder == "structured" and schema is not None:
                try:
                    raw_json = _json.loads(raw_text)
                except (ValueError, TypeError):
                    raw_json = None
            prompt_tokens = len(getattr(out, "prompt_token_ids", []) or [])
            completion_tokens = len(getattr(gen, "token_ids", []) or [])
            results.append(ModelOutput(
                raw_text=raw_text,
                raw_json=raw_json,
                latency_s=per_call_latency,
                metadata={
                    "model_id": self.model_id,
                    "decoder": decoder,
                    "structured_supported": True,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": getattr(gen, "finish_reason", None),
                    "batch_size": len(outputs),
                    "batch_wall_s": wall,
                },
            ))
        return results
