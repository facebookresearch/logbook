"""Adapter for Qwen/Qwen3-Omni-30B-A3B-Instruct via mainstream vLLM 0.21+.

The model is natively supported as ``Qwen3OmniMoeForConditionalGeneration``
(thinker auto-selected for text-out). Mainstream vLLM (no ``vllm-omni``
fork) is the working path on our AWS H100 cluster after we discovered
the V1 engine silently kills the EngineCore during MoE encoder warmup;
the official Qwen3-Omni cookbook works around this by forcing the V0
engine via ``VLLM_USE_V1=0``, and that's what we do here.

INSTALL: see ``$REPO/envs/las-gpu128`` — torch 2.11+cu128, vllm 0.21.0,
transformers 5.9.0. Built from the vllm 0.21.0 + cu129 wheel channel
(see project NOTES for the exact ``uv pip install`` recipe).

PROMPT FORMAT: do not hand-roll. ``AutoProcessor.apply_chat_template``
renders the correct ``<|audio_bos|><|AUDIO|><|audio_eos|>`` placeholders.

STRUCTURED OUTPUT: ``StructuredOutputsParams(json=schema)`` enforces a
JSON schema at decode time (xgrammar backend in vLLM 0.21).
"""

from __future__ import annotations

import json as _json
import os
import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput
from long_audio.inference.thinking_budget import ThinkingMixin


# HuggingFace repo ID — pass --model-id <local-path> to use a local symlink.
DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

DEFAULT_SYSTEM = (
    "You are an expert audio analyst for wearable-device recordings of a "
    "single domestic resident. Produce a temporal segmentation of human "
    "activities."
)


class Qwen3OmniVLLMAdapter(ModelAdapter, ThinkingMixin):
    """Qwen3-Omni-30B-A3B-Instruct via mainstream vLLM 0.21+ with V0 engine.

    Args:
        model_id: HF repo id or local path. Defaults to local
            ``~/storage/models/Qwen3-Omni-30B-A3B-Instruct``.
        gpu_memory_utilization: cookbook default 0.95 on a single H100.
        max_model_len: 32768 per cookbook; bump only if your chunks
            actually need more.
        tensor_parallel_size: 1 fits 30B-A3B on a single 80 GB H100
            (loads ~59 GB weights + ~12 GB KV cache).
        max_num_seqs: cookbook default 1 (low concurrency, keeps KV cache
            from being over-provisioned for batched serving we don't need).
        enable_thinking: flow into apply_chat_template. True only makes
            sense with the Thinking variant of the model_id.
        thinking_budget_enforce: s1-style hard cap on reasoning tokens.
            0 = off (default). See long_audio.inference.thinking_budget.
    """

    name = "qwen3-omni"

    # ThinkingMixin provides THINK_START="<think>" / THINK_END="</think>"
    # defaults; Qwen 3 uses those verbatim -- no override needed.

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 1,
        enable_thinking: bool = False,
        thinking_budget_enforce: int = 0,
    ):
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self._init_thinking(enable_thinking, thinking_budget_enforce)
        self._llm = None
        self._processor = None
        self._sampling_params_cls = None
        self._structured_outputs_cls = None

    def load(self) -> None:
        if self._llm is not None:
            return
        # MUST be set before importing vllm. The V1 engine silently kills
        # EngineCore during MoE multimodal warmup on this model; V0 works.
        # Assignment (not setdefault): AWS Slurm environment has
        # VLLM_USE_V1=1 preset upstream, which setdefault would defer to —
        # observed in egolife_A_qwen3-omni-120832 where V1 booted despite
        # the setdefault. Force V0 unconditionally.
        os.environ["VLLM_USE_V1"] = "0"
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._sampling_params_cls = SamplingParams
        self._structured_outputs_cls = StructuredOutputsParams

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        # ThinkingMixin: no-op when thinking_budget_enforce == 0.
        self._resolve_think_end_tokens(
            getattr(self._processor, "tokenizer", self._processor),
        )

        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            trust_remote_code=True,
            # Cookbook-style: leave all modalities enabled so the warmup
            # path matches what Qwen team tested. V0 doesn't OOM on this.
            limit_mm_per_prompt={"audio": 3, "video": 3, "image": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            dtype="bfloat16",
            seed=1234,
        )
        # ThinkingMixin: registers ThinkingBudgetLogitsProcessor class at
        # engine init when thinking_budget_enforce > 0; empty dict otherwise.
        kwargs.update(self._thinking_llm_kwargs())
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
            messages, tokenize=False, add_generation_prompt=True,
            # enable_thinking flows into the jinja template as a variable;
            # the Thinking variant's template branches on it.
            enable_thinking=self.enable_thinking,
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
        if self.enable_thinking:
            # See vllm_text.VLLMTextAdapter.generate for the same rationale.
            # Qwen3-Omni's <think>/</think> aren't flagged special so it
            # worked accidentally under skip_special_tokens=True; Qwen3-
            # Omni-Thinking may tag them differently. Only opt-in when
            # thinking is on so non-thinking Stage-A captioning keeps the
            # cleaner default output.
            sp_kwargs["skip_special_tokens"] = False
        if decoder == "structured" and schema is not None:
            sp_kwargs["structured_outputs"] = self._structured_outputs_cls(
                json=schema["schema"]
            )
        extra_args = self._thinking_extra_args()
        if extra_args is not None:
            # V1 LogitsProcessor is engine-registered + batch-aware —
            # each batch slot tracks its own state via extra_args +
            # BatchUpdate hooks, so N>1 is supported.
            sp_kwargs["extra_args"] = extra_args
            # Extend max_tokens: caller's max_new_tokens is treated as the
            # output-only budget when enforcement is on; add the enforce
            # budget so there is room for the answer after forced THINK_END.
            sp_kwargs["max_tokens"] = self.thinking_budget_enforce + max_new_tokens
        sampling_params = self._sampling_params_cls(**sp_kwargs)

        # vLLM batches natively when given a list. ``max_num_seqs`` on the
        # LLM constructor caps concurrent sequences on-GPU — bump it to
        # match or exceed your batch size for Stage A captioning, else
        # vLLM serializes the batch internally.
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
        # Attribute wall time evenly across the batch — vLLM doesn't
        # expose per-request timing and a batched call's individual
        # latencies are not meaningful anyway.
        per_call_latency = wall / max(1, len(outputs))

        results: list[ModelOutput] = []
        for out in outputs:
            gen = out.outputs[0]
            full_text = gen.text
            # Peel <think>...</think> prefix off the visible output via
            # ThinkingMixin. Trace is None when thinking was off or the
            # marker was absent.
            thinking_trace, raw_text = self._split_thinking(full_text)
            # Direct token-id slice preserves thoughts_tokens + response
            # == completion_tokens (see ThinkingMixin._thoughts_tokens_from_ids).
            thoughts_tokens = self._thoughts_tokens_from_ids(
                getattr(gen, "token_ids", None)
            )
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
                    "thoughts_tokens": thoughts_tokens,
                    "finish_reason": getattr(gen, "finish_reason", None),
                    "batch_size": len(outputs),
                    "batch_wall_s": wall,
                    **self._thinking_metadata(),
                },
                thinking_trace=thinking_trace,
            ))
        return results
