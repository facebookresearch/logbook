# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Open-weight HF text LLMs as cascade Stage B alternatives to Gemini.

Wraps the standard vLLM ``LLM.chat()`` API. Each subclass hard-codes its
HF model_id + the tensor-parallel size that fits on H100 80GB.

THINKING DISABLED
We pass ``chat_template_kwargs={'enable_thinking': False}`` through every
``LLM.chat()`` call. Templates that don't reference the kwarg ignore it
silently; thinking-aware templates (Qwen3, Olmo-3.1, …) honor it and skip
the extended-reasoning prefix. This keeps the cascade Stage B comparable
to Gemini-Flash-no-thinking.

REGISTRY (in ``scripts/run_cascade.py``)
    "qwen2.5-72b"   → Qwen2_5_72BTextAdapter
    "llama3.3-70b"  → Llama3_3_70BTextAdapter
    "olmo3.1-32b"   → Olmo3_1_32BTextAdapter
    "k2-v2"         → K2_V2TextAdapter
    "qwen3-32b"     → Qwen3_32BTextAdapter
    "gemma4-31b"    → Gemma4_31BTextAdapter
"""

from __future__ import annotations

import json as _json
import os
import time
from typing import Any

from long_audio.inference.thinking_budget import ThinkingMixin
from long_audio.inference.models.base import (
    Decoder,
    ModelOutput,
    TextLLMAdapter,
)


def _resolve_lora_rank(adapter_dir: str, override: int | None) -> int:
    """max_lora_rank for the vLLM engine — must be >= the adapter's ``r``.

    Uses ``override`` if given, else reads ``r`` from the adapter's
    ``adapter_config.json`` (PEFT writes it there), falling back to 64.
    """
    if override:
        return int(override)
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            return int(_json.load(f).get("r", 64))
    except (OSError, ValueError, TypeError):
        return 64


class VLLMTextAdapter(TextLLMAdapter, ThinkingMixin):
    """vLLM-served HF text LLM, chat-template-rendered prompt.

    Concrete model + TP set by subclasses via class attrs
    ``DEFAULT_MODEL_ID`` and ``DEFAULT_TENSOR_PARALLEL_SIZE``.
    Thinking-mode support (enable + s1-style budget enforcement) is
    mixed in from ``ThinkingMixin`` — subclasses whose chat template
    uses non-``<think>`` markers override the class constants.
    """

    name = "vllm-text"
    DEFAULT_MODEL_ID: str = ""
    DEFAULT_TENSOR_PARALLEL_SIZE: int = 1
    DEFAULT_MAX_MODEL_LEN: int = 32768
    DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.9
    # Forced HF config overrides for LLM(). Dense text bases leave this None
    # (no-op). Multimodal bases (see Gemma4_31BTextAdapter) set
    # ``{"architectures": [...]}`` so vLLM instantiates the FULL
    # conditional-generation arch — required for LoRA adapters that carry
    # vision/audio-tower weights (see _build_llm_kwargs).
    HF_OVERRIDES: dict[str, Any] | None = None

    def __init__(
        self,
        model_id: str | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int = 32,
        adapter_dir: str | None = None,
        max_lora_rank: int | None = None,
        hf_overrides: dict[str, Any] | None = None,
        enable_thinking: bool = False,
        thinking_budget_enforce: int = 0,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        if not self.model_id:
            raise ValueError(
                f"{type(self).__name__}: model_id is required (no default)."
            )
        self.tensor_parallel_size = (
            tensor_parallel_size or self.DEFAULT_TENSOR_PARALLEL_SIZE
        )
        self.max_model_len = max_model_len or self.DEFAULT_MAX_MODEL_LEN
        self.gpu_memory_utilization = (
            gpu_memory_utilization or self.DEFAULT_GPU_MEMORY_UTILIZATION
        )
        self.max_num_seqs = max_num_seqs
        # Optional PEFT LoRA adapter: serve the base model + a LoRA on top
        # (vLLM enable_lora + LoRARequest). ``max_lora_rank`` must be >= the
        # adapter's r; default resolves it from the adapter_config.json.
        self.adapter_dir = adapter_dir
        self.max_lora_rank = (
            _resolve_lora_rank(adapter_dir, max_lora_rank)
            if adapter_dir
            else None
        )
        # Constructor arg wins over the class-attr default (lets a caller /
        # CLI correct the arch name without a code change).
        self.hf_overrides = (
            hf_overrides if hf_overrides is not None else self.HF_OVERRIDES
        )
        self._init_thinking(enable_thinking, thinking_budget_enforce)
        self._llm = None
        self._SamplingParams = None
        self._lora_request = None

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Assemble the ``LLM()`` kwargs. Split out of ``load()`` (and free of
        any ``vllm`` import) so the wiring is unit-testable on CPU."""
        llm_kwargs: dict[str, Any] = dict(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        # Force the FULL (conditional-generation) architecture for multimodal
        # bases served text-only. A cascade LoRA trained with
        # target_modules="all-linear" on the full HF model (gemma-4 loads via
        # AutoModelForCausalLM WITH its vision/audio towers) carries
        # vision_tower.* weights. vLLM's LoRA loader validates every adapter
        # module against the loaded arch's supported modules and rejects the
        # adapter if a tower module is absent ("expected target modules {...}
        # but received ['vision_tower...']"). Instantiating the full arch makes
        # those modules exist so the adapter loads. Dense bases: None -> no-op.
        if self.hf_overrides:
            llm_kwargs["hf_overrides"] = self.hf_overrides
        if self.adapter_dir:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
        # ThinkingMixin: registers ThinkingBudgetLogitsProcessor class at
        # engine init when thinking_budget_enforce > 0; empty dict otherwise.
        llm_kwargs.update(self._thinking_llm_kwargs())
        return llm_kwargs

    def load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        llm_kwargs = self._build_llm_kwargs()
        if self.adapter_dir:
            from vllm.lora.request import LoRARequest

            # (name, int_id, path); reused for every generate call.
            self._lora_request = LoRARequest(
                "cascade_ft", 1, self.adapter_dir
            )
        self._llm = LLM(**llm_kwargs)
        # ThinkingMixin: no-op when enforcement is off.
        self._resolve_think_end_tokens(self._llm.get_tokenizer())

    def unload(self) -> None:
        self._llm = None
        self._SamplingParams = None
        self._lora_request = None

    def supports_structured(self) -> bool:
        # vLLM supports xgrammar JSON-schema decoding, but cascade Stage B
        # already runs json_repair on the free-form output, matching the
        # Gemini-Flash-no-thinking baseline. Stay freeform for parity.
        return False

    def generate(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "freeform",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        if self._llm is None:
            raise RuntimeError("Call load() first.")

        sp_kwargs: dict[str, Any] = dict(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        if self.enable_thinking:
            # vLLM defaults to skip_special_tokens=True, which silently
            # eats models whose thinking channel is delimited by
            # special-flagged tokens (Gemma-4: <|channel>=100,
            # <channel|>=101 both special). That strips the whole
            # thought block from out.text before _split_thinking sees
            # it — the trace looks empty even though the model actually
            # generated 1000+ reasoning tokens. Only opt-in when
            # thinking is on so non-thinking baseline runs keep the
            # cleaner default output.
            sp_kwargs["skip_special_tokens"] = False
        extra_args = self._thinking_extra_args()
        if extra_args is not None:
            sp_kwargs["extra_args"] = extra_args
            # max_new_tokens is treated as output-only when enforcing;
            # extend total cap so the answer fits after forced THINK_END.
            sp_kwargs["max_tokens"] = self.thinking_budget_enforce + max_new_tokens
        sampling = self._SamplingParams(**sp_kwargs)
        messages = [{"role": "user", "content": prompt}]

        t0 = time.time()
        # chat_template_kwargs is forwarded into apply_chat_template; for
        # thinking-aware templates (Qwen3, Olmo-3.1, Gemma-4-Thinking)
        # this gates the extended-reasoning prefix. Templates that don't
        # reference the kwarg (Qwen 2.5, Llama) ignore it silently.
        chat_kwargs: dict[str, Any] = dict(
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            use_tqdm=False,
        )
        if self._lora_request is not None:
            chat_kwargs["lora_request"] = self._lora_request
        outputs = self._llm.chat([messages], **chat_kwargs)
        latency = time.time() - t0

        out = outputs[0].outputs[0] if outputs and outputs[0].outputs else None
        full_text = (out.text or "") if out is not None else ""
        finish_reason = getattr(out, "finish_reason", None) if out else None

        # Peel the leading <THINK_START>...<THINK_END> block off via
        # ThinkingMixin. Trace is None when thinking was off / absent.
        thinking_trace, raw_text = self._split_thinking(full_text)
        # Direct token-id slice of the trace region — preserves the
        # invariant thoughts_tokens + response_tokens == completion_tokens
        # (no re-tokenization drift). See ThinkingMixin._thoughts_tokens_from_ids.
        thoughts_tokens = self._thoughts_tokens_from_ids(
            out.token_ids if out is not None else None
        )

        # Soft attempt at structured parse — caller's json_repair handles
        # the messier cases when raw_text is wrapped in ```json fences etc.
        raw_json: dict | None = None
        try:
            raw_json = _json.loads(raw_text)
        except (ValueError, TypeError):
            raw_json = None

        prompt_tokens = (
            len(outputs[0].prompt_token_ids) if outputs and outputs[0].prompt_token_ids else None
        )
        completion_tokens = (
            len(out.token_ids) if out is not None and out.token_ids else None
        )

        return ModelOutput(
            raw_text=raw_text,
            raw_json=raw_json,
            latency_s=latency,
            metadata={
                "model_id": self.model_id,
                "tensor_parallel_size": self.tensor_parallel_size,
                "decoder": "freeform",
                "structured_supported": False,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thoughts_tokens": thoughts_tokens,
                "finish_reason": finish_reason,
                "adapter_dir": self.adapter_dir,
                **self._thinking_metadata(),
            },
            thinking_trace=thinking_trace,
        )


class Qwen2_5_72BTextAdapter(VLLMTextAdapter):
    """Qwen 2.5 72B Instruct — non-thinking dense decoder."""

    name = "qwen2.5-72b"
    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
    DEFAULT_TENSOR_PARALLEL_SIZE = 4


class Llama3_3_70BTextAdapter(VLLMTextAdapter):
    """Llama 3.3 70B Instruct — non-thinking dense decoder."""

    name = "llama3.3-70b"
    DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
    DEFAULT_TENSOR_PARALLEL_SIZE = 4


class Olmo3_1_32BTextAdapter(VLLMTextAdapter):
    """Olmo 3.1 32B Instruct — Allen AI; thinking-capable template,
    explicitly disabled via chat_template_kwargs."""

    name = "olmo3.1-32b"
    DEFAULT_MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"
    DEFAULT_TENSOR_PARALLEL_SIZE = 2


class K2_V2TextAdapter(VLLMTextAdapter):
    """LLM360 K2-V2 Instruct."""

    name = "k2-v2"
    DEFAULT_MODEL_ID = "LLM360/K2-V2-Instruct"
    DEFAULT_TENSOR_PARALLEL_SIZE = 4


class Qwen3_32BTextAdapter(VLLMTextAdapter):
    """Qwen 3 32B dense — thinking-capable template, explicitly disabled
    via chat_template_kwargs (inherited from VLLMTextAdapter)."""

    name = "qwen3-32b"
    DEFAULT_MODEL_ID = "Qwen/Qwen3-32B"
    DEFAULT_TENSOR_PARALLEL_SIZE = 2


class Gemma4_31BTextAdapter(VLLMTextAdapter):
    """Google Gemma 4 31B Instruction-tuned — multimodal base served
    text-only. Ships with a chain-of-thought template whose boundary
    tokens differ from the Qwen/Olmo ``<think>``/``</think>`` default
    (see THINK_START/THINK_END overrides below).

    TP=4 (not 2 like Olmo-3.1-32B) because Gemma 4 has heterogeneous
    head dims (head_dim=256, global_head_dim=512) that inflate the KV
    cache per token — vLLM force-selects the TRITON_ATTN backend and
    the ~62 GB weights + KV don't fit on 2×H100 80GB with the default
    max_model_len=32768. TP=4 splits weight+KV across four GPUs
    comfortably.

    FULL MULTIMODAL ARCH
    Gemma-4 is a multimodal base — hf_overrides forces the
    conditional-generation architecture at load time so the loaded arch
    matches the training-time model tree. Text-only chat still works:
    we simply never send image/audio inputs.

    LORA SERVING — MERGE-FIRST (vLLM 0.21.0 workaround)
    vLLM 0.21.0's multimodal-LoRA path silently drops every
    ``language_model.*`` LoRA entry at merge time (confirmed via A/B
    test: same engine, same prompt, WITH-LoRARequest vs WITHOUT is
    BYTE-IDENTICAL). Enabling ``enable_tower_connector_lora=True`` to
    activate the full mm-LoRA path crashes engine init with a
    ``torch.empty(None)`` bug in the tower punica wrapper.

    Workaround: consume a PRE-MERGED checkpoint (base + LoRA folded in
    via peft.merge_and_unload) as a plain model, no LoRARequest. When a
    caller passes ``adapter_dir=X``, this class transparently swaps to
    serving ``X/merged/`` and clears the adapter path; if that dir is
    missing, it errors with the exact merge command to run.
    """

    name = "gemma4-31b"
    DEFAULT_MODEL_ID = "google/gemma-4-31B-it"
    DEFAULT_TENSOR_PARALLEL_SIZE = 4
    HF_OVERRIDES = {"architectures": ["Gemma4ForConditionalGeneration"]}
    # Gemma-4 chain-of-thought is wrapped in a "thought channel", NOT the
    # Qwen/Olmo <think>/</think> pair. With enable_thinking=True the model
    # generates: "<|channel>thought\n<reasoning>\n<channel|><visible answer>".
    # Tokenization (Gemma-4 tokenizer):
    #   "<|channel>thought" -> [100, 45518]  (2 tokens)
    #   "<channel|>"        -> [101]         (1 token — clean s1 stop)
    THINK_START: str = "<|channel>thought"
    THINK_END: str = "<channel|>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.adapter_dir:
            merged = os.path.join(self.adapter_dir, "merged")
            if not os.path.isdir(merged):
                raise RuntimeError(
                    f"Gemma-4 LoRA adapter cannot be served directly — vLLM "
                    f"0.21.0 silently drops language_model.* LoRA entries for "
                    f"Gemma4ForConditionalGeneration. Merge the adapter into a "
                    f"plain checkpoint first:\n"
                    f"  ADAPTER_DIR={self.adapter_dir} \\\n"
                    f"    sbatch slurm_scripts/merge_lora.sbatch\n"
                    f"(writes {merged}/), then re-run inference."
                )
            print(
                f"[Gemma4] adapter {self.adapter_dir} -> serving pre-merged "
                f"model {merged} (base+LoRA merged, no LoRARequest)",
                flush=True,
            )
            self.model_id = merged
            self.adapter_dir = None
            self.max_lora_rank = None
            self._lora_request = None
