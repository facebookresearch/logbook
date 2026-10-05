"""HF-native inference adapter for the fine-tuned Qwen3-Omni thinker.

``Qwen3OmniHFAdapter`` wraps a loaded ``Qwen3OmniMoeForConditionalGeneration``
(+ its processor) behind the standard :class:`~long_audio.inference.models.base.
ModelAdapter` interface, so the fine-tuned model reuses the exact chunking /
prompting / stitching path of the baseline inference runner
(``long_audio.inference.chunk_runner.run_inference``). That guarantees the
fine-tuned eval numbers are apples-to-apples with the frozen-model baselines
and land in the same ``eval_segmentation.json`` format.

The model was SFT'd to emit the segmentation JSON directly, so structured
grammar decoding is unnecessary: ``supports_structured()`` is ``False`` and the
runner pipes the raw text through ``json_repair`` (decoder ``"freeform"``).

GPU-only: ``generate`` and ``load_finetuned_model`` require the 30B checkpoint +
transformers/peft on CUDA. The module imports cheaply (torch only) so the eval
driver + tests can import it without the heavy deps.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


class Qwen3OmniHFAdapter(ModelAdapter):
    """Adapter around a loaded (fine-tuned) Qwen3-Omni model for generation.

    Args:
        model: a loaded ``Qwen3OmniMoeForConditionalGeneration`` (talker
            disabled). LoRA may be live (training callback) or merged
            (standalone eval) — either way generation uses the adapted weights.
        processor: the matching ``Qwen3OmniMoeProcessor``.
        name: adapter name recorded in run metadata.
        input_device: device to place processor tensors on. Defaults to the
            thinker's first-parameter device; accelerate dispatch hooks re-route
            across shards under ``device_map="auto"``.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        name: str = "qwen3-omni-ft",
        input_device: Any = None,
    ):
        self.model = model
        self.processor = processor
        self.name = name
        self._input_device = input_device
        self._loaded = True

    def load(self) -> None:  # already loaded by the caller
        self._loaded = True

    def supports_structured(self) -> bool:
        # SFT'd to emit JSON directly; no grammar constraint -> freeform decode.
        return False

    def _device(self):
        if self._input_device is not None:
            return self._input_device
        thinker = getattr(self.model, "thinker", self.model)
        try:
            return next(thinker.parameters()).device
        except StopIteration:
            return None

    def _generate_ids(self, inputs: dict, gen_kwargs: dict):
        # Qwen-Omni's generate returns text ids (talker disabled). Some
        # versions accept return_audio=False; fall back if not.
        try:
            return self.model.generate(**inputs, return_audio=False, **gen_kwargs)
        except TypeError:
            return self.model.generate(**inputs, **gen_kwargs)

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
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=[text], audio=[audio], sampling_rate=sample_rate, return_tensors="pt"
        )
        dev = self._device()
        if dev is not None:
            inputs = inputs.to(dev)
        # The processor emits audio ``input_features`` in float32, but the model
        # is loaded in bf16 -> the audio tower's conv2d bias is bf16 and rejects
        # a float32 input ("Input type (float) and bias type (c10::BFloat16)
        # should be the same"). Cast the batch to the model dtype; BatchFeature
        # .to(dtype=...) only touches floating tensors, so input_ids / masks
        # stay integer. (Canonical Qwen-Omni inference pattern: inputs.to(dtype).)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is None:
            thinker = getattr(self.model, "thinker", self.model)
            first_param = next(thinker.parameters(), None)
            model_dtype = first_param.dtype if first_param is not None else None
        if model_dtype is not None and hasattr(inputs, "to"):
            inputs = inputs.to(dtype=model_dtype)

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.time()
        with torch.no_grad():
            out = self._generate_ids(inputs, gen_kwargs)
        latency = time.time() - t0
        if isinstance(out, (tuple, list)):
            out = out[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        raw_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return ModelOutput(
            raw_text=raw_text,
            raw_json=None,
            latency_s=latency,
            metadata={"adapter": self.name},
        )


def load_finetuned_model(
    model_id: str,
    adapter_dir: str | Path | None,
    *,
    device_map: str | dict | None = "auto",
    attn_implementation: str = "flash_attention_2",
    merge: bool = True,
):
    """Load the base Qwen3-Omni + (optionally) a saved LoRA adapter for inference.

    Loads the bf16 base, disables the talker, and — when ``adapter_dir`` is
    provided — attaches the PEFT adapter and (by default) merges it into the
    base for faster generation. Returns the top-level model ready for
    :class:`Qwen3OmniHFAdapter`.

    When ``adapter_dir`` is ``None`` the base is returned as-is (no peft
    involvement) — used by the base HF-native serving path (``run_e2e.py
    --engine hf``) to sidestep vLLM's max_model_len cap and reach the model's
    native max_position_embeddings=40960 for long-chunk inference.

    Both the LoRA deltas and the ``modules_to_save`` projector live inside the
    adapter checkpoint, so ``PeftModel.from_pretrained`` restores them together;
    ``merge_and_unload`` then folds the LoRA into the base linears and swaps the
    trained projector in for its frozen original.
    """
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    # ``torch_dtype`` (not ``dtype``) for compatibility with older transformers
    # pins; matches ``build_model``.
    top = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    top.disable_talker()

    if adapter_dir is None:
        # Base HF path: no LoRA to attach. Enable KV cache and clear the stale
        # device map so subsequent readers see a clean state.
        top.thinker.config.use_cache = True
        if hasattr(top, "hf_device_map"):
            top.hf_device_map = None
        return top

    from peft import PeftModel

    # peft 0.19.1's transformers-v5 adapter-load conversion is incompatible with
    # transformers 5.9.0: convert_peft_adapter_state_dict_for_transformers() calls
    # WeightConverter(distributed_operation=..., quantization_operation=...), but
    # tf 5.9.0's WeightConverter.__init__ only accepts (source_patterns,
    # target_patterns, operations) -> TypeError on load. Our adapter is saved in
    # classic peft key format (base_model.model.*.lora_A/B + modules_to_save
    # projector) and the base is loaded normally (not tf-v5 tensor-parallel), so
    # the classic load path is exactly what we want. Force-disable the v5
    # conversion for the duration of the load, then restore it. (Save side never
    # applies the conversion, so this is symmetric.)
    import peft.utils.save_and_load as _peft_sl

    _prev_ge_v5 = _peft_sl.is_transformers_ge_v5
    # In some peft versions ``is_transformers_ge_v5`` is a bool; in others it's
    # a callable. Preserve the type of the patched value so downstream call
    # sites (``is_transformers_ge_v5()`` vs ``if is_transformers_ge_v5:``) keep
    # working regardless. See training code review #9.
    if callable(_prev_ge_v5):
        _peft_sl.is_transformers_ge_v5 = lambda: False
    else:
        _peft_sl.is_transformers_ge_v5 = False
    try:
        peft_thinker = PeftModel.from_pretrained(top.thinker, str(adapter_dir))
    finally:
        _peft_sl.is_transformers_ge_v5 = _prev_ge_v5
    top.thinker = peft_thinker.merge_and_unload() if merge else peft_thinker
    top.thinker.config.use_cache = True
    # ``top.hf_device_map`` (set by ``from_pretrained(device_map="auto")``)
    # still references the pre-merge module tree; clear it so any future reader
    # gets a clean ``None`` rather than stale keys pointing at freed modules.
    # See training code review #14.
    if hasattr(top, "hf_device_map"):
        top.hf_device_map = None
    return top
