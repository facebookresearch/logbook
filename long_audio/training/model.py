# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Omni model loading for E2E fine-tuning (HF-native, not vLLM).

Loads ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` in bf16 with ``device_map=
"auto"`` and wires the trainable surface: LoRA on the LLM attention
q/k/v/o projections (resolved as an explicit name list because PEFT
0.19.x mishandles a regex ``target_modules`` under ``task_type=
"CAUSAL_LM"``) plus ``modules_to_save`` on ``thinker.audio_tower.proj1
/proj2`` so the projector trains as full modules and rides inside the
adapter checkpoint. Everything else (audio encoder, vision, talker) is
frozen. Entry point: :func:`build_model`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Targeting constants + import-light selection helpers (unit-tested).
# ---------------------------------------------------------------------------

#: PEFT ``target_modules`` regex, matched with ``re.fullmatch`` against module
#: names *relative to the thinker* (we wrap ``model.thinker``, so the LLM decoder
#: is ``model.layers.<i>...``). ``model.layers`` is unique to the text LLM: the
#: audio encoder lives under ``audio_tower.layers.*`` and vision under
#: ``visual.*``, so neither is caught — which is the whole point of anchoring on
#: the path instead of the bare ``q_proj`` suffix (that would also wrap the
#: frozen audio encoder's ``q/k/v_proj``).
LORA_TARGET_REGEX = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"

#: Module names (thinker-relative) declared as PEFT ``modules_to_save`` so the
#: projector trains as full non-LoRA modules and is written into the adapter
#: checkpoint. ``proj1`` (d_model->d_model) + ``proj2`` (d_model->LLM hidden) are
#: the audio-token -> LLM-embedding bridge, the only trainable part of the
#: otherwise-frozen audio tower.
PROJECTOR_MODULE_NAMES = ["audio_tower.proj1", "audio_tower.proj2"]

#: Substrings identifying projector params by name. Matches both the plain param
#: names and PEFT's ``modules_to_save`` wrapping
#: (``...audio_tower.proj1.modules_to_save.default.weight``).
PROJECTOR_PARAM_KEYS = ("audio_tower.proj1.", "audio_tower.proj2.")


def is_projector_param(name: str) -> bool:
    """True if ``name`` is a projector param (raw or ``modules_to_save``-wrapped)."""
    return any(k in name for k in PROJECTOR_PARAM_KEYS)


def is_lora_param(name: str) -> bool:
    """True if ``name`` is a PEFT LoRA adapter param (``lora_A``/``lora_B``…)."""
    return "lora_" in name


def lora_targets_module(module_name: str) -> bool:
    """True if ``module_name`` (thinker-relative) is a LoRA target.

    Mirrors what PEFT does with a string ``target_modules`` (``re.fullmatch``),
    so tests can assert the regex hits the LLM attention and misses the audio /
    vision encoders without instantiating PEFT.
    """
    return re.fullmatch(LORA_TARGET_REGEX, module_name) is not None


def trainable_parameter_summary(module: Any) -> dict[str, Any]:
    """Break down trainable params into LoRA vs projector vs other.

    Returns element counts + a ``trainable_pct`` of the total. Useful as a
    startup sanity print (catches e.g. the projector being silently re-frozen,
    or LoRA accidentally landing on the audio encoder).
    """
    total = lora = projector = other = 0
    for name, p in module.named_parameters():
        n = p.numel()
        total += n
        if not p.requires_grad:
            continue
        if is_lora_param(name):
            lora += n
        elif is_projector_param(name):
            projector += n
        else:
            other += n
    trainable = lora + projector + other
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "projector": projector,
        "other": other,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def thinker_device_map(top_model: Any) -> dict[str, Any] | None:
    """Derive a thinker-relative ``hf_device_map`` from the top-level model.

    ``from_pretrained(device_map="auto")`` sets ``hf_device_map`` on the
    top-level ``Qwen3OmniMoeForConditionalGeneration`` only, with keys like
    ``"thinker.model.layers.3"`` / ``"talker..."``. Since we hand the *thinker*
    (peft-wrapped) to ``Trainer``, and Trainer detects model-parallel by reading
    ``model.hf_device_map`` (transformers ``trainer.py``: >1 non-cpu device ->
    ``is_model_parallel=True`` -> skips DataParallel), we must re-expose the map
    scoped to the thinker.

    Returns the stripped sub-map (``"thinker."`` prefix removed, talker/code2wav
    keys dropped), or ``None`` if the model was not loaded with a device map
    (single-device / CPU) — in which case Trainer handles placement normally.
    """
    dmap = getattr(top_model, "hf_device_map", None)
    if not dmap:
        return None
    sub: dict[str, Any] = {}
    for key, dev in dmap.items():
        if key == "thinker" or key.startswith("thinker."):
            sub[key[len("thinker.") :] if key != "thinker" else ""] = dev
    return sub or None


# ---------------------------------------------------------------------------
# Config + model construction (heavy deps imported lazily).
# ---------------------------------------------------------------------------


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


def select_lora_target_names(module: Any) -> list[str]:
    """Resolve the explicit list of attention modules to LoRA-adapt.

    Selects module names with :data:`LORA_TARGET_REGEX` (``re.fullmatch``, which
    hits only the LLM ``model.layers.<i>.self_attn.(q|k|v|o)_proj``). We hand
    PEFT this explicit list rather than the regex string because PEFT 0.19.x
    mishandles a str ``target_modules`` under ``task_type="CAUSAL_LM"`` (iterates
    it as characters). Sorted for determinism.
    """
    return sorted({n for n, _ in module.named_modules() if lora_targets_module(n)})


def build_lora_config(lora: LoraSettings, target_modules: list[str]):
    """Construct the ``peft.LoraConfig`` for LLM-attention-only adaptation.

    ``target_modules`` must be the explicit list resolved by
    :func:`select_lora_target_names` (a regex string is unsafe — see its
    docstring). NOTE: we deliberately do NOT use PEFT ``"all-linear"`` for E2E:
    on the Qwen3-Omni MoE thinker it resolves to no LoRA targets ("No modules
    were targeted for adaptation") — the experts are ``nn.Parameter`` and the
    omni layout under ``device_map="auto"`` yields an empty set — so E2E targets
    the LLM attention explicitly. (Cascade text-only training uses all-linear.)
    """
    from peft import LoraConfig

    if not target_modules:
        raise ValueError(
            "build_lora_config: empty target_modules — the LoRA-target selection "
            "matched no attention modules (has the model layout changed?)."
        )
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
        # Train the projector as full non-LoRA modules; PEFT saves them into the
        # adapter checkpoint so best-keeping restores the best projector too.
        modules_to_save=list(PROJECTOR_MODULE_NAMES),
    )


def load_processor(model_id: str):
    """Load the Qwen3-Omni processor (``Qwen3OmniMoeProcessor``)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def build_model(
    model_id: str,
    *,
    lora: LoraSettings | None = None,
    device_map: str | dict | None = "auto",
    gradient_checkpointing: bool = True,
    attn_implementation: str = "flash_attention_2",
    router_aux_loss_coef: float | None = None,
):
    """Load Qwen3-Omni for E2E fine-tuning and return ``(peft_thinker, top_model)``.

    ``peft_thinker`` is what you pass to ``Trainer`` (its ``forward`` takes
    ``labels`` -> ``loss``); ``top_model`` is retained so the caller can access
    the config / keep the object graph alive.

    Steps (see module docstring): bf16 load -> ``disable_talker()`` ->
    ``get_peft_model`` with LoRA on the LLM attention + the projector as
    ``modules_to_save`` (this alone defines the trainable surface: everything
    else is frozen) -> grad-ckpt + ``enable_input_require_grads`` +
    ``use_cache=False`` -> propagate a thinker-scoped device map so Trainer runs
    naive model-parallel (not DataParallel).
    """
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    lora = lora or LoraSettings()

    # ``torch_dtype`` (not ``dtype``) for compatibility with older transformers
    # pins (las-enclap venv is 4.29, which doesn't recognise the ``dtype`` alias
    # introduced in 4.45+). Both are accepted on recent transformers.
    top = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    # Drop the speech synthesiser — we do audio-in / text-out only.
    top.disable_talker()

    thinker = top.thinker
    thinker.config.use_cache = False
    if router_aux_loss_coef is not None:
        # Best-effort: enable the MoE load-balancing aux term. The thinker adds
        # `router_aux_loss_coef * aux_loss` to the LM loss when its forward runs
        # with output_router_logits=True.
        thinker.config.text_config.router_aux_loss_coef = router_aux_loss_coef
        thinker.config.text_config.output_router_logits = True

    from peft import get_peft_model

    # LoRA (attention) + modules_to_save (projector) fully define the trainable
    # surface; get_peft_model freezes the encoder, experts, vision, and lm_head.
    # Explicit target list (NOT "all-linear"): peft's all-linear yields "No
    # modules were targeted" on this omni/MoE thinker under device_map="auto".
    target_names = select_lora_target_names(thinker)
    if not target_names:
        raise RuntimeError(
            "No LoRA target modules matched on the thinker — the model layout "
            "may have changed (expected model.layers.*.self_attn.[qkvo]_proj)."
        )
    peft_thinker = get_peft_model(thinker, build_lora_config(lora, target_names))

    # Verify PROJECTOR_MODULE_NAMES actually landed on real modules. Silent-
    # freeze failure: if audio_tower's projector was renamed upstream,
    # ``modules_to_save`` matches nothing and the audio-token->LLM bridge stays
    # frozen without any user-visible error. See training code review #8.
    _assert_projector_wrapped(peft_thinker, thinker)

    if gradient_checkpointing:
        # Order matters: enable_input_require_grads so grad-ckpt has a grad-
        # requiring input even though the embedding/base is frozen.
        peft_thinker.enable_input_require_grads()
        thinker.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Make Trainer treat this as model-parallel (skip DataParallel) when the
    # top-level was sharded across >1 GPU.
    sub_map = thinker_device_map(top)
    if sub_map is not None:
        peft_thinker.hf_device_map = sub_map
        peft_thinker.is_parallelizable = True
        peft_thinker.model_parallel = True

    # Under naive MP (device_map != None), the modules_to_save deep-copy of the
    # projector should inherit device placement via accelerate hooks. In some
    # peft x accelerate combinations the copy silently lands on CPU, which then
    # blows up mid-forward with a device mismatch. Catch it at build time. See
    # training code review #7.
    _assert_trainable_not_on_cpu(peft_thinker, device_map)

    return peft_thinker, top


def _assert_projector_wrapped(peft_thinker: Any, thinker: Any) -> None:
    """Fail loud if PROJECTOR_MODULE_NAMES didn't wrap any trainable params."""
    proj_trainable = sum(
        p.numel()
        for n, p in peft_thinker.named_parameters()
        if p.requires_grad and is_projector_param(n)
    )
    if proj_trainable > 0:
        return
    audio_children: list[str] = []
    audio_tower = getattr(thinker, "audio_tower", None)
    if audio_tower is not None:
        try:
            audio_children = [n for n, _ in audio_tower.named_children()]
        except (AttributeError, TypeError):
            audio_children = []
    raise RuntimeError(
        "build_model: PROJECTOR_MODULE_NAMES "
        f"{PROJECTOR_MODULE_NAMES} matched no trainable params after PEFT "
        "wrap — modules_to_save didn't hit the expected "
        "audio_tower.proj1/proj2 targets. Model layout has drifted. "
        f"Actual audio_tower children: {audio_children or '<not accessible>'}"
    )


def _assert_trainable_not_on_cpu(peft_thinker: Any, device_map: Any) -> None:
    """Fail loud if any trainable param is on CPU when a GPU device_map was requested."""
    if device_map is None:
        return
    if isinstance(device_map, str) and device_map == "cpu":
        return
    cpu_trainable: list[tuple[str, tuple[int, ...]]] = []
    for name, p in peft_thinker.named_parameters():
        if p.requires_grad and p.device.type == "cpu":
            cpu_trainable.append((name, tuple(p.shape)))
    if not cpu_trainable:
        return
    raise RuntimeError(
        f"build_model: {len(cpu_trainable)} trainable param(s) landed on CPU "
        f"despite device_map={device_map!r} — likely a modules_to_save + "
        "accelerate hook mismatch that would blow up mid-forward with a "
        "device mismatch. First offenders (name, shape): "
        f"{cpu_trainable[:5]}"
    )


def install_router_aux_diagnostic(peft_thinker: Any, coef: float) -> Any:
    """Register a one-shot forward hook that logs ``aux_loss`` vs ``loss``.

    Diagnostic to confirm ``--router-aux-loss-coef`` actually contributes to
    the Trainer's loss: on the first forward call, prints both fields from the
    output plus ``aux_loss * coef``. Fires once, then removes itself (so
    gradient checkpointing's re-forwards don't spam the log). Returns the hook
    handle for callers that want to remove it manually. See training code
    review #10.
    """
    state = {"fired": False, "handle": None}

    def _get(o: Any, key: str) -> Any:
        if isinstance(o, dict):
            return o.get(key)
        return getattr(o, key, None)

    def _hook(_module, _inputs, output):
        if state["fired"]:
            return
        state["fired"] = True
        loss = _get(output, "loss")
        aux = _get(output, "aux_loss")
        contrib = None
        if aux is not None:
            try:
                contrib = float(aux) * float(coef)
            except (TypeError, ValueError):
                contrib = None
        print(
            f"[router-aux-diagnostic] first-step loss={loss!r} "
            f"aux_loss={aux!r} coef={coef} aux*coef={contrib}",
            flush=True,
        )
        handle = state["handle"]
        if handle is not None:
            handle.remove()

    state["handle"] = peft_thinker.register_forward_hook(_hook)
    return state["handle"]


# ---------------------------------------------------------------------------
# Checkpoint save (LoRA adapter + projector — both handled by PEFT).
# ---------------------------------------------------------------------------


def save_trainable(peft_thinker: Any, out_dir: str | Path) -> None:
    """Persist the trainable weights: LoRA adapter + projector, in one artifact.

    ``save_pretrained`` writes both the LoRA deltas and the ``modules_to_save``
    projector into ``adapter_model.safetensors`` (+ ``adapter_config.json``), so
    a checkpoint is fully round-trippable via ``PeftModel.from_pretrained`` (see
    ``long_audio.training.infer.load_finetuned_model``). The frozen base is not
    saved — reload it from ``model_id`` and re-attach.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_thinker.save_pretrained(str(out_dir))
