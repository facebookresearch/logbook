# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Compute real 24h-SINS FLOPs for every model in the long-audio benchmark.

Uses actual prediction outputs (tokenized with each model's own tokenizer)
plus audio-encoder FLOPs (Whisper/Qwen-Omni audio module/CLAP) on top of
LLM FLOPs. MoE arms use published active-param counts.

Canonical workload = first 24h of SINS: 144 10-min windows for text-LLM
Stage-B and E2E audio LLMs; 8640 10-sec sub-chunks for captioners.

Requires the enclap venv (enclap + msclap-cap + transformers) and 1 GPU.
Output goes to ``--out`` (or ``$FLOPS_OUT``) as a per-model breakdown.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Populated by main() from --runs-root / --out (or $RUNS_DIR / $FLOPS_OUT).
RUNS: Path | None = None
OUT_PATH: Path | None = None

# --- Canonical 24h-of-SINS shape --------------------------------------------
N_WIN_10MIN = 144        # cascB text-LLM + E2E audio LLM
N_SUB_10S   = 8640       # captioners
SR = 16_000
DUMMY_10S    = np.zeros(SR * 10, dtype=np.float32)
DUMMY_10MIN  = np.zeros(SR * 600, dtype=np.float32)


# --- Model registry ---------------------------------------------------------
# n_active_B: published active params per forward (in billions). For MoE
# models this is smaller than total; for dense it equals total. For API-
# only models where the vendor doesn't disclose, set None + note.
#
# tokenizer_id: HF hub id (or local path) for the tokenizer we use to count
# tokens. For API-only models where the vendor's tokenizer isn't open-source,
# fall back to a proxy (e.g. tiktoken cl100k / gpt-4o) and note the caveat.
#
# hf_model_id: HF hub id used to load the full *ForConditionalGeneration
# for per-family full-model prefill FLOPs measurement (compute_e2e /
# compute_captioner). encoder architecture, chunking behavior, and audio-
# token rate are all discovered from the actual model — no encoder_kind
# hint needed.
#
# For CLAP entries only: encoder_kind + decoder_kind are used by
# compute_clap for the audit record + BART/GPT-2 decoder param count.

REGISTRY: dict[str, dict[str, Any]] = {
    # ---- Text-LLMs (Stage-B cascade) — no audio encoder ---------------------
    # n_active_B measured via .parameters() on meta-tensor HF init at runtime
    # (count_params_from_hf, cheap — config only). Open-weights only; Gemini
    # and Luna keep n_active_B=None with no hf_model_id because the weights
    # aren't published — token counts still computed with the tokenizer proxy.
    "text_llm:k2-v2":          {"n_active_B": None, "tokenizer_id": "LLM360/K2-V2-Instruct",       "hf_model_id": "LLM360/K2-V2-Instruct",       "n_note": "LLM360 K2-V2 (dense Llama-family; NOT Moonshot Kimi K2)"},
    "text_llm:olmo3.1-32b":    {"n_active_B": None, "tokenizer_id": "allenai/Olmo-3.1-32B-Instruct","hf_model_id": "allenai/Olmo-3.1-32B-Instruct"},
    "text_llm:qwen2.5-72b":    {"n_active_B": None, "tokenizer_id": "Qwen/Qwen2.5-72B-Instruct",   "hf_model_id": "Qwen/Qwen2.5-72B-Instruct"},
    "text_llm:llama3.3-70b":   {"n_active_B": None, "tokenizer_id": "meta-llama/Llama-3.3-70B-Instruct","hf_model_id": "meta-llama/Llama-3.3-70B-Instruct"},
    "text_llm:qwen3-32b":      {"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-32B",              "hf_model_id": "Qwen/Qwen3-32B"},
    "text_llm:gemma4-31b":     {"n_active_B": None, "tokenizer_id": "google/gemma-4-31B-it",       "hf_model_id": "google/gemma-4-31B-it"},
    "text_llm:gemini":         {"n_active_B": None, "tokenizer_id": "google/gemma-4-31B-it",       "hf_model_id": None, "n_note": "Gemini 3.5 Flash — undisclosed weights; tokenizer proxy = Gemma-4"},
    "text_llm:luna":           {"n_active_B": None, "tokenizer_id": "openai-community/gpt2",       "hf_model_id": None, "n_note": "GPT-5.6 (Luna) — undisclosed weights; tokenizer proxy = GPT-2"},

    # ---- E2E audio LLMs -----------------------------------------------------
    # n_active_B always measured via .parameters() at runtime (dense arch is
    # the norm; Qwen3-Omni MoE returns total 30B — we report it honestly and
    # let the reader interpret vs the disclosed 3B-active spec).
    "e2e:qwen3-omni":          {"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "hf_model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "n_note": "MoE — .parameters() returns TOTAL (30B); active is ~3B (128 experts × top-8)"},
    "e2e:qwen2.5-omni":        {"n_active_B": None, "tokenizer_id": "Qwen/Qwen2.5-Omni-7B", "hf_model_id": "Qwen/Qwen2.5-Omni-7B"},
    "e2e:af3-hf":              {"n_active_B": None, "tokenizer_id": "nvidia/audio-flamingo-3-hf", "hf_model_id": "nvidia/audio-flamingo-3-hf"},

    # ---- Captioners (non-CLAP) — 10s sub-chunks ----------------------------
    "cap:af3-hf":              {"n_active_B": None, "tokenizer_id": "nvidia/audio-flamingo-3-hf", "hf_model_id": "nvidia/audio-flamingo-3-hf"},
    "cap:qwen2.5-omni":        {"n_active_B": None, "tokenizer_id": "Qwen/Qwen2.5-Omni-7B", "hf_model_id": "Qwen/Qwen2.5-Omni-7B"},
    "cap:qwen3-omni":          {"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "hf_model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "n_note": "MoE — .parameters() returns TOTAL (30B)"},
    "cap:qwen3-omni-captioner":{"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-Omni-30B-A3B-Captioner", "hf_model_id": "Qwen/Qwen3-Omni-30B-A3B-Captioner", "n_note": "MoE — .parameters() returns TOTAL"},

    # ---- CLAP captioners — encoder + autoregressive decoder ----------------
    "clap:enclap":             {"n_active_B": None, "tokenizer_id": "facebook/bart-base", "adapter": "long_audio.inference.models.enclap.EnClapLargeAdapter", "encoder_kind": "clap-large", "decoder_kind": "bart-base", "encoder_input_s": 10, "n_note": "N will be measured from .parameters() at load"},
    "clap:msclap-cap":         {"n_active_B": None, "tokenizer_id": "openai-community/gpt2", "adapter": "long_audio.inference.models.msclap.MSClapCapAdapter", "encoder_kind": "msclap", "decoder_kind": "gpt2", "encoder_input_s": 10, "n_note": "N will be measured from .parameters() at load"},

    # ---- FT (fine-tuned) arms — inference-only cost, delta from base -------
    # LoRA-merged / adapter FT preserves architecture (merge_lora bakes
    # adapters in; vLLM LoRA at inference doesn't change shapes), so N and
    # prefill FLOPs are IDENTICAL to base. Adding these arms costs zero extra
    # GPU work: same hf_model_id → _PARAM_COUNT_CACHE + _ENCODER_FLOPS_CACHE
    # both hit for whichever base arm ran first. Only token counts (from the
    # FT cell's chunk_*.json) differ, driving all delta FLOPs.
    #
    # `cell_override` steers compute_{text_llm,e2e} to the FT cell instead of
    # the default {cascB_{model} / cascA_af3-hf / sins} / {e2e_{model} / sins}.
    "text_llm:gemma4-31b_ft":  {"n_active_B": None, "tokenizer_id": "google/gemma-4-31B-it",       "hf_model_id": "google/gemma-4-31B-it",       "cell_override": "cascB_gemma4-31b_ft/cascA_af3-hf/sins",  "n_note": "merged full-FT (adapter=None); N + shape identical to base gemma4-31b"},
    "text_llm:qwen3-32b_ft":   {"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-32B",              "hf_model_id": "Qwen/Qwen3-32B",              "cell_override": "cascB_qwen3-32b_ft/cascA_af3-hf/sins",   "n_note": "base + LoRA adapter; N + shape identical to base qwen3-32b"},
    "e2e:qwen3-omni_ft":       {"n_active_B": None, "tokenizer_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "hf_model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "cell_override": "e2e_qwen3-omni_ft/sins", "n_note": "FT adapter; architecture identical to base qwen3-omni (MoE 128×top-8) — prefill_flops_per_chunk reused via _ENCODER_FLOPS_CACHE"},
}


# --- Helpers ----------------------------------------------------------------

def load_tokenizer(tokenizer_id: str):
    """AutoTokenizer.from_pretrained with trust_remote_code. No tiktoken
    fallback — closed-weights entries (Gemini, Luna) list a proxy
    open-weights tokenizer_id in REGISTRY (Gemma-4 / GPT-2) and MUST resolve.
    A silent tiktoken fallback would misreport token counts."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)


def import_adapter(path: str):
    """Resolve 'a.b.c.ClassName' to a class object."""
    mod_path, cls_name = path.rsplit(".", 1)
    __import__(mod_path)
    return getattr(sys.modules[mod_path], cls_name)


# Cache: per-family prefill FLOPs per (hf_model_id, seconds). Loading a
# 30 B model to GPU per call is expensive; a single (family, duration)
# measurement is content-independent so we reuse across e2e / captioner /
# per-cell calls.
_ENCODER_FLOPS_CACHE: dict[tuple[str, int], dict] = {}


def measure_family_full_prefill_flops(hf_model_id: str, seconds: int) -> dict:
    """Measure per-family FLOPs by feeding the FULL model (encoder +
    connector + LLM) one forward pass over `seconds` seconds of audio,
    with no autoregressive generation. Returns:
        {
            "prefill_flops": int,      # calflops on the whole forward
            "n_total_params": int,     # sum of .parameters()
            "n_encoder_params": int,   # audio_tower / audio_encoder submodule
            "n_total_params_raw": int, # sum(model.parameters()) — MoE-inflated
            "n_total_params_active": int, # MoE-contracted + .talker filtered
            "n_encoder_params": int,   # audio_tower / audio_encoder submodule
            "n_decoder_active_params": int,  # n_total_active - n_encoder
            "moe": bool, "moe_hparams": tuple | None,
        }

    One calflops call over the whole prefill path captures encoder +
    connector + LLM-prefill correctly for both chunking regimes:
        * AF-family: processor tiles 10-min audio into 30-s slices, so
          the "prefill" here is effectively 20 × 30-s encoder passes +
          one LLM prefill over the resulting audio tokens.
        * Qwen-Omni AuT: single variable-length forward with block-wise
          attention (sub-linear scaling — direct measurement only).

    The autoregressive tail is NOT measured here (would inflate the
    number by up to 2·N_dec·completion_tokens for each generated token).
    Callers add that term as 2 * n_decoder_active_params * completion_tokens.
    Uses count_params_from_hf's MoE-aware active count (fused-expert
    walker + .talker filter) so the autoregressive term isn't inflated
    by inactive experts for MoE models like Qwen3-Omni.

    HARD FAILS on any calflops error or memory failure — no fallback.
    Loads the full model on GPU in bfloat16 (30B ≈ 60 GB → needs 4×H100
    for Qwen3-Omni; AF3 8B fits on 1 H100).
    """
    key = (hf_model_id, seconds)
    if key in _ENCODER_FLOPS_CACHE:
        return _ENCODER_FLOPS_CACHE[key]

    import importlib
    import numpy as np
    import transformers
    from transformers import AutoConfig, AutoProcessor
    # Use torch's native FlopCounterMode (TorchDispatchMode-based) instead
    # of calflops for the full-model path — calflops's nn.functional
    # monkey-patching interferes with accelerate's device-alignment hooks
    # on sharded models (Qwen3-Omni-30B hits cross-device tensor errors
    # inside every RMSNorm / Linear during the shard-crossing forward).
    # FlopCounterMode is op-level, sharding-transparent, and torch.no_grad
    # -compatible.
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.flop_counter as _fc

    # Patch sdpa_flop_count for GQA: default asserts Q/K shape parity,
    # but Qwen3-Omni uses grouped-query attention where Q_heads > K_heads.
    # Standard SDPA formula: 4 · batch_prod · Q_heads · seqQ · seqK · head_dim
    # (softmax O(Q_heads·seqQ·seqK) is negligible vs matmul; skip it).
    def _sdpa_flop_count_gqa(query_shape, key_shape, value_shape):
        q_heads, s_q, d_q = query_shape[-3:]
        _, s_k, _ = key_shape[-3:]
        batch_prod = 1
        for x in query_shape[:-3]:
            batch_prod *= x
        return 4 * batch_prod * q_heads * s_q * s_k * d_q
    _fc.sdpa_flop_count = _sdpa_flop_count_gqa

    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    arch = cfg.architectures[0]
    # Prefer the top-level export; fall through to the arch's own module.
    # Suffix rewrite handles configs that name the bare Model wrapper (e.g.
    # Qwen2.5-Omni: arch='Qwen2_5OmniModel', but only
    # 'Qwen2_5OmniForConditionalGeneration' has a real forward()).
    candidates = [arch]
    if arch.endswith("Model"):
        candidates.append(arch[:-len("Model")] + "ForConditionalGeneration")
        candidates.append(arch[:-len("Model")] + "ForCausalLM")
    else:
        candidates.append(arch + "ForConditionalGeneration")
        candidates.append(arch + "ForCausalLM")
    model_cls = next((getattr(transformers, n) for n in candidates
                      if hasattr(transformers, n)), None)
    if model_cls is None:
        mt = cfg.model_type
        mod = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
        model_cls = next((getattr(mod, n) for n in candidates
                          if hasattr(mod, n)), None)
    if model_cls is None:
        raise RuntimeError(
            f"Cannot resolve model class for {hf_model_id} "
            f"(arch={cfg.architectures[0]!r}, tried={candidates})"
        )

    # Load full model on GPU in bf16.
    # - device_map="cuda:0" (single GPU): required for AF3 because its
    #   audio→LLM connector mixes tensors across shards, tripping
    #   "expected all tensors to be on the same device" during forward.
    #   AF3 fits on one H100.
    # - device_map="auto" (sharded): Qwen3-Omni-30B doesn't fit on a single
    #   80 GB H100 (thinker+talker ≈ 70 GB weights + activations); keep
    #   sharded and let HF's device dispatch handle cross-device forward.
    #   The audio-tower + thinker path used for FLOPs stays internally
    #   consistent, so the same-device error doesn't fire here.
    # device_map for the forward measurement:
    # - AF3: 'cuda:0' — connector on sharded model crashes.
    # - Qwen-Omni: 'auto' — 30B thinker + 600s audio activations don't fit
    #   on one 80 GB H100. FlopCounterMode is sharding-transparent (unlike
    #   calflops), so cross-device forwards work fine here.
    device_map = "auto" if cfg.model_type.startswith("qwen") else "cuda:0"
    print(f"    [prefill-flops {hf_model_id}] loading full model in bf16 "
          f"(device_map={device_map!r}) …", flush=True)
    model = model_cls.from_pretrained(
        hf_model_id, torch_dtype=torch.bfloat16, device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # Locate audio encoder submodule for the param-count split.
    encoder, encoder_path = None, None
    for holder in ("", "thinker.", "model."):
        base = model
        for part in holder.split("."):
            if part:
                base = getattr(base, part)
        for attr in ("audio_tower", "audio_encoder"):
            if hasattr(base, attr):
                encoder = getattr(base, attr); encoder_path = f"{holder}{attr}"; break
        if encoder is not None: break
    if encoder is None:
        raise RuntimeError(
            f"No audio_tower / audio_encoder submodule found in {hf_model_id} "
            f"(top-level children: {[n for n, _ in model.named_children()]})"
        )

    n_total_raw = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    # MoE-aware ACTIVE param count for the 2·N_dec·completion autoregressive
    # term. count_params_from_hf handles MoE contraction (top-k experts only)
    # + .talker filter (TTS branch — irrelevant for our text-out FLOPs).
    pc = count_params_from_hf(hf_model_id)
    n_total_active = pc["active"]
    n_dec_active = n_total_active - n_enc
    if n_dec_active <= 0:
        raise RuntimeError(
            f"n_dec_active came out {n_dec_active} for {hf_model_id} "
            f"(active_total={n_total_active}, encoder={n_enc}). Sanity-check "
            "the MoE walker or the encoder submodule locator."
        )

    # Build the multimodal prompt EXACTLY like the model's inference code:
    # chat template with an audio placeholder → processor with the SINGULAR
    # `audio=` key (plural `audios=` silently drops audio for every family
    # in this set and yields a text-only forward that crashes downstream).
    proc = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
    wav = np.zeros(16000 * seconds, dtype=np.float32)
    chat_text = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "audio", "audio": "placeholder"},
            {"type": "text",  "text":  "Describe."},
        ]}],
        add_generation_prompt=True, tokenize=False,
    )
    feat = proc(
        text=[chat_text], audio=[wav],
        sampling_rate=16000, return_tensors="pt",
    )
    if not any(k in feat for k in ("input_features", "audio_features",
                                    "feature_attention_mask")):
        raise RuntimeError(
            f"Processor for {hf_model_id} produced no audio-feature key; "
            f"got {sorted(feat.keys())}. Text-only forward would not "
            "exercise the audio encoder + connector."
        )

    # Move tensor inputs to the model's device, AND cast float tensors to
    # the model's dtype (bfloat16). Processor emits `input_features` as
    # float32 by default; the model's audio-tower Conv2d weights are bf16
    # → cross-dtype forward crashes with "Input type (float) and bias type
    # (BFloat16) should be the same". Int tensors (input_ids, attention_mask)
    # must stay int. Matches HF reference: `inputs.to(model.device).to(model.dtype)`.
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = {}
    for k, v in feat.items():
        if not hasattr(v, "to"):
            inputs[k] = v
            continue
        tv = v.to(first_device)
        if tv.is_floating_point():
            tv = tv.to(model_dtype)
        inputs[k] = tv
    # Disable KV cache via config (not kwargs — calflops would try to .to()
    # the bool). All these models have .config.use_cache; no need to guard.
    model.config.use_cache = False

    # Qwen-Omni families: the top-level `ForConditionalGeneration.forward`
    # routes through BOTH thinker (text out, what we care about) and talker
    # (TTS). Without `speaker=...` args, the talker path hits stub nn.Modules
    # and crashes with `_forward_unimplemented`. Feed calflops the thinker
    # submodule directly — matches the FLOPs we actually pay per generated
    # text token, and skips the TTS branch entirely.
    forward_target = model.thinker if hasattr(model, "thinker") else model
    if forward_target is not model:
        # Free talker/token2wav weights before forward — they're TTS-only,
        # not invoked here, but 30B-A3B loads them anyway (~10 GB in bf16)
        # which puts the forward over 80 GB on 600s audio activations.
        for attr in ("talker", "token2wav"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        import gc; gc.collect(); torch.cuda.empty_cache()
        print(f"    [prefill-flops {hf_model_id}] routing forward through "
              f".thinker (freed talker + token2wav, "
              f"GPU mem now {torch.cuda.memory_allocated()/1e9:.1f}GB)",
              flush=True)

    print(f"    [prefill-flops {hf_model_id} @ {seconds}s] running one forward "
          f"(input keys: {sorted(inputs.keys())})", flush=True)
    _counter = FlopCounterMode(display=False)
    with _counter, torch.no_grad():
        _ = forward_target(**inputs)
    flops = _counter.get_total_flops()
    if flops <= 0:
        raise RuntimeError(
            f"FlopCounterMode returned {flops} for {hf_model_id} @ {seconds}s."
        )
    result = {
        "prefill_flops": int(flops),
        "n_total_params_raw": int(n_total_raw),  # sum(model.parameters()) - MoE inflated
        "n_total_params_active": int(n_total_active),  # MoE-contracted, .talker filtered
        "n_encoder_params": int(n_enc),
        "n_decoder_active_params": int(n_dec_active),
        "moe": pc["moe"],
        "moe_hparams": pc["moe_hparams"],
    }
    print(f"    [prefill-flops {hf_model_id} @ {seconds}s] "
          f"encoder_path={encoder_path}  "
          f"N_total_raw={n_total_raw/1e9:.2f}B  N_active={n_total_active/1e9:.2f}B  "
          f"N_enc={n_enc/1e9:.3f}B  N_dec_active={n_dec_active/1e9:.2f}B  "
          f"prefill_FLOPs={int(flops)/1e12:.2f}T",
          flush=True)
    _ENCODER_FLOPS_CACHE[key] = result
    del model, encoder; torch.cuda.empty_cache()
    return result


def find_text_decoder(model: torch.nn.Module, decoder_kind: str) -> torch.nn.Module | None:
    """Same idea, for CLAP-family decoders (BART, GPT-2)."""
    candidates = {
        "bart-base": ["bart_model", "bart", "decoder", "text_decoder", "language_model"],
        "gpt2":       ["gpt2", "gpt_model", "decoder", "text_decoder", "language_model"],
    }.get(decoder_kind, [])
    for name in candidates:
        m = getattr(model, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def count_active_params(module: torch.nn.Module) -> int:
    """Sum of all trainable + non-trainable params. Note: for MoE this is
    TOTAL, not active — the registry's n_active_B overrides when set."""
    return sum(p.numel() for p in module.parameters())


_PARAM_COUNT_CACHE: dict[str, dict[str, int]] = {}

def _find_moe_hyperparams(cfg) -> tuple[int | None, int | None]:
    """Walk config tree looking for MoE hyperparams. Returns
    (num_experts, num_experts_per_tok) or (None, None) for dense models.

    For multimodal MoE (Qwen3-Omni) the fields live several levels deep —
    thinker_config.text_config.num_experts = 128. We prefer the THINKER
    path since that's the main-LM invoked on every audio-in→text-out
    forward. Talker (TTS-only) has different num_experts_per_tok and is
    never invoked in our use case.
    """
    def probe(obj):
        n = getattr(obj, "num_experts", None) or getattr(obj, "n_routed_experts", None)
        k = getattr(obj, "num_experts_per_tok", None) or getattr(obj, "num_experts_per_token", None)
        return (n, k) if (n and k) else (None, None)

    # BFS with explicit priority: thinker/text/llm ranked ABOVE talker so
    # we don't accidentally read the talker's routing config.
    from collections import deque
    seen = set()
    queue = deque([cfg])
    PRIORITY_SUBS = ("text_config", "thinker_config", "llm_config", "language_config")
    LOW_PRIORITY_SUBS = ("talker_config",)
    while queue:
        c = queue.popleft()
        if id(c) in seen: continue
        seen.add(id(c))
        n, k = probe(c)
        if n and k:
            return n, k
        for sub in PRIORITY_SUBS:
            if hasattr(c, sub) and getattr(c, sub) is not None:
                queue.append(getattr(c, sub))
        for sub in LOW_PRIORITY_SUBS:
            if hasattr(c, sub) and getattr(c, sub) is not None:
                queue.append(getattr(c, sub))
    return None, None


def count_params_from_hf(hf_model_id: str) -> dict | None:
    """Load HF model on meta-tensor (no weights, no GPU) and return
    {"total": N, "active": N_active, "moe": bool, "moe_hparams": (n_exp, top_k) | None}.

    For dense models: active == total.
    For MoE: active = total - Σ (n_experts - top_k) × per_expert_params
    over all nn.ModuleList submodules whose length matches config.num_experts.
    Cached per hf_model_id (shared across e2e + captioner + text_llm calls).
    """
    if hf_model_id in _PARAM_COUNT_CACHE:
        return _PARAM_COUNT_CACHE[hf_model_id]
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel
    import torch.nn as nn
    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    # AutoModel misses conditional-generation subclasses. config.architectures[0]
    # names the exact HF class (e.g. "Qwen3OmniMoeForConditionalGeneration").
    # Some classes (Qwen2_5OmniModel) live in transformers.models.<mtype>.*
    # but aren't re-exported at top level — search there too.
    import importlib
    model_cls = None
    base_names = list(getattr(cfg, "architectures", None) or [])
    # Some configs list a base class (e.g. Qwen2_5OmniModel) that isn't
    # actually instantiable — try common suffix variants too.
    candidate_names: list[str] = []
    for n in base_names:
        candidate_names.append(n)
        if not n.endswith("ForConditionalGeneration"):
            candidate_names.append(n.rstrip("Model") + "ForConditionalGeneration")
        if not n.endswith("ForCausalLM"):
            candidate_names.append(n.rstrip("Model") + "ForCausalLM")
    model_type = getattr(cfg, "model_type", None)
    for arch_name in candidate_names:
        c = getattr(transformers, arch_name, None)
        if c is not None:
            model_cls = c; break
        if model_type:
            for sub in (f"transformers.models.{model_type}.modeling_{model_type}",
                        f"transformers.models.{model_type}"):
                try:
                    mod = importlib.import_module(sub)
                    c = getattr(mod, arch_name, None)
                    if c is not None:
                        model_cls = c; break
                except ImportError:
                    pass
            if model_cls is not None:
                break
    # Concrete HF classes (LlamaForCausalLM, Qwen3OmniMoeForConditionalGeneration, …)
    # take config in __init__; only AutoModel* has from_config.
    with init_empty_weights():
        if model_cls is not None:
            m = model_cls(cfg)
        else:
            m = AutoModel.from_config(cfg, trust_remote_code=True)
    # Filter out inactive submodules that don't fire on our forward path.
    # Qwen{2.5,3}-Omni wrap thinker + talker: the talker is TTS-only, not
    # invoked for audio-in→text-out inference. Whole-model .parameters()
    # counts talker experts too, inflating total ~5B and shared ~1.75B
    # for Qwen3-Omni. Skip params whose name contains ".talker.".
    SKIP_SUBSTR = (".talker.",)
    def _keep(pname: str) -> bool:
        return not any(s in pname for s in SKIP_SUBSTR)
    total = sum(p.numel() for pname, p in m.named_parameters() if _keep(pname))
    n_exp, top_k = _find_moe_hyperparams(cfg)
    active = total
    moe_source = ""
    if n_exp and top_k:
        # Two strategies: (a) walk ModuleList[num_experts] and take
        # per_expert = mod[0].parameters() — standard HF MoE layout;
        # (b) fused-expert layout (Qwen3-Omni packs experts into a
        # single tensor of shape [num_experts, ...]) — walk parameters
        # by name, treat any param whose name matches r"\.experts\."
        # or ".experts_" as MoE-scoped. Both strategies respect
        # SKIP_SUBSTR so talker experts don't inflate the moe count.
        inactive = 0
        for name, mod in m.named_modules():
            if any(s in name for s in SKIP_SUBSTR): continue
            if isinstance(mod, nn.ModuleList) and len(mod) == n_exp and len(mod) > 0:
                per_expert = sum(p.numel() for p in mod[0].parameters())
                inactive += (n_exp - top_k) * per_expert
                moe_source = "ModuleList[num_experts]"
        if inactive == 0:
            # Fused-expert fallback: identify by param name
            moe_params = 0
            for pname, p in m.named_parameters():
                if not _keep(pname): continue
                if ".experts." in pname or ".experts_" in pname or "moe_experts" in pname:
                    moe_params += p.numel()
            if moe_params > 0:
                inactive = moe_params * (n_exp - top_k) // n_exp
                moe_source = "fused-experts (name pattern)"
        active = total - inactive
    result = {
        "total": total, "active": active,
        "moe": bool(n_exp and top_k),
        "moe_hparams": (n_exp, top_k) if (n_exp and top_k) else None,
        "moe_source": moe_source if (n_exp and top_k) else None,
    }
    _PARAM_COUNT_CACHE[hf_model_id] = result
    tag = f"MoE {n_exp}×top-{top_k} via {moe_source or 'no-match'}" if result["moe"] else "dense"
    print(f"    [.parameters() meta-tensor {hf_model_id}] "
          f"total={total/1e9:.3f}B active={active/1e9:.3f}B ({tag})",
          flush=True)
    return result


def calflops_forward(module: torch.nn.Module, args) -> int:
    """Run calflops. Casts args to list — calflops does
    `args[i] = wrap(args[i])` internally, which raises TypeError on a tuple
    ("does not support item assignment"). No except: any failure must
    surface with its full traceback."""
    from calflops import calculate_flops
    flops, _macs, _params = calculate_flops(
        model=module, args=list(args),
        print_results=False, print_detailed=False, output_as_string=False,
    )
    return int(flops)


# --- Read predictions from desc-v4 ------------------------------------------

def sins_chunks_24h(cell_dir: Path, n_want: int = N_WIN_10MIN) -> list[dict]:
    """Load the first n_want chunk_*.json files (canonical first-24h)."""
    paths = sorted(cell_dir.glob("chunk_*.json"))[:n_want]
    return [json.loads(p.read_text()) for p in paths]


def sins_sub_chunk_captions(cascA_dir: Path, arm: str) -> list[str]:
    """For non-CLAP captioners: read the actual per-sub-chunk generated
    captions from the arm's descriptions.jsonl. Returns first 8640 captions.
    Missing jsonl or malformed lines raise — no silent skip."""
    jsonl = cascA_dir / f"{arm}.descriptions.jsonl"
    if not jsonl.exists():
        raise RuntimeError(
            f"sins_sub_chunk_captions[{arm}]: descriptions jsonl missing: {jsonl}"
        )
    caps = []
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cap = row.get("description") or row.get("caption") or ""
            if cap:
                caps.append(cap)
            if len(caps) >= N_SUB_10S:
                break
    return caps


# --- Per-family FLOPs computation -------------------------------------------

def compute_text_llm(key: str, cfg: dict) -> dict:
    """Text-only Stage-B: read cascB_{model}/cascA_af3-hf/sins/ chunk_*.json,
    tokenize the actual prompt + completion with the model's own tokenizer,
    sum, then apply 2N * total_tokens."""
    model = key.split(":", 1)[1]
    # FT arms set cell_override; base arms use the default cascB_{model}/cascA_af3-hf/sins.
    cell = RUNS / cfg["cell_override"] if cfg.get("cell_override") else RUNS / f"cascB_{model}" / "cascA_af3-hf" / "sins"
    if not cell.is_dir():
        raise RuntimeError(f"compute_text_llm[{model}]: cell dir missing: {cell}")

    tokenizer = load_tokenizer(cfg["tokenizer_id"])
    chunks = sins_chunks_24h(cell)
    if len(chunks) < N_WIN_10MIN:
        print(f"  [warn] {model}: only {len(chunks)}/{N_WIN_10MIN} chunks available", flush=True)

    sum_prompt = 0
    sum_completion = 0
    for c in chunks:
        prompt_text = c.get("prompt") or ""
        completion_text = c.get("raw_text") or ""
        sum_prompt += len(tokenizer.encode(prompt_text))
        sum_completion += len(tokenizer.encode(completion_text))

    n_active = cfg.get("n_active_B")
    n_active_source = "hardcoded"
    if n_active is None:
        # Measure via .parameters() on meta-tensor HF init. Skip only when
        # the model is genuinely closed-weights (hf_model_id=None → Gemini,
        # Luna) — in that case we still report token counts but skip FLOPs.
        hf_id = cfg.get("hf_model_id")
        if hf_id is None:
            return {
                "model": model, "family": "text_llm",
                "n_active_B": None, "n_note": cfg.get("n_note", ""),
                "prompt_toks_24h": sum_prompt, "completion_toks_24h": sum_completion,
                "total_toks_24h": sum_prompt + sum_completion,
                "language_model_pf_24h": None,
                "total_pf_24h": None,
                "note": "N undisclosed — token counts computed, FLOPs skipped",
            }
        pc = count_params_from_hf(hf_id)
        if pc is None:
            raise RuntimeError(f"count_params_from_hf failed for {hf_id}")
        n_active = pc["active"] / 1e9
        n_total_B = pc["total"] / 1e9
        n_active_source = (
            f".parameters() meta-tensor from {hf_id} "
            f"(MoE {pc['moe_hparams'][0]}×top-{pc['moe_hparams'][1]})"
            if pc["moe"] else
            f".parameters() meta-tensor from {hf_id} (dense)"
        )

    total_toks = sum_prompt + sum_completion
    flops = 2 * (n_active * 1e9) * total_toks
    return {
        "model": model, "family": "text_llm",
        "n_active_B": n_active, "n_total_B": n_total_B, "n_active_source": n_active_source,
        "n_note": cfg.get("n_note", ""),
        "tokenizer_id": cfg["tokenizer_id"],
        "prompt_toks_24h": sum_prompt, "completion_toks_24h": sum_completion,
        "total_toks_24h": total_toks,
        "audio_encoder_pf_24h": 0.0,
        "language_model_pf_24h": flops / 1e15,
        "total_pf_24h": flops / 1e15,
        "source_cell": str(cell.relative_to(RUNS)),
    }


def compute_e2e(key: str, cfg: dict) -> dict:
    """E2E audio LLM: audio_encoder FLOPs + LLM FLOPs (2N * text_prompt +
    audio_embed + completion), using actual prompt/completion from cell
    chunk_*.json. Encoder FLOPs are measured via direct HF hub load
    (WhisperModel etc.) — we no longer try to fish the encoder out of the
    adapter's internals (unreliable across adapter classes)."""
    model = key.split(":", 1)[1]
    # FT arms set cell_override; base arms use the default e2e_{model}/sins.
    cell = RUNS / cfg["cell_override"] if cfg.get("cell_override") else RUNS / f"e2e_{model}" / "sins"
    if not cell.is_dir():
        raise RuntimeError(f"compute_e2e[{model}]: cell dir missing: {cell}")

    chunks = sins_chunks_24h(cell)
    tokenizer = load_tokenizer(cfg["tokenizer_id"])

    # 1. Text prompt + completion tokens from actual predictions
    sum_prompt_text = 0
    sum_completion = 0
    for c in chunks:
        sum_prompt_text += len(tokenizer.encode(c.get("prompt") or ""))
        sum_completion += len(tokenizer.encode(c.get("raw_text") or ""))

    # 2. n_active: measured via .parameters() on a meta-tensor HF load
    #    (no weight download, no GPU memory). Registry can override with
    #    a fixed n_active_B if we ever need to pin a specific number.
    n_active = cfg.get("n_active_B")
    n_active_source = "hardcoded"
    n_total_B: float | None = n_active
    if n_active is None:
        pc = count_params_from_hf(cfg["hf_model_id"])
        if pc is None:
            raise RuntimeError(f"count_params_from_hf failed for {cfg['hf_model_id']}")
        n_active = pc["active"] / 1e9
        n_total_B = pc["total"] / 1e9
        n_active_source = (
            f".parameters() meta-tensor from {cfg['hf_model_id']} "
            f"(MoE {pc['moe_hparams'][0]}×top-{pc['moe_hparams'][1]})"
            if pc["moe"] else
            f".parameters() meta-tensor from {cfg['hf_model_id']} (dense)"
        )

    # 3. Encoder FLOPs — direct HF load, cached across models
    # Two-part FLOPs decomposition:
    #   (a) One full-model prefill FLOPs measurement — feeds the whole
    #       *ForConditionalGeneration one forward pass with the 10-min
    #       audio input and no generation. Captures encoder + connector +
    #       LLM-prefill in a single calflops call (correct for AF-family's
    #       30-s tiling AND for Qwen-Omni AuT's variable-length forward).
    #   (b) Autoregressive tail — 2 × N_decoder × completion_tokens, since
    #       each generated output token adds one incremental decoder-only
    #       forward beyond what (a) already covered.
    m = measure_family_full_prefill_flops(cfg["hf_model_id"], seconds=600)
    prefill_flops_per_chunk = m["prefill_flops"]
    n_dec_active = m["n_decoder_active_params"]  # LLM + connector (audio encoder excluded)

    prefill_pf_24h = prefill_flops_per_chunk * N_WIN_10MIN / 1e15
    autoreg_flops_24h = 2 * n_dec_active * sum_completion
    autoreg_pf_24h = autoreg_flops_24h / 1e15

    return {
        "model": model, "family": "e2e",
        "n_active_B": n_active, "n_total_B": n_total_B,
        "n_decoder_B": n_dec_active / 1e9,
        "n_encoder_B": m["n_encoder_params"] / 1e9,
        "n_active_source": n_active_source, "n_note": cfg.get("n_note", ""),
        "tokenizer_id": cfg["tokenizer_id"],
        "text_prompt_toks_24h": sum_prompt_text,
        "completion_toks_24h": sum_completion,
        "prefill_flops_per_chunk": prefill_flops_per_chunk,
        "prefill_pf_24h": prefill_pf_24h,
        "autoregressive_pf_24h": autoreg_pf_24h,
        "total_pf_24h": prefill_pf_24h + autoreg_pf_24h,
        "source_cell": str(cell.relative_to(RUNS)),
    }


def compute_captioner(key: str, cfg: dict) -> dict:
    """Non-CLAP captioner: same shape as e2e but on 10-s sub-chunks. Reads
    real captions from cascA_{arm}/sins/{arm}.descriptions.jsonl."""
    model = key.split(":", 1)[1]
    arm_dir = RUNS / f"cascA_{model}" / "sins"
    if not arm_dir.is_dir():
        raise RuntimeError(f"compute_captioner[{model}]: cascA dir missing: {arm_dir}")

    tokenizer = load_tokenizer(cfg["tokenizer_id"])
    captions = sins_sub_chunk_captions(arm_dir, model)[:N_SUB_10S]
    if len(captions) < N_SUB_10S:
        print(f"  [warn] {model}: only {len(captions)}/{N_SUB_10S} captions", flush=True)

    # Completion tokens: tokenize each real caption
    sum_completion = sum(len(tokenizer.encode(c)) for c in captions)
    # Text-prompt tokens per sub-chunk: near-zero for pure captioners (usually
    # "Describe this audio." ≈ 5). Sample the actual prompt if adapter stashed
    # it; otherwise assume 5 * 8640.
    text_prompt_toks_per_sub = 5
    sum_prompt_text = text_prompt_toks_per_sub * N_SUB_10S

    # n_active: measured via .parameters() on meta-tensor HF load. Cache
    # hits across e2e + captioner registries (same hf_model_id).
    n_active = cfg.get("n_active_B")
    n_active_source = "hardcoded"
    n_total_B: float | None = n_active
    if n_active is None:
        pc = count_params_from_hf(cfg["hf_model_id"])
        if pc is None:
            raise RuntimeError(f"count_params_from_hf failed for {cfg['hf_model_id']}")
        n_active = pc["active"] / 1e9
        n_total_B = pc["total"] / 1e9
        n_active_source = (
            f".parameters() meta-tensor from {cfg['hf_model_id']} "
            f"(MoE {pc['moe_hparams'][0]}×top-{pc['moe_hparams'][1]})"
            if pc["moe"] else
            f".parameters() meta-tensor from {cfg['hf_model_id']} (dense)"
        )

    # Two-part FLOPs decomposition (same as compute_e2e):
    #   prefill: one full-model forward per 10-s sub-chunk (calflops)
    #   autoregressive: 2 × N_decoder × completion_tokens
    m = measure_family_full_prefill_flops(cfg["hf_model_id"], seconds=10)
    prefill_flops_per_sub = m["prefill_flops"]
    n_dec_active = m["n_decoder_active_params"]

    prefill_pf_24h = prefill_flops_per_sub * N_SUB_10S / 1e15
    autoreg_flops_24h = 2 * n_dec_active * sum_completion
    autoreg_pf_24h = autoreg_flops_24h / 1e15

    return {
        "model": model, "family": "captioner",
        "n_active_B": n_active, "n_total_B": n_total_B,
        "n_decoder_B": n_dec_active / 1e9,
        "n_encoder_B": m["n_encoder_params"] / 1e9,
        "n_active_source": n_active_source, "n_note": cfg.get("n_note", ""),
        "tokenizer_id": cfg["tokenizer_id"],
        "text_prompt_toks_24h": sum_prompt_text,
        "completion_toks_24h": sum_completion,
        "prefill_flops_per_sub_chunk": prefill_flops_per_sub,
        "prefill_pf_24h": prefill_pf_24h,
        "autoregressive_pf_24h": autoreg_pf_24h,
        "total_pf_24h": prefill_pf_24h + autoreg_pf_24h,
        "source_cell": str(arm_dir.relative_to(RUNS)),
    }


class _ClapModelProxy(torch.nn.Module):
    """Wraps LAION-CLAP's clap_model so calflops can measure via forward().
    The real .forward returns None; the encoder path is
    .get_audio_embedding_from_data(x, use_tensor=True)."""

    def __init__(self, clap_model):
        super().__init__()
        self.clap_model = clap_model

    def forward(self, x):
        return self.clap_model.get_audio_embedding_from_data(x, use_tensor=True)


def _measure_encoder_flops(name: str, module: torch.nn.Module,
                           shape: tuple, cuda: bool = True) -> int:
    """Run calflops on `module` with input tensor of the given `shape`.
    Single known-good shape per call — no candidate-list guessing.
    HARD FAILS on any calflops error so a stale shape is caught
    immediately."""
    if cuda and torch.cuda.is_available():
        module = module.cuda().eval()
    x = torch.zeros(*shape, dtype=torch.float32)
    if cuda and torch.cuda.is_available():
        x = x.cuda()
    f = calflops_forward(module, (x,))
    if f <= 0:
        raise RuntimeError(
            f"[clap-enc {name}] calflops returned {f} for shape={shape}."
        )
    print(f"    [clap-enc {name}] calflops shape={shape} → {f/1e9:.3f} GFLOPs",
          flush=True)
    return f


def compute_clap(key: str, cfg: dict) -> dict:
    """CLAP: load the ACTUAL inference adapter (EnClapLargeAdapter,
    MSClapCapAdapter) — same load path as production inference — and split
    params into encoder vs decoder from its internals. FLOPs analytical:
    encoder = 2N × seq_len (one forward per 10-s sub-chunk),
    decoder = 2N × completion_toks (autoregressive over real captions).

    Layout of the loaded adapters (verified by probe):
      EnClap.self._enclap has:
        .encodec       (EncodecModel,                      15M) — codec front end
        .clap_model    (CLAP_Module w/ audio+text branches, 158M) — LAION-CLAP
        .model         (EnClapBartForConditionalGeneration, 441M) — BART decoder
        → encoder = encodec + clap_model; decoder = model
      MSClap.self._model has:
        .clapcap       (ClapCaptionModel,                   227M) — encoder+GPT-2 combined
        → we can't cleanly split; report combined N and note it in the record.
    """
    import torch.nn as nn
    model = key.split(":", 1)[1]
    arm_dir = RUNS / f"cascA_{model}" / "sins"
    tokenizer = load_tokenizer(cfg["tokenizer_id"])  # BART / GPT-2
    captions = sins_sub_chunk_captions(arm_dir, model)[:N_SUB_10S]
    sum_completion = sum(len(tokenizer.encode(c)) for c in captions)

    # No try/except — adapter load failures must crash so the traceback is
    # visible. Silent fallback to `n_enc_params=None, flops=0` produced
    # wrong-but-plausible numbers.
    if model == "enclap":
        from long_audio.inference.models.enclap import EnClapLargeAdapter
        a = EnClapLargeAdapter()
        a.load()
        enc = a._enclap
        n_enc_params = (
            sum(p.numel() for p in enc.encodec.parameters())
            + sum(p.numel() for p in enc.clap_model.parameters())
        )
        n_dec_params = sum(p.numel() for p in enc.model.parameters())
        load_note = "EnClapLargeAdapter().load(); enc=encodec+clap_model, dec=model"
        # Known-good shapes from prior probes:
        #   encodec:    (1, 1, 240000) = 24 kHz mono × 10 s
        #   clap_model: (1, 480000)     = 48 kHz mono × 10 s (LAION-CLAP)
        f_enc = _measure_encoder_flops(
            "enclap.encodec", enc.encodec, shape=(1, 1, 240000),
        )
        f_clap = _measure_encoder_flops(
            "enclap.clap_model", _ClapModelProxy(enc.clap_model),
            shape=(1, 480000),
        )
        flops_per_enc_fwd = f_enc + f_clap
        enc_source = "encodec@(1,1,240000) + clap_model@(1,480000)"
        del a, enc
    elif model == "msclap-cap":
        from long_audio.inference.models.msclap import (
            MSClapCapAdapter, _install_torchaudio_soundfile_shim,
        )
        _install_torchaudio_soundfile_shim()
        a = MSClapCapAdapter()
        a.load()
        clapcap = a._model.clapcap
        # MSClap ClapCaptionModel structure:
        #   .clap         (msclap.models.clap.AudioEncoder)   → audio encoder
        #   .clap_project (msclap.models.mapper.TransformerMapper) → connector
        #   .gpt          (transformers ... GPT2LMHeadModel)  → text decoder
        n_enc_params = (
            sum(p.numel() for p in clapcap.clap.parameters())
            + sum(p.numel() for p in clapcap.clap_project.parameters())
        )
        n_dec_params = sum(p.numel() for p in clapcap.gpt.parameters())
        load_note = "MSClapCapAdapter().load(); enc=clapcap.clap+.clap_project, dec=clapcap.gpt"
        # Known-good shapes from prior probes:
        #   clap:         (1, 441000) = 44.1 kHz mono × 10 s (MS-CLAP waveform)
        #   clap_project: (1, 1024)   = CLAP embedding dim
        f_clap = _measure_encoder_flops(
            "msclap.clap", clapcap.clap, shape=(1, 441000),
        )
        f_proj = _measure_encoder_flops(
            "msclap.clap_project", clapcap.clap_project, shape=(1, 1024),
        )
        flops_per_enc_fwd = f_clap + f_proj
        enc_source = "clap@(1,441000) + clap_project@(1,1024)"
        del a, clapcap
    else:
        raise RuntimeError(f"compute_clap: unknown model {model!r}")

    flops_enc_24h = flops_per_enc_fwd * N_SUB_10S

    # Decoder FLOPs: 2N × (audio-derived + completion) tokens.
    # Audio-derived per-sub-chunk:
    #   MSClap → clapcap.prefix_length (probed = 40): clap_project outputs
    #     40 prefix tokens prepended to GPT-2 input; GPT-2 processes them.
    #   EnClap → 750: encodec at 24 kHz outputs codes at 75 Hz (24000/320
    #     hop) → 750 codes per 10-s sub-chunk feed BART encoder. BART decoder
    #     cross-attends over these encoder hidden states. Adding as decoder
    #     T under-approximates cross-attention (real cost is
    #     2·d_dec·T_enc·T_dec per layer) but captures the fact that the
    #     decoder does spend compute on the audio-derived context.
    audio_prefix_per_sub = {"enclap": 750, "msclap-cap": 40}.get(model, 0)
    audio_prefix_toks_24h = audio_prefix_per_sub * N_SUB_10S
    flops_dec_24h = 2 * (n_dec_params or 0) * (sum_completion + audio_prefix_toks_24h)

    return {
        "model": model, "family": "clap",
        "n_enc_params": n_enc_params, "n_dec_params": n_dec_params,
        "n_params_source": load_note,
        "encoder_flops_source": enc_source,
        "tokenizer_id": cfg["tokenizer_id"],
        "decoder_kind": cfg["decoder_kind"],
        "avg_caption_toks": sum_completion / max(1, len(captions)),
        "n_captions": len(captions),
        "audio_prefix_toks_per_sub": audio_prefix_per_sub,
        "audio_prefix_toks_24h": audio_prefix_toks_24h,
        "completion_toks_24h": sum_completion,
        "flops_per_encoder_forward": flops_per_enc_fwd,
        "audio_encoder_pf_24h": flops_enc_24h / 1e15,
        "language_model_pf_24h": flops_dec_24h / 1e15,
        "total_pf_24h": (flops_enc_24h + flops_dec_24h) / 1e15,
        "source_cell": str(arm_dir.relative_to(RUNS)),
    }


# --- Main -------------------------------------------------------------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Measure per-family FLOPs for the desc-v4 run tree.")
    p.add_argument("--runs-root", default=os.environ.get("RUNS_DIR"),
                   help="Root of the desc-v4 runs tree (contains cascB_*/, e2e_*/, cascA_*/). "
                        "Falls back to $RUNS_DIR.")
    p.add_argument("--out", default=os.environ.get("FLOPS_OUT"),
                   help="Where to write the flops JSON. Falls back to $FLOPS_OUT.")
    args = p.parse_args()
    if not args.runs_root:
        p.error("--runs-root is required (or set $RUNS_DIR).")
    if not args.out:
        p.error("--out is required (or set $FLOPS_OUT).")

    global RUNS, OUT_PATH
    RUNS = Path(args.runs_root)
    OUT_PATH = Path(args.out)

    # FAMILIES env: comma-list filter (default all). Useful for running
    # CLAP in las-enclap venv (has older transformers 4.29) while running
    # everything else in las-all (transformers 5.9).
    families_env = os.environ.get("FAMILIES", "text_llm,e2e,cap,clap")
    families_filter = {f.strip() for f in families_env.split(",") if f.strip()}
    print(f"FAMILIES filter: {sorted(families_filter)}", flush=True)

    dispatch: dict[str, Callable[[str, dict], dict]] = {
        "text_llm": compute_text_llm,
        "e2e":      compute_e2e,
        "cap":      compute_captioner,
        "clap":     compute_clap,
    }
    results = {"text_llm": [], "e2e": [], "cap": [], "clap": []}
    t_start = time.time()
    for key, cfg in REGISTRY.items():
        family = key.split(":", 1)[0]
        if family not in families_filter:
            continue
        print(f"\n=== {key} ===", flush=True)
        t0 = time.time()
        # No dispatch-level try/except — any exception must propagate so we
        # see the full traceback. Prior "safeguard" turned real errors into
        # {"error": ...} JSON rows that let the pipeline continue with
        # missing / wrong FLOPs and made debugging impossible.
        out = dispatch[family](key, cfg)
        out["_wall_s"] = round(time.time() - t0, 1)
        results[family].append(out)
        print(f"  → {out.get('total_pf_24h', out.get('error'))}", flush=True)

    results["_meta"] = {
        "run_ts_source": str(RUNS),
        "canonical_24h_windows_10min": N_WIN_10MIN,
        "canonical_24h_subchunks_10s": N_SUB_10S,
        "wall_s": round(time.time() - t_start, 1),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
