# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""FLOPs measurement for models where we can't use the 2N × tokens formula:

  * AF3-hf: probe audio encoder to count EXACT audio-embed
    tokens per 10-sec chunk (their run metadata doesn't record it).
  * EnCLAP and MSCLAP-cap: full model forward via calflops for one
    10-sec audio input; scale to 24h.

Requires las-enclap venv (has enclap + msclap-cap installed) + a GPU
allocation (all model loads use CUDA).

Output: /tmp/flops_direct_24h.json with per-model FLOPs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- shared: 10s dummy audio (silence at 16kHz — enough to measure
# structural encoder cost; audio content doesn't change FLOPs)
SR = 16_000
DUMMY_10S = np.zeros(SR * 10, dtype=np.float32)


def probe_af(name: str, adapter_cls) -> dict:
    """Load AF adapter, run one 10s inference, capture audio_tokens from
    the adapter's own logging or by wrapping the encoder call.
    """
    print(f"\n=== probing {name} ===")
    a = adapter_cls()
    a.load()
    # We want the audio-token count; simplest is to run generate() on a
    # tiny audio and read the adapter's internal token accounting.
    t0 = time.time()
    out = a.generate(
        audio=DUMMY_10S, sample_rate=SR,
        prompt="Describe this audio.", decoder="freeform",
        max_new_tokens=5, temperature=0.0,
    )
    dur = time.time() - t0
    print(f"  {name}: {dur:.1f}s to run 1 chunk. metadata={out.metadata}")
    n_audio = out.metadata.get("prompt_tokens") or 0
    if n_audio == 0:
        # Fall back to counting via adapter internals — model_id-specific
        # hook. For now report absence.
        print(f"  {name}: prompt_tokens not exposed by adapter; falling back")
        return {"model": name, "audio_tokens_per_10s": None,
                "note": "adapter didn't expose prompt_tokens"}
    # AF prompt token = text prompt (few tokens) + audio embed. Rough
    # subtraction to isolate audio (text prompt "Describe this audio."
    # is ~5 tokens).
    audio_toks = n_audio - 5
    print(f"  {name}: ~audio_tokens_per_10s = {audio_toks}")
    return {"model": name, "audio_tokens_per_10s": audio_toks,
            "total_prompt_tokens": n_audio,
            "completion_tokens": out.metadata.get("completion_tokens")}


def profile_clap(name: str, adapter_cls) -> dict:
    """calflops profile of a CLAP-based captioner. Full 10s forward pass.

    The adapter's model has: audio encoder (CLAP) + text decoder (BART or GPT-2).
    calflops treats the full forward as one op — includes both.
    """
    from calflops import calculate_flops

    print(f"\n=== profiling {name} (calflops) ===")
    a = adapter_cls()
    a.load()
    # Extract the underlying nn.Module from the adapter. Each adapter
    # exposes a different attribute (._model, .model, .bart_model, ...).
    # Try known attributes.
    model = None
    for attr in ("_model", "model", "bart_model", "captioner"):
        m = getattr(a, attr, None)
        if isinstance(m, torch.nn.Module):
            model = m
            break
    if model is None:
        return {"model": name, "flops_pf_10s": None,
                "note": "could not extract nn.Module from adapter"}
    # calflops needs input shape or example inputs; audio adapters take
    # raw audio tensor. Try with a torch tensor input.
    audio_t = torch.from_numpy(DUMMY_10S).unsqueeze(0).cuda()
    try:
        flops, macs, params = calculate_flops(
            model=model, args=(audio_t,),
            print_results=False, print_detailed=False, output_as_string=False,
        )
        # calflops returns FLOPs as a float. Convert to petaFLOPs per chunk.
        flops_pf_chunk = flops / 1e15
        # For 24h of SINS at 10-sec = 8640 chunks
        flops_pf_24h = flops_pf_chunk * 8640
        print(f"  {name}: {flops:.2e} FLOPs/10s ({flops_pf_chunk:.3e} PF/chunk) "
              f"× 8640 chunks = {flops_pf_24h:.2f} PF/24h  (params={params})")
        return {
            "model": name, "params": params,
            "flops_per_10s_chunk": flops,
            "flops_pf_per_chunk": flops_pf_chunk,
            "flops_pf_24h": flops_pf_24h,
        }
    except Exception as e:
        return {"model": name, "flops_pf_10s": None,
                "note": f"calflops failed: {type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    from long_audio.inference.models.audio_flamingo3_hf import AudioFlamingo3HFAdapter
    from long_audio.inference.models.enclap import EnClapLargeAdapter
    from long_audio.inference.models.msclap import MSClapCapAdapter

    results = {"af_probe": [], "clap_flops": []}
    for name, cls in [("af3-hf", AudioFlamingo3HFAdapter)]:
        try:
            results["af_probe"].append(probe_af(name, cls))
        except Exception as e:
            results["af_probe"].append({"model": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    for name, cls in [("enclap", EnClapLargeAdapter),
                      ("msclap-cap", MSClapCapAdapter)]:
        try:
            results["clap_flops"].append(profile_clap(name, cls))
        except Exception as e:
            results["clap_flops"].append({"model": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    out_path = Path("/tmp/flops_direct_24h.json")
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
