"""Merge a PEFT LoRA adapter into its base model weights and save the result
as a plain HF checkpoint that vLLM can serve without a LoRARequest.

Motivating case: vLLM 0.21.0's multimodal-LoRA path silently drops
``language_model.*`` LoRA entries for Gemma4ForConditionalGeneration
(confirmed via A/B test — WITH-LoRARequest and WITHOUT-LoRARequest outputs
are byte-identical). The workaround is to serve a pre-merged checkpoint.

Also works for dense text bases (qwen/llama/olmo/k2) as a generic offline
merge, though those don't strictly need it — vLLM serves their LoRA adapters
via LoRARequest without issue.

Usage:
    python scripts/merge_lora.py --adapter-dir <path> [--output-dir <path>]

Convention: output-dir defaults to ``<adapter_dir>/merged/`` so
Gemma4_31BTextAdapter (see long_audio/inference/models/vllm_text.py) can
auto-resolve it without an extra config knob.
"""
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True,
                    help="PEFT adapter dir (has adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the merged model. Default: <adapter_dir>/merged/")
    ap.add_argument("--base-model-id", default=None,
                    help="HF id of the base model. Default: read from adapter_config.json's base_model_name_or_path")
    args = ap.parse_args()

    output_dir = args.output_dir or os.path.join(args.adapter_dir, "merged")

    cfg_path = os.path.join(args.adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    base_id = args.base_model_id or cfg["base_model_name_or_path"]

    print(f"[merge_lora] adapter: {args.adapter_dir}", flush=True)
    print(f"[merge_lora] base:    {base_id}", flush=True)
    print(f"[merge_lora] output:  {output_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    print("[merge_lora] loading base model (bfloat16, device_map=auto)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[merge_lora]   loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("[merge_lora] loading adapter...", flush=True)
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    print(f"[merge_lora]   loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("[merge_lora] merging LoRA into base (merge_and_unload)...", flush=True)
    merged = peft_model.merge_and_unload()
    print(f"[merge_lora]   merged in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print("[merge_lora] saving merged model...", flush=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    print(f"[merge_lora]   saved in {time.time()-t0:.1f}s", flush=True)

    # Prefer AutoProcessor (multimodal): it wraps the tokenizer + image/audio
    # preprocessors so vLLM can find the processor_config.json /
    # preprocessor_config.json it needs to construct a feature extractor for
    # multimodal architectures like Gemma4ForConditionalGeneration. Fall back
    # to AutoTokenizer for dense text bases that don't have a processor.
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
        proc.save_pretrained(output_dir)
        print("[merge_lora] saved AutoProcessor (tokenizer + preprocessor)", flush=True)
    except Exception as e:
        print(f"[merge_lora] AutoProcessor unavailable ({type(e).__name__}: {e}); "
              f"falling back to AutoTokenizer", flush=True)
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        tok.save_pretrained(output_dir)

    # Copy the adapter's chat_template.jinja (matches training-time template).
    src_ct = os.path.join(args.adapter_dir, "chat_template.jinja")
    if os.path.exists(src_ct):
        dst_ct = os.path.join(output_dir, "chat_template.jinja")
        with open(src_ct) as f:
            ct = f.read()
        with open(dst_ct, "w") as f:
            f.write(ct)
        print(f"[merge_lora] copied chat_template.jinja ({len(ct)} bytes)", flush=True)

    print(f"[merge_lora] DONE — merged model at {output_dir}", flush=True)


if __name__ == "__main__":
    main()
