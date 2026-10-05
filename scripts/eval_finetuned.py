"""Standalone eval for a fine-tuned Qwen3-Omni LoRA checkpoint on Ego4D.

Loads the bf16 base + a saved LoRA adapter + projector (from ``train_e2e.py``'s
output dir), runs generation over an Ego4D split through the shared inference +
eval path, and writes an ``eval_segmentation.json`` in the SAME format as
``scripts/eval_segmentation.py --dataset ego4d`` — so fine-tuned numbers sit next to the
frozen baselines directly.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_finetuned.py \\
        --adapter-dir runs/sft/e2e_qwen3-omni-<TS> \\
        --split val --chunk-min 10 --max-videos 50 \\
        --output runs/sft/e2e_qwen3-omni-<TS>/eval_val.json

Run as a single process (naive model parallel via ``device_map="auto"``); do not
launch under ``accelerate`` / ``torchrun``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.eval._json import NumpyEncoder as _NumpyEncoder
from long_audio.eval.segmentation import format_headline, format_pred_missing_line
from long_audio.training.evaluate import run_ego4d_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adapter-dir", required=True,
                   help="Dir with the LoRA adapter + projector.pt (train_e2e output).")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Base HF repo/path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--split", default="val", help="Ego4D split (val/test).")
    p.add_argument("--chunk-min", type=float, default=10.0)
    p.add_argument("--time-unit", choices=["second", "minute"], default="minute")
    p.add_argument("--no-description", action="store_true")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Cap #videos (smoke / quick eval).")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    # Eval-set filters (default to the canonical Ego4D eval set).
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--min-duration-s", type=float, default=600.0)
    p.add_argument("--pass-id", default="1")
    p.add_argument("--single-pass", action="store_true",
                   help="Don't require both annotator passes.")
    p.add_argument("--no-merge", action="store_true",
                   help="Keep LoRA un-merged (slower generation).")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--output", default=None,
                   help="Output JSON (default: <adapter-dir>/eval_<split>.json).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from long_audio.training.infer import Qwen3OmniHFAdapter, load_finetuned_model
    from long_audio.training.model import load_processor

    print(f"[eval_finetuned] loading {args.model_id} + adapter {args.adapter_dir} …", flush=True)
    top = load_finetuned_model(
        args.model_id,
        args.adapter_dir,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        merge=not args.no_merge,
    )
    processor = load_processor(args.model_id)
    adapter = Qwen3OmniHFAdapter(top, processor, name=f"ft:{Path(args.adapter_dir).name}")

    agg = run_ego4d_eval(
        adapter,
        args.manifest,
        args.split,
        chunk_minutes=args.chunk_min,
        time_unit=args.time_unit,
        with_description=not args.no_description,
        max_videos=args.max_videos,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_name=adapter.name,
        min_events=args.min_events,
        min_duration_s=args.min_duration_s,
        require_both_passes=not args.single_pass,
        pass_id=args.pass_id,
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path(args.adapter_dir) / f"eval_{args.split}.json"
    )
    out_path.write_text(json.dumps(agg, indent=2, cls=_NumpyEncoder))
    print(f"[eval_finetuned] Wrote {out_path}", flush=True)
    print(f"  n_videos={agg['n_videos']}  n_data_points={agg['n_data_points']}", flush=True)
    print(format_headline(agg), flush=True)
    print(format_pred_missing_line(agg["pred_missing"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
