# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Main inference entrypoint — chunk runner + model adapter + dataset selection.

Usage:
    # SINS smoke (one chunk)
    python scripts/run_e2e.py qwen3-omni --dataset sins --smoke

    # Full SINS run
    python scripts/run_e2e.py qwen3-omni --dataset sins --full

    # Ego4D smoke on the first 3 videos passing the dataset filter
    python scripts/run_e2e.py qwen3-omni --dataset ego4d --smoke --max-videos 3

    # Full Ego4D run
    python scripts/run_e2e.py qwen3-omni --dataset ego4d --full

Run outputs go to:
  SINS:  <repo>/runs/<UTC>/<model_name>/{chunk_*.json, summary.json}
  Ego4D: <repo>/runs/<UTC>/<model_name>/ego4d/<uid>/{chunk_*.json, summary.json}

The <repo>/runs/ dir is typically a symlink to actual storage (see README).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.inference.chunk_runner import run_inference

REPO_ROOT = Path(__file__).resolve().parent.parent
SINS_MONO_FLAC = REPO_ROOT / "datasets" / "SINS" / "mono.flac"
EGO4D_MANIFEST = REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
EGOLIFE_MANIFEST = REPO_ROOT / "datasets" / "egolife" / "manifest.json"
RUNS_DIR = REPO_ROOT / "runs"


MODELS = {
    "qwen2.5-omni": ("long_audio.inference.models.qwen2_5_omni",
                     "Qwen2_5OmniVLLMAdapter"),
    "qwen3-omni":   ("long_audio.inference.models.qwen3_omni",
                     "Qwen3OmniVLLMAdapter"),
    "af3-hf":       ("long_audio.inference.models.audio_flamingo3_hf",
                     "AudioFlamingo3HFAdapter"),
    "gemini":       ("long_audio.inference.models.gemini", "GeminiAdapter"),
    "fake":         ("long_audio.inference.models.fake", "FakeAdapter"),
}


# Per-dataset label/hint resolution. Keep this thin — the heavy lifting
# (label tuples, hint dicts, label-set sanity) lives with the dataset.
def _dataset_config(name: str) -> dict:
    if name == "sins":
        from long_audio.datasets.sins.schema import SINS_LABEL_HINTS, SINS_LABELS
        return {
            "labels": SINS_LABELS,
            "label_hints": SINS_LABEL_HINTS,
            "dataset_name": "SINS",
            "t0_iso": "2017-01-30T04:38:36.435Z",
        }
    if name == "ego4d":
        from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
        return {
            "labels": ATUS_LABELS,
            "label_hints": ATUS_HINTS,
            "dataset_name": "Ego4D",
            "t0_iso": "1970-01-01T00:00:00Z",  # per-video timeline starts at 0
        }
    if name == "egolife":
        # EgoLife re-exports Ego4D's ATUS-6 taxonomy verbatim for
        # cross-dataset comparability. Bump in lockstep with Ego4D.
        from long_audio.datasets.egolife.schema import EGOLIFE_HINTS, EGOLIFE_LABELS
        return {
            "labels": EGOLIFE_LABELS,
            "label_hints": EGOLIFE_HINTS,
            "dataset_name": "EgoLife",
            "t0_iso": "1970-01-01T00:00:00Z",  # per-session timeline starts at 0
        }
    raise ValueError(f"Unknown dataset: {name!r}")


# Adapters that accept ``max_num_seqs`` (the vLLM max-concurrent-sequences
# knob). Used by ``load_adapter`` to reject the flag loudly when paired
# with a non-vLLM adapter, instead of silently dropping it.
_VLLM_ADAPTERS = {"qwen2.5-omni", "qwen3-omni"}


# Adapters that accept the Gemini `thinking_budget` int kwarg (native
# reasoning-token cap on the google-genai SDK).
_GEMINI_THINKING_ADAPTERS = {"gemini"}
# Adapters that accept `enable_thinking` + `thinking_budget_enforce`
# (s1-style kwargs on ThinkingMixin-using adapters).
_VLLM_THINKING_ADAPTERS = {"qwen3-omni"}


def load_adapter(
    name: str,
    model_id: str | None = None,
    max_num_seqs: int | None = None,
    thinking_budget: int = 0,
    enable_thinking: bool = False,
    thinking_budget_enforce: int = 0,
):
    """Instantiate the adapter with explicit per-flag routing.

    Adapter-specific tuning flags (``max_num_seqs``, ``thinking_budget``,
    ``enable_thinking``, ``thinking_budget_enforce``) are routed only
    to adapters whose ``__init__`` accepts them. Mismatches error out
    loudly rather than getting silently swallowed.
    """
    module_path, cls_name = MODELS[name]
    mod = __import__(module_path, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    kwargs: dict = {}
    if model_id:
        kwargs["model_id"] = model_id
    if max_num_seqs is not None:
        if name not in _VLLM_ADAPTERS:
            raise SystemExit(
                f"--max-num-seqs={max_num_seqs} is not supported by adapter "
                f"{name!r}. Only vLLM adapters accept it: "
                f"{sorted(_VLLM_ADAPTERS)}."
            )
        kwargs["max_num_seqs"] = max_num_seqs
    if thinking_budget > 0:
        if name not in _GEMINI_THINKING_ADAPTERS:
            raise SystemExit(
                f"--thinking-budget={thinking_budget} is only supported by "
                f"Gemini adapters: {sorted(_GEMINI_THINKING_ADAPTERS)}. "
                f"Got adapter={name!r}. For Qwen/Gemma s1-style budget, "
                "use --thinking-enable + --thinking-budget-enforce."
            )
        kwargs["thinking_budget"] = thinking_budget
    if enable_thinking or thinking_budget_enforce > 0:
        if name not in _VLLM_THINKING_ADAPTERS:
            raise SystemExit(
                f"--thinking-enable / --thinking-budget-enforce not supported "
                f"by adapter {name!r}. vLLM thinking-aware adapters: "
                f"{sorted(_VLLM_THINKING_ADAPTERS)}. For Gemini, use "
                "--thinking-budget."
            )
        kwargs["enable_thinking"] = enable_thinking
        if thinking_budget_enforce > 0:
            kwargs["thinking_budget_enforce"] = thinking_budget_enforce
    return cls(**kwargs)


def thinking_label_suffix(
    thinking_budget: int, enable_thinking: bool, thinking_budget_enforce: int,
) -> str:
    """Compute the model_label suffix from the three thinking flags.

    Semantics (matches launcher's per-cell dir naming):
      * thinking_budget > 0 (Gemini native)      -> f"_thinking{N}"
      * enable_thinking + enforce > 0 (s1)       -> f"_thinking{N}"
      * enable_thinking + enforce == 0 (boolean) -> "_thinking"
      * everything else                          -> ""
    """
    if thinking_budget > 0:
        return f"_thinking{thinking_budget}"
    if enable_thinking:
        if thinking_budget_enforce > 0:
            return f"_thinking{thinking_budget_enforce}"
        return "_thinking"
    return ""


def _load_dataset(args):
    """Build the dataset from CLI args."""
    if args.dataset == "sins":
        from long_audio.datasets.sins import SINSDataset
        ds = SINSDataset(data_dir=SINS_MONO_FLAC.parent)
        if args.audio:
            # Thin override: keep the canonical manifest (annotations, duration)
            # but swap the audio path. The override file SHOULD have the same
            # duration as the manifest; if it doesn't, audio_window_s will
            # clamp inference to the manifest's duration.
            ds.manifest["audio_path"] = str(args.audio)
        return ds
    if args.dataset == "ego4d":
        from long_audio.datasets.ego4d import Ego4DDataset
        manifest_path = Path(args.manifest) if args.manifest else EGO4D_MANIFEST
        if args.eval_set:
            return Ego4DDataset.eval_set(manifest_path, split=args.split)
        return Ego4DDataset(
            manifest_path=manifest_path,
            min_events=args.min_events,
            min_duration_s=args.min_duration_s,
            split=args.split,
        )
    if args.dataset == "egolife":
        from long_audio.datasets.egolife.dataset import EgoLifeDataset
        manifest_path = Path(args.manifest) if args.manifest else EGOLIFE_MANIFEST
        if args.eval_set:
            # EgoLife has a single annotator pass, no train/val/test split
            # (per dataset shape). ``eval_set`` here just applies the
            # canonical ``min_events`` / ``min_duration_s`` filter.
            return EgoLifeDataset.eval_set(manifest_path)
        return EgoLifeDataset(
            manifest_path=manifest_path,
            min_events=args.min_events,
            min_duration_s=args.min_duration_s,
        )
    raise ValueError(f"Unknown dataset {args.dataset!r}")


def _run(args, adapter, dcfg, utc) -> int:
    """Single inference loop over ``dataset.iter_items()``.

    SINS yields 1 item under ``./``; Ego4D yields N items under
    ``ego4d/<uid>/``. ``run_index.json`` is only written for multi-item
    datasets, at the longest common-prefix dir of the per-item subdirs.
    """
    import json

    dataset = _load_dataset(args)
    items = list(dataset.iter_items())
    n_total = len(items)
    if args.max_videos is not None:
        items = items[:args.max_videos]
    n_items = len(items)
    if n_items == 0:
        raise RuntimeError(f"Dataset {args.dataset!r} has 0 items after filtering.")

    model_label = getattr(args, "model_label", args.model)
    run_model_dir = Path(args.output_root) / utc / f"e2e_{model_label}"
    run_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_e2e] dataset={args.dataset} items={n_items}/{n_total} "
          f"run dir: {run_model_dir}", flush=True)

    summaries: list[dict] = []
    for idx, item in enumerate(items):
        out_dir = run_model_dir / item.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        # Uid-level fast-skip: if summary.json already exists AND --force is
        # not set, hydrate the row from disk and continue. Enables targeted
        # re-runs (delete uids you want to redo, re-fire the sbatch; only
        # the missing ones re-infer). --force ignores existing outputs.
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not getattr(args, "force", False):
            s = json.loads(summary_path.read_text())
            print(f"[run_e2e] skip {item.id}: summary.json exists", flush=True)
            summaries.append({
                "id": item.id, "duration": item.duration,
                "n_chunks": s.get("n_chunks"),
                "n_chunks_completed": s.get("n_chunks_completed"),
                "n_segments": s.get("n_segments_total"),
                "rt_factor": s.get("rt_factor"),
                "wall_s": s.get("total_inference_s"),
                "summary_path": str(summary_path), "skipped": True,
            })
            continue
        if n_items > 1:
            print(f"[run_e2e] ({idx+1}/{n_items}) {item.id} "
                  f"duration={item.duration:.0f}s", flush=True)
        summary = run_inference(
            item.audio_path, adapter, out_dir, dcfg["labels"],
            label_hints=dcfg["label_hints"],
            chunk_minutes=args.chunk_min,
            decoder=args.decoder,
            max_chunks=args.max_chunks if args.smoke else None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prompt_mode=args.prompt_mode,
            context_mode=args.context,
            time_unit=args.time_unit,
            dataset_name=dcfg["dataset_name"],
            t0_iso=dcfg["t0_iso"],
            audio_offset_s=item.audio_offset_s,
            audio_window_s=item.duration,
            with_description=args.emit_description,
        )
        summaries.append({
            "id": item.id,
            "duration": item.duration,
            "n_chunks": summary.n_chunks,
            "n_chunks_completed": summary.n_chunks_completed,
            "n_segments": summary.n_segments_total,
            "rt_factor": summary.rt_factor,
            "wall_s": summary.total_inference_s,
            "summary_path": str(out_dir / "summary.json"),
        })

    if n_items == 1:
        s = summaries[0]
        print(f"[run_e2e] DONE. {s['n_chunks_completed']}/{s['n_chunks']} chunks, "
              f"{s['n_segments']} segments, RT factor {s['rt_factor']:.1f}x. "
              f"Summary: {s['summary_path']}", flush=True)
        return 0

    # Multi-item: aggregate. Index lives at the common parent of all
    # per-item subdirs (Ego4D's ``ego4d/`` dir).
    common_parent = items[0].subdir.parent
    index_dir = run_model_dir / common_parent
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "run_index.json"
    index_payload = {
        "model": model_label,
        "dataset": args.dataset,
        "n_videos": n_items,
        "videos": summaries,
    }
    # Record the exact FT checkpoint so the evaluated model is unambiguous.
    if getattr(args, "adapter_dir", None):
        index_payload["adapter_dir"] = args.adapter_dir
        index_payload["base_model_id"] = args.model_id or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    index_path.write_text(json.dumps(index_payload, indent=2))
    total_wall = sum(s["wall_s"] for s in summaries)
    total_audio = sum(s["duration"] for s in summaries)
    print(f"[run_e2e] DONE. {n_items} items, "
          f"audio={total_audio:.0f}s, wall={total_wall:.1f}s, "
          f"aggregate RT factor {total_audio/max(total_wall,1e-6):.1f}x. "
          f"Index: {index_path}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODELS.keys()))
    parser.add_argument("--dataset", choices=["sins", "ego4d", "egolife"], default="sins",
                        help="Which dataset to run on (default: sins).")
    parser.add_argument(
        "--audio", default=None,
        help=f"SINS only: single mono FLAC to run inference on. "
             f"(Default: {SINS_MONO_FLAC})",
    )
    parser.add_argument(
        "--manifest", default=None,
        help=f"Ego4D only: path to the manifest.json built by manifest_builder.py. "
             f"(Default: {EGO4D_MANIFEST})",
    )
    parser.add_argument(
        "--output-root", default=str(RUNS_DIR),
        help="Root dir; per-run subdirs are <UTC>/<model_name>/(ego4d/<uid>/)?.",
    )
    parser.add_argument(
        "--chunk-min", type=float, default=10.0,
        help="Chunk size in minutes. AF3 trained at 10; Qwen-Omni claims up to 600.",
    )
    parser.add_argument("--decoder", choices=["structured", "freeform"], default="structured")
    parser.add_argument("--prompt-mode", choices=["structured", "simple"], default="structured")
    parser.add_argument("--model-id", default=None,
                        help="Override HF repo id / local path.")
    parser.add_argument(
        "--adapter-dir", default=None,
        help="Fine-tuned LoRA + projector checkpoint dir (train_e2e.py output). "
             "When set, runs the FINE-TUNED model: loads the base (--model-id, "
             "default Qwen3-Omni) + this adapter via the HF-native "
             "Qwen3OmniHFAdapter (NOT vLLM), writes to e2e_<model>_ft/, and "
             "forces --decoder freeform. The absolute checkpoint path is "
             "recorded in run_index.json so the evaluated checkpoint is "
             "unambiguous. Run single-process (naive MP); not accelerate/torchrun.",
    )
    parser.add_argument(
        "--attn-implementation", default=None,
        help="FT-adapter path only: HF attn impl for load_finetuned_model. "
             "Default: auto-detect (flash_attention_2 if flash_attn is "
             "importable, else sdpa — e.g. the las-train venv). Pass "
             "explicitly (e.g. 'sdpa' / 'eager') to override.",
    )
    parser.add_argument(
        "--device-map", default="auto",
        help="FT-adapter path only: device_map for load_finetuned_model "
             "('auto' = naive model parallel across visible GPUs).",
    )
    parser.add_argument(
        "--engine", choices=["vllm", "hf"], default="vllm",
        help="Inference engine for BASE Qwen3-Omni (no --adapter-dir). "
             "'vllm' (default) is the fast VLLMAdapter path but is hard-capped "
             "at max_model_len=32768 — long chunks >~30 min blow that cap. "
             "'hf' uses HF Transformers native (load_finetuned_model with "
             "adapter_dir=None + Qwen3OmniHFAdapter), respecting the model's "
             "config max_position_embeddings=40960; ~5-10x slower but unlocks "
             "the ~45 min chunk. Ignored when --adapter-dir is set (the FT "
             "path is always HF native, regardless of --engine).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--force", action="store_true",
                        help="Re-run uids even if summary.json exists. "
                             "Default: fast-skip uids with existing summary "
                             "for targeted re-runs after selective delete.")
    parser.add_argument("--temperature", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true",
                       help="Run only --max-chunks chunks per audio (default 1).")
    group.add_argument("--full", action="store_true",
                       help="Run all chunks.")
    parser.add_argument("--max-chunks", type=int, default=1,
                        help="With --smoke: number of chunks per audio.")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Ego4D only: cap number of videos to run.")
    parser.add_argument("--min-events", type=int, default=2,
                        help="Ego4D/EgoLife filter: minimum merged-runs in pass-1 to include. "
                             "Default 2 matches Ego4DDataset.EVAL_DEFAULTS / EgoLifeDataset.EVAL_DEFAULTS.")
    parser.add_argument("--min-duration-s", type=float, default=600.0,
                        help="Ego4D filter: minimum audio duration in seconds.")
    parser.add_argument("--eval-set", action="store_true",
                        help="Ego4D: use the canonical eval-set filter "
                             "(Ego4DDataset.EVAL_DEFAULTS) instead of --min-events "
                             "/ --min-duration-s.")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Ego4D: filter to one split. Default = all splits (use for "
             "inter-annotator stats); pass `test` for inference / eval runs.",
    )
    parser.add_argument("--context", choices=["none", "prev"], default="none")
    parser.add_argument("--time-unit", choices=["second", "minute"], default="second")
    parser.add_argument(
        "--emit-description", action="store_true",
        help="Per-segment 1-2 sentence sub-activity narrative in addition to "
             "{label,start,end}. Default off keeps prompt + schema byte-identical "
             "to the seg-only baseline.",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="vLLM adapters only: override the LLM's max_num_seqs. Note that "
             "chunk_runner already processes one chunk at a time (no parallel "
             "batching across chunks), so in E2E this knob does NOT speed "
             "things up — it shapes vLLM's KV-cache preallocation. Lower "
             "value -> more KV headroom per sequence -> safer for long "
             "outputs (e.g. when --emit-description inflates generations). "
             "Errors out if paired with a non-vLLM adapter (Gemini, …).",
    )
    # ---- thinking mode (Gemini native + Qwen/Gemma s1) --------------------
    parser.add_argument(
        "--thinking-budget", type=int, default=0,
        help="Gemini only: sets thinking_config.thinking_budget on the "
             "google-genai config. 0 = off (default). Positive int allows up "
             "to N reasoning tokens on top of max_new_tokens. Rejects when "
             "used with a non-Gemini adapter -- for Qwen/Gemma s1-style hard "
             "cap use --thinking-enable + --thinking-budget-enforce.",
    )
    parser.add_argument(
        "--thinking-enable", action="store_true",
        help="Qwen3-Omni-Thinking / Gemma-4-thinking only: enable the "
             "chat-template <think>...</think> prefix. Only meaningful with "
             "a Thinking-variant model_id.",
    )
    parser.add_argument(
        "--thinking-budget-enforce", type=int, default=0,
        help="Qwen3-Omni / Gemma-4 only: s1-style forced-termination cap on "
             "reasoning tokens. Requires --thinking-enable. 0 = off. Positive "
             "int attaches a LogitsProcessor that forces </think> after N "
             "reasoning tokens if the model has not naturally emitted it.",
    )
    args = parser.parse_args()

    # Sanity-check the thinking-mode flag combinations.
    if args.thinking_budget_enforce > 0 and not args.thinking_enable:
        parser.error(
            "--thinking-budget-enforce requires --thinking-enable "
            "(the chat template must emit <think> for the LogitsProcessor "
            "to have anything to close)."
        )
    if args.thinking_budget > 0 and (args.thinking_enable or args.thinking_budget_enforce > 0):
        parser.error(
            "--thinking-budget is Gemini-only; combining it with "
            "--thinking-enable / --thinking-budget-enforce (Qwen/Gemma) is "
            "a config error."
        )

    dcfg = _dataset_config(args.dataset)
    # Honor RUN_TS env var so multi-arm batch submissions can land under a
    # single shared dir. Fall back to a fresh stamp for ad-hoc invocations.
    utc = os.environ.get("RUN_TS") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    if args.adapter_dir:
        # Fine-tuned path: base + LoRA/projector via HF-native adapter. Writes
        # to e2e_<model>_ft/ and forces freeform (the SFT model emits JSON
        # directly, so no grammar-guided decode).
        from long_audio.training.infer import Qwen3OmniHFAdapter, load_finetuned_model
        from long_audio.training.model import load_processor

        base_id = args.model_id or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        args.adapter_dir = str(Path(args.adapter_dir).resolve())
        args.model_label = f"{args.model}_ft"
        args.decoder = "freeform"
        # Auto-detect the attention impl unless explicitly overridden: prefer
        # flash_attention_2 when flash_attn is importable, else fall back to
        # sdpa (e.g. the las-train venv, which ships without flash_attn).
        if not args.attn_implementation:
            try:
                import flash_attn  # noqa: F401

                args.attn_implementation = "flash_attention_2"
            except Exception:
                args.attn_implementation = "sdpa"
        print(f"[run_e2e] FT path: base={base_id} adapter={args.adapter_dir} "
              f"(attn={args.attn_implementation}, device_map={args.device_map})",
              flush=True)
        top = load_finetuned_model(
            base_id, args.adapter_dir,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        adapter = Qwen3OmniHFAdapter(
            top, load_processor(base_id), name=args.model_label,
        )
    elif args.engine == "hf" and args.model == "qwen3-omni":
        # Base HF-native path: same loader as the FT branch above but with
        # adapter_dir=None (no LoRA). Unlocks the model's native
        # max_position_embeddings=40960 for long-chunk inference that vLLM's
        # max_model_len=32768 rejects. Slower (~5-10x per token vs vLLM);
        # only use when you need chunks that would blow the vLLM cap.
        from long_audio.training.infer import Qwen3OmniHFAdapter, load_finetuned_model
        from long_audio.training.model import load_processor

        base_id = args.model_id or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        # No suffix: HF and vLLM paths for the base model share output dir
        # (same weights, same tokenizer, same generation config). Bare
        # `e2e_qwen3-omni/` — the engine split is a serving detail, not a
        # model variant.
        args.model_label = args.model
        if not args.attn_implementation:
            try:
                import flash_attn  # noqa: F401
                args.attn_implementation = "flash_attention_2"
            except Exception:
                args.attn_implementation = "sdpa"
        print(f"[run_e2e] base HF path: base={base_id} (no adapter) "
              f"(attn={args.attn_implementation}, device_map={args.device_map})",
              flush=True)
        top = load_finetuned_model(
            base_id, None,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        adapter = Qwen3OmniHFAdapter(
            top, load_processor(base_id), name=args.model_label,
        )
    else:
        # Base path: label may gain a `_thinking[N]` suffix based on the
        # thinking-mode flags. See thinking_label_suffix for semantics.
        args.model_label = args.model + thinking_label_suffix(
            args.thinking_budget,
            args.thinking_enable,
            args.thinking_budget_enforce,
        )
        adapter = load_adapter(
            args.model, model_id=args.model_id,
            max_num_seqs=args.max_num_seqs,
            thinking_budget=args.thinking_budget,
            enable_thinking=args.thinking_enable,
            thinking_budget_enforce=args.thinking_budget_enforce,
        )
    print(f"[run_e2e] loading adapter {adapter.name} …", flush=True)
    adapter.load()

    return _run(args, adapter, dcfg, utc)


if __name__ == "__main__":
    raise SystemExit(main())
