# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-2 cascade fine-tuning: text-only LoRA SFT.

Fine-tunes an open-weight chat LLM to map fixed Stage-A captions to the
segmentation JSON consumed by the cascade Stage B runner. This is the text-only
analogue of ``scripts/train_e2e.py``:

    python scripts/train_cascade.py \\
        --model qwen3-32b \\
        --manifest datasets/ego4d/annotated_manifest.json \\
        --captions-root runs/cascA_inference/af3-hf \\
        --output-dir runs/sft/cascaded_qwen3-32b-$(date -u +%Y%m%dT%H%M%SZ) \\
        --train-split train --val-split val \\
        --window-min 10 --time-unit minute

Only PEFT adapter weights are saved. Reload the frozen base model and attach the
adapter for generation/eval.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.training.stage2_data import Stage2SFTDataset, Stage2SFTExample

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "sft"


@dataclass(frozen=True)
class Stage2ModelPreset:
    """HF model preset for Stage-2 text-only training."""

    alias: str
    model_id: str
    tensor_parallel_size: int
    default_device_map: str = "auto"


MODEL_PRESETS: dict[str, Stage2ModelPreset] = {
    "qwen2.5-72b": Stage2ModelPreset(
        alias="qwen2.5-72b",
        model_id="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=4,
    ),
    "qwen3-32b": Stage2ModelPreset(
        alias="qwen3-32b",
        model_id="Qwen/Qwen3-32B",
        tensor_parallel_size=2,
    ),
    "gemma4-31b": Stage2ModelPreset(
        alias="gemma4-31b",
        model_id="google/gemma-4-31B-it",
        tensor_parallel_size=4,
    ),
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()


def default_output_dir(model_alias: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(DEFAULT_OUTPUT_ROOT / f"cascaded_{_slug(model_alias)}-{ts}")


def _base_is_gemma(model_id: str) -> bool:
    """True if the HF base model id is a Gemma-family model.

    Gemma-4's cascade LoRA cannot be served by vLLM 0.21.0 as a LoRARequest —
    the multimodal-LoRA path silently drops every language_model.* entry
    (confirmed via A/B test: WITH-LoRARequest and WITHOUT are byte-identical).
    Workaround is to pre-merge the LoRA into a plain checkpoint; we do that
    inline at end of training when the base is Gemma. See
    long_audio/inference/models/vllm_text.py :: Gemma4_31BTextAdapter."""
    return "gemma" in model_id.lower()


def resolve_model(args: argparse.Namespace) -> tuple[str, str, Stage2ModelPreset | None]:
    """Return ``(model_alias, model_id, preset_or_none)``."""
    preset = MODEL_PRESETS.get(args.model)
    if preset is not None:
        return preset.alias, args.model_id or preset.model_id, preset
    if not args.model_id:
        raise SystemExit(
            f"--model {args.model!r} is not a known preset; pass --model-id."
        )
    return _slug(args.model), args.model_id, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model / data.
    p.add_argument(
        "--model",
        default="qwen3-32b",
        help=f"Model preset alias ({', '.join(MODEL_PRESETS)}) or a custom name.",
    )
    p.add_argument("--model-id", default=None, help="Override HF repo/local path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument(
        "--captions-root",
        default=None,
        help=(
            "Stage-A captions root. Default: "
            "runs/cascA_inference/<describe-model>."
        ),
    )
    p.add_argument("--describe-model", default="af3-hf")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Checkpoint/adapters output. Default: runs/sft/cascaded_<model>-<UTC>.",
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default=None)
    p.add_argument("--window-min", type=float, default=10.0)
    p.add_argument("--time-unit", choices=["second", "minute"], default="minute")
    p.add_argument("--no-description", action="store_true")
    p.add_argument("--context", choices=["none", "prev"], default="none")
    p.add_argument("--pass-id", default="1")
    p.add_argument("--min-events", type=int, default=1)
    p.add_argument("--min-duration-s", type=float, default=0.0)
    p.add_argument(
        "--skip-missing-captions",
        action="store_true",
        help="Skip manifest videos without Stage-A JSONLs.",
    )
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    # Tokenization.
    p.add_argument("--max-length", type=int, default=32768)
    # LoRA.
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Optimisation.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    # Runtime/checkpointing.
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", default=None,
                   help="HF attn impl. Default: auto-detect (flash_attention_2 if "
                        "flash_attn is importable, else sdpa — e.g. the las-train "
                        "venv ships without flash_attn).")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logger", choices=["wandb", "tensorboard", "none"], default="wandb")
    p.add_argument("--wandb-project", default="long-audio-segmentation")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets/tokenizer/collator config and exit before model load.",
    )
    p.add_argument(
        "--init-only",
        action="store_true",
        help="Load model, attach LoRA, print trainable params, and exit before Trainer.",
    )
    return p.parse_args(argv)


def _chat_template(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply a chat template, disabling thinking where tokenizers support it."""
    kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=False,
    )
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _messages(ex: Stage2SFTExample, *, include_assistant: bool) -> list[dict]:
    msgs = [{"role": "user", "content": ex.prompt}]
    if include_assistant:
        msgs.append({"role": "assistant", "content": ex.target_text})
    return msgs


class Stage2TextSFTCollator:
    """Tokenize Stage-2 examples and mask prompt tokens in ``labels``.

    Model-agnostic — works with any chat template (Qwen2.5, Qwen3, Gemma, …)
    with no hardcoded role marker:

      * The response boundary is found by **content divergence**: for each
        example the full sequence (user + assistant) is tokenized alongside a
        perturbed copy whose assistant content differs at its first character.
        Everything the template emits *before* the response — user turn,
        assistant-turn opener, and any scaffold — is identical in both, so the
        length of their common token-prefix is exactly where the response begins
        in the full sequence. This handles both Qwen3 (empty ``<think>`` block
        rendered *inside* the assistant message) and Gemma-4 (a ``<|channel>
        thought`` scaffold present only in the generation prompt, NOT the
        message) — cases where neither a fixed marker nor the prompt-rendering
        length gives the right boundary.
      * Padding is masked via ``attention_mask == 0``, not ``input_ids ==
        pad_id`` (which would zero out a response-terminating EOS when
        ``pad_token == eos_token``). The boundary is computed on each row's
        unpadded real tokens and applied at its first non-pad position, so it is
        correct for left- OR right-padding.
      * Overlength examples raise a loud ``RuntimeError`` rather than being
        silently truncated (which would chop the target JSON tail and its EOS,
        corrupting the loss and teaching the model to emit unclosed JSON).
    """

    def __init__(self, tokenizer: Any, *, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _perturbed_messages(example: Stage2SFTExample) -> list[dict]:
        """Same messages, but the assistant content is changed at char 0 so the
        response region (and only it) diverges from the real rendering."""
        msgs = _messages(example, include_assistant=True)
        content = msgs[-1]["content"]
        first = content[:1]
        alt = "a" if first != "a" else "b"
        perturbed = list(msgs)
        perturbed[-1] = {"role": "assistant", "content": alt + content}
        return perturbed

    def __call__(self, examples: Sequence[Stage2SFTExample]) -> dict:
        full_texts = [
            _chat_template(
                self.tokenizer,
                _messages(ex, include_assistant=True),
                add_generation_prompt=False,
            )
            for ex in examples
        ]
        perturbed_texts = [
            _chat_template(
                self.tokenizer,
                self._perturbed_messages(ex),
                add_generation_prompt=False,
            )
            for ex in examples
        ]

        # Tokenize WITHOUT truncation so we can detect overlength examples and
        # fail loud. Silent truncation would drop the target JSON tail (and its
        # EOS) — model would train to emit unclosed JSON.
        batch = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        real_lens = batch["attention_mask"].sum(dim=1).tolist()
        overlong = [(i, L) for i, L in enumerate(real_lens) if L > self.max_length]
        if overlong:
            raise RuntimeError(
                f"Stage2TextSFTCollator: {len(overlong)} example(s) exceed "
                f"max_length={self.max_length}: {overlong}. Reduce --window-min "
                "or filter oversize examples at dataset construction — silent "
                "truncation would corrupt the loss."
            )

        perturbed_ids = self.tokenizer(
            perturbed_texts, padding=False, truncation=False
        )["input_ids"]

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        for row in range(len(examples)):
            mask_row = attention_mask[row].bool()
            real_ids = input_ids[row][mask_row].tolist()
            # Index of the first real token (0 for right-pad, >0 for left-pad).
            nz = mask_row.nonzero()
            content_start = int(nz[0].item()) if nz.numel() else 0

            pert = perturbed_ids[row]
            n = 0
            limit = min(len(real_ids), len(pert))
            while n < limit and real_ids[n] == pert[n]:
                n += 1
            if not 0 < n < len(real_ids):
                raise RuntimeError(
                    "Stage2TextSFTCollator: could not locate the assistant "
                    f"response boundary for row {row} (n={n}, real_len="
                    f"{len(real_ids)}). Chat-template / tokenizer issue — the "
                    "response is empty or diverges immediately; investigate "
                    "before training."
                )
            labels[row, : content_start + n] = -100

        # Pad-mask via attention_mask (see class docstring: safer than input_ids
        # == pad_id when pad shares the eos id).
        labels[attention_mask == 0] = -100
        batch["labels"] = labels
        return batch


def _subset(ds, limit: int | None):
    if limit is None:
        return ds
    from torch.utils.data import Subset

    n = min(limit, len(ds))
    return Subset(ds, list(range(n)))


def build_datasets(args: argparse.Namespace):
    common = dict(
        captions_root=args.captions_root,
        describe_model=args.describe_model,
        window_minutes=args.window_min,
        time_unit=args.time_unit,
        pass_id=args.pass_id,
        with_description=not args.no_description,
        context_mode=args.context,
        skip_missing_captions=args.skip_missing_captions,
        min_events=args.min_events,
        min_duration_s=args.min_duration_s,
    )
    train_ds = Stage2SFTDataset(args.manifest, split=args.train_split, **common)
    train_ds = _subset(train_ds, args.max_train_samples)
    val_ds = None
    if args.val_split:
        val_ds = Stage2SFTDataset(args.manifest, split=args.val_split, **common)
        val_ds = _subset(val_ds, args.max_val_samples)
    return train_ds, val_ds


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


#: perception-tower path segments to keep LoRA off (multimodal bases).
_TOWER_PARTS = frozenset({"vision_tower", "audio_tower"})


def _is_tower_name(name: str) -> bool:
    """True if any dot-separated segment of ``name`` is a perception tower."""
    return bool(_TOWER_PARTS & set(name.split(".")))


def _strip_tower_lora(peft_model: Any) -> int:
    """Unwrap LoRA from perception-tower modules after an ``"all-linear"``
    ``get_peft_model`` so the saved adapter carries NO ``vision_tower`` /
    ``audio_tower`` entries.

    Rationale: multimodal bases (gemma-4 via ``AutoModelForCausalLM``) expose
    vision/audio towers whose ``nn.Linear`` layers ``"all-linear"`` LoRA-wraps.
    Those are dead in text-only SFT AND make the adapter unservable by vLLM text
    engines — vLLM rejects tower LoRA targets, and its ``enable_tower_connector_lora``
    path is broken in 0.21.0. LoRA is zero-initialised (``lora_B`` = 0), so
    unwrapping a tower LoRA back to its ``base_layer`` is an exact identity op.

    We (1) replace each tower ``LoraLayer`` with its original ``base_layer`` so no
    tower LoRA weights are saved, and (2) drop the tower names from the adapter's
    ``target_modules`` (PEFT resolves ``"all-linear"`` to a concrete name set that
    otherwise still lists the towers) so ``adapter_config.json`` is clean too.

    Returns the number of tower modules unwrapped; dense text bases (qwen/llama)
    have no towers -> ``0`` (no-op).
    """
    from peft.tuners.lora import LoraLayer

    to_strip = [
        name
        for name, mod in peft_model.named_modules()
        if isinstance(mod, LoraLayer) and _is_tower_name(name)
    ]
    for name in to_strip:
        parent_name, _, child = name.rpartition(".")
        parent = peft_model.get_submodule(parent_name)
        setattr(parent, child, peft_model.get_submodule(name).base_layer)

    # PEFT stores all-linear as a resolved concrete name set; strip the towers
    # from it so nothing references them in the saved config.
    for cfg in peft_model.peft_config.values():
        tm = cfg.target_modules
        if isinstance(tm, (set, list, tuple)):
            cfg.target_modules = type(tm)(t for t in tm if not _is_tower_name(t))

    return len(to_strip)


def build_model(
    model_id: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device_map: str,
    attn_implementation: str,
    gradient_checkpointing: bool,
    bf16: bool,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if bf16 else None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # LoRA targeting is always PEFT "all-linear" (LoRA on every nn.Linear;
    # lm_head auto-excluded; MoE experts are nn.Parameter, so skipped), then the
    # perception towers are stripped post-hoc by _strip_tower_lora. This is the
    # only supported path — simpler than enumerating text modules ourselves, and
    # works for any arch (dense bases have no towers, so the strip is a no-op).
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    peft_model = get_peft_model(model, lora_cfg)

    n_stripped = _strip_tower_lora(peft_model)
    print(
        f"[train_cascade] all-linear -> stripped LoRA from {n_stripped} "
        "vision/audio tower modules (adapter is text-decoder only)",
        flush=True,
    )

    if gradient_checkpointing:
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return peft_model


def trainable_parameter_summary(model: Any) -> dict[str, float | int]:
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }




def logger_report_to(logger: str) -> list[str]:
    if logger == "none":
        return []
    return [logger]


def run_name_from_output_dir(output_dir: str) -> str:
    return Path(output_dir).name


def wandb_config(args: argparse.Namespace, *, model_alias: str, model_id: str, n_train: int | None = None, n_val: int | None = None) -> dict[str, Any]:
    keys = [
        "manifest", "captions_root", "describe_model", "train_split", "val_split",
        "window_min", "time_unit", "context", "pass_id", "min_events",
        "min_duration_s", "max_length", "lora_r", "lora_alpha", "lora_dropout",
        "lr", "weight_decay", "warmup_ratio", "per_device_batch_size", "grad_accum",
        "epochs", "max_steps", "lr_scheduler", "seed", "bf16", "device_map",
        "attn_implementation",
    ]
    cfg = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    cfg.update({
        "task": "cascade_sft",
        "model_alias": model_alias,
        "model_id": model_id,
        "output_dir": args.output_dir,
        "run_name": run_name_from_output_dir(args.output_dir),
        "train_examples": n_train,
        "val_examples": n_val,
    })
    return cfg


def maybe_init_wandb(
    args: argparse.Namespace,
    *,
    model_alias: str,
    model_id: str,
    n_train: int | None = None,
    n_val: int | None = None,
):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - env/dependency issue
        raise SystemExit("--logger wandb requires the wandb package; install long-audio[train].") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name_from_output_dir(args.output_dir),
        group=f"cascade/{model_alias}",
        job_type="train_cascade",
        config=wandb_config(
            args,
            model_alias=model_alias,
            model_id=model_id,
            n_train=n_train,
            n_val=n_val,
        ),
    )


def maybe_log_adapter_artifact(wandb_run: Any, output_dir: str, *, name: str, artifact_type: str) -> None:
    if wandb_run is None:
        return
    import wandb

    out = Path(output_dir)
    if not out.exists():
        return
    artifact = wandb.Artifact(name=_slug(name), type=artifact_type)
    artifact.add_dir(str(out))
    wandb_run.log_artifact(artifact)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_alias, model_id, preset = resolve_model(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(model_alias)
    if args.val_split and args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
            f"({args.eval_steps}) for load_best_model_at_end."
        )

    print(
        f"[train_cascade] model={model_alias} model_id={model_id} "
        f"captions_root={args.captions_root} out={args.output_dir}",
        flush=True,
    )
    print(
        "[train_cascade] lora targets: all-linear (vision/audio towers "
        "stripped post-hoc)",
        flush=True,
    )

    train_ds, val_ds = build_datasets(args)
    n_train = len(train_ds)
    n_val = len(val_ds) if val_ds is not None else None
    print(
        f"[train_cascade] train examples: {n_train}"
        + (f" | val: {n_val}" if n_val is not None else ""),
        flush=True,
    )

    wandb_run = maybe_init_wandb(
        args,
        model_alias=model_alias,
        model_id=model_id,
        n_train=n_train,
        n_val=n_val,
    )
    try:
        tokenizer = load_tokenizer(model_id)
        collator = Stage2TextSFTCollator(tokenizer, max_length=args.max_length)
        if args.dry_run:
            ex = train_ds[0] if len(train_ds) else None
            if ex is not None:
                batch = collator([ex])
                keep = batch["labels"][0] != -100
                label_tail = batch["labels"][0][keep][-8:].tolist() if keep.any() else []
                input_tail = batch["input_ids"][0][keep][-8:].tolist() if keep.any() else []
                print(
                    f"[train_cascade] dry-run batch input shape="
                    f"{tuple(batch['input_ids'].shape)} labels={tuple(batch['labels'].shape)}",
                    flush=True,
                )
                print(
                    f"[train_cascade] dry-run assistant input_ids tail={input_tail} "
                    f"labels tail={label_tail}",
                    flush=True,
                )
            print("[train_cascade] dry-run complete before model load.", flush=True)
            return 0

        from transformers import Trainer, TrainingArguments

        # Auto-detect attn implementation: prefer flash_attention_2 when available,
        # fall back to sdpa (e.g. the las-all venv ships without flash_attn).
        # Mirrors the same pattern in train_e2e.py / run_e2e.py.
        if not args.attn_implementation:
            try:
                import flash_attn  # noqa: F401
                args.attn_implementation = "flash_attention_2"
            except Exception:
                args.attn_implementation = "sdpa"
        print(f"[train_cascade] loading model {model_id} "
              f"(attn={args.attn_implementation}, device_map={args.device_map}) ...", flush=True)
        model = build_model(
            model_id,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            bf16=args.bf16,
        )
        summ = trainable_parameter_summary(model)
        print(
            f"[train_cascade] trainable params: {summ['trainable']:,} / {summ['total']:,} "
            f"({summ['trainable_pct']:.3f}%)",
            flush=True,
        )
        if args.init_only:
            print("[train_cascade] init-only complete before Trainer/training.", flush=True)
            return 0

        have_val = val_ds is not None
        targs = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            lr_scheduler_type=args.lr_scheduler,
            bf16=args.bf16,
            gradient_checkpointing=False,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            eval_strategy=("steps" if have_val else "no"),
            eval_steps=args.eval_steps,
            load_best_model_at_end=have_val,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=logger_report_to(args.logger),
            run_name=run_name_from_output_dir(args.output_dir),
            label_names=["labels"],
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )

        print("[train_cascade] starting training...", flush=True)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        maybe_log_adapter_artifact(
            wandb_run,
            args.output_dir,
            name=run_name_from_output_dir(args.output_dir),
            artifact_type="cascade-adapter",
        )
        print(f"[train_cascade] DONE. Saved PEFT adapter to {args.output_dir}.", flush=True)

        # Gemma-only: also save a pre-merged (base + LoRA folded) checkpoint
        # at <output_dir>/merged/. vLLM 0.21.0's multimodal-LoRA path silently
        # drops language_model.* LoRA entries for Gemma4ForConditionalGeneration,
        # so serving requires a merged checkpoint (see
        # long_audio/inference/models/vllm_text.py :: Gemma4_31BTextAdapter).
        # Doing the merge here (model is already loaded) saves a ~60 GB re-load
        # vs the standalone scripts/merge_lora.py post-training. Skipped for
        # dense text bases — their LoRA adapters serve fine via vLLM LoRARequest.
        if _base_is_gemma(model_id):
            merged_dir = os.path.join(args.output_dir, "merged")
            print(f"[train_cascade] Gemma base detected -> merging LoRA for serving "
                  f"-> {merged_dir}", flush=True)
            t0 = time.time()
            merged = model.merge_and_unload()
            print(f"[train_cascade]   merge_and_unload in {time.time()-t0:.1f}s",
                  flush=True)
            os.makedirs(merged_dir, exist_ok=True)
            t0 = time.time()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            # AutoProcessor over AutoTokenizer: Gemma4ForConditionalGeneration
            # is multimodal, and vLLM constructs a feature extractor at load
            # time that needs processor_config.json / preprocessor_config.json.
            # Fall back to the tokenizer already-loaded upstream if the base
            # doesn't expose an AutoProcessor.
            try:
                from transformers import AutoProcessor
                proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                proc.save_pretrained(merged_dir)
            except Exception as e:
                print(f"[train_cascade]   AutoProcessor unavailable "
                      f"({type(e).__name__}: {e}); saving tokenizer only",
                      flush=True)
                tokenizer.save_pretrained(merged_dir)
            # Copy the adapter's chat_template.jinja so vLLM picks up the
            # training-time template. save_pretrained on the PeftModel writes
            # it to output_dir at line above; mirror into merged_dir.
            ct_src = os.path.join(args.output_dir, "chat_template.jinja")
            if os.path.exists(ct_src):
                ct_dst = os.path.join(merged_dir, "chat_template.jinja")
                with open(ct_src) as f:
                    ct = f.read()
                with open(ct_dst, "w") as f:
                    f.write(ct)
            print(f"[train_cascade]   saved merged model in {time.time()-t0:.1f}s",
                  flush=True)
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
