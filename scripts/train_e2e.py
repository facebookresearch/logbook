# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""E2E fine-tuning entry point: bf16 LoRA on Qwen3-Omni for Ego4D segmentation.

Fine-tunes ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` in a single process with
bf16 base sharded via ``device_map="auto"`` (naive model parallel — do
NOT launch under ``accelerate`` / ``torchrun``). Trainable surface is
LoRA on LLM attention + the audio-tower projector; SFT examples come
from :class:`long_audio.training.data.Ego4DSFTDataset`. Only the LoRA
adapter + projector are checkpointed; run generation eval separately via
``scripts/eval_finetuned.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.training.data import Ego4DSFTDataset, Qwen3OmniSFTCollator
from long_audio.training.model import (
    LoraSettings,
    build_model,
    install_router_aux_diagnostic,
    load_processor,
    save_trainable,
    trainable_parameter_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "sft"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()


def _model_slug(model_id: str) -> str:
    lo = model_id.lower()
    if "qwen3-omni" in lo:
        return "qwen3-omni"
    return _slug(model_id.rsplit("/", 1)[-1])


def default_output_dir(model_id: str) -> str:
    import time

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(DEFAULT_OUTPUT_ROOT / f"e2e_{_model_slug(model_id)}-{ts}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / data.
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF repo or local path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output-dir", default=None, help="Checkpoints + adapter output. Default: runs/sft/e2e_<model>-<UTC>.")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default=None,
                   help="If set (e.g. 'val'), run periodic eval loss.")
    p.add_argument("--chunk-min", type=float, default=10.0,
                   help="Audio chunk length in minutes (5/10/20/30). 5-min ~= "
                        "classification; >=10-min teaches segmentation.")
    p.add_argument("--time-unit", choices=["second", "minute"], default="minute",
                   help="Target-timestamp grammar; must match inference/eval.")
    p.add_argument("--no-description", action="store_true",
                   help="Drop per-segment description from the target JSON.")
    p.add_argument("--min-events", type=int, default=1,
                   help="Training-lenient default (keep more data).")
    p.add_argument("--min-duration-s", type=float, default=0.0)
    p.add_argument("--pass-id", default="1", help="Ego4D narration pass for gold.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap #chunk-examples (smoke runs).")
    # LoRA.
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--router-aux-loss-coef", type=float, default=None,
                   help="Enable MoE load-balancing aux loss with this coef "
                        "(best-effort; default: leave model config as-is).")
    # Optimisation.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If >0, overrides --epochs.")
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    # Checkpointing / logging.
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-metrics", action="store_true",
                   help="Run the 4-metric segmentation eval (generation) on the "
                        "val split during training, in addition to eval loss. "
                        "Requires --val-split.")
    p.add_argument("--eval-max-videos", type=int, default=8,
                   help="Cap #val videos for the in-training metric eval (speed).")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a Trainer checkpoint dir to resume from.")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", default=None,
                   help="HF attn impl. Default: auto-detect (flash_attention_2 if "
                        "flash_attn is importable, else sdpa — e.g. the las-train "
                        "venv ships without flash_attn).")
    p.add_argument("--device-map", default="auto",
                   help="'auto' (naive MP) or 'cuda:0' for single-GPU.")
    p.add_argument("--logger", choices=["wandb", "tensorboard", "none"], default="wandb")
    p.add_argument("--wandb-project", default="long-audio-segmentation")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args(argv)



def logger_report_to(logger: str) -> list[str]:
    if logger == "none":
        return []
    return [logger]


def run_name_from_output_dir(output_dir: str) -> str:
    return Path(output_dir).name


def wandb_config(args: argparse.Namespace, *, n_train: int | None = None, n_val: int | None = None) -> dict:
    keys = [
        "model_id", "manifest", "train_split", "val_split", "chunk_min",
        "time_unit", "pass_id", "min_events", "min_duration_s", "lora_r",
        "lora_alpha", "lora_dropout", "router_aux_loss_coef", "lr",
        "weight_decay", "warmup_ratio", "per_device_batch_size", "grad_accum",
        "epochs", "max_steps", "lr_scheduler", "seed", "device_map",
        "attn_implementation", "eval_metrics", "eval_max_videos",
    ]
    cfg = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    cfg.update({
        "task": "e2e_sft",
        "model_arch": _model_slug(args.model_id),
        "output_dir": args.output_dir,
        "run_name": run_name_from_output_dir(args.output_dir),
        "train_examples": n_train,
        "val_examples": n_val,
    })
    return cfg


def maybe_init_wandb(args: argparse.Namespace, *, n_train: int | None = None, n_val: int | None = None):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - env/dependency issue
        raise SystemExit("--logger wandb requires the wandb package; install long-audio[train].") from exc

    model_arch = _model_slug(args.model_id)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name_from_output_dir(args.output_dir),
        group=f"e2e/{model_arch}",
        job_type="train_e2e",
        config=wandb_config(args, n_train=n_train, n_val=n_val),
    )


def maybe_log_adapter_artifact(wandb_run, output_dir: str, *, name: str, artifact_type: str) -> None:
    if wandb_run is None:
        return
    import wandb

    out = Path(output_dir)
    if not out.exists():
        return
    artifact = wandb.Artifact(name=_slug(name), type=artifact_type)
    artifact.add_dir(str(out))
    wandb_run.log_artifact(artifact)


def build_datasets(args):
    common = dict(
        chunk_minutes=args.chunk_min,
        time_unit=args.time_unit,
        pass_id=args.pass_id,
        with_description=not args.no_description,
        load_audio=True,
        min_events=args.min_events,
        min_duration_s=args.min_duration_s,
    )
    train_ds = Ego4DSFTDataset(args.manifest, split=args.train_split, **common)
    if args.max_train_samples is not None:
        from torch.utils.data import Subset

        n = min(args.max_train_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))  # contiguous head for smoke runs
    val_ds = None
    if args.val_split:
        val_ds = Ego4DSFTDataset(args.manifest, split=args.val_split, **common)
    return train_ds, val_ds


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_id)
    from transformers import Trainer, TrainingArguments

    class _ModelParallelTrainer(Trainer):
        """Trainer tolerant of naive model parallelism (device_map='auto').

        With the model sharded across GPUs, the loss is computed on the last
        shard (where lm_head lives, e.g. cuda:3), but Trainer's loss-accumulation
        guard is unconditional: it requires the per-step loss on args.device
        (cuda:0) and raises "Calculated loss must be on the original device"
        otherwise. Move the detached per-step loss back to args.device to satisfy
        it. (Defined here, not at module scope, so importing this script needs no
        transformers.) This is the known device_map + Trainer friction that
        DeepSpeed/FSDP would sidestep.
        """

        def training_step(self, *step_args, **step_kwargs):
            loss = super().training_step(*step_args, **step_kwargs)
            return loss.to(self.args.device)

    # Auto-detect attn implementation: prefer flash_attention_2 when available,
    # fall back to sdpa (e.g. the las-train venv ships without flash_attn).
    # Mirrors the same pattern in run_e2e.py for FT inference.
    if not args.attn_implementation:
        try:
            import flash_attn  # noqa: F401
            args.attn_implementation = "flash_attention_2"
        except Exception:
            args.attn_implementation = "sdpa"
    print(f"[train_e2e] loading model {args.model_id} "
          f"(attn={args.attn_implementation}, device_map={args.device_map}) ...", flush=True)
    peft_thinker, top = build_model(
        args.model_id,
        lora=LoraSettings(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout),
        device_map=args.device_map,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )
    summ = trainable_parameter_summary(peft_thinker)
    print(
        f"[train_e2e] trainable params: {summ['trainable']:,} / {summ['total']:,} "
        f"({summ['trainable_pct']:.3f}%) - lora={summ['lora']:,} "
        f"projector={summ['projector']:,} other={summ['other']:,}",
        flush=True,
    )
    if summ["projector"] == 0:
        raise RuntimeError(
            "Projector params are not trainable - the freeze/LoRA wiring is "
            "broken (audio_tower.proj1/proj2 should carry gradient)."
        )
    if summ["other"] != 0:
        raise RuntimeError(
            "Unexpected trainable params outside LoRA+projector "
            f"({summ['other']:,}); LoRA likely leaked onto the frozen encoder."
        )

    processor = load_processor(args.model_id)
    train_ds, val_ds = build_datasets(args)
    n_train = len(train_ds)
    n_val = len(val_ds) if val_ds is not None else None
    print(
        f"[train_e2e] train examples: {n_train}"
        + (f" | val: {n_val}" if n_val is not None else ""),
        flush=True,
    )
    collator = Qwen3OmniSFTCollator(processor)

    have_val = val_ds is not None
    if have_val and args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
            f"({args.eval_steps}) so load_best_model_at_end can align save/eval "
            "checkpoints."
        )

    wandb_run = maybe_init_wandb(args, n_train=n_train, n_val=n_val)
    try:
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
            bf16=True,
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

        trainer = _ModelParallelTrainer(
            model=peft_thinker,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )

        if args.eval_metrics:
            if not args.val_split:
                raise SystemExit("--eval-metrics requires --val-split (e.g. 'val').")
            from long_audio.training.evaluate import Ego4DEvalCallback
            from long_audio.training.infer import Qwen3OmniHFAdapter

            eval_cb = Ego4DEvalCallback(
                Qwen3OmniHFAdapter(top, processor),
                args.manifest,
                args.val_split,
                chunk_minutes=args.chunk_min,
                time_unit=args.time_unit,
                with_description=not args.no_description,
                max_videos=args.eval_max_videos,
                model_name="qwen3-omni-ft",
                min_events=args.min_events,
                min_duration_s=args.min_duration_s,
                pass_id=args.pass_id,
            )
            eval_cb.bind_trainer(trainer)
            trainer.add_callback(eval_cb)
            print(
                f"[train_e2e] segmentation-metric eval on '{args.val_split}' "
                f"(<= {args.eval_max_videos} videos) each eval step",
                flush=True,
            )

        if args.router_aux_loss_coef is not None:
            # One-shot forward hook that logs loss / aux_loss / aux*coef on the
            # first training step, then removes itself. Confirms the MoE
            # load-balancing aux term is actually contributing to the Trainer's
            # loss.
            install_router_aux_diagnostic(peft_thinker, args.router_aux_loss_coef)

        print("[train_e2e] starting training...", flush=True)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        save_trainable(peft_thinker, args.output_dir)
        maybe_log_adapter_artifact(
            wandb_run,
            args.output_dir,
            name=run_name_from_output_dir(args.output_dir),
            artifact_type="e2e-adapter",
        )
        print(
            f"[train_e2e] DONE. Saved LoRA adapter + projector to {args.output_dir}. "
            "Run generation eval separately via scripts/eval_finetuned.py.",
            flush=True,
        )
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
