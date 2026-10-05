# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the E2E training entry point (scripts/train_e2e.py).

CPU-testable surface is just ``parse_args`` — model load, the Trainer loop, and
best-checkpoint selection are GPU/Trainer-integration behavior (covered by the
AWS smoke test). Training now trains + saves only; generation eval is a separate
workflow (``scripts/eval_finetuned.py``), so the post-training-eval CLI is gone.
``scripts/`` is not a package, so the module is loaded by path.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

_TRAIN_E2E = Path(__file__).resolve().parents[1] / "scripts" / "train_e2e.py"


def _load_train_e2e():
    spec = importlib.util.spec_from_file_location("train_e2e_under_test", _TRAIN_E2E)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_e2e = _load_train_e2e()


class ParseArgsTests(unittest.TestCase):
    def test_core_defaults(self):
        args = train_e2e.parse_args([])
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.chunk_min, 10.0)
        self.assertEqual(args.time_unit, "minute")
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.per_device_batch_size, 1)
        self.assertEqual(args.grad_accum, 8)
        self.assertEqual(args.epochs, 1.0)
        self.assertIsNone(args.val_split)
        self.assertFalse(args.eval_metrics)  # in-training callback stays opt-in

    def test_default_output_dir_uses_sft_e2e_prefix(self):
        out = train_e2e.default_output_dir(train_e2e.DEFAULT_MODEL_ID)
        self.assertIn("/runs/sft/e2e_qwen3-omni-", out)

    def test_post_training_eval_cli_removed(self):
        # Simplification: training no longer runs generation eval, so these
        # flags must be gone (both as attributes and as accepted CLI args).
        args = train_e2e.parse_args(["--output-dir", "/tmp/x"])
        for attr in (
            "skip_final_eval",
            "final_eval_max_videos",
            "final_eval_min_events",
            "final_eval_min_duration_s",
            "final_eval_single_pass",
        ):
            self.assertFalse(hasattr(args, attr), f"{attr} should be removed")
        with self.assertRaises(SystemExit):
            train_e2e.parse_args(["--output-dir", "/tmp/x", "--skip-final-eval"])

    def test_logger_flags_and_report_to(self):
        args = train_e2e.parse_args([
            "--logger", "tensorboard",
            "--wandb-project", "proj",
            "--wandb-entity", "team",
        ])
        self.assertEqual(args.logger, "tensorboard")
        self.assertEqual(args.wandb_project, "proj")
        self.assertEqual(args.wandb_entity, "team")
        self.assertEqual(train_e2e.logger_report_to("wandb"), ["wandb"])
        self.assertEqual(train_e2e.logger_report_to("tensorboard"), ["tensorboard"])
        self.assertEqual(train_e2e.logger_report_to("none"), [])

    def test_wandb_config_and_disabled_init(self):
        if importlib.util.find_spec("wandb") is None:
            self.skipTest("wandb not installed")
        old_mode = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "disabled"
        try:
            args = train_e2e.parse_args([
                "--output-dir", "/tmp/e2e_qwen3-omni-test",
                "--logger", "wandb",
            ])
            cfg = train_e2e.wandb_config(args, n_train=2, n_val=1)
            self.assertEqual(cfg["task"], "e2e_sft")
            self.assertEqual(cfg["model_arch"], "qwen3-omni")
            self.assertEqual(cfg["train_examples"], 2)
            run = train_e2e.maybe_init_wandb(args, n_train=2, n_val=1)
            self.assertIsNotNone(run)
            run.finish()
        finally:
            if old_mode is None:
                os.environ.pop("WANDB_MODE", None)
            else:
                os.environ["WANDB_MODE"] = old_mode

    def test_helpers_removed(self):
        # The post-training eval orchestration functions are gone.
        self.assertFalse(hasattr(train_e2e, "run_final_eval"))
        self.assertFalse(hasattr(train_e2e, "write_split_evals"))


if __name__ == "__main__":
    unittest.main()
