"""Tests for scripts/train_cascade.py helpers that do not load real models."""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch

from long_audio.training.stage2_data import Stage2SFTExample
from scripts import train_cascade

_HAS_PEFT = importlib.util.find_spec("peft") is not None


class _FakeTokenizer:
    """Whitespace tokenizer with a PERSISTENT per-instance vocab (like a real
    tokenizer: token ids are stable across calls), so the collator's
    content-divergence comparison across two renderings is valid."""

    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = "<pad>"
    _seed = {"<|im_start|>assistant": 777}

    def __init__(self):
        self._vocab = dict(self._seed)

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = max(self._vocab.values()) + 1
        return self._vocab[tok]

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant\n":
            return [777]
        return [self._id(t) for t in text.split()]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        assert tokenize is False
        prefix = "user: prompt <|im_start|>assistant"
        if messages and messages[-1]["role"] == "assistant":
            return prefix + f" {messages[-1]['content']}"
        return prefix

    def _row_ids(self, text, truncation, max_length):
        toks = text.split()
        if truncation and max_length is not None:
            toks = toks[:max_length]
        return [self._id(t) for t in toks]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None):
        # Mirror real HF: return_tensors=None -> python lists (flat for a single
        # str input); return_tensors="pt" -> padded tensors.
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        ids = [self._row_ids(t, truncation, max_length) for t in texts]
        if return_tensors == "pt":
            max_len = max(len(x) for x in ids)
            att = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
            ids = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(att, dtype=torch.long),
            }
        if single:
            return {"input_ids": ids[0], "attention_mask": [1] * len(ids[0])}
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

    def save_pretrained(self, path):  # pragma: no cover - not used
        return None


class _FakePadEqualsEosTokenizer(_FakeTokenizer):
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token = "<eos>"
    _seed = {"<eos>": 0, "<|im_start|>assistant": 777}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = _FakeTokenizer.apply_chat_template(
            self, messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize, **kwargs
        )
        if messages and messages[-1]["role"] == "assistant":
            text += " <eos>"
        return text


class _FakeThinkTokenizer(_FakeTokenizer):
    """Mimics Qwen3 (enable_thinking=False): the chat template injects an empty
    ``<think> </think>`` block inside the rendered assistant message, so the
    response starts two tokens later. Content-divergence must skip the block."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "user: prompt <|im_start|>assistant <think> </think>"
        if messages and messages[-1]["role"] == "assistant":
            text += f" {messages[-1]['content']}"
        return text


class _FakeGemma4Tokenizer(_FakeTokenizer):
    """Mimics a Gemma chat template: uses ``<start_of_turn>model`` as the
    assistant-turn opener — there is NO ``<|im_start|>assistant`` marker at all,
    and no <think> scaffold. Exercises the model-agnostic (marker-free) boundary."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "<start_of_turn>user prompt <end_of_turn> <start_of_turn>model"
        if messages and messages[-1]["role"] == "assistant":
            text += f" {messages[-1]['content']} <end_of_turn>"
        return text


class TrainStage2ConfigTests(unittest.TestCase):
    def test_known_presets(self):
        self.assertEqual(
            train_cascade.MODEL_PRESETS["qwen2.5-72b"].model_id,
            "Qwen/Qwen2.5-72B-Instruct",
        )
        self.assertEqual(
            train_cascade.MODEL_PRESETS["qwen3-32b"].model_id,
            "Qwen/Qwen3-32B",
        )
        self.assertEqual(
            train_cascade.MODEL_PRESETS["gemma4-31b"].model_id,
            "google/gemma-4-31B-it",
        )

    def test_parse_args_and_resolve_model(self):
        args = train_cascade.parse_args([
            "--model", "qwen3-32b",
            "--captions-root", "runs/cascA_inference/af-next-captioner/ego4d/train",
            "--init-only",
        ])
        alias, model_id, preset = train_cascade.resolve_model(args)
        self.assertEqual(alias, "qwen3-32b")
        self.assertEqual(model_id, "Qwen/Qwen3-32B")
        self.assertTrue(args.init_only)
        self.assertIsNotNone(preset)

    def test_custom_model_requires_model_id(self):
        args = train_cascade.parse_args([
            "--model", "custom",
            "--captions-root", "runs/cascA_inference/af-next-captioner/ego4d/train",
        ])
        with self.assertRaises(SystemExit):
            train_cascade.resolve_model(args)

    def test_logger_flags_and_report_to(self):
        args = train_cascade.parse_args([
            "--model", "qwen3-32b",
            "--logger", "tensorboard",
            "--wandb-project", "proj",
            "--wandb-entity", "team",
        ])
        self.assertEqual(args.logger, "tensorboard")
        self.assertEqual(args.wandb_project, "proj")
        self.assertEqual(args.wandb_entity, "team")
        self.assertEqual(train_cascade.logger_report_to("wandb"), ["wandb"])
        self.assertEqual(train_cascade.logger_report_to("tensorboard"), ["tensorboard"])
        self.assertEqual(train_cascade.logger_report_to("none"), [])

    def test_wandb_config_and_disabled_init(self):
        if importlib.util.find_spec("wandb") is None:
            self.skipTest("wandb not installed")
        old_mode = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "disabled"
        try:
            args = train_cascade.parse_args([
                "--model", "qwen3-32b",
                "--output-dir", "/tmp/cascaded_qwen3-32b-test",
                "--logger", "wandb",
            ])
            alias, model_id, _ = train_cascade.resolve_model(args)
            cfg = train_cascade.wandb_config(
                args, model_alias=alias, model_id=model_id, n_train=2, n_val=1
            )
            self.assertEqual(cfg["task"], "cascade_sft")
            self.assertEqual(cfg["model_alias"], "qwen3-32b")
            self.assertEqual(cfg["train_examples"], 2)
            run = train_cascade.maybe_init_wandb(
                args, model_alias=alias, model_id=model_id, n_train=2, n_val=1
            )
            self.assertIsNotNone(run)
            run.finish()
        finally:
            if old_mode is None:
                os.environ.pop("WANDB_MODE", None)
            else:
                os.environ["WANDB_MODE"] = old_mode


class Stage2TextSFTCollatorTests(unittest.TestCase):
    def test_masks_prompt_tokens_and_keeps_assistant_tokens(self):
        ex = Stage2SFTExample(
            uid="v",
            window_index=0,
            n_windows=1,
            split="train",
            window_start_s=0.0,
            window_end_s=600.0,
            prompt="prompt tokens here",
            target_text='{"segments": []}',
            target_obj={"segments": []},
            n_segments=0,
            gold_segments=[],
            descriptions=[],
            descriptions_path="fake.descriptions.jsonl",
            time_unit="minute",
        )
        batch = train_cascade.Stage2TextSFTCollator(_FakeTokenizer(), max_length=64)([ex])
        labels = batch["labels"][0]
        self.assertTrue((labels[:3] == -100).all())  # user prompt + assistant marker
        self.assertTrue((labels[3:] != -100).any())

    def test_masks_qwen3_think_scaffold(self):
        # Regression for the qwen3-32b collator crash: the empty <think> block the
        # template injects after the assistant marker must be MASKED (part of the
        # prompt scaffold), not trained on, and the boundary cross-check must pass.
        ex = Stage2SFTExample(
            uid="v",
            window_index=0,
            n_windows=1,
            split="train",
            window_start_s=0.0,
            window_end_s=600.0,
            prompt="prompt tokens here",
            target_text='{"segments": []}',
            target_obj={"segments": []},
            n_segments=0,
            gold_segments=[],
            descriptions=[],
            descriptions_path="fake.descriptions.jsonl",
            time_unit="minute",
        )
        coll = train_cascade.Stage2TextSFTCollator(_FakeThinkTokenizer(), max_length=64)
        batch = coll([ex])
        labels = batch["labels"][0]
        # prompt(2) + assistant opener(1) + think scaffold(2) = 5 tokens masked;
        # the model-agnostic boundary is the prompt-rendering length.
        self.assertTrue((labels[:5] == -100).all())  # incl. the <think></think> block
        self.assertTrue((labels[5:] != -100).any())  # response JSON is trainable

    def test_masks_gemma_style_no_marker(self):
        # Regression for gemma4-31b: the collator must work with a chat template
        # that has NO <|im_start|>assistant marker (Gemma uses <start_of_turn>
        # model). Must not raise, and must mask exactly the prompt rendering.
        ex = Stage2SFTExample(
            uid="v",
            window_index=0,
            n_windows=1,
            split="train",
            window_start_s=0.0,
            window_end_s=600.0,
            prompt="prompt tokens here",
            target_text='{"segments": []}',
            target_obj={"segments": []},
            n_segments=0,
            gold_segments=[],
            descriptions=[],
            descriptions_path="fake.descriptions.jsonl",
            time_unit="minute",
        )
        coll = train_cascade.Stage2TextSFTCollator(_FakeGemma4Tokenizer(), max_length=64)
        batch = coll([ex])  # must NOT raise despite the missing Qwen marker
        labels = batch["labels"][0]
        # prompt rendering = "<start_of_turn>user prompt <end_of_turn>
        # <start_of_turn>model" -> 4 whitespace tokens masked; response after.
        self.assertTrue((labels[:4] == -100).all())
        self.assertTrue((labels[4:] != -100).any())

    def test_terminal_eos_label_survives_when_pad_equals_eos(self):
        ex = Stage2SFTExample(
            uid="v",
            window_index=0,
            n_windows=1,
            split="train",
            window_start_s=0.0,
            window_end_s=600.0,
            prompt="prompt tokens here",
            target_text='{"segments": []}',
            target_obj={"segments": []},
            n_segments=0,
            gold_segments=[],
            descriptions=[],
            descriptions_path="fake.descriptions.jsonl",
            time_unit="minute",
        )
        tok = _FakePadEqualsEosTokenizer()
        batch = train_cascade.Stage2TextSFTCollator(tok, max_length=64)([ex])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        last_real = int(batch["attention_mask"][0].sum().item()) - 1

        self.assertEqual(input_ids[last_real].item(), tok.eos_token_id)
        self.assertEqual(labels[last_real].item(), tok.eos_token_id)
        self.assertNotEqual(labels[last_real].item(), -100)


def _fake_multimodal():
    """gemma-4-like: a text decoder (plain nn.Linear attn+MLP) + vision/audio
    towers with nn.Linear layers + an lm_head. Enough of the CAUSAL_LM surface
    for get_peft_model."""
    import torch.nn as nn

    def block():
        b = nn.Module()
        b.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(4, 4, bias=False))
        b.mlp = nn.Module()
        for p in ("gate_proj", "up_proj", "down_proj"):
            setattr(b.mlp, p, nn.Linear(4, 4, bias=False))
        return b

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([block()])
            # perception towers with nn.Linear contaminants
            self.vision_tower = nn.Module()
            self.vision_tower.encoder = nn.Module()
            self.vision_tower.encoder.down_proj = nn.Linear(4, 4, bias=False)
            self.audio_tower = nn.Module()
            self.audio_tower.q_proj = nn.Linear(4, 4, bias=False)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def prepare_inputs_for_generation(self, *a, **k):
            return {}

        def forward(self, x):
            return x

    return Fake()


@unittest.skipUnless(_HAS_PEFT, "peft not installed (training box only)")
class StripTowerLoraTests(unittest.TestCase):
    """all-linear + _strip_tower_lora must leave a text-decoder-only adapter:
    no vision/audio tower LoRA in the saved state_dict OR config, text LoRA
    still trainable."""

    def _peft_all_linear(self):
        from peft import LoraConfig, get_peft_model

        return get_peft_model(
            _fake_multimodal(),
            LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                       task_type="CAUSAL_LM"),
        )

    def test_strip_removes_tower_lora_keeps_text(self):
        from peft.utils.save_and_load import get_peft_model_state_dict

        pm = self._peft_all_linear()
        n = train_cascade._strip_tower_lora(pm)
        self.assertEqual(n, 2)  # vision_tower.encoder.down_proj + audio_tower.q_proj

        sd = get_peft_model_state_dict(pm)
        tower = [k for k in sd if "vision_tower" in k or "audio_tower" in k]
        self.assertEqual(tower, [], "no tower LoRA weights may be saved")
        text = [k for k in sd if "lora_" in k]
        self.assertTrue(text, "text-decoder LoRA weights must remain")

        # config target_modules must not name the towers either
        tm = pm.peft_config["default"].target_modules
        self.assertFalse(any(train_cascade._is_tower_name(t) for t in tm))

        # text LoRA still trainable; no tower param trainable
        trainable = [n for n, p in pm.named_parameters()
                     if "lora_" in n and p.requires_grad]
        self.assertTrue(trainable)
        self.assertFalse(any(train_cascade._is_tower_name(n) for n in trainable))

    def test_strip_is_noop_on_dense_model(self):
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        class Dense(nn.Module):  # no towers
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList()
                lyr = nn.Module()
                lyr.self_attn = nn.Module()
                lyr.self_attn.q_proj = nn.Linear(4, 4, bias=False)
                lyr.self_attn.v_proj = nn.Linear(4, 4, bias=False)
                self.model.layers.append(lyr)
                self.lm_head = nn.Linear(4, 8, bias=False)

            def get_output_embeddings(self):
                return self.lm_head

            def prepare_inputs_for_generation(self, *a, **k):
                return {}

            def forward(self, x):
                return x

        pm = get_peft_model(
            Dense(),
            LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                       task_type="CAUSAL_LM"),
        )
        self.assertEqual(train_cascade._strip_tower_lora(pm), 0)


class IsTowerNameTests(unittest.TestCase):
    def test_matches_dot_segments_only(self):
        self.assertTrue(train_cascade._is_tower_name("a.vision_tower.b"))
        self.assertTrue(train_cascade._is_tower_name("x.audio_tower.q_proj"))
        self.assertFalse(train_cascade._is_tower_name("model.layers.0.q_proj"))
        # substring, not a full segment -> not a tower
        self.assertFalse(train_cascade._is_tower_name("my_vision_tower_x.p"))


if __name__ == "__main__":
    unittest.main()
