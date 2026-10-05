"""Tests for the Qwen3-Omni fine-tuning wiring (long_audio.training.model).

These are import-light: they exercise the LoRA-targeting / projector-targeting /
param-selection / device-map logic against a tiny synthetic module tree that
mirrors the real thinker's attribute layout (``audio_tower.proj1/proj2``, an
audio encoder attention with ``q/k/v_proj``, ``visual``, and
``model.layers.<i>.self_attn.(q|k|v|o)_proj``). No 30B checkpoint, no GPU. Tests
that need PEFT itself (``build_lora_config``) are guarded with ``skipUnless`` —
peft only exists on the training box.
"""

from __future__ import annotations

import importlib.util
import unittest

import torch.nn as nn

from long_audio.training.model import (
    LORA_TARGET_REGEX,
    PROJECTOR_MODULE_NAMES,
    build_lora_config,
    is_lora_param,
    is_projector_param,
    lora_targets_module,
    select_lora_target_names,
    thinker_device_map,
    trainable_parameter_summary,
    LoraSettings,
)

_HAS_PEFT = importlib.util.find_spec("peft") is not None


class _LLMAttn(nn.Module):
    """Text-LLM attention: the LoRA target. Includes fake lora_A/lora_B to
    simulate the post-``get_peft_model`` state without importing peft."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.lora_A = nn.Linear(8, 4, bias=False)
        self.lora_B = nn.Linear(4, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _LLMAttn()
        self.mlp = nn.Linear(8, 8, bias=False)


class _TextModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])


class _AudioAttn(nn.Module):
    """Audio encoder attention: q/k/v_proj collide by NAME with the LLM but
    must NOT be LoRA'd (uses out_proj, not o_proj)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 4)


class _AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_out = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([_AudioAttn()])
        self.proj1 = nn.Linear(4, 4)   # projector part 1
        self.proj2 = nn.Linear(4, 6)   # projector part 2 (-> LLM hidden)


class _Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


class _FakeThinker(nn.Module):
    """Mirrors Qwen3OmniMoeThinkerForConditionalGeneration's attribute paths."""

    def __init__(self):
        super().__init__()
        self.audio_tower = _AudioTower()
        self.visual = _Vision()
        self.model = _TextModel()
        self.lm_head = nn.Linear(8, 16, bias=False)


class ParamPredicateTests(unittest.TestCase):
    def test_is_projector_param(self):
        self.assertTrue(is_projector_param("audio_tower.proj1.weight"))
        self.assertTrue(is_projector_param("base_model.model.audio_tower.proj2.bias"))
        # PEFT modules_to_save wrapping: the trainable copy must still match...
        self.assertTrue(is_projector_param(
            "base_model.model.audio_tower.proj1.modules_to_save.default.weight"
        ))
        # ...and so does the frozen original copy (the requires_grad filter in
        # trainable_parameter_summary is what excludes it from the counts).
        self.assertTrue(is_projector_param(
            "base_model.model.audio_tower.proj2.original_module.weight"
        ))
        self.assertFalse(is_projector_param("audio_tower.conv_out.weight"))
        self.assertFalse(is_projector_param("model.layers.0.self_attn.q_proj.weight"))

    def test_is_lora_param(self):
        self.assertTrue(is_lora_param("model.layers.0.self_attn.q_proj.lora_A.default.weight"))
        self.assertFalse(is_lora_param("model.layers.0.self_attn.q_proj.weight"))

    def test_projector_module_names(self):
        self.assertEqual(
            PROJECTOR_MODULE_NAMES, ["audio_tower.proj1", "audio_tower.proj2"]
        )


class LoraTargetRegexTests(unittest.TestCase):
    def test_regex_hits_llm_attn_only(self):
        thinker = _FakeThinker()
        matched = [
            name for name, _ in thinker.named_modules() if lora_targets_module(name)
        ]
        # Exactly the 4 attn projections in each of the 2 LLM layers = 8.
        self.assertEqual(len(matched), 8)
        for name in matched:
            self.assertRegex(name, r"^model\.layers\.\d+\.self_attn\.[qkvo]_proj$")

    def test_regex_misses_audio_and_vision(self):
        thinker = _FakeThinker()
        for name, _ in thinker.named_modules():
            if name.startswith("audio_tower") or name.startswith("visual"):
                self.assertFalse(
                    lora_targets_module(name),
                    msg=f"{name} must not be a LoRA target",
                )

    def test_regex_does_not_match_lora_submodules_or_mlp(self):
        self.assertFalse(lora_targets_module("model.layers.0.self_attn.lora_A"))
        self.assertFalse(lora_targets_module("model.layers.0.mlp"))
        # And it's a fullmatch (no partial-name leakage).
        self.assertFalse(lora_targets_module("x.model.layers.0.self_attn.q_proj"))
        self.assertEqual(
            LORA_TARGET_REGEX,
            r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        )

    def test_select_lora_target_names_resolves_explicit_list(self):
        # We hand PEFT the resolved list (not the regex string), so this is the
        # exact set of modules LoRA will wrap. Must be the 8 LLM-attn projs only,
        # sorted, with no audio/vision leakage.
        names = select_lora_target_names(_FakeThinker())
        self.assertEqual(
            names,
            sorted([
                f"model.layers.{i}.self_attn.{p}"
                for i in (0, 1) for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]),
        )
        self.assertTrue(all(n.startswith("model.layers.") for n in names))


class TrainableSummaryTests(unittest.TestCase):
    def test_summary_splits_lora_projector_other(self):
        thinker = _FakeThinker()
        # Post-peft state: lora + projector trainable, nothing else.
        for n, p in thinker.named_parameters():
            p.requires_grad_(is_lora_param(n) or is_projector_param(n))
        summ = trainable_parameter_summary(thinker)
        self.assertEqual(summ["other"], 0)
        self.assertGreater(summ["lora"], 0)
        self.assertGreater(summ["projector"], 0)
        self.assertEqual(summ["trainable"], summ["lora"] + summ["projector"])
        self.assertAlmostEqual(
            summ["trainable_pct"], 100.0 * summ["trainable"] / summ["total"]
        )

    def test_summary_flags_encoder_leak(self):
        thinker = _FakeThinker()
        # Simulate LoRA leaking onto the audio encoder attention.
        for n, p in thinker.named_parameters():
            p.requires_grad_(False)
        thinker.audio_tower.layers[0].q_proj.weight.requires_grad_(True)
        summ = trainable_parameter_summary(thinker)
        self.assertGreater(summ["other"], 0)  # the guard in train_e2e trips on this


@unittest.skipUnless(_HAS_PEFT, "peft not installed (training box only)")
class LoraConfigTests(unittest.TestCase):
    """The projector must ride in the adapter via ``modules_to_save`` (so
    best-keeping restores it), and LoRA must target the resolved attention list
    (a regex string is unsafe under CAUSAL_LM — see model.py)."""

    def test_lora_config_uses_explicit_list_and_modules_to_save(self):
        names = select_lora_target_names(_FakeThinker())
        cfg = build_lora_config(LoraSettings(r=8, alpha=16, dropout=0.0), names)
        self.assertEqual(list(cfg.target_modules), names)
        self.assertEqual(list(cfg.modules_to_save), PROJECTOR_MODULE_NAMES)
        self.assertEqual(cfg.r, 8)
        self.assertEqual(cfg.lora_alpha, 16)

    def test_empty_target_modules_raises(self):
        with self.assertRaises(ValueError):
            build_lora_config(LoraSettings(), [])


class DeviceMapTests(unittest.TestCase):
    def test_strips_thinker_prefix_and_drops_talker(self):
        class _Top:
            hf_device_map = {
                "thinker.audio_tower": 0,
                "thinker.model.layers.0": 0,
                "thinker.model.layers.1": 1,
                "thinker.lm_head": 1,
                "talker": 1,
                "code2wav": 1,
            }

        sub = thinker_device_map(_Top())
        self.assertEqual(
            sub,
            {
                "audio_tower": 0,
                "model.layers.0": 0,
                "model.layers.1": 1,
                "lm_head": 1,
            },
        )
        # No talker/code2wav keys leak through.
        self.assertNotIn("talker", sub)

    def test_none_when_no_device_map(self):
        class _Top:
            pass

        self.assertIsNone(thinker_device_map(_Top()))


if __name__ == "__main__":
    unittest.main()
