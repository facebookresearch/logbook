"""CPU tests for the cascade Stage-B LoRA wiring in vllm_text.py.

Only the LoRA-rank resolution + adapter bookkeeping are exercised — vLLM
itself is imported lazily inside ``load()``, so these run without GPU/vLLM.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

try:
    from long_audio.inference.models.vllm_text import (
        Gemma4_31BTextAdapter,
        Qwen3_32BTextAdapter,
        _resolve_lora_rank,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@unittest.skipUnless(HAS_VLLM, "vllm not installed")
class ResolveLoraRankTests(unittest.TestCase):
    def _adapter_dir(self, r):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": r, "peft_type": "LORA"}, f)
        return d

    def test_reads_r_from_config(self):
        self.assertEqual(_resolve_lora_rank(self._adapter_dir(32), None), 32)

    def test_override_wins(self):
        self.assertEqual(_resolve_lora_rank(self._adapter_dir(32), 8), 8)

    def test_missing_config_falls_back_to_64(self):
        self.assertEqual(_resolve_lora_rank(tempfile.mkdtemp(), None), 64)


@unittest.skipUnless(HAS_VLLM, "vllm not installed")
class VLLMTextAdapterLoraInitTests(unittest.TestCase):
    def test_no_adapter_leaves_lora_unset(self):
        a = Qwen3_32BTextAdapter()
        self.assertIsNone(a.adapter_dir)
        self.assertIsNone(a.max_lora_rank)
        self.assertIsNone(a._lora_request)

    def test_adapter_dir_resolves_rank(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 16}, f)
        a = Qwen3_32BTextAdapter(adapter_dir=d)
        self.assertEqual(a.adapter_dir, d)
        self.assertEqual(a.max_lora_rank, 16)  # >= adapter r, for vLLM engine

    def test_max_lora_rank_override(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 16}, f)
        a = Qwen3_32BTextAdapter(adapter_dir=d, max_lora_rank=64)
        self.assertEqual(a.max_lora_rank, 64)


@unittest.skipUnless(HAS_VLLM, "vllm not installed")
class HfOverridesTests(unittest.TestCase):
    """Multimodal bases force the full arch via hf_overrides; dense bases don't.

    Exercises _build_llm_kwargs (no vLLM import) so it runs on CPU.
    """

    def test_gemma4_forces_conditional_generation_arch(self):
        kwargs = Gemma4_31BTextAdapter()._build_llm_kwargs()
        self.assertEqual(
            kwargs["hf_overrides"],
            {"architectures": ["Gemma4ForConditionalGeneration"]},
        )

    def test_dense_base_has_no_hf_overrides(self):
        kwargs = Qwen3_32BTextAdapter()._build_llm_kwargs()
        self.assertNotIn("hf_overrides", kwargs)

    def test_constructor_override_wins(self):
        override = {"architectures": ["SomeOtherArch"]}
        a = Gemma4_31BTextAdapter(hf_overrides=override)
        self.assertEqual(a._build_llm_kwargs()["hf_overrides"], override)
        # And a dense base can be forced multimodal from the caller too.
        b = Qwen3_32BTextAdapter(hf_overrides=override)
        self.assertEqual(b._build_llm_kwargs()["hf_overrides"], override)

    def test_adapter_dir_adds_lora_kwargs(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 32}, f)
        kwargs = Gemma4_31BTextAdapter(adapter_dir=d)._build_llm_kwargs()
        self.assertTrue(kwargs["enable_lora"])
        self.assertEqual(kwargs["max_lora_rank"], 32)
        # Both the LoRA wiring AND the forced multimodal arch must coexist —
        # that combination is the whole point of this fix.
        self.assertIn("hf_overrides", kwargs)


if __name__ == "__main__":
    unittest.main()
