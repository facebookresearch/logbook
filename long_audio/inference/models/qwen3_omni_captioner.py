"""Adapter for ``Qwen/Qwen3-Omni-30B-A3B-Captioner``.

Same vLLM-based architecture as ``Qwen3-Omni-30B-A3B-Instruct``; only
the HF model_id differs. Subclass the base adapter; override the default
model_id (and the ``name`` registry slug) so the run dirs / metadata
record it as a distinct arm.

Intended slot: cascade Stage A (per-clip captioning). The captioner
variant was post-trained specifically for short-audio dense captioning.
"""

from __future__ import annotations

from long_audio.inference.models.qwen3_omni import Qwen3OmniVLLMAdapter


DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Captioner"


class Qwen3OmniCaptionerVLLMAdapter(Qwen3OmniVLLMAdapter):
    """Qwen3-Omni-30B-A3B-Captioner via mainstream vLLM 0.21+ (V0 engine).

    Same constructor surface as :class:`Qwen3OmniVLLMAdapter`; only the
    default ``model_id`` differs.
    """

    name = "qwen3-omni-captioner"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
