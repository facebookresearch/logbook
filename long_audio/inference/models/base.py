# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Abstract adapters for chunk-level inference.

``ModelAdapter`` wraps a single audio LLM (Qwen3-Omni via transformers,
AudioFlamingo3 via NVIDIA's client, …). ``TextLLMAdapter`` wraps a
single text-in / text-out LLM (Gemini text, OpenAI, local LLM, …), used
by the Stage-B cascade where descriptions replace audio.

The runners (``chunk_runner.py``, ``cascade.py``) talk only to these
interfaces, which keeps orchestration model-agnostic.

DECODING MODES
``decoder="structured"`` asks the adapter to constrain output to the
JSON schema (vLLM grammar-guided / outlines / lm-format-enforcer / Gemini
response_schema). Adapters that cannot do this should return False from
``supports_structured()``; callers asking for structured against an
unstructured adapter raise loudly (no silent fallback).

``decoder="freeform"`` returns whatever the model emits as text; the
runner pipes it through ``json_repair`` downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Decoder = Literal["structured", "freeform"]


@dataclass
class ModelOutput:
    """Raw output from a single ``generate`` call."""

    raw_text: str
    raw_json: dict | None = None
    # Per-call timing info, populated by the adapter if cheap to measure.
    latency_s: float | None = None
    # Free-form, e.g. {"prompt_tokens": ..., "completion_tokens": ...}.
    metadata: dict = field(default_factory=dict)
    # Extended-reasoning trace when the model was run with thinking mode
    # (Gemini `thinking_budget>0`, Qwen 3 / Gemma `enable_thinking=True`).
    # Adapters that extract the reasoning block populate this; downstream
    # tooling uses it for cost/quality diagnostics without polluting the
    # user-visible ``raw_text``. None when thinking was off or the
    # adapter did not parse a trace out of the response.
    thinking_trace: str | None = None


class ModelAdapter(ABC):
    """Thin wrapper around one model + framework.

    Subclasses must implement ``load``, ``generate``, ``supports_structured``,
    and ``name``. ``unload`` is optional; default is a no-op. Adapters should
    be safe to ``load`` once and reuse across many ``generate`` calls.
    """

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None:
        """Load weights and prepare for inference. Idempotent."""

    def unload(self) -> None:
        """Free GPU/CPU resources. Default: no-op."""

    @abstractmethod
    def supports_structured(self) -> bool:
        """Whether this adapter can constrain output to a JSON schema."""

    @abstractmethod
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        """Run one inference call.

        Args:
            audio: shape (n_samples,) float32 mono audio in [-1, 1].
            sample_rate: samples per second.
            prompt: full text prompt (see long_audio.inference.prompt).
            schema: vLLM-style ``{"name", "schema", "strict"}`` dict. Ignored
                when ``decoder="freeform"`` or ``supports_structured() is False``.
            decoder: ``"structured"`` to apply the schema; ``"freeform"`` for
                free-form text.
            max_new_tokens: cap on the model's output token count.
            temperature: sampling temperature; 0.0 = greedy.
        """

    def generate_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int,
        prompts: list[str],
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[ModelOutput]:
        """Run N inference calls. Adapters that support true GPU batching
        (vLLM ``llm.generate([req,...])``, HF ``processor(text=[..], audio=[..])``)
        override this; the default loops ``generate()`` so every adapter
        satisfies the API without extra work. Stage-A captioning at 10-s
        chunks is where the batched path actually pays off — direct E2E
        at 10-min chunks already amortizes per-chunk overhead.
        """
        if len(audios) != len(prompts):
            raise ValueError(
                f"generate_batch: len(audios)={len(audios)} != len(prompts)={len(prompts)}"
            )
        return [
            self.generate(
                a, sample_rate, p,
                schema=schema, decoder=decoder,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            for a, p in zip(audios, prompts)
        ]


class TextLLMAdapter(ABC):
    """Thin wrapper around one text-in / text-out LLM.

    Used by the Stage-B cascade: ``describe.py`` runs an audio
    ``ModelAdapter`` per chunk to produce one-sentence descriptions,
    then ``cascade.py`` hands a window of timestamped descriptions to a
    ``TextLLMAdapter`` that returns the segmentation JSON.

    The interface mirrors ``ModelAdapter`` minus the audio params, so
    callers can swap Gemini → Claude → OpenAI → a local LLM with no
    orchestration changes.
    """

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None:
        """Set up SDK client / load weights. Idempotent."""

    def unload(self) -> None:
        """Free resources. Default: no-op."""

    @abstractmethod
    def supports_structured(self) -> bool:
        """Whether this adapter can constrain output to a JSON schema."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        """Run one text-only inference call.

        Same contract as ``ModelAdapter.generate`` minus audio: returns
        a ``ModelOutput`` whose ``raw_text`` (and, for structured
        decoders, ``raw_json``) carry the model's reply.
        """
