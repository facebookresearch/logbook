"""Test-only adapter that returns canned segmentations for offline runs.

Use for chunk_runner unit/integration tests that should never touch a GPU.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np

from long_audio.inference.models.base import (
    Decoder,
    ModelAdapter,
    ModelOutput,
    TextLLMAdapter,
)


class FakeAdapter(ModelAdapter):
    """Adapter that returns whatever you give it.

    Args:
        responses: list of dicts (schema-shaped) OR raw strings, one per
            ``generate`` call, cycled if exhausted.
        gen_callback: optional callable(audio, sr, prompt, **kwargs) -> dict
            for tests that want to react to the chunk content. Overrides
            ``responses`` when provided.
        latency_s: synthetic per-call latency.
        structured: what supports_structured() returns.
        name: adapter name for logging.
    """

    def __init__(
        self,
        responses: list | None = None,
        gen_callback: Callable | None = None,
        latency_s: float = 0.0,
        structured: bool = True,
        name: str = "fake",
    ):
        self.responses = list(responses) if responses else [{"segments": []}]
        self.gen_callback = gen_callback
        self.latency_s = latency_s
        self._structured = structured
        self.name = name
        self._loaded = False
        self._call_count = 0

    def load(self) -> None:
        self._loaded = True

    def supports_structured(self) -> bool:
        return self._structured

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
        if not self._loaded:
            raise RuntimeError("Adapter not loaded; call load() first.")
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.gen_callback is not None:
            payload = self.gen_callback(
                audio, sample_rate, prompt,
                schema=schema, decoder=decoder,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        else:
            payload = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        if isinstance(payload, dict):
            return ModelOutput(
                raw_text=json.dumps(payload), raw_json=payload, latency_s=self.latency_s,
                metadata={"call": self._call_count, "adapter": self.name},
            )
        return ModelOutput(
            raw_text=str(payload), raw_json=None, latency_s=self.latency_s,
            metadata={"call": self._call_count, "adapter": self.name},
        )


class FakeTextLLMAdapter(TextLLMAdapter):
    """Text-only counterpart to :class:`FakeAdapter`.

    Same responses-list / gen_callback contract; used by Stage-B
    (``cascade.py``) tests so they don't need GOOGLE_API_KEY.
    """

    def __init__(
        self,
        responses: list | None = None,
        gen_callback: Callable | None = None,
        latency_s: float = 0.0,
        structured: bool = True,
        name: str = "fake-text",
    ):
        self.responses = list(responses) if responses else [{"segments": []}]
        self.gen_callback = gen_callback
        self.latency_s = latency_s
        self._structured = structured
        self.name = name
        self._loaded = False
        self._call_count = 0

    def load(self) -> None:
        self._loaded = True

    def supports_structured(self) -> bool:
        return self._structured

    def generate(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Adapter not loaded; call load() first.")
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.gen_callback is not None:
            payload = self.gen_callback(
                prompt, schema=schema, decoder=decoder,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        else:
            payload = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        if isinstance(payload, dict):
            return ModelOutput(
                raw_text=json.dumps(payload), raw_json=payload, latency_s=self.latency_s,
                metadata={"call": self._call_count, "adapter": self.name},
            )
        return ModelOutput(
            raw_text=str(payload), raw_json=None, latency_s=self.latency_s,
            metadata={"call": self._call_count, "adapter": self.name},
        )
