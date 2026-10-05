"""Adapter for Google Gemini multimodal (audio-in) via ``google-genai``.

Sends audio as a multimodal Part (FLAC/WAV/MP3) and supports
JSON-schema-constrained output via ``response_schema`` +
``response_mime_type="application/json"``. Remote-only — no local
weights. ``GOOGLE_API_KEY`` is read at ``load()`` time; ``model_id`` is
REQUIRED at construction (no default) to force explicit model choice.

Audio is passed through as-is (no truncation). Public entry points:
:class:`GeminiAdapter` (audio-in) and :class:`GeminiTextAdapter`
(text-only, used by the cascade Stage-B).
"""

from __future__ import annotations

import io
import json as _json
import os
import time
from typing import Any

import numpy as np
import soundfile

from long_audio.inference.models.base import (
    Decoder,
    ModelAdapter,
    ModelOutput,
    TextLLMAdapter,
)


class GeminiAdapter(ModelAdapter):
    """Google Gemini via the ``google-genai`` SDK (remote API).

    Args:
        model_id: Gemini model name. REQUIRED — no default. Pass the
            exact id you want (e.g. ``gemini-2.5-flash``).
        timeout_s: per-call timeout passed to the SDK.
    """

    name = "gemini"

    def __init__(
        self,
        model_id: str,
        timeout_s: int = 180,
        thinking_budget: int = 0,
    ):
        self.model_id = model_id
        self.timeout_s = timeout_s
        # Extended-reasoning cap. 0 = thinking off (default).
        # Positive int = allow up to N reasoning tokens.
        # CAVEAT: thinking tokens count against max_output_tokens on the
        # current SDK — callers using thinking_budget>0 should inflate
        # max_new_tokens to preserve visible-output headroom
        # (rule of thumb: max_new_tokens = expected_output + thinking_budget).
        self.thinking_budget = thinking_budget
        self._client = None
        self._types_mod = None

    def load(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY env var is required for GeminiAdapter. "
                "Set it from your Google AI Studio account."
            )

        from google import genai
        from google.genai import types

        self._types_mod = types
        self._client = genai.Client(**self._client_kwargs(api_key=api_key, types_mod=types))

    def _client_kwargs(self, *, api_key: str, types_mod) -> dict[str, Any]:
        """Hook for subclasses to inject custom endpoint / API version.

        Wires ``self.timeout_s`` to ``HttpOptions.timeout`` so the SDK
        raises after ``timeout_s`` seconds instead of hanging indefinitely
        on a stalled socket (SDK default is effectively infinite; SINS
        runs have observed 40+ min TCP-read hangs). Subclasses that
        override this must preserve the timeout wiring or set their own.
        """
        return {
            "api_key": api_key,
            "http_options": types_mod.HttpOptions(
                timeout=self.timeout_s * 1000,  # SDK expects milliseconds
            ),
        }

    def _invoke(self, prompt: str, audio_part, config):
        """Hook for subclasses to wrap the API call (e.g. with retries).

        The public adapter just calls the SDK once; transient-error handling
        is the user's problem (the SDK already does some internal retries).
        Subclasses can override to add tenacity / a custom retry policy.
        """
        return self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt, audio_part],
            config=config,
        )

    def unload(self) -> None:
        # No GPU / weights; just drop the client reference.
        self._client = None
        self._types_mod = None

    def supports_structured(self) -> bool:
        return True

    @staticmethod
    def _encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode mono float32 audio in [-1, 1] to in-memory FLAC bytes."""
        buf = io.BytesIO()
        soundfile.write(buf, audio, sample_rate, format="FLAC")
        return buf.getvalue()

    def _build_config(
        self,
        *,
        schema: dict | None,
        decoder: Decoder,
        max_new_tokens: int,
        temperature: float,
    ):
        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_new_tokens,
            # thinking_budget from the adapter kwarg (default 0 = off).
            # See __init__ for the accounting caveat re: max_output_tokens.
            # include_thoughts=True is required for the SDK to return
            # reasoning parts (part.thought=True); without it the
            # response contains no thought parts and thinking_trace
            # cannot be recovered from the API response.
            "thinking_config": self._types_mod.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
        }
        if decoder == "structured" and schema is not None:
            # `schema` is the vLLM-style {"name", "schema", "strict"} wrapper
            # produced by `make_segmentation_schema`. The actual JSON Schema
            # lives under .schema.
            cfg_kwargs["response_schema"] = schema["schema"]
            cfg_kwargs["response_mime_type"] = "application/json"
        return self._types_mod.GenerateContentConfig(**cfg_kwargs)

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
        if self._client is None:
            raise RuntimeError("Call load() first.")
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}")
        audio = audio.astype(np.float32, copy=False)

        flac_bytes = self._encode_flac(audio, sample_rate)
        audio_part = self._types_mod.Part.from_bytes(
            data=flac_bytes, mime_type="audio/flac"
        )
        config = self._build_config(
            schema=schema, decoder=decoder,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        t0 = time.time()
        response = self._invoke(prompt, audio_part, config)
        attempt = getattr(response, "_long_audio_attempts", 1)
        latency = time.time() - t0

        raw_text = (response.text or "") if hasattr(response, "text") else ""
        raw_json: dict | None = None
        if decoder == "structured" and schema is not None:
            try:
                raw_json = _json.loads(raw_text)
            except (ValueError, TypeError):
                raw_json = None

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        thoughts_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        thinking_trace = _extract_gemini_thinking_trace(candidates)

        return ModelOutput(
            raw_text=raw_text,
            raw_json=raw_json,
            latency_s=latency,
            metadata={
                "model_id": self.model_id,
                "decoder": decoder,
                "structured_supported": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thoughts_tokens": thoughts_tokens,
                "thinking_budget": self.thinking_budget,
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "retry_attempts": attempt,
            },
            thinking_trace=thinking_trace,
        )


def _extract_gemini_thinking_trace(candidates) -> str | None:
    """Pull the reasoning text out of a Gemini response's parts.

    In the google-genai SDK, thinking output shows up as ``Part``
    objects whose ``.thought`` attribute is truthy. Concatenate their
    ``.text`` across all candidates. Returns None if the response
    contained no thought parts (thinking was off, or the model didn't
    surface any).
    """
    parts_out: list[str] = []
    for cand in candidates or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                t = getattr(part, "text", None)
                if t:
                    parts_out.append(t)
    return "".join(parts_out) if parts_out else None


class GeminiTextAdapter(TextLLMAdapter):
    """Text-only Gemini via the ``google-genai`` SDK.

    Same SDK, same auth, same response-shape handling as
    :class:`GeminiAdapter` — just without the audio Part. Used by
    ``long_audio.inference.cascade`` to consume Stage-A descriptions
    and emit segmentation JSON.

    Args:
        model_id: Gemini model name. REQUIRED — same loud-fail discipline
            as :class:`GeminiAdapter`.
        timeout_s: per-call timeout passed to the SDK.
    """

    name = "gemini-text"

    def __init__(
        self,
        model_id: str,
        timeout_s: int = 180,
        thinking_budget: int = 0,
    ):
        self.model_id = model_id
        self.timeout_s = timeout_s
        # See GeminiAdapter.__init__ for the accounting caveat.
        self.thinking_budget = thinking_budget
        self._client = None
        self._types_mod = None

    def load(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY env var is required for GeminiTextAdapter. "
                "Set it from your Google AI Studio account."
            )
        from google import genai
        from google.genai import types

        self._types_mod = types
        self._client = genai.Client(**self._client_kwargs(api_key=api_key, types_mod=types))

    def _client_kwargs(self, *, api_key: str, types_mod) -> dict[str, Any]:
        """Hook for subclasses to inject custom endpoint / API version.

        Wires ``self.timeout_s`` to ``HttpOptions.timeout`` so the SDK
        raises after ``timeout_s`` seconds instead of hanging indefinitely
        on a stalled socket (SDK default is effectively infinite; SINS
        runs have observed 40+ min TCP-read hangs). Subclasses that
        override this must preserve the timeout wiring or set their own.
        """
        return {
            "api_key": api_key,
            "http_options": types_mod.HttpOptions(
                timeout=self.timeout_s * 1000,  # SDK expects milliseconds
            ),
        }

    def unload(self) -> None:
        self._client = None
        self._types_mod = None

    def supports_structured(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        if self._client is None:
            raise RuntimeError("Call load() first.")

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_new_tokens,
            "thinking_config": self._types_mod.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
        }
        if decoder == "structured" and schema is not None:
            cfg_kwargs["response_schema"] = schema["schema"]
            cfg_kwargs["response_mime_type"] = "application/json"
        config = self._types_mod.GenerateContentConfig(**cfg_kwargs)

        t0 = time.time()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt],
            config=config,
        )
        latency = time.time() - t0

        raw_text = (response.text or "") if hasattr(response, "text") else ""
        raw_json: dict | None = None
        if decoder == "structured" and schema is not None:
            try:
                raw_json = _json.loads(raw_text)
            except (ValueError, TypeError):
                raw_json = None

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        thoughts_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        thinking_trace = _extract_gemini_thinking_trace(candidates)

        return ModelOutput(
            raw_text=raw_text,
            raw_json=raw_json,
            latency_s=latency,
            metadata={
                "model_id": self.model_id,
                "decoder": decoder,
                "structured_supported": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thoughts_tokens": thoughts_tokens,
                "thinking_budget": self.thinking_budget,
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
            },
            thinking_trace=thinking_trace,
        )
