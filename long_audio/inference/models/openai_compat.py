# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter for OpenAI Chat Completions and any OpenAI-compatible endpoint.

Covers OpenAI, vLLM's OpenAI server, and other Chat-Completions-compatible
LLM endpoints. Concrete deployments differ only in ``model_id``,
``base_url`` (default: OpenAI's public API), and API key env var
(default: ``OPENAI_API_KEY``). ``OPENAI_BASE_URL`` env is honored when
no constructor arg is given.

``model_id`` is REQUIRED at construction (no default) to force explicit
model choice.
"""

from __future__ import annotations

import json as _json
import os
import time
from typing import Any

from long_audio.inference.models.base import (
    Decoder,
    ModelOutput,
    TextLLMAdapter,
)


class OpenAICompatTextAdapter(TextLLMAdapter):
    """Text-only Chat Completions client against any OpenAI-compatible
    endpoint.

    Args:
        model_id: Model name that the target server routes on. REQUIRED
            — no default. Pass the exact id you want.
        base_url: Override the endpoint URL. Falls back to ``OPENAI_BASE_URL``
            env var, then to the SDK's default (public api.openai.com).
        timeout_s: per-call timeout passed to the SDK.
    """

    name = "openai-compat-text"

    # GPT-5 / reasoning-model family reject any `temperature != 1` (their
    # default). Subclasses that target such models should set this False;
    # generate() will then omit the field and let the server use its own
    # default. Standard chat models (GPT-4o etc.) accept any T; leave True.
    SUPPORTS_TEMPERATURE: bool = True

    # Cap the reasoning budget for reasoning-model families (o1/o3/gpt-5).
    # None => don't send the field (standard chat models). Subclasses that
    # target reasoning models should set this so hidden reasoning tokens
    # don't silently eat the max_completion_tokens budget and leave
    # user-visible text empty — parallel to Gemini's
    # ThinkingConfig(thinking_budget=0). Accepted values are
    # model-dependent (commonly ``'none'`` / ``'low'`` / ``'medium'`` /
    # ``'high'``).
    REASONING_EFFORT: str | None = None

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        timeout_s: int = 120,
    ):
        self.model_id = model_id
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._client = None

    # Hook for subclasses to change which env var(s) hold the key.
    def _resolve_api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY env var is required for {type(self).__name__}."
            )
        return api_key

    # Hook for subclasses to change which env var holds the base URL default.
    def _resolve_base_url(self) -> str | None:
        return self.base_url or os.environ.get("OPENAI_BASE_URL")

    # Hook for subclasses to wrap the API call (e.g. with retries). The
    # public adapter calls the SDK once; transient-error handling is the
    # user's problem (the SDK already does some internal retries).
    def _invoke(self, messages: list[dict], **create_kwargs):
        return self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **create_kwargs,
        )

    def load(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": self._resolve_api_key(),
            "timeout": self.timeout_s,
        }
        base_url = self._resolve_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def unload(self) -> None:
        self._client = None

    def supports_structured(self) -> bool:
        # OpenAI Chat Completions supports json_schema-constrained output
        # via `response_format={"type": "json_schema", ...}`. Compatible
        # servers (vLLM etc.) mostly do too.
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

        messages = [{"role": "user", "content": prompt}]
        create_kwargs: dict[str, Any] = {
            # Uses ``max_completion_tokens``; subclass and override
            # generate() if you need to hit an older server that still
            # requires ``max_tokens``.
            "max_completion_tokens": max_new_tokens,
        }
        if self.SUPPORTS_TEMPERATURE:
            create_kwargs["temperature"] = temperature
        if self.REASONING_EFFORT is not None:
            create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
        if decoder == "structured" and schema is not None:
            # `schema` is the vLLM-style {"name", "schema", "strict"} wrapper
            # produced by `make_segmentation_schema`. Chat Completions'
            # json_schema response_format takes the same three fields.
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "output"),
                    "schema": schema["schema"],
                    "strict": bool(schema.get("strict", True)),
                },
            }

        t0 = time.time()
        response = self._invoke(messages, **create_kwargs)
        attempt = getattr(response, "_long_audio_attempts", 1)
        latency = time.time() - t0

        choice = response.choices[0] if response.choices else None
        raw_text = (choice.message.content or "") if choice and choice.message else ""
        raw_json: dict | None = None
        if decoder == "structured" and schema is not None:
            try:
                raw_json = _json.loads(raw_text)
            except (ValueError, TypeError):
                raw_json = None

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

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
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "retry_attempts": attempt,
            },
        )
