# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Generic inference components for long-audio LLM segmentation.

Dataset-specific label sets, hints, and prompt renderers live in
``long_audio.datasets.<dataset>.{schema,prompt}``. The components here
are dataset-agnostic.

- schema: generic ``make_segmentation_schema(labels, time_unit)`` and
  parsing helpers (``parse_segments``, ``parse_timestamp``).
- prompt: chunk-aware ``render_prompt(chunk, labels, label_hints, ...)``
  template; dataset modules supply ``labels`` and (optionally) hints.
- chunk_runner: shared chunk-iter wrapper that any ``ModelAdapter`` plugs into.
  Free-text outputs from non-structured adapters get parsed via the
  ``json_repair`` library inline — no chunk-wide-guess heuristics.
- models/*: per-model adapters (Qwen2.5-Omni, Qwen3-Omni, AudioFlamingo3, ...).
"""

from long_audio.inference.prompt import (
    ChunkContext,
    ContextMode,
    render_label_block,
    render_prompt,
    render_simple_classification_prompt,
)
from long_audio.inference.schema import (
    TimeUnit,
    make_segmentation_schema,
    parse_segments,
    parse_timestamp,
)

__all__ = [
    "ChunkContext",
    "ContextMode",
    "TimeUnit",
    "make_segmentation_schema",
    "parse_segments",
    "parse_timestamp",
    "render_label_block",
    "render_prompt",
    "render_simple_classification_prompt",
]
