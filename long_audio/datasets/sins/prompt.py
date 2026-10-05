"""SINS-specific prompt renderer.

The generic chunk-aware prompt rendering lives in
``long_audio.inference.prompt``; this module is a thin wrapper that
binds in the SINS label set + per-label hints (both from
``long_audio.datasets.sins.schema``).
"""

from __future__ import annotations

from long_audio.datasets.sins.schema import SINS_LABEL_HINTS, SINS_LABELS
from long_audio.inference.prompt import ChunkContext, ContextMode, render_prompt
from long_audio.inference.schema import TimeUnit


def render_sins_prompt(
    chunk: ChunkContext,
    *,
    context_mode: ContextMode = "none",
    prev_label: str | None = None,
    time_unit: TimeUnit = "second",
    granularity_hint: str = "Use whatever segment granularity matches what you hear.",
    with_description: bool = False,
) -> str:
    """SINS-flavored prompt: full label set + per-label hints."""
    return render_prompt(
        chunk,
        labels=SINS_LABELS,
        label_hints=SINS_LABEL_HINTS,
        context_mode=context_mode,
        prev_label=prev_label,
        time_unit=time_unit,
        granularity_hint=granularity_hint,
        with_description=with_description,
    )
