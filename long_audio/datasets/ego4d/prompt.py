"""Ego4D-specific prompt helper — ATUS class block + thin renderer wrapper.

The system instruction and task/output structure are shared with all other
datasets in ``long_audio.inference.prompt``. Only the label set + per-label
descriptions are Ego4D-specific.
"""

from __future__ import annotations

from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
from long_audio.inference.prompt import ChunkContext, ContextMode, render_prompt
from long_audio.inference.schema import TimeUnit


def render_ego4d_prompt(
    chunk: ChunkContext,
    *,
    context_mode: ContextMode = "none",
    prev_label: str | None = None,
    time_unit: TimeUnit = "second",
    granularity_hint: str = "Use whatever segment granularity matches what you hear.",
    with_description: bool = False,
) -> str:
    """Ego4D-flavored prompt: full ATUS class set + per-class descriptions."""
    return render_prompt(
        chunk,
        labels=ATUS_LABELS,
        label_hints=ATUS_HINTS,
        context_mode=context_mode,
        prev_label=prev_label,
        time_unit=time_unit,
        granularity_hint=granularity_hint,
        with_description=with_description,
    )
