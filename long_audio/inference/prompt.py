"""Generic chunk-aware prompt template for long-audio segmentation.

Renders a chunk-aware prompt with:
  - chunk metadata header (wall-clock range only; chunk index/count
    intentionally omitted so the model treats each chunk independently
    and prompts are reusable across single- vs multi-chunk runs)
  - optional minimal prior-context line ("Immediately before this chunk,
    the activity was: <label>"; deliberately label-only — no timestamps,
    no cumulative totals — so the model gets a tiny continuity hint
    without any leaked statistics from prior predictions)
  - label set with one-line definitions (caller supplies the dataset's
    label tuple + optional per-label hint dict)
  - output-format spec matching schema.make_segmentation_schema

Dataset-specific label tuples + hint dicts live with their dataset
schema module (e.g. ``long_audio.datasets.sins.schema:SINS_LABEL_HINTS``).

TIMESTAMP CONVENTION
The model emits timestamps relative to the start of the *current chunk*
(00:00 = chunk start). The chunk runner adds the chunk's absolute
``chunk_start_s`` to lift each segment onto the wall-clock timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from long_audio.inference.schema import TimeUnit


@dataclass
class ChunkContext:
    """Per-chunk metadata used in the prompt header."""

    chunk_index: int  # 0-based
    n_chunks: int
    chunk_start_wallclock_s: float  # seconds since dataset T0
    chunk_end_wallclock_s: float
    t0_iso: str = "1970-01-01T00:00:00Z"  # override per-dataset
    dataset_name: str = "dataset"


def _fmt_mmss(seconds: float) -> str:
    """Format a non-negative number of seconds as MM:SS, capped at 99:59."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m > 99:
        m = 99
        s = 59
    return f"{m:02d}:{s:02d}"


def render_label_block(
    labels: tuple[str, ...],
    label_hints: dict[str, str] | None = None,
) -> str:
    """Render the label set with optional one-line hints."""
    lines = ["Allowed event classes (use these labels exactly):"]
    for lbl in labels:
        hint = (label_hints or {}).get(lbl, "")
        if hint:
            lines.append(f"  - {lbl}: {hint}")
        else:
            lines.append(f"  - {lbl}")
    return "\n".join(lines)


SYSTEM_INSTRUCTION = (
    "You are an expert audio analyst for wearable-device recordings of a single "
    "person's activities. You receive long audio recordings, "
    "to produce a temporal segmentation of the wearer's activities."
)

SYSTEM_INSTRUCTION_CASCADE = (
    "You are an expert audio analyst for wearable-device recordings of a single "
    "person's activities. You receive a series of timestamped descriptions, "
    "to produce a temporal segmentation of the wearer's activities."
)


# Context modes the chunk runner can pass through:
#   "none" — no prior-context line. Every chunk is independent.
#   "prev" — single line: the immediately previous predicted label only,
#            with no timestamps, no totals. Minimal continuity hint.
ContextMode = Literal["none", "prev"]


def _render_prev_block(prev_label: str | None) -> str:
    """One-line continuity hint for ``context_mode == 'prev'``."""
    if not prev_label:
        return ""
    return (
        "\n\n# Prior context\n"
        f"Immediately before this chunk, the predicted activity was: {prev_label}."
    )


def render_simple_classification_prompt(
    chunk: ChunkContext,
    labels: tuple[str, ...],
) -> str:
    """Short Q-A prompt for models that struggle with structured-JSON output.

    Asks the model to name the most prominent activity in the clip from the
    allowed label set. The chunk runner's free-text label-scan fallback then
    converts the response into a single chunk-spanning segment.
    """
    label_list = ", ".join(labels)
    return (
        "Listen to this short audio clip recorded by a wearable device in a "
        "private home. Identify the single dominant human activity in the clip.\n\n"
        f"Choose exactly one label from this list: {label_list}.\n"
        "Answer with just the label word, lowercase, no extra text. "
        "If nothing distinctive is happening, answer 'absence'."
    )


def _chunk_durations(chunk: ChunkContext) -> tuple[float, int]:
    s = max(0.0, chunk.chunk_end_wallclock_s - chunk.chunk_start_wallclock_s)
    return s, int(round(s / 60.0))


_DESCRIPTION_BULLET = (
    '  - "description": 1-2 sentences, present tense, naming the concrete '
    "actions and objects in this segment. Add detail beyond the event label; "
    "infer plausible specifics when evidence is partial. Refer to the "
    "recording subject as \"the wearer\".\n"
)


def _render_output_spec(
    time_unit: TimeUnit,
    chunk_duration_min: int,
    *,
    anchor: str,
    with_description: bool = False,
) -> str:
    """Output-format block — identical between E2E and cascade Stage B
    except for the anchor word ("chunk" for E2E, "window" for Stage B).

    ``with_description=True`` appends a ``description`` bullet to the
    per-segment field list (matches ``make_segmentation_schema(...,
    with_description=True)``).
    """
    desc_bullet = _DESCRIPTION_BULLET if with_description else ""
    if time_unit == "second":
        return (
            "\n\n# Output format\n"
            'Return a JSON object with one key, "segments", whose value is an '
            "array of objects each with:\n"
            f'  - "start": MM:SS *relative to the start of this {anchor}* '
            f"(00:00 = {anchor} start)\n"
            f'  - "end":   MM:SS *relative to the start of this {anchor}*\n'
            '  - "event": one of the allowed event classes below\n'
            f"{desc_bullet}"
            "Spans must be non-overlapping and chronologically ordered. "
            "Do not output anything outside the JSON object."
        )
    return (
        "\n\n# Output format\n"
        'Return a JSON object with one key, "segments", whose value is an '
        "array of objects each with:\n"
        f'  - "start": integer minute (0..{chunk_duration_min}) '
        f'*relative to the start of this {anchor}* ("0" = {anchor} start)\n'
        f'  - "end":   integer minute (0..{chunk_duration_min}) '
        f"*relative to the start of this {anchor}*\n"
        '  - "event": one of the allowed event classes below\n'
        f"{desc_bullet}"
        "Each end must be strictly greater than its start. Spans must "
        "be non-overlapping and chronologically ordered. Do not output "
        "anything outside the JSON object."
    )


def render_prompt(
    chunk: ChunkContext,
    labels: tuple[str, ...],
    *,
    label_hints: dict[str, str] | None = None,
    context_mode: ContextMode = "none",
    prev_label: str | None = None,
    granularity_hint: str = "Use whatever segment granularity matches what you hear.",
    time_unit: TimeUnit = "second",
    with_description: bool = False,
) -> str:
    """Render the full prompt text (audio is passed separately to the API).

    Args:
        chunk: per-chunk metadata.
        labels: allowed label set for the dataset.
        label_hints: optional ``{label: one-line description}`` dict;
            falls through to a bare-label list if absent.
        context_mode: ``"none"`` for no prior-chunk context, ``"prev"`` to
            include only the immediately-previous predicted activity label.
        prev_label: the previous chunk's last predicted label. Ignored
            when ``context_mode == "none"``; required (else no-op) when
            ``context_mode == "prev"``.
        granularity_hint: free-form steer on segment granularity (used
            only in ``time_unit == "second"`` mode; minute mode hard-
            caps granularity at 1-minute boundaries so the hint is moot).
        time_unit: ``"second"`` → MM:SS timestamps;
            ``"minute"`` → integer-minute timestamps (0..chunk_minutes).
            Minute mode caps the maximum segment count and is useful
            when fine-second outputs over-segment (e.g. Q3).

    Returns:
        A complete text prompt ready to send alongside the chunk audio.
    """
    cd_s, cd_min = _chunk_durations(chunk)
    prev_block = _render_prev_block(prev_label) if context_mode == "prev" else ""
    label_block = "\n\n# " + render_label_block(labels, label_hints)

    if time_unit == "second":
        task = (
            "\n\n# Task\n"
            f"The recording is {_fmt_mmss(cd_s)} long ({cd_s:.0f}s). "
            "Segment it into non-overlapping, contiguous spans covering every "
            "moment from start to end using labels from the list below (no gaps). "
            + granularity_hint
        )
    else:
        task = (
            "\n\n# Task\n"
            f"The recording is {cd_min} minutes long. "
            "Segment it into non-overlapping, contiguous spans at *whole-minute* "
            f"boundaries, covering every minute from 0 to {cd_min} "
            "using labels from the list below (no gaps). Each segment must span "
            "at least one full minute; do not emit sub-minute segments."
        )
    output_spec = _render_output_spec(
        time_unit,
        cd_min,
        anchor="chunk",
        with_description=with_description,
    )
    return f"{SYSTEM_INSTRUCTION}{prev_block}{task}{output_spec}{label_block}"


def _render_description_lines(
    descriptions: list[dict],
    time_unit: TimeUnit,
    chunk_start_s: float,
) -> str:
    """Format Stage-A captions as [MM:SS-MM:SS] lines relative to the chunk.

    ``descriptions`` carries absolute wall-clock seconds (Stage-A writes
    absolute coords); subtract ``chunk_start_s`` so the timestamps in the
    prompt are chunk-relative — matching the timestamp convention the
    segmentation model is asked to emit.
    """
    lines = []
    for d in descriptions:
        rel_start = float(d["start_s"]) - chunk_start_s
        rel_end = float(d["end_s"]) - chunk_start_s
        if time_unit == "second":
            t = f"[{_fmt_mmss(rel_start)}-{_fmt_mmss(rel_end)}]"
        else:
            t = f"[{int(rel_start // 60)}-{int(rel_end // 60)}]"
        lines.append(f"{t} {d['description'].strip()}")
    return "\n".join(lines)


def render_prompt_from_descriptions(
    chunk: ChunkContext,
    descriptions: list[dict],
    labels: tuple[str, ...],
    *,
    label_hints: dict[str, str] | None = None,
    context_mode: ContextMode = "none",
    prev_label: str | None = None,
    granularity_hint: str = "Use whatever segment granularity matches the descriptions.",
    time_unit: TimeUnit = "second",
    with_description: bool = False,
) -> str:
    """Stage-B prompt: same skeleton as ``render_prompt`` but audio is
    replaced by a timestamped list of Stage-A descriptions.

    ``descriptions`` is a list of ``{"start_s", "end_s", "description"}``
    dicts in absolute wall-clock seconds covering this chunk's window.
    The rendered timestamps inside the prompt are chunk-relative, matching
    the chunk-relative convention the segmentation model is asked to emit.
    """
    cd_s, cd_min = _chunk_durations(chunk)
    prev_block = _render_prev_block(prev_label) if context_mode == "prev" else ""
    label_block = "\n\n# " + render_label_block(labels, label_hints)
    desc_block = "\n\n# Audio descriptions\n" + _render_description_lines(
        descriptions, time_unit, chunk.chunk_start_wallclock_s
    )

    if time_unit == "second":
        task = (
            "\n\n# Task\n"
            f"The descriptions span {_fmt_mmss(cd_s)} ({cd_s:.0f}s) "
            f"across {len(descriptions)} consecutive clips. Segment the timeline "
            "into non-overlapping, contiguous spans covering every moment from "
            "start to end using labels from the list below (no gaps). "
            + granularity_hint
        )
    else:
        task = (
            "\n\n# Task\n"
            f"The descriptions span {cd_min} minutes across "
            f"{len(descriptions)} consecutive clips. Segment the timeline "
            "into non-overlapping, contiguous spans at *whole-minute* "
            f"boundaries, covering every minute from 0 to {cd_min} "
            "using labels from the list below (no gaps). Each segment must span "
            "at least one full minute; do not emit sub-minute segments."
        )
    output_spec = _render_output_spec(
        time_unit,
        cd_min,
        anchor="window",
        with_description=with_description,
    )
    return (
        f"{SYSTEM_INSTRUCTION_CASCADE}{prev_block}{task}{output_spec}"
        f"{label_block}{desc_block}"
    )
