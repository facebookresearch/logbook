# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Generic JSON schema for grammar-constrained segmentation outputs.

Dataset-agnostic builder + timestamp parser. Callers pass the label set
explicitly (dataset-specific label sets live with the dataset module).
Timestamps in the schema are chunk-relative; the chunk runner converts
back to absolute time when stitching. ``time_unit="second"`` uses MM:SS,
``time_unit="minute"`` uses an integer 0..N. Absence gaps must be
represented as explicit ``absence`` segments so the prediction timeline
is gap-free.
"""

from __future__ import annotations

import re
from typing import Literal


TimeUnit = Literal["second", "minute"]

# Second-level: MM:SS anchored, MM in 0..59, SS in 0..59. Models
# occasionally emit HH:MM:SS for long chunks — kept as a separate
# optional pattern below for parsing tolerance.
_TIMESTAMP_PATTERN_SECOND = r"^([0-5]?\d):([0-5]\d)$"
# Minute-level: 1-2 digit integer minute. The prompt steers the actual
# range (0..N for an N-minute chunk); the regex stays tolerant of
# leading zeros for grammar-friendliness.
_TIMESTAMP_PATTERN_MINUTE = r"^\d{1,2}$"


def _timestamp_pattern(time_unit: TimeUnit) -> str:
    if time_unit == "second":
        return _TIMESTAMP_PATTERN_SECOND
    if time_unit == "minute":
        return _TIMESTAMP_PATTERN_MINUTE
    raise ValueError(f"Unknown time_unit: {time_unit!r}")


def make_segmentation_schema(
    labels: tuple[str, ...],
    time_unit: TimeUnit = "second",
    *,
    with_description: bool = False,
) -> dict:
    """Return a JSON schema for the segmentation output.

    Args:
        labels: allowed event labels (enum). Required — supply your
            dataset's label tuple (e.g. ``SINS_LABELS``).
        time_unit: "second" → MM:SS pattern; "minute" → 1-2 digit
            integer-minute pattern. The chunk runner picks the matching
            ``parse_segments(..., time_unit=...)``.
        with_description: when True, each segment additionally requires a
            free-text ``description`` (1-2 sentence sub-activity narrative).
            Default False keeps the schema byte-identical to the seg-only
            baseline.

    The schema enforces:
      - top-level object with a "segments" array
      - each segment has "start", "end", "event" (and "description" when
        ``with_description=True``)
      - no extra properties allowed
    """
    pattern = _timestamp_pattern(time_unit)
    ts_desc_unit = "MM:SS" if time_unit == "second" else "integer minute"
    seg_properties: dict = {
        "start": {
            "type": "string",
            "pattern": pattern,
            "description": f"Segment start as {ts_desc_unit} relative to chunk start",
        },
        "end": {
            "type": "string",
            "pattern": pattern,
            "description": f"Segment end as {ts_desc_unit} relative to chunk start",
        },
        "event": {
            "type": "string",
            "enum": list(labels),
        },
    }
    required = ["start", "end", "event"]
    if with_description:
        seg_properties["description"] = {
            "type": "string",
            "description": (
                "1-2 sentence sub-activity narrative for this span "
                "(what the person is doing)."
            ),
            "minLength": 1,
            "maxLength": 400,
        }
        required.append("description")
    return {
        "name": "segmentation",
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": seg_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
        "strict": True,
    }


# --- Parsing helpers (post-vLLM, in case schema enforcement is bypassed) ---


_TS_RE_SECOND = re.compile(_TIMESTAMP_PATTERN_SECOND)
_HHMMSS_RE = re.compile(r"^(\d{1,2}):([0-5]\d):([0-5]\d)$")
_TS_RE_MINUTE = re.compile(_TIMESTAMP_PATTERN_MINUTE)


def parse_timestamp(ts, time_unit: TimeUnit = "second") -> float:
    """Parse a timestamp into seconds.

    Second mode accepts MM:SS or H:MM:SS strings only.
    Minute mode accepts a bare 1-2 digit integer-minute string ("0".."99")
    OR a raw int/float — freeform-mode models often emit unquoted integers
    (the structured-decoding schema forces strings, but with no enforcement
    the model picks whichever form is natural).
    """
    if time_unit == "minute":
        # Tolerate raw int/float from freeform-mode models. (bool is a subclass
        # of int in Python — explicitly excluded so True/False don't sneak in.)
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            if ts < 0:
                raise ValueError(f"Negative minute timestamp: {ts!r}")
            return float(ts) * 60.0
        if isinstance(ts, str) and (m := _TS_RE_MINUTE.match(ts)):
            return int(m.group(0)) * 60.0
        raise ValueError(f"Unparseable minute timestamp: {ts!r}")
    # Default: second mode.
    if isinstance(ts, str):
        if m := _TS_RE_SECOND.match(ts):
            return int(m.group(1)) * 60 + int(m.group(2))
        if m := _HHMMSS_RE.match(ts):
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    raise ValueError(f"Unparseable timestamp: {ts!r}")


def parse_segments(
    raw: dict,
    labels: tuple[str, ...],
    chunk_start_s: float = 0.0,
    time_unit: TimeUnit = "second",
) -> list[dict]:
    """Convert a model output dict into the eval-suite event format.

    Args:
        raw: dict matching the schema (second- or minute-grain).
        labels: allowed label set; segments with other labels are dropped.
            Required — supply your dataset's label tuple.
        chunk_start_s: absolute wall-clock-seconds-from-T0 at which this
            chunk's audio begins. Chunk-relative timestamps in the model
            output get offset by this value.
        time_unit: ``"second"`` to parse MM:SS / HH:MM:SS; ``"minute"``
            to parse bare integer minutes.

    Returns:
        list of {"label": str, "start": float, "end": float} in
        absolute wall-clock seconds, suitable for the eval suite.
    """
    if not isinstance(raw, dict) or "segments" not in raw:
        return []
    out: list[dict] = []
    allowed = set(labels)
    for seg in raw["segments"]:
        try:
            label = seg["event"]
            if label not in allowed:
                continue
            start = parse_timestamp(seg["start"], time_unit) + chunk_start_s
            end = parse_timestamp(seg["end"], time_unit) + chunk_start_s
            if end < start:
                continue
            entry: dict = {"label": label, "start": start, "end": end}
            desc = seg.get("description")
            if isinstance(desc, str) and desc:
                entry["description"] = desc
            out.append(entry)
        except (KeyError, ValueError, TypeError):
            continue
    return out
