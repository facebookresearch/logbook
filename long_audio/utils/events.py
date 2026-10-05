# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared event-list helpers.

An "event list" is a list of dicts ``{"<label_key>": str, "start": float,
"end": float}`` where the label key is usually ``"label"`` (eval), ``"event"``
(Ego4D manifest), or ``"activity"`` (SINS mono builder). All helpers in this
module accept a ``key=`` argument so each caller can use its own schema
without re-shaping the events.

These primitives consolidate roughly five copies of merge / gap-fill /
overlap-resolution code that grew across ``eval/metrics.py``,
``inference/chunk_runner.py``, ``datasets/ego4d/manifest_builder.py``,
and ``datasets/sins/mono_builder.py`` during the prototyping era.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def coalesce_runs(
    events: list[dict],
    key: str = "label",
    tol_s: float = 1e-6,
) -> list[dict]:
    """Merge adjacent same-label events into single events.

    Two events are 'adjacent' if ``e2.start - e1.end <= tol_s`` (a small
    epsilon absorbs floating-point dust from chunk-aligned predictions).
    Assumes events are non-overlapping on each side; overlap arbitration
    lives in :func:`resolve_overlaps` instead.

    Args:
        events: list of event dicts with at least ``"start"``, ``"end"``
            and the configured label ``key``.
        key: dict key used as the label (``"label"``, ``"event"``,
            ``"activity"``, ...).
        tol_s: max gap to still treat as adjacent.

    Returns:
        New list of merged events (input is not modified). Returned dicts
        are shallow copies of the inputs with ``"end"`` extended; all
        other fields are inherited from the first event in each run.
    """
    if not events:
        return []
    es = sorted(events, key=lambda e: e["start"])
    out: list[dict] = [dict(es[0])]
    for e in es[1:]:
        prev = out[-1]
        if e[key] == prev[key] and (e["start"] - prev["end"]) <= tol_s:
            prev["end"] = e["end"]
        else:
            out.append(dict(e))
    return out


def fill_gaps(
    events: list[dict],
    start: float,
    end: float,
    gap_label: str,
    key: str = "label",
) -> list[dict]:
    """Insert synthetic ``gap_label`` events to make the timeline gap-free.

    Walks ``events`` in chronological order, emitting a ``gap_label`` event
    for every region in ``[start, end]`` not covered by an input event.
    Input events past ``end`` are clipped; events before ``start`` are
    clipped or dropped if they fall entirely outside the window. Adjacent
    same-label events in the input are NOT merged here (use
    :func:`coalesce_runs` first if needed).

    Args:
        events: list of event dicts.
        start, end: timeline bounds (seconds).
        gap_label: label value to insert for missing regions.
        key: dict key used as the label.

    Returns:
        New list of events that fully covers ``[start, end]``.
    """
    out: list[dict] = []
    cursor = start
    for s in sorted(events, key=lambda x: (x["start"], x["end"])):
        st = float(s["start"])
        en = float(s["end"])
        if st > cursor + 1e-6:
            out.append({key: gap_label, "start": cursor, "end": min(st, end)})
        st = max(st, start)
        en = min(en, end)
        if en > st + 1e-6:
            out.append({**s, "start": st, "end": en})
            cursor = max(cursor, en)
    if end > cursor + 1e-6:
        out.append({key: gap_label, "start": cursor, "end": end})
    return out


def assert_gt_contiguous(
    events: list[dict], duration_s: float, name: str = "ground_truth"
) -> None:
    """Raise ``ValueError`` if ``events`` does not fully cover [0, duration_s]
    with non-overlapping, contiguous spans.

    GT is expected to be a complete annotation of the timeline. Any gap or
    overlap is a manifest bug — surface it loudly rather than masking it
    with silent default-label fill.

    This is the DATA-PREP-side GT contract assertion: callers should
    invoke it once when finalizing the GT manifest. Eval-time code does
    NOT call this — by the time GT reaches a metric, the contract is
    assumed to hold.

    Events here use the manifest's ``start``/``end`` keys (seconds).
    """
    if duration_s <= 0:
        return
    if not events:
        raise ValueError(
            f"{name} is empty but duration_s={duration_s}s "
            "(GT must fully cover the timeline)"
        )
    es = sorted(events, key=lambda e: e["start"])
    if es[0]["start"] > 1e-6:
        raise ValueError(
            f"{name} starts at {es[0]['start']}s, expected 0.0 "
            "(GT must cover [0, duration_s])"
        )
    cur_end = es[0]["end"]
    for i in range(1, len(es)):
        s = es[i]["start"]
        if s > cur_end + 1e-6:
            raise ValueError(
                f"{name} has gap of {s - cur_end:.6f}s between "
                f"idx {i-1} (ends {cur_end}s) and idx {i} (starts {s}s)"
            )
        if s < cur_end - 1e-6:
            raise ValueError(
                f"{name} has overlap of {cur_end - s:.6f}s between "
                f"idx {i-1} (ends {cur_end}s) and idx {i} (starts {s}s)"
            )
        cur_end = max(cur_end, es[i]["end"])
    if cur_end < duration_s - 1e-6:
        raise ValueError(
            f"{name} ends at {cur_end}s but duration_s={duration_s}s "
            f"(trailing gap of {duration_s - cur_end:.6f}s)"
        )


def resolve_overlaps(
    events: list[dict],
    priority_fn: Callable[[str], int],
    min_duration_s: float = 1.0,
    gap_label: str | None = None,
    key: str = "label",
    duration_s: float | None = None,
) -> list[dict]:
    """Resolve overlapping events into a single non-overlapping stream.

    Sweep-line algorithm shared between SINS annotation cleaning and the
    eval-time single-stream construction:

      1. Collect every unique transition point (event start, event end,
         and optionally the bounds ``[0, duration_s]``).
      2. For each interval between consecutive transitions, take the
         events active over the midpoint:
           * If none → emit ``gap_label`` (or skip if ``gap_label`` is None).
           * Else → winner is the event with the highest ``priority_fn``;
             ties broken by latest-starting (later preempts earlier).
      3. Drop sub-``min_duration_s`` spans by extending the previous
         span's end across them (sliver suppression for dense overlaps).
      4. Merge adjacent same-label spans.

    Args:
        events: list of event dicts.
        priority_fn: ``label -> int`` (higher wins).
        min_duration_s: spans shorter than this get absorbed into the
            previous span. ``0`` disables.
        gap_label: label to insert in uncovered regions. ``None`` leaves
            gaps as gaps.
        key: dict key used as the label.
        duration_s: if provided, the output covers ``[0, duration_s]``;
            otherwise the output covers ``[min start, max end]`` of the
            input events.

    Returns:
        New list of non-overlapping events.
    """
    if not events and duration_s is None:
        return []
    if not events and gap_label is None:
        return []
    if not events:
        return [{key: gap_label, "start": 0.0, "end": float(duration_s)}]

    extent_end = duration_s if duration_s is not None else max(e["end"] for e in events)
    extent_start = 0.0 if duration_s is not None else min(e["start"] for e in events)

    transitions: set[float] = {extent_start, extent_end}
    for e in events:
        transitions.add(float(e["start"]))
        transitions.add(float(e["end"]))
    points = sorted(p for p in transitions if extent_start <= p <= extent_end)

    raw: list[dict] = []
    for i in range(len(points) - 1):
        seg_start, seg_end = points[i], points[i + 1]
        if seg_end <= seg_start:
            continue
        mid = 0.5 * (seg_start + seg_end)
        active = [e for e in events if e["start"] <= mid < e["end"]]
        if not active:
            if gap_label is not None:
                raw.append({key: gap_label, "start": seg_start, "end": seg_end})
            continue
        max_pri = max(priority_fn(e[key]) for e in active)
        candidates = [e for e in active if priority_fn(e[key]) == max_pri]
        winner = max(candidates, key=lambda e: e["start"])
        raw.append({**winner, "start": seg_start, "end": seg_end})

    if min_duration_s > 0:
        absorbed: list[dict] = []
        for seg in raw:
            if (seg["end"] - seg["start"]) < min_duration_s and absorbed:
                absorbed[-1] = {**absorbed[-1], "end": seg["end"]}
            else:
                absorbed.append(seg)
        raw = absorbed

    merged: list[dict] = []
    for seg in raw:
        if (
            merged
            and merged[-1][key] == seg[key]
            and abs(merged[-1]["end"] - seg["start"]) < 1e-9
        ):
            merged[-1] = {**merged[-1], "end": seg["end"]}
        else:
            merged.append(seg)
    return merged


def merge_intervals(
    intervals: list[tuple[float, float]],
    max_gap: float = 0.0,
) -> list[tuple[float, float]]:
    """Merge a list of (start, end) intervals, joining any two whose gap
    (``next.start - prev.end``) is ``<= max_gap``. Overlaps merge even
    with ``max_gap=0``.

    Args:
        intervals: any order; sorted internally.
        max_gap: extend merges to include tolerable adjacent-gap joins.
            Default 0 = only overlapping intervals merge.

    Returns:
        Disjoint intervals in chronological order.
    """
    if not intervals:
        return []
    xs = sorted(intervals)
    merged: list[list[float]] = [list(xs[0])]
    for s, e in xs[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def intersect_intervals(
    a: tuple[float, float],
    b_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Intersect a single interval ``a = (a_start, a_end)`` with a
    (possibly unsorted) list of intervals. Returns a chronologically
    sorted list of overlap segments.
    """
    a_s, a_e = a
    if a_e <= a_s:
        return []
    out: list[tuple[float, float]] = []
    for b_s, b_e in sorted(b_intervals):
        if b_e <= a_s:
            continue
        if b_s >= a_e:
            break
        out.append((max(a_s, b_s), min(a_e, b_e)))
    return out


def tile_by_midpoint(
    items: list[dict],
    total_s: float,
    tile_s: float,
    *,
    start_key: str = "start",
    end_key: str = "end",
    min_tail_s: float = 60.0,
) -> list[tuple[float, float, list[dict]]]:
    """Partition ``items`` into non-overlapping tiles of length ``tile_s``
    covering ``[0, total_s)`` using the midpoint rule.

    Each item is assigned to the single tile that contains its temporal
    midpoint (``(item.start + item.end) / 2``). Items are never split or
    double-counted. Every tile boundary is emitted, even if the tile ends
    up empty — callers decide whether an empty tile is an error.

    Tail handling (parameterized by ``min_tail_s``):
      - If the remainder ``total_s % tile_s`` is >= ``min_tail_s``, the
        tail is emitted as its own short tile (final tile has width in
        ``[min_tail_s, tile_s)``).
      - If the remainder is < ``min_tail_s``, the tail is dropped
        entirely — the last emitted tile ends at ``n_full * tile_s``,
        NOT at ``total_s``. Trailing content (~0-60 s of audio /
        captions) is silently truncated.
      - Exception: if ``total_s < tile_s`` the whole session becomes a
        single tile spanning ``[0, total_s)``.

    Rationale for chopping sub-60s tails: for EgoLife's ~2 s DenseCaption
    fragments, a tail under 60 s risks having <10 fragments — small
    enough that midpoint-based assignment may yield zero and trip the
    empty-window raise downstream. The trade-off is 8.4 h (3.25 %) of
    total audio lost across 216 sessions.

    Args:
        items: list of dicts with numeric ``start_key`` and ``end_key`` fields.
        total_s: session / timeline duration in seconds. Must be > 0.
        tile_s: tile length in seconds. Must be > 0.
        start_key, end_key: dict keys for item start/end times.
        min_tail_s: minimum tail duration to emit as its own tile.
            Default 60. Set to 0 to always emit the tail (may be tiny).

    Returns:
        ``[(tile_start, tile_end, [items in tile]), ...]`` — one entry per
        tile, in chronological order.
    """
    if total_s <= 0:
        raise ValueError(f"total_s must be positive, got {total_s}")
    if tile_s <= 0:
        raise ValueError(f"tile_s must be positive, got {tile_s}")

    n_full = int(total_s // tile_s)
    remainder = total_s - n_full * tile_s
    if n_full == 0:
        # session shorter than one tile — collapse into a single tile
        boundaries: list[tuple[float, float]] = [(0.0, total_s)]
    else:
        boundaries = [(k * tile_s, (k + 1) * tile_s) for k in range(n_full)]
        if remainder >= min_tail_s:
            boundaries.append((n_full * tile_s, total_s))
        # else: chop the sub-min_tail_s tail

    tiles: list[tuple[float, float, list[dict]]] = []
    for t_start, t_end in boundaries:
        members = [
            it for it in items
            if t_start <= (it[start_key] + it[end_key]) / 2.0 < t_end
        ]
        tiles.append((t_start, t_end, members))
    return tiles


def midpoint_boundaries(
    slices: list[dict],
    duration: float,
    *,
    carry: dict[str, str],
    snap_edges: bool = True,
    name: str = "midpoint_boundaries",
) -> list[dict]:
    """Resolve adjacent overlapping/gapped slices into a disjoint timeline.

    Sets the boundary between two adjacent slices to the midpoint of their
    overlap/gap (``(prev.end + cur.start) / 2``), producing non-overlapping
    spans over ``[0, duration]``. Slices are sorted internally by
    ``(start, -end)``. This is the shared core behind Ego4D ``actions``
    assembly (``scripts/data/ego4d/annotate_manifest._apply_midpoint_
    boundaries``) and the training gold timeline
    (``long_audio.training.data.disjoint_slice_segments``); see
    :func:`tile_by_midpoint` for the *fixed-tile* variant.

    The first span's start is clamped to ``>= 0`` and the last span's end to
    ``<= duration``. With ``snap_edges=True`` (default) the first start and last
    end are additionally forced to exactly ``0.0`` / ``duration`` so the output
    covers ``[0, duration]`` gap-free (float-drift cleanup); ``snap_edges=False``
    leaves the clamped values (caller-does-its-own-snap contract).

    Args:
        slices: dicts with numeric ``start`` / ``end`` plus every source key
            named in ``carry``.
        duration: timeline length (seconds).
        carry: mapping ``output_key -> source_key`` of fields to copy from each
            slice onto its output span (e.g. ``{"event": "event"}`` or
            ``{"event": "event", "description": "summary"}``).
        snap_edges: force exact ``[0, duration]`` edges (see above).
        name: prefix for the degenerate-span error message.

    Returns:
        ``[{"start", "end", **carry}, ...]`` sorted by start, one per input
        slice (no same-label coalescing; run :func:`coalesce_runs` after if the
        caller wants merged runs).

    Raises:
        ValueError: if a computed span is degenerate (``end <= start``). By the
            upstream coverage contract this should not happen.
    """
    if not slices:
        return []
    ss = sorted(slices, key=lambda s: (float(s["start"]), -float(s["end"])))
    n = len(ss)
    out: list[dict] = []
    for i, s in enumerate(ss):
        st = float(s["start"])
        en = float(s["end"])
        if i > 0:
            st = (float(ss[i - 1]["end"]) + float(s["start"])) / 2.0
        if i + 1 < n:
            en = (float(s["end"]) + float(ss[i + 1]["start"])) / 2.0
        if i == 0:
            st = max(0.0, st)
        if i == n - 1:
            en = min(float(duration), en)
        if en <= st:
            raise ValueError(
                f"{name}: degenerate span at idx {i}: st={st}, en={en} "
                "(should not happen given upstream coverage)."
            )
        rec = {"start": st, "end": en}
        for out_key, in_key in carry.items():
            rec[out_key] = s[in_key]
        out.append(rec)
    if snap_edges:
        out[0] = {**out[0], "start": 0.0}
        out[-1] = {**out[-1], "end": float(duration)}
    return out
