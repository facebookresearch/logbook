# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 3 of the SINS data-prep pipeline: mono.flac + final manifest.json.

Reads ``<data_dir>/nodes.json`` (written by ``build_flacs.py``) for
per-node FLAC paths + raw apartment-level annotations. Cleans the
annotations into a single room-free event stream + per-node listening
trace, streams the per-node FLACs through a 2 s linear crossfade into
``<data_dir>/mono.flac``, and writes the final
``<data_dir>/manifest.json``.

Cleaning algorithm:
  1. Sweep-line over the union of all event start/end transition points.
  2. Per interval, pick a winner — class hierarchy (specific > "other"
     > "absence"), latest-start tiebreak.
  3. Drop the room suffix; keep `activity` only.
  4. Prepend leading "other" (Node 1) if first event starts > 0.
  5. Drop sub-1s slivers by extending the previous span.
  6. Assign per-segment node via `pick_node(activity, room)`.
  7. Coalesce adjacent same-node spans → `node_trace`.
  8. Coalesce adjacent same-activity spans → `annotations` (room-free).

Output ``manifest.json``: top-level ``annotations``, ``node_trace``,
``duration``, ``audio_path`` (mono.flac), `crossfade_s`, `sample_rate`.
Verified via `assert_gt_contiguous`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from long_audio.utils.events import assert_gt_contiguous, coalesce_runs


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "SINS"


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

LIVING_KITCHEN_ACTIVITIES = frozenset({"cooking", "dishwashing", "eating"})
DEFAULT_NODE_WHEN_ALL_ABSENT = 1


def pick_node(activity: str, room: str) -> int:
    """Map (activity, room) → SINS node id (1..13).

    Rules:
      - cooking / dishwashing / eating at room=living → Node 4 (kitchen-side)
      - everything else at room=living or room=hall → Node 1
      - room=bathroom → Node 13
      - room=wcroom (toilet) → Node 11
      - room=bedroom → Node 9
    """
    if activity in LIVING_KITCHEN_ACTIVITIES and room == "living":
        return 4
    if room in ("living", "hall"):
        return 1
    if room == "bathroom":
        return 13
    if room == "wcroom":
        return 11
    if room == "bedroom":
        return 9
    raise ValueError(f"Unknown SINS room {room!r}")


def _event_class(activity: str) -> int:
    """Priority class for arbitration. Higher wins.

    2 = specific activity, 1 = "other", 0 = "absence".
    """
    if activity == "absence":
        return 0
    if activity == "other":
        return 1
    return 2


def _drop_short_spans(raw: list[dict], min_duration_s: float = 1.0) -> list[dict]:
    """Remove spans shorter than min_duration_s by extending the previous
    span's end across them. Sweep-line transitions on dense overlap zones
    produce sub-1s slivers that are noise, not signal.
    """
    out: list[dict] = []
    for seg in raw:
        if (seg["end"] - seg["start"]) < min_duration_s and out:
            out[-1] = {**out[-1], "end": seg["end"]}
        else:
            out.append(seg)
    return out


def clean_annotations(
    annots: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Resolve overlapping room-tagged events into a singular event stream
    plus a per-node listening trace.

    Args:
        annots: list of {"label": "<activity>:<room>", "start": float,
            "end": float}. Label must be ``activity:room``; raises if not.

    Returns:
        (annotations, node_trace):
          - annotations: room-free event stream
              [{"start", "end", "activity"}, ...]
            adjacent same-activity merged.
          - node_trace: per-node listening trace
              [{"start", "end", "node"}, ...]
            adjacent same-node merged.

    Arbitration:
      - Class hierarchy: specific > "other" > "absence". Higher wins.
      - Within same class: latest-start wins (later preempts earlier).
      - No events covering a moment → emit `absence` mapped to the
        default node.
    """
    if not annots:
        return [], []

    events = []
    for a in annots:
        label = a["label"]
        if ":" not in label:
            raise ValueError(f"Bad label (expected 'activity:room'): {label!r}")
        activity, room = label.split(":", 1)
        activity = activity.strip().lower().replace(" ", "_")
        room = room.strip().lower()
        events.append({
            "start": float(a["start"]),
            "end": float(a["end"]),
            "activity": activity,
            "room": room,
            "class": _event_class(activity),
        })

    points: list[float] = sorted({p for e in events for p in (e["start"], e["end"])})

    raw: list[dict] = []
    for i in range(len(points) - 1):
        seg_start, seg_end = points[i], points[i + 1]
        if seg_end <= seg_start:
            continue
        mid = 0.5 * (seg_start + seg_end)
        active = [e for e in events if e["start"] <= mid < e["end"]]
        if not active:
            raw.append({
                "start": seg_start, "end": seg_end,
                "activity": "absence", "room": "living",
            })
            continue
        max_cls = max(e["class"] for e in active)
        cands = [e for e in active if e["class"] == max_cls]
        winner = max(cands, key=lambda e: e["start"])
        room = winner["room"]
        if winner["activity"] == "absence":
            room = "living"
        raw.append({
            "start": seg_start, "end": seg_end,
            "activity": winner["activity"], "room": room,
        })

    if raw and raw[0]["start"] > 0:
        raw.insert(0, {
            "start": 0.0, "end": raw[0]["start"],
            "activity": "other", "room": "living",
        })

    raw = _drop_short_spans(raw, min_duration_s=1.0)

    for seg in raw:
        seg["node"] = pick_node(seg["activity"], seg["room"])

    node_trace = coalesce_runs(
        [{"start": seg["start"], "end": seg["end"], "node": f"Node{seg['node']}"}
         for seg in raw],
        key="node",
    )
    # Emit the unified schema key (`event`) at the I/O boundary; coalescing
    # internally still uses the per-seg `activity` key (intermediate var).
    annotations = coalesce_runs(
        [{"start": seg["start"], "end": seg["end"], "event": seg["activity"]}
         for seg in raw],
        key="event",
    )
    return annotations, node_trace


# ---------------------------------------------------------------------------
# Audio assembly
# ---------------------------------------------------------------------------

def _read_window(path: Path, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
    """Read [start_s, end_s) from a mono FLAC. Returns float32."""
    start_frame = int(round(start_s * sample_rate))
    end_frame = int(round(end_s * sample_rate))
    n_frames = max(0, end_frame - start_frame)
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    audio, sr = soundfile.read(
        str(path), start=start_frame, frames=n_frames, dtype="float32", always_2d=False,
    )
    if sr != sample_rate:
        raise ValueError(f"sr mismatch for {path}: {sr} vs {sample_rate}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.shape[0] < n_frames:
        audio = np.pad(audio, (0, n_frames - audio.shape[0]))
    return audio.astype(np.float32, copy=False)


def build_mono_stream(
    node_trace: list[dict],
    node_paths: dict[str, Path],
    output_path: Path,
    *,
    sample_rate: int = 16000,
    crossfade_s: float = 2.0,
    timeline_end_s: float | None = None,
    chunk_s: float = 300.0,
    log_every: int = 12,
) -> dict:
    """Stream a single mono FLAC from per-node FLACs, following node_trace.
    2s linear crossfade at each node transition. Returns summary dict.
    """
    if not node_trace:
        raise ValueError("node_trace is empty")

    last_end = max(s["end"] for s in node_trace)
    total_end = last_end if timeline_end_s is None else min(last_end, timeline_end_s)
    if total_end <= 0:
        raise ValueError("timeline has zero duration")

    half_xf = max(0.0, crossfade_s / 2.0)
    chunk_samples = int(round(chunk_s * sample_rate))
    total_samples = int(round(total_end * sample_rate))
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(
        f"[build_manifest] {len(node_trace)} spans, "
        f"duration {total_end/3600:.2f}h ({total_samples} samples), "
        f"{n_chunks} chunks of {chunk_s:.0f}s, crossfade {crossfade_s:.1f}s",
        flush=True,
    )
    print(f"[build_manifest] output: {output_path}", flush=True)

    with soundfile.SoundFile(
        str(output_path), mode="w",
        samplerate=sample_rate, channels=1,
        subtype="PCM_16", format="FLAC",
    ) as wout:
        cursor_s = 0.0
        chunk_i = 0
        while cursor_s < total_end:
            chunk_start = cursor_s
            chunk_end = min(cursor_s + chunk_s, total_end)
            buf = np.zeros(int(round((chunk_end - chunk_start) * sample_rate)),
                           dtype=np.float32)

            for j, span in enumerate(node_trace):
                if span["end"] <= chunk_start:
                    continue
                if span["start"] >= chunk_end:
                    break
                seg_start = max(span["start"], chunk_start)
                seg_end = min(span["end"], chunk_end)
                if seg_end <= seg_start:
                    continue
                node_path = node_paths[span["node"]]

                read_start = seg_start
                read_end = seg_end
                head_fade = 0.0
                tail_fade = 0.0
                if j > 0 and abs(span["start"] - seg_start) < 1e-9 and span["start"] >= chunk_start:
                    head_fade = min(half_xf, (span["end"] - span["start"]) / 2.0)
                    read_start = span["start"] - head_fade
                if j + 1 < len(node_trace) and abs(span["end"] - seg_end) < 1e-9 and span["end"] <= chunk_end:
                    next_span = node_trace[j + 1]
                    tail_fade = min(
                        half_xf,
                        (span["end"] - span["start"]) / 2.0,
                        (next_span["end"] - next_span["start"]) / 2.0,
                    )
                    read_end = span["end"] + tail_fade

                read_start = max(read_start, chunk_start)
                read_end = min(read_end, chunk_end)
                if read_end <= read_start:
                    continue

                seg_audio = _read_window(node_path, read_start, read_end, sample_rate)
                rel_start = int(round((read_start - chunk_start) * sample_rate))
                n_seg = seg_audio.shape[0]
                if head_fade > 0 and read_start <= span["start"]:
                    ramp_end_s = span["start"] + head_fade
                    ramp_n = int(round((min(ramp_end_s, read_end) - read_start) * sample_rate))
                    if ramp_n > 0:
                        ramp = np.linspace(0.0, 1.0, ramp_n, endpoint=False, dtype=np.float32)
                        seg_audio[:ramp_n] = seg_audio[:ramp_n] * ramp
                if tail_fade > 0 and read_end >= span["end"]:
                    ramp_start_s = span["end"] - tail_fade
                    ramp_start_in_seg = int(round((max(ramp_start_s, read_start) - read_start) * sample_rate))
                    ramp_n = n_seg - ramp_start_in_seg
                    if ramp_n > 0:
                        ramp = np.linspace(1.0, 0.0, ramp_n, endpoint=False, dtype=np.float32)
                        seg_audio[ramp_start_in_seg:] = seg_audio[ramp_start_in_seg:] * ramp

                end_in_buf = rel_start + n_seg
                end_in_buf = min(end_in_buf, buf.shape[0])
                buf[rel_start : end_in_buf] += seg_audio[: end_in_buf - rel_start]

            wout.write(buf)
            cursor_s = chunk_end
            chunk_i += 1
            if chunk_i % log_every == 0 or chunk_i == n_chunks:
                elapsed = time.time() - t0
                pct = 100.0 * cursor_s / total_end
                rate = cursor_s / max(elapsed, 1e-6)
                eta = (total_end - cursor_s) / max(rate, 1e-6)
                print(
                    f"[build_manifest] chunk {chunk_i}/{n_chunks} ({pct:5.1f}%) "
                    f"elapsed {elapsed/60:.1f}m, rate {rate:.0f}x rt, ETA {eta/60:.1f}m",
                    flush=True,
                )

    elapsed = time.time() - t0
    out_gb = output_path.stat().st_size / 1e9
    print(f"[build_manifest] DONE in {elapsed/60:.1f}m. "
          f"Wrote {output_path} ({out_gb:.2f} GB)", flush=True)

    return {
        "output_audio": str(output_path),
        "sample_rate": sample_rate,
        "duration_s": total_end,
        "n_spans": len(node_trace),
        "crossfade_s": crossfade_s,
        "elapsed_s": elapsed,
        "size_bytes": output_path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Pipeline + CLI
# ---------------------------------------------------------------------------

def _load_node_inputs(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (node_manifests, annotations_raw) from build_flacs.py's nodes.json."""
    nodes_path = data_dir / "nodes.json"
    if not nodes_path.exists():
        raise FileNotFoundError(
            f"{nodes_path} not found. Run scripts/data/sins/build_flacs.py first."
        )
    m = json.loads(nodes_path.read_text())
    nodes = m["nodes"]  # die noisily if key absent
    annotations = nodes[0]["annotations"]
    return nodes, annotations


def build_mono_and_manifest(
    output_dir: Path,
    node_manifests: list[dict],
    annotations_raw: list[dict],
    *,
    sample_rate: int = 16000,
    crossfade_s: float = 2.0,
) -> dict:
    """Build mono.flac from per-node FLACs + raw annotations and write the
    top-level manifest. Returns the build_mono_stream summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sins.build_mono_and_manifest] building mono.flac → {output_dir}")

    node_paths = {
        f"Node{n['node_id']}": Path(n["audio_path"]) for n in node_manifests
    }
    annotations, node_trace = clean_annotations(annotations_raw)
    if not node_trace:
        raise RuntimeError(
            "clean_annotations returned empty node_trace; nothing to build"
        )

    # Cap timeline at the shortest per-node FLAC so reads don't overflow.
    min_dur = min(
        soundfile.info(str(p)).frames / sample_rate for p in node_paths.values()
    )
    timeline_end_s = min(min_dur, node_trace[-1]["end"])

    mono_path = output_dir / "mono.flac"
    summary = build_mono_stream(
        node_trace=node_trace,
        node_paths=node_paths,
        output_path=mono_path,
        sample_rate=sample_rate,
        crossfade_s=crossfade_s,
        timeline_end_s=timeline_end_s,
    )

    assert_gt_contiguous(
        annotations, duration_s=summary["duration_s"], name="SINS annotations",
    )

    manifest = {
        "dataset": "SINS",
        "description": (
            "SINS — annotation-driven mono stream (single source of truth). "
            "node_trace records which per-node FLAC was used for each span; "
            f"{crossfade_s:.1f} s linear crossfade at each node transition."
        ),
        "audio_path": str(mono_path),
        "sample_rate": sample_rate,
        "duration": summary["duration_s"],
        "crossfade_s": crossfade_s,
        "annotations": annotations,
        "node_trace": [
            {"start": s["start"], "end": s["end"], "node": s["node"]}
            for s in node_trace
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[sins.build_mono_and_manifest] wrote {manifest_path} "
        f"({len(annotations)} merged event spans, {len(node_trace)} node spans)"
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="SINS dataset directory (default: datasets/SINS).")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--crossfade-s", type=float, default=2.0)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    nodes, annotations = _load_node_inputs(data_dir)
    print(f"using {len(nodes)} nodes, {len(annotations)} raw annotation events")

    summary = build_mono_and_manifest(
        output_dir=data_dir,
        node_manifests=nodes,
        annotations_raw=annotations,
        sample_rate=args.sample_rate,
        crossfade_s=args.crossfade_s,
    )
    print(f"\nDONE. Summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
