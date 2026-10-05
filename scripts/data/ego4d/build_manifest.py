# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 3 of the Ego4D data-prep pipeline: build the deterministic manifest.

Reads ``ego4d.json``, ``narration.json``, ``moments_*.json`` and Stage 1's
``audio_manifest.json``; writes ``manifest.json`` with one record per
surviving uid (audio metadata, both narration passes as
``passes.{1,2}.slices``, moments, and a stable train/val/test split
derived from ``sha256(video_source::fb_participant_id)``). Deterministic
and CPU-only. Uids without audio, without both narration passes, with
interior narration coverage gaps, or with <2 slices in either pass are
dropped with a per-uid WARN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict


# Universities to keep: the four with > 10 h of audio in the canonical
# eval-set TEST split (frl 24.6h, iiith 32.2h, kaust 10.4h, minnesota
# 10.8h). The other six contribute < 10 h test each and inflate per-uni
# variance on the eval. Restricting here (Stage 3) means Stage 4's OLMo
# pass doesn't waste inference on slices we'll drop downstream. Override
# with ``--include-universities a b c`` (or ``all`` to skip the filter).
INCLUDED_UNIVERSITIES = ("frl_track_1_public", "iiith", "kaust", "minnesota")
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from long_audio.utils.paths import to_repo_relative  # noqa: E402


# ============================================================================
# Inlined helpers (formerly long_audio.datasets.ego4d.manifest_builder).
# ============================================================================

_LEAD_SHIFT_THRESHOLD_S = 0.1
_COVERAGE_DRIFT_TOL_S = 0.010  # 10 ms: sub-second drift between adjacent
# narration windows is float artifact, not a
# real coverage gap.


def compute_lead_shift(
    narration_entry: dict[str, Any] | None,
) -> tuple[float, float]:
    """Return ``(lead_start, lead_shift)`` for one video's narration.

    ``lead_start`` is the offset in the original mp4 where narration begins.
    ``lead_shift`` is the amount to subtract from each narration timestamp
    so the manifest's t=0 corresponds to original t=lead_start.

    Threshold rule:
      - first_summary_start < 0.1 → (0.0, 0.0) (artifact, snap to 0).
      - first_summary_start >= 0.1 → (first, first) (narrator started late).
    """
    if not narration_entry:
        return 0.0, 0.0
    starts: list[float] = []
    for pass_key in ("narration_pass_1", "narration_pass_2"):
        summaries = narration_entry[pass_key]["summaries"]
        if summaries:
            starts.append(float(summaries[0]["start_sec"]))
    if not starts:
        return 0.0, 0.0
    first_summary_start = min(starts)
    if first_summary_start < _LEAD_SHIFT_THRESHOLD_S:
        return 0.0, 0.0
    return first_summary_start, first_summary_start


def _shift_slices(
    raw_slices: list[dict],
    lead_shift: float,
    uid: str = "<unknown>",
) -> list[dict]:
    """Subtract ``lead_shift`` from each slice's start/end. Drops slices
    that fall entirely pre-shift (en <= 0); clamps partial-leading slices
    to start at 0. Raises if ALL slices are pre-shift (upstream filter bug)."""
    if lead_shift == 0.0:
        return list(raw_slices)
    out: list[dict] = []
    n_dropped = 0
    for s in raw_slices:
        st = float(s["start_sec"]) - lead_shift
        en = float(s["end_sec"]) - lead_shift
        if en <= 0:
            n_dropped += 1
            continue
        if st < 0:
            st = 0.0
        new = dict(s)
        new["start_sec"] = st
        new["end_sec"] = en
        out.append(new)
    if n_dropped:
        print(
            f"[build_manifest] WARN uid={uid} _shift_slices: dropped "
            f"{n_dropped} slice(s) entirely before lead_shift={lead_shift:.3f}",
            flush=True,
        )
    if n_dropped and not out:
        raise ValueError(
            f"_shift_slices: uid={uid} all {n_dropped} slices fell before "
            f"lead_shift={lead_shift:.3f}. Upstream must filter."
        )
    return out


def _dedupe_summaries(
    recs: list[dict[str, Any]],
    uid: str = "<unknown>",
) -> tuple[list[dict[str, Any]], int]:
    """Drop degenerate (end <= start) + containment-duplicate windows.

    The clamp in _shift_slices can create new containment-duplicates by
    collapsing multiple pre-shift slices at start=0 post-shift, so this
    pass is meant to run AFTER _shift_slices.

    Returns (kept, n_dropped_total). Kept are in original source order so
    downstream batching stays positionally meaningful.
    """
    if not recs:
        return [], 0
    n_input = len(recs)
    clean: list[dict[str, Any]] = []
    for r in recs:
        if r["end_sec"] <= r["start_sec"]:
            print(
                f"[build_manifest] WARN uid={uid} _dedupe_summaries: degenerate "
                f"slice start_sec={r['start_sec']} >= end_sec={r['end_sec']} — dropped",
                flush=True,
            )
            continue
        clean.append(r)
    indexed = list(enumerate(clean))
    indexed.sort(key=lambda ir: (ir[1]["start_sec"], -ir[1]["end_sec"]))
    kept: list[tuple[int, dict[str, Any]]] = []
    for orig_idx, r in indexed:
        if kept and r["end_sec"] <= kept[-1][1]["end_sec"] + 1e-6:
            continue
        kept.append((orig_idx, r))
    kept.sort(key=lambda ir: ir[0])
    return [r for _, r in kept], n_input - len(kept)


def _interior_gap(slices: list[dict]) -> tuple[bool, str]:
    """Return (has_gap, reason). Slices must be post-shift, dedupe'd,
    non-degenerate, with at least 1 entry. Detects ONLY interior
    discontinuities — leading/trailing edges can be trimmed by audio
    offset/duration adjustment, so they don't count.
    """
    if len(slices) < 2:
        return False, ""
    sorted_s = sorted(slices, key=lambda s: s["start_sec"])
    cursor = sorted_s[0]["end_sec"]
    for i in range(1, len(sorted_s)):
        s = sorted_s[i]
        gap = s["start_sec"] - cursor
        if gap > _COVERAGE_DRIFT_TOL_S:
            return True, (
                f"interior gap of {gap:.3f}s at slice {i} "
                f"(prev end {cursor:.3f}s, this start {s['start_sec']:.3f}s)"
            )
        cursor = max(cursor, s["end_sec"])
    return False, ""


def _build_pass_slices(
    summaries: list[dict],
    lead_shift: float,
    uid: str,
    pass_num: int,
) -> tuple[list[dict] | None, str]:
    """Return (slices, error). Slices are post-shift, dedupe'd,
    coverage-checked narration windows for one pass. error is "" on success.
    """
    recs: list[dict] = []
    for item in summaries:
        raw = item["summary_text"].strip()
        if not raw:
            continue
        recs.append(
            {
                "raw_summary": raw,
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "annotation_uid": item["annotation_uid"],
            }
        )
    if not recs:
        return None, "no non-empty summaries"
    try:
        shifted = _shift_slices(recs, lead_shift, uid=f"{uid}/pass{pass_num}")
    except ValueError as e:
        return None, f"shift failed: {e}"
    shifted, _ = _dedupe_summaries(shifted, uid=f"{uid}/pass{pass_num}")
    if len(shifted) < 2:
        return None, f"only {len(shifted)} slice(s) post-shift dedupe"
    has_gap, reason = _interior_gap(shifted)
    if has_gap:
        return None, reason
    sorted_s = sorted(shifted, key=lambda s: (s["start_sec"], s["end_sec"]))
    # Emit unified-schema keys (start/end) at the I/O boundary. Internal
    # helpers still use start_sec/end_sec — only the on-disk JSON unifies.
    out = [
        {
            "start": float(s["start_sec"]),
            "end": float(s["end_sec"]),
            "raw_summary": s["raw_summary"],
            "annotation_uid": s["annotation_uid"],
        }
        for s in sorted_s
    ]
    return out, ""


def _load_moments(
    moments_paths: list[Path],
    video_uids: set[str],
    full_offsets: dict[str, float],
    durations: dict[str, float],
) -> dict[str, list[dict]]:
    """Load moments from train+val files, pool per uid, shift + clip to
    manifest [0, duration] coords.

    Each label's video_*_time is in the original mp4 timeline; we subtract
    `full_offsets[uid]` (the COMBINED lead_shift + cross-pass t_start)
    to land in manifest coords, then clip to [0, durations[uid]].
    """
    data: dict[str, list[dict]] = {}
    for path in moments_paths:
        moments = json.loads(Path(path).read_text())
        for vid in moments["videos"]:
            uid = vid["video_uid"]
            if uid not in video_uids:
                continue
            offset = full_offsets[uid]
            dur = durations[uid]
            events: list[dict] = []
            for clip in vid["clips"]:
                for annot in clip["annotations"]:
                    for label in annot["labels"]:
                        # Guard on the same fields we actually use below —
                        # start_time / end_time are a different (clip-relative)
                        # pair on the moments schema and can pass this check
                        # while video_start_time / video_end_time are degenerate.
                        vst = float(label["video_start_time"])
                        vet = float(label["video_end_time"])
                        if vst >= vet:
                            continue
                        st = vst - offset
                        en = vet - offset
                        st = max(0.0, st)
                        en = min(float(dur), en)
                        if en <= st:
                            continue
                        events.append(
                            {
                                "start": st,
                                "end": en,
                                "label": label["label"]
                                .replace("_", " ")
                                .replace("/", "or"),
                            }
                        )
            if uid in data:
                data[uid].extend(events)
            else:
                data[uid] = events
    for uid in data:
        data[uid].sort(key=lambda e: e["start"])
    return data


def _assign_split(video_source: str, fb_participant_id: str) -> str:
    """Deterministic per-participant 60/20/20 split.

    Stateless: a participant's split depends only on (video_source,
    fb_participant_id), so it survives filter / pipeline changes as long
    as the participant still survives.
    """
    key = f"{video_source}::{fb_participant_id}".encode()
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


# ============================================================================
# Main pipeline.
# ============================================================================


def build_manifest(
    *,
    ego4d_json_path: Path,
    audio_manifest_path: Path,
    narration_path: Path,
    moments_paths: list[Path],
    audio_root: Path,
    output_path: Path,
    sample_rate: int = 16000,
    included_universities: tuple[str, ...] | None = INCLUDED_UNIVERSITIES,
) -> dict:
    """Run Stage 3 end-to-end. Writes manifest.json + prints summary log."""
    ego4d = json.loads(Path(ego4d_json_path).read_text())
    audio_manifest = json.loads(Path(audio_manifest_path).read_text())
    narration = json.loads(Path(narration_path).read_text())

    audio_by_uid = audio_manifest["videos"]
    # Map uid -> raw mp4 audio duration (from ego4d.json's video_metadata);
    # used solely to attribute dropped audio hours in the run summary log.
    uid_to_raw_audio_seconds: dict[str, float] = {}
    for v in ego4d["videos"]:
        d = v["video_metadata"]["audio_duration_sec"]
        uid_to_raw_audio_seconds[v["video_uid"]] = float(d) if d else 0.0

    drop_counts: Counter[str] = Counter()
    drop_seconds: dict[str, float] = defaultdict(float)
    kept_records: list[dict] = []
    full_offsets: dict[str, float] = {}  # lead_shift + cross-pass t_start
    durations: dict[str, float] = {}

    def _drop(uid: str, reason: str) -> None:
        drop_counts[reason] += 1
        drop_seconds[reason] += uid_to_raw_audio_seconds.get(uid, 0.0)

    included = set(included_universities) if included_universities else None

    for v in ego4d["videos"]:
        uid = v["video_uid"]

        # 0. university filter (cheap; runs first so OLMo never sees slices
        #    from universities we'll drop downstream)
        if included is not None and v.get("video_source") not in included:
            _drop(uid, "video_source_not_included")
            continue

        # 1. participant id (key always present in ego4d.json; value can be None)
        pid = v["fb_participant_id"]
        if not pid:
            _drop(uid, "no_fb_participant_id")
            continue

        # 2. audio (uid might not be in audio_manifest if Stage 1 skipped it)
        a = audio_by_uid.get(uid)
        if a is None:
            _drop(uid, "uid_not_in_audio_manifest")
            continue
        raw_dur = a["raw_audio_dur_s"]
        if not raw_dur or float(raw_dur) <= 0:
            _drop(uid, "no_raw_audio_dur_s")
            continue
        raw_dur = float(raw_dur)

        # 3. narration (uid might not be in narration.json, or a pass
        #    key may be entirely absent — distinct from an empty
        #    summaries array, but both fail the BOTH-passes requirement).
        narr = narration.get(uid)
        if narr is None:
            _drop(uid, "uid_not_in_narration")
            continue
        if "narration_pass_1" not in narr or not narr["narration_pass_1"].get(
            "summaries"
        ):
            _drop(uid, "pass1_no_summaries")
            continue
        if "narration_pass_2" not in narr or not narr["narration_pass_2"].get(
            "summaries"
        ):
            _drop(uid, "pass2_no_summaries")
            continue

        _, lead_shift = compute_lead_shift(narr)

        # 4 + 5. per-pass shift / dedupe / coverage check / min-slice
        pass1, err1 = _build_pass_slices(
            narr["narration_pass_1"]["summaries"],
            lead_shift,
            uid,
            1,
        )
        if pass1 is None:
            tag = err1.split(":", 1)[0]
            _drop(uid, f"pass1_failed: {tag}")
            continue
        pass2, err2 = _build_pass_slices(
            narr["narration_pass_2"]["summaries"],
            lead_shift,
            uid,
            2,
        )
        if pass2 is None:
            tag = err2.split(":", 1)[0]
            _drop(uid, f"pass2_failed: {tag}")
            continue

        # Cross-pass intersection: timeline lives where BOTH passes cover.
        t_start_post_shift = max(pass1[0]["start_sec"], pass2[0]["start_sec"])
        t_end_post_shift = min(pass1[-1]["end_sec"], pass2[-1]["end_sec"])
        if t_end_post_shift - t_start_post_shift <= 0:
            _drop(uid, "cross_pass_intersection_empty")
            continue

        # Re-coordinate to [0, duration] manifest space by subtracting
        # t_start_post_shift. Drop slices that fall entirely outside this
        # window; clip partial-edge slices.
        def _reframe(slices: list[dict]) -> list[dict]:
            out: list[dict] = []
            for s in slices:
                st = s["start_sec"] - t_start_post_shift
                en = s["end_sec"] - t_start_post_shift
                if en <= 0 or st >= (t_end_post_shift - t_start_post_shift):
                    continue
                st = max(0.0, st)
                en = min(t_end_post_shift - t_start_post_shift, en)
                if en <= st:
                    continue
                out.append(
                    {
                        "start_sec": st,
                        "end_sec": en,
                        "raw_summary": s["raw_summary"],
                        "annotation_uid": s["annotation_uid"],
                    }
                )
            return out

        pass1_reframed = _reframe(pass1)
        pass2_reframed = _reframe(pass2)
        if len(pass1_reframed) < 2 or len(pass2_reframed) < 2:
            _drop(uid, "post_reframe_too_few_slices")
            continue

        audio_offset_s = lead_shift + t_start_post_shift
        duration = t_end_post_shift - t_start_post_shift

        # If the manifest window extends past the raw FLAC, clip-and-keep:
        # the trailing narration past EOF can't be matched against any audio,
        # so we trim ``duration`` to what audio we actually have and drop the
        # tail of pass1/pass2 slices the same way ``_reframe`` would.
        # Drop only if the start offset is itself past EOF (no audio at all).
        audio_avail_s = raw_dur - audio_offset_s
        if audio_avail_s <= 0.05:
            _drop(uid, "audio_offset_beyond_eof")
            continue
        if duration > audio_avail_s + 0.05:
            print(
                f"[build_manifest] WARN uid={uid} clip manifest_window: "
                f"duration {duration:.3f}s -> {audio_avail_s:.3f}s "
                f"(raw_dur={raw_dur:.3f}s, audio_offset={audio_offset_s:.3f}s)",
                flush=True,
            )
            duration = audio_avail_s
            pass1_reframed = [
                {**s, "end_sec": min(s["end_sec"], duration)}
                for s in pass1_reframed
                if s["start_sec"] < duration
            ]
            pass2_reframed = [
                {**s, "end_sec": min(s["end_sec"], duration)}
                for s in pass2_reframed
                if s["start_sec"] < duration
            ]
            if len(pass1_reframed) < 2 or len(pass2_reframed) < 2:
                _drop(uid, "post_clip_too_few_slices")
                continue

        video_source = v["video_source"]
        split = _assign_split(video_source, pid)

        full_offsets[uid] = audio_offset_s
        durations[uid] = duration

        kept_records.append(
            {
                "uid": uid,
                "audio_path": to_repo_relative(
                    audio_root / f"{uid}.flac", "[ego4d.build]"
                ),
                "audio_offset_s": audio_offset_s,
                "duration": duration,
                "sample_rate": sample_rate,
                "scenarios": v["scenarios"],
                "fb_participant_id": pid,
                "video_source": video_source,
                "split": split,
                "passes": {
                    "1": {"slices": pass1_reframed},
                    "2": {"slices": pass2_reframed},
                },
            }
        )

    # Moments: pool per uid, shift by full_offset, clip to duration.
    kept_uids = {r["uid"] for r in kept_records}
    moments_per_uid = _load_moments(
        moments_paths,
        kept_uids,
        full_offsets,
        durations,
    )
    for r in kept_records:
        r["moments"] = moments_per_uid.get(r["uid"], [])

    # Summary stats for log.
    split_counts: Counter[str] = Counter(r["split"] for r in kept_records)
    split_seconds: dict[str, float] = defaultdict(float)
    for r in kept_records:
        split_seconds[r["split"]] += r["duration"]
    source_split: dict[str, Counter] = defaultdict(Counter)
    for r in kept_records:
        source_split[r["video_source"]][r["split"]] += 1

    manifest = {
        "dataset": "Ego4D",
        "description": (
            "Ego4D Stage 3 (deterministic) manifest. No OLMo. Each surviving "
            "uid carries post-shift, dedupe'd, coverage-verified narration "
            "slices for both passes + per-uid moments + split assignment. "
            "All `start_sec`/`end_sec` in `passes` and `moments` live in "
            "manifest-local [0, duration] coordinates; `audio_offset_s` is "
            "the raw-FLAC seek offset where manifest t=0 begins. "
            "Stage 4 (annotate_manifest.py) reads this and emits "
            "annotated_manifest.json with per-slice event + facts + per-"
            "pass coalesced actions."
        ),
        "audio_root": str(audio_root),
        "sample_rate": sample_rate,
        "n_videos": len(kept_records),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_config": {
            "stratify_by": "video_source",
            "group_by": "fb_participant_id",
            "ratios": {"train": 60, "val": 20, "test": 20},
            "method": "deterministic sha256 bucket per (video_source, pid)",
        },
        "videos": kept_records,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2))

    # --- summary log ---
    print()
    print("=" * 80)
    print(f"[build_manifest] DONE. {len(kept_records)} videos kept.")
    print(f"  output: {output_path}")
    total_kept_h = sum(r["duration"] for r in kept_records) / 3600
    print(f"  total kept audio: {total_kept_h:.1f} h (manifest-window)")
    print()
    print("Dropped uids by reason:")
    total_dropped = sum(drop_counts.values())
    total_dropped_h = sum(drop_seconds.values()) / 3600
    for reason, n in drop_counts.most_common():
        hrs = drop_seconds[reason] / 3600
        print(f"  {n:5d} ({hrs:7.1f}h)  {reason}")
    print(f"  {total_dropped:5d} ({total_dropped_h:7.1f}h)  TOTAL DROPPED")
    print()
    print("Split distribution (videos):")
    for split in ("train", "val", "test"):
        n = split_counts[split]
        hrs = split_seconds[split] / 3600
        print(f"  {split:5s}: {n:5d} videos ({hrs:7.1f}h)")
    print()
    print("Per-source split distribution (videos):")
    print(f"  {'video_source':25s} {'train':>7} {'val':>7} {'test':>7}  total")
    for src in sorted(source_split):
        c = source_split[src]
        print(
            f"  {src:25s} {c['train']:7d} {c['val']:7d} {c['test']:7d}  "
            f"{c['train']+c['val']+c['test']}"
        )
    print("=" * 80)

    return manifest


class MissingVideoError(KeyError):
    """Raised when a public-manifest uid isn't in the user's narration.json."""


class MissingAnnotationError(KeyError):
    """Raised when a public-manifest slice's (uid, annotation_uid) isn't
    in the user's narration.json for the expected pass."""


def _build_narration_index(
    narration: dict,
) -> dict[tuple[str, str, str], list[tuple[float, str]]]:
    """Return ``{(uid, pass_num, annotation_uid): [(start_sec, text), ...]}``
    from narration.json. Pass_num is "1" or "2". Only non-empty summaries
    are indexed.

    Values are lists because Ego4D's narration.json occasionally reuses
    the same ``annotation_uid`` for multiple distinct ``summary_text``
    entries within one (uid, pass). The join logic in
    :func:`build_manifest_from_public` resolves the collision by picking
    the candidate whose original ``start_sec`` is closest to the public
    slice's ``start + audio_offset_s`` (the shift Stage 3 subtracted).
    """
    out: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for uid, narr in narration.items():
        for pass_num in ("1", "2"):
            pkey = f"narration_pass_{pass_num}"
            if pkey not in narr:
                continue
            for item in narr[pkey].get("summaries", []):
                text = item["summary_text"].strip()
                if not text:
                    continue
                key = (uid, pass_num, item["annotation_uid"])
                out[key].append((float(item["start_sec"]), text))
    return out


def build_manifest_from_public(
    *,
    public_manifest_path: Path,
    narration_path: Path,
    moments_paths: list[Path],
    audio_root: Path,
    output_path: Path,
) -> dict:
    """Rehydrate a full-shape manifest from the public manifest + the
    user's own Ego4D download (narration.json, moments files).

    All filtering decisions (uid set, splits, slice boundaries, actions)
    come from the public manifest verbatim. The user's narration.json
    supplies the per-slice ``raw_summary`` text; the user's moments
    files supply ``moments`` per uid. ``audio_path`` is rewritten to
    absolute under ``audio_root``.

    Raises :class:`MissingVideoError` if a public uid isn't in the
    user's narration.json, or :class:`MissingAnnotationError` if any
    (uid, pass, annotation_uid) isn't present. Pass ``moments_paths=[]``
    to skip moments repopulation (eval will see an empty moments list).
    """
    public = json.loads(Path(public_manifest_path).read_text())
    narration = json.loads(Path(narration_path).read_text())
    narr_index = _build_narration_index(narration)

    public_videos = public["videos"]
    public_uids = {v["uid"] for v in public_videos}

    missing_in_narration = sorted(public_uids - set(narration.keys()))
    if missing_in_narration:
        raise MissingVideoError(
            f"{len(missing_in_narration)} public uid(s) not in your "
            f"narration.json ({narration_path}). First few: "
            f"{missing_in_narration[:5]}. Make sure you downloaded the "
            f"full Ego4D v2 annotations (`ego4d --output_directory "
            f"<DIR> --datasets annotations`)."
        )

    full_offsets = {v["uid"]: float(v["audio_offset_s"]) for v in public_videos}
    durations = {v["uid"]: float(v["duration"]) for v in public_videos}

    moments_per_uid: dict[str, list[dict]] = {}
    if moments_paths:
        moments_per_uid = _load_moments(
            moments_paths,
            public_uids,
            full_offsets,
            durations,
        )

    kept_records: list[dict] = []
    for v in public_videos:
        uid = v["uid"]
        audio_offset_s = float(v["audio_offset_s"])
        rehydrated_passes: dict[str, dict] = {}
        for pkey, pdata in v["passes"].items():
            new_slices = []
            for s in pdata["slices"]:
                annot_uid = s["annotation_uid"]
                key = (uid, pkey, annot_uid)
                candidates = narr_index.get(key)
                if not candidates:
                    raise MissingAnnotationError(
                        f"public slice (uid={uid}, pass={pkey}, "
                        f"annotation_uid={annot_uid}) not in your "
                        f"narration.json. The slice is listed in the "
                        f"public manifest but its text can't be joined "
                        f"back — either narration.json is from a different "
                        f"Ego4D release than the one used to build the "
                        f"public manifest, or it's incomplete."
                    )
                if len(candidates) == 1:
                    text = candidates[0][1]
                else:
                    # Rare: Ego4D reuses an annotation_uid for multiple
                    # distinct summaries. Pick the one whose original
                    # narration start_sec is closest to the public slice's
                    # start remapped into original narration coords.
                    expected = float(s["start"]) + audio_offset_s
                    text = min(
                        candidates, key=lambda c: abs(c[0] - expected)
                    )[1]
                new_slices.append(
                    {
                        "start": float(s["start"]),
                        "end": float(s["end"]),
                        "annotation_uid": annot_uid,
                        "raw_summary": text,
                    }
                )
            rehydrated_passes[pkey] = {
                "slices": new_slices,
                "actions": pdata["actions"],
            }

        kept_records.append(
            {
                "uid": uid,
                "audio_path": to_repo_relative(
                    audio_root / f"{uid}.flac", "[ego4d.rehydrate]"
                ),
                "audio_offset_s": float(v["audio_offset_s"]),
                "duration": float(v["duration"]),
                "sample_rate": v["sample_rate"],
                "scenarios": v["scenarios"],
                "fb_participant_id": v["fb_participant_id"],
                "video_source": v["video_source"],
                "split": v["split"],
                "passes": rehydrated_passes,
                "moments": moments_per_uid.get(uid, []),
            }
        )

    manifest = {
        "dataset": public["dataset"],
        "description": (
            "Ego4D manifest rehydrated from manifest.public.json + the "
            "user's own Ego4D download. Shape matches a from-scratch "
            "Stage-3 build, additionally carries OLMo-derived "
            "`passes.*.actions` from the public manifest (so Stage 4 can "
            "skip classify and only run AFG to populate per-slice "
            "`facts` for description-quality eval)."
        ),
        "audio_root": str(audio_root),
        "sample_rate": public["sample_rate"],
        "n_videos": len(kept_records),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rehydrated_from": str(public_manifest_path),
        "split_config": public.get("split_config"),
        "videos": kept_records,
    }
    manifest = {k: v for k, v in manifest.items() if v is not None}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2))

    print()
    print("=" * 80)
    print(f"[build_manifest] REHYDRATED from {public_manifest_path}")
    print(f"  wrote {len(kept_records)} videos to {output_path}")
    print(f"  moments: {sum(len(r['moments']) for r in kept_records)} total "
          f"across {sum(1 for r in kept_records if r['moments'])} videos "
          f"({'disabled' if not moments_paths else f'from {len(moments_paths)} file(s)'})")
    print("=" * 80)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    EGO4D_DIR = REPO_ROOT / "datasets" / "ego4d"
    p.add_argument("--ego4d", type=Path, default=EGO4D_DIR / "ego4d.json")
    p.add_argument(
        "--audio-manifest",
        type=Path,
        default=EGO4D_DIR / "audio" / "audio_manifest.json",
    )
    p.add_argument(
        "--narration",
        type=Path,
        default=EGO4D_DIR / "v2" / "annotations" / "narration.json",
    )
    p.add_argument(
        "--moments",
        type=Path,
        nargs="*",
        default=[
            EGO4D_DIR / "v2" / "annotations" / "moments_train.json",
            EGO4D_DIR / "v2" / "annotations" / "moments_val.json",
        ],
    )
    p.add_argument("--audio-root", type=Path, default=EGO4D_DIR / "audio")
    p.add_argument("--output", type=Path, default=EGO4D_DIR / "manifest.json")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--include-universities",
        nargs="+",
        default=list(INCLUDED_UNIVERSITIES),
        help=f"Only keep videos with video_source in this list. Pass "
        f"`--include-universities all` to disable the filter. "
        f"Default: {list(INCLUDED_UNIVERSITIES)}",
    )
    p.add_argument(
        "--public-manifest",
        type=Path,
        default=None,
        help="Rehydrate from the public manifest instead of running the "
        "full Stage-3 build. All filtering, splits, slice boundaries, and "
        "per-pass actions are taken from the public manifest; raw_summary "
        "+ moments are joined from the user's own narration.json / moments "
        "files. Skips the ego4d.json university+pid filter and the "
        "audio_manifest existence check.",
    )
    args = p.parse_args()

    if args.public_manifest is not None:
        build_manifest_from_public(
            public_manifest_path=args.public_manifest,
            narration_path=args.narration,
            moments_paths=list(args.moments),
            audio_root=args.audio_root,
            output_path=args.output,
        )
        return 0

    if args.include_universities == ["all"]:
        included = None
    else:
        included = tuple(args.include_universities)

    build_manifest(
        ego4d_json_path=args.ego4d,
        audio_manifest_path=args.audio_manifest,
        narration_path=args.narration,
        moments_paths=list(args.moments),
        audio_root=args.audio_root,
        output_path=args.output,
        sample_rate=args.sample_rate,
        included_universities=included,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
