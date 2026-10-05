"""Stage 5 of the EgoLife data-prep pipeline: assemble the final manifest.json.

Reads Stage 2's ``audio_manifest.json`` (per-session FLAC paths and
wall-clock windows), Stage 3's translated DenseCaption fragments, and
Stage 4's ``activity_labels.json`` (per-session actions runs). Emits
``<egolife>/manifest.json`` in Ego4D's ``passes.{1}.actions`` +
``passes.{1}.summaries`` shape (single pass — EgoLife has one annotator
source). CPU-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from long_audio.utils.events import tile_by_midpoint
from long_audio.utils.paths import to_repo_relative

DEFAULT_AUDIO_MANIFEST = REPO_ROOT / "datasets" / "egolife" / "audio" / "audio_manifest.json"
DEFAULT_TRANSLATED_ROOT = REPO_ROOT / "datasets" / "egolife" / "translated" / "DenseCaption"
DEFAULT_ACTIVITY_LABELS = REPO_ROOT / "datasets" / "egolife" / "activity_labels.json"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "egolife" / "manifest.json"

# Non-overlapping 5-min tiles for summaries[], matching Stage 4's
# classifier window granularity. actions[] and summaries[] share the
# same tile boundaries via tile_by_midpoint.
SUMMARY_WINDOW_S = 300.0


def _load_translated(root: Path) -> dict[tuple[str, str], list[dict]]:
    """Load bilingual DenseCaption fragments, absolute-wall-clock-timed.

    Raises on any unparseable jsonl filename — Stage 3's output structure
    is fully determined, unexpected names indicate a schema drift."""
    _FN = re.compile(r"^A\d_[A-Z]+_DAY\d+_(\d{2})(\d{2})(\d{2})(\d{2})$")
    by_pd: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for participant_dir in sorted(root.glob("A?_*")):
        if not participant_dir.is_dir():
            continue
        for day_dir in sorted(participant_dir.glob("DAY?")):
            for jl in sorted(day_dir.glob("*.jsonl")):
                m = _FN.match(jl.stem)
                if not m:
                    raise ValueError(
                        f"unparseable translated jsonl filename: {jl}"
                    )
                h, mi, se, ff = (int(x) for x in m.groups())
                hour_offset = h * 3600 + mi * 60 + se + ff / 100.0
                with jl.open(encoding="utf-8") as f:
                    for line in f:
                        e = json.loads(line)
                        e["abs_start_s"] = hour_offset + e["start_s"]
                        e["abs_end_s"] = hour_offset + e["end_s"]
                        by_pd[(participant_dir.name, day_dir.name)].append(e)
    for k in by_pd:
        by_pd[k].sort(key=lambda x: x["abs_start_s"])
    return by_pd


def _fragments_in_session(
    fragments: list[dict],
    session_wall_start_s: float,
    session_dur_s: float,
) -> list[dict]:
    """Return session-local fragments ``[{start, end, text_en, text_zh}, ...]``.

    Fragments must have both ``text_en`` and ``text_zh`` — raises KeyError
    otherwise (Stage 3 guarantees this)."""
    wall_end = session_wall_start_s + session_dur_s
    out = []
    for f in fragments:
        if f["abs_end_s"] <= session_wall_start_s:
            continue
        if f["abs_start_s"] >= wall_end:
            break
        s = max(0.0, f["abs_start_s"] - session_wall_start_s)
        e = min(session_dur_s, f["abs_end_s"] - session_wall_start_s)
        out.append({
            "start": s, "end": e,
            "text_en": f["text_en"],
            "text_zh": f["text_zh"],
        })
    return out


def _build_summaries(
    fragments: list[dict],
    session_dur_s: float,
    stage4_windows: list[dict],
    window_s: float = SUMMARY_WINDOW_S,
) -> list[dict]:
    """Aggregate fragments into non-overlapping 5-min tiles + attach the
    per-tile ``facts[]`` produced by Stage 4's AFG step.

    Tiles share boundaries with Stage 4's classifier/AFG windows so
    ``actions[]``, ``summaries[]``, and ``stage4_windows`` all align 1:1
    by construction (all three consume ``tile_by_midpoint`` with the
    same session_dur_s + window_s). Enforced with a length assert.

    Raises on empty tiles — an empty summary means an upstream coverage
    or alignment bug (same rationale as annotate_manifest._window_fragments)."""
    if not fragments:
        raise ValueError(
            f"no translated fragments provided (session_dur_s={session_dur_s})"
        )
    tiles = tile_by_midpoint(fragments, session_dur_s, window_s)
    if len(tiles) != len(stage4_windows):
        raise RuntimeError(
            f"Stage 4 window count ({len(stage4_windows)}) != Stage 5 tile "
            f"count ({len(tiles)}) for session_dur_s={session_dur_s:.1f}. "
            f"tile_by_midpoint must produce identical partitions given the "
            f"same (session_dur_s, window_s)."
        )
    summaries = []
    for (t_start, t_end, tile_frags), w in zip(tiles, stage4_windows):
        # Boundary sanity check: Stage 4 and Stage 5 tile edges should match.
        if abs(t_start - w["start"]) > 1e-6 or abs(t_end - w["end"]) > 1e-6:
            raise RuntimeError(
                f"tile boundary mismatch: stage5=[{t_start:.6f}-{t_end:.6f}] "
                f"vs stage4=[{w['start']:.6f}-{w['end']:.6f}]"
            )
        # text_en for downstream eval is Stage 4's CLEANED (wearer-aligned)
        # text, so AFG atoms + summary prose stay consistent. Raw
        # concatenation stays available on Stage 4's windows for provenance.
        text_en = w["cleaned_text"]
        text_zh = " ".join(f["text_zh"] for f in tile_frags)
        if not text_en:
            raise ValueError(
                f"empty summary tile [{t_start:.1f}-{t_end:.1f}] in "
                f"session_dur_s={session_dur_s:.1f} — DenseCaption coverage "
                f"gap or wall-clock alignment drift"
            )
        summaries.append({
            "start": t_start, "end": t_end,
            "text_en": text_en, "text_zh": text_zh,
            "facts": w["facts"],
        })
    return summaries


class MissingVideoError(KeyError):
    """Public manifest references a uid absent from the user's local
    audio_manifest.json or activity_labels.json."""


def build_manifest_from_public(
    *,
    public_manifest_path: Path,
    audio_manifest_path: Path,
    translated_root: Path,
    activity_labels_path: Path,
    output_path: Path,
    summary_window_s: float = SUMMARY_WINDOW_S,
) -> dict:
    """Rehydrate the full EgoLife manifest from the public manifest plus
    the user's own Stage-3 (translated DenseCaption) and Stage-4
    (activity_labels.json) outputs.

    The public manifest is the source of truth for the uid set, tile
    boundaries, and per-pass ``actions``. Text (``text_en``, ``text_zh``)
    and ``facts`` are populated per-tile from the user's own data.
    Validates that tile boundaries align with what Stage 4 / 5 would
    produce from the user's data and raises on mismatch.
    """
    public = json.loads(Path(public_manifest_path).read_text())
    audio_manifest = json.loads(Path(audio_manifest_path).read_text())
    activity_labels = json.loads(Path(activity_labels_path).read_text())
    activity_by_uid = activity_labels["sessions"]
    audio_by_uid = {s["uid"]: s for s in audio_manifest["sessions"]}

    frags_by_pd = _load_translated(translated_root)

    public_uids = [v["uid"] for v in public["videos"]]
    missing_audio = [u for u in public_uids if u not in audio_by_uid]
    if missing_audio:
        raise MissingVideoError(
            f"{len(missing_audio)} public uid(s) absent from your "
            f"audio_manifest.json ({audio_manifest_path}). First few: "
            f"{missing_audio[:5]}. Rerun scripts/data/egolife/"
            f"extract_audio.py over your EgoLife download."
        )
    missing_activity = [u for u in public_uids if u not in activity_by_uid]
    if missing_activity:
        raise MissingVideoError(
            f"{len(missing_activity)} public uid(s) absent from your "
            f"activity_labels.json ({activity_labels_path}). First few: "
            f"{missing_activity[:5]}. Rerun scripts/data/egolife/"
            f"annotate_manifest.py."
        )

    videos: list[dict[str, Any]] = []
    for pub_v in public["videos"]:
        uid = pub_v["uid"]
        sess = audio_by_uid[uid]
        act = activity_by_uid[uid]

        if act["error"] is not None:
            raise RuntimeError(
                f"session {uid} has Stage-4 error={act['error']!r}; the "
                f"public manifest shipped it as a survivor, so your Stage-4 "
                f"run disagrees — rerun annotate_manifest.py."
            )

        pd_key = (sess["participant"], sess["day"])
        if pd_key not in frags_by_pd:
            raise MissingVideoError(
                f"session {uid}: no translated fragments for {pd_key}. "
                f"Rerun scripts/data/egolife/translate_captions.py."
            )
        session_frags = _fragments_in_session(
            frags_by_pd[pd_key],
            sess["wall_clock_start_s"],
            sess["raw_audio_dur_s"],
        )
        summaries = _build_summaries(
            session_frags,
            sess["raw_audio_dur_s"],
            act["windows"],
            summary_window_s,
        )

        pub_summaries = pub_v["passes"]["1"]["summaries"]
        if len(summaries) != len(pub_summaries):
            raise RuntimeError(
                f"uid={uid}: tile count mismatch (built {len(summaries)}, "
                f"public {len(pub_summaries)}). Your Stage-4 window_s or "
                f"session_dur_s doesn't match the public manifest's."
            )
        for s, p in zip(summaries, pub_summaries):
            if abs(s["start"] - p["start"]) > 1e-6 or abs(s["end"] - p["end"]) > 1e-6:
                raise RuntimeError(
                    f"uid={uid}: tile boundary mismatch "
                    f"(built [{s['start']:.6f}-{s['end']:.6f}], "
                    f"public [{p['start']:.6f}-{p['end']:.6f}])"
                )

        videos.append({
            "uid": uid,
            "audio_path": to_repo_relative(
                sess["audio_path"], "[egolife.rehydrate]"
            ),
            "sample_rate": public["sample_rate"],
            "duration": sess["raw_audio_dur_s"],
            "audio_offset_s": 0.0,
            "wall_clock_start": sess["wall_clock_start"],
            "wall_clock_end": sess["wall_clock_end"],
            "scenarios": ["egolife_home"],
            "participant": sess["participant"],
            "day": sess["day"],
            "session_idx": sess["session_idx"],
            "clip_offsets": sess["clips"],
            "passes": {
                "1": {
                    "actions": pub_v["passes"]["1"]["actions"],
                    "summaries": summaries,
                }
            },
        })

    manifest = {
        "dataset": public["dataset"],
        "version": public.get("version", "v1"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": public["sample_rate"],
        "channels": audio_manifest["channels"],
        "n_videos": len(videos),
        "summary_window_s": summary_window_s,
        "rehydrated_from": str(public_manifest_path),
        "videos": videos,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2))
    print(
        f"[build] REHYDRATED from {public_manifest_path}: "
        f"{len(videos)} videos -> {output_path}",
        flush=True,
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-manifest", type=Path, default=DEFAULT_AUDIO_MANIFEST)
    p.add_argument("--translated-root", type=Path, default=DEFAULT_TRANSLATED_ROOT)
    p.add_argument("--activity-labels", type=Path, default=DEFAULT_ACTIVITY_LABELS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summary-window-s", type=float, default=SUMMARY_WINDOW_S)
    p.add_argument(
        "--public-manifest",
        type=Path,
        default=None,
        help="Rehydrate from the public manifest instead of running the "
        "full Stage-5 build. Public manifest supplies uid set + tile "
        "boundaries + per-pass actions; your own translated fragments + "
        "activity_labels.json supply text + facts.",
    )
    args = p.parse_args()

    if args.public_manifest is not None:
        build_manifest_from_public(
            public_manifest_path=args.public_manifest,
            audio_manifest_path=args.audio_manifest,
            translated_root=args.translated_root,
            activity_labels_path=args.activity_labels,
            output_path=args.output,
            summary_window_s=args.summary_window_s,
        )
        return 0

    audio_manifest = json.loads(args.audio_manifest.read_text())
    activity_labels = json.loads(args.activity_labels.read_text())
    activity_by_uid = activity_labels["sessions"]

    print(f"[build] loading translated fragments from {args.translated_root}",
          flush=True)
    frags_by_pd = _load_translated(args.translated_root)

    # Verify Stage 2 <-> Stage 4 uid alignment. Any mismatch is a bug in
    # one of the upstream stages, not a runtime condition to swallow.
    stage2_uids = {s["uid"] for s in audio_manifest["sessions"]}
    stage4_uids = set(activity_by_uid.keys())
    if stage2_uids != stage4_uids:
        missing_in_stage4 = stage2_uids - stage4_uids
        extra_in_stage4 = stage4_uids - stage2_uids
        raise RuntimeError(
            f"audio_manifest and activity_labels uid sets differ. "
            f"missing_in_stage4={sorted(missing_in_stage4)[:5]}... "
            f"extra_in_stage4={sorted(extra_in_stage4)[:5]}..."
        )

    # Sessions Stage 4 dropped with a NAMED, EXPECTED failure mode are the
    # only permitted skips. Any other error is a bug — raise.
    ALLOWED_STAGE4_DROP_REASONS = {"parse_failure"}

    videos: list[dict[str, Any]] = []
    n_dropped_by_reason: dict[str, int] = {r: 0 for r in ALLOWED_STAGE4_DROP_REASONS}
    for sess in audio_manifest["sessions"]:
        uid = sess["uid"]
        act = activity_by_uid[uid]  # KeyError = uid mismatch (already guarded)

        stage4_error = act["error"]  # required field
        if stage4_error is not None:
            if stage4_error not in ALLOWED_STAGE4_DROP_REASONS:
                raise RuntimeError(
                    f"session {uid} has unexpected Stage 4 error: "
                    f"{stage4_error!r}. Allowed: {ALLOWED_STAGE4_DROP_REASONS}"
                )
            n_dropped_by_reason[stage4_error] += 1
            continue

        actions = act["actions"]
        if not actions:
            raise RuntimeError(
                f"session {uid} has Stage 4 error=None but empty actions[] "
                f"— Stage 4 invariant violated"
            )

        pd_key = (sess["participant"], sess["day"])
        if pd_key not in frags_by_pd:
            raise RuntimeError(
                f"session {uid} has no translated fragments for {pd_key} "
                f"— Stage 3 output missing"
            )
        session_frags = _fragments_in_session(
            frags_by_pd[pd_key],
            sess["wall_clock_start_s"],
            sess["raw_audio_dur_s"],
        )
        summaries = _build_summaries(
            session_frags, sess["raw_audio_dur_s"],
            act["windows"], args.summary_window_s,
        )

        videos.append({
            "uid": uid,
            "audio_path": to_repo_relative(
                sess["audio_path"], "[egolife.build]"
            ),
            "sample_rate": audio_manifest["sample_rate"],
            "duration": sess["raw_audio_dur_s"],
            "audio_offset_s": 0.0,
            "wall_clock_start": sess["wall_clock_start"],
            "wall_clock_end": sess["wall_clock_end"],
            "scenarios": ["egolife_home"],
            "participant": sess["participant"],
            "day": sess["day"],
            "session_idx": sess["session_idx"],
            "clip_offsets": sess["clips"],
            "passes": {
                "1": {
                    "actions": actions,
                    "summaries": summaries,
                }
            },
            "moments": [],
        })

    manifest = {
        "dataset": "egolife",
        "version": "v1",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": audio_manifest["sample_rate"],
        "channels": audio_manifest["channels"],
        "n_videos": len(videos),
        "n_dropped_by_reason": n_dropped_by_reason,
        "summary_window_s": args.summary_window_s,
        "videos": videos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    drop_summary = ", ".join(
        f"{n} {r}" for r, n in n_dropped_by_reason.items() if n > 0
    ) or "no drops"
    print(
        f"[build] DONE. {len(videos)} sessions in manifest "
        f"(Stage-4 drops: {drop_summary}). "
        f"-> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
