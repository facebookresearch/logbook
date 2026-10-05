# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 2 of the EgoLife data-prep pipeline: intersect audio ↔ caption
coverage, split into sessions, extract FLAC.

For each participant-day, sorts the 30-second mp4 clips, computes real
inter-clip audio gaps, intersects them with DenseCaption coverage
intervals, and emits one FLAC per resulting session via ``ffmpeg``
concat-demuxer. Sessions are ``A{i}_{NAME}_DAY{d}_S{k:02d}``. Outputs
per-session FLACs plus ``<output_root>/audio_manifest.json`` describing
the sessions and their constituent clips.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from long_audio.utils.paths import to_repo_relative  # noqa: E402

EGOLIFE_DIR = REPO_ROOT / "datasets" / "egolife"
DEFAULT_RAW = EGOLIFE_DIR / "raw"
DEFAULT_OUTPUT = EGOLIFE_DIR / "audio"

SAMPLE_RATE = 16000  # Hz, matches Ego4D / SINS
CHANNELS = 1         # mono
AUDIO_GAP_THRESHOLD_S = 1.0
CAPTION_MERGE_GAP_S = 60.0

# DAY{d}_A{i}_{NAME}_{HHMMSSFF}.mp4
_FN_RE = re.compile(r"^DAY(\d+)_A(\d)_([A-Z]+)_(\d{6,8})(?:_cur)?\.mp4$")
# SRT filename: A{i}_{NAME}_DAY{d}_{HHMMSSFF}.srt
_SRT_FN_RE = re.compile(r"^A\d_[A-Z]+_DAY\d+_(\d{2})(\d{2})(\d{2})(\d{2})$")
# SRT timestamp: HH:MM:SS,mmm --> HH:MM:SS,mmm
_SRT_TS_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


from long_audio.utils.events import intersect_intervals, merge_intervals


def _resolve_ffmpeg() -> str:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not on PATH and imageio_ffmpeg not installed."
        ) from e


def _resolve_ffprobe(ffmpeg: str) -> str:
    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError(
        f"ffprobe not on PATH and not found next to ffmpeg at {ffmpeg}"
    )


def _parse_filename(name: str) -> tuple[int, str, str, float] | None:
    """Return (day, participant, name, wall_clock_start_s) or None if unmatched."""
    m = _FN_RE.match(name)
    if not m:
        return None
    day, p_idx, p_name, ts = m.groups()
    ts = ts.zfill(8)  # normalize to HHMMSSFF (8 chars, FF = centiseconds)
    h, mi, se, ff = int(ts[0:2]), int(ts[2:4]), int(ts[4:6]), int(ts[6:8])
    wall_s = h * 3600 + mi * 60 + se + ff / 100.0
    return int(day), f"A{p_idx}_{p_name}", p_name, wall_s


def _probe_duration(ffprobe: str, path: Path) -> float:
    """Return the audio duration in seconds. Raises on any failure — no
    silent None, per data-prep fail-loudly discipline."""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path}: returncode={proc.returncode} "
            f"stderr={proc.stderr.strip()[:300]!r}"
        )
    return float(proc.stdout.strip())


def _fmt_wall(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def _discover_clips(
    raw_root: Path,
    ffprobe: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return ``{(participant, day): [{filename, wall_start_s, duration_s, path}, ...]}``.

    Discovers all mp4s under the raw snapshot, parses filenames, probes
    real per-clip durations. Clips are sorted by wall-clock start.
    """
    print(f"[extract] scanning {raw_root}", flush=True)
    files: list[Path] = []
    for participant_dir in sorted(raw_root.glob("A?_*")):
        if not participant_dir.is_dir():
            continue
        for day_dir in sorted(participant_dir.glob("DAY?")):
            files.extend(sorted(day_dir.glob("*.mp4")))
    print(f"[extract] discovered {len(files)} mp4 files", flush=True)

    parsed: list[tuple[Path, int, str, float]] = []
    for f in files:
        info = _parse_filename(f.name)
        if info is None:
            raise ValueError(
                f"unparseable mp4 filename: {f}. "
                f"Expected DAY{{d}}_A{{i}}_{{NAME}}_{{HHMMSSFF}}.mp4"
            )
        day, participant, _, wall_s = info
        parsed.append((f, day, participant, wall_s))

    print(f"[extract] probing durations for {len(parsed)} clips "
          f"(parallel workers used later; this pass is single-threaded)",
          flush=True)
    by_pd: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for i, (path, day, participant, wall_s) in enumerate(parsed):
        dur = _probe_duration(ffprobe, path)  # raises on failure
        by_pd[(participant, f"DAY{day}")].append({
            "filename": path.name,
            "path": str(path),
            "wall_start_s": wall_s,
            "duration_s": dur,
        })
        if (i + 1) % 500 == 0:
            print(f"[extract] probed {i + 1}/{len(parsed)}", flush=True)

    # sort each participant-day by wall_start
    for k in by_pd:
        by_pd[k].sort(key=lambda c: c["wall_start_s"])
    print(f"[extract] {len(by_pd)} participant-days after probing", flush=True)
    return by_pd


def _split_into_audio_sessions(
    clips: list[dict[str, Any]],
    gap_threshold_s: float,
) -> list[list[dict[str, Any]]]:
    """Split a participant-day's chronologically-sorted clips into
    audio-only sessions at any real audio gap > ``gap_threshold_s``.
    Later intersected with caption coverage in ``_intersect_with_captions``.

    Real gap between clip i and clip i+1 =
        clips[i+1].wall_start_s - (clips[i].wall_start_s + clips[i].duration_s)
    """
    if not clips:
        return []
    sessions: list[list[dict[str, Any]]] = [[clips[0]]]
    for prev, curr in zip(clips, clips[1:]):
        gap = curr["wall_start_s"] - (prev["wall_start_s"] + prev["duration_s"])
        if gap > gap_threshold_s:
            sessions.append([curr])
        else:
            sessions[-1].append(curr)
    return sessions


def _parse_srt_ts_only(text: str, src: Path) -> list[tuple[float, float]]:
    """Return (start_s, end_s) for every subtitle entry in an SRT file.
    Timestamp-only — we don't care about the Chinese text at Stage 2.
    Raises on malformed entries."""
    out: list[tuple[float, float]] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        try:
            int(lines[i].strip())  # index line
        except ValueError as e:
            raise ValueError(f"{src}: expected int index at line {i + 1}") from e
        i += 1
        if i >= len(lines):
            raise ValueError(f"{src}: EOF after index, expected timestamp")
        m = _SRT_TS_RE.match(lines[i])
        if not m:
            raise ValueError(f"{src}: expected SRT timestamp at line {i + 1}")
        h1, mi1, s1, ms1, h2, mi2, s2, ms2 = m.groups()
        start = int(h1) * 3600 + int(mi1) * 60 + int(s1) + int(ms1) / 1000.0
        end = int(h2) * 3600 + int(mi2) * 60 + int(s2) + int(ms2) / 1000.0
        out.append((start, end))
        i += 1
        while i < len(lines) and lines[i].strip():
            i += 1
    return out


def _load_caption_coverage(
    raw_root: Path,
    merge_gap_s: float,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Read DenseCaption SRT files (Chinese-side, timestamps only) and
    build merged caption-coverage intervals per (participant, day).

    Each SRT filename encodes the wall-clock hour (``A1_JAKE_DAY1_HHMMSSFF``).
    Absolute wall-clock times are computed by adding the hour offset to
    each entry's SRT-file-local timestamp. Fragments joined into disjoint
    coverage intervals via ``merge_intervals(..., max_gap=merge_gap_s)``.
    """
    dc_root = raw_root / "EgoLifeCap" / "DenseCaption"
    by_pd: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for pdir in sorted(dc_root.glob("A?_*")):
        if not pdir.is_dir():
            continue
        for ddir in sorted(pdir.glob("DAY?")):
            for srt in sorted(ddir.glob("*.srt")):
                m = _SRT_FN_RE.match(srt.stem)
                if not m:
                    raise ValueError(f"unparseable SRT filename: {srt}")
                h, mi, se, ff = (int(x) for x in m.groups())
                hour_offset = h * 3600 + mi * 60 + se + ff / 100.0
                entries = _parse_srt_ts_only(srt.read_text(encoding="utf-8"), srt)
                for start, end in entries:
                    by_pd[(pdir.name, ddir.name)].append(
                        (hour_offset + start, hour_offset + end)
                    )
    coverage: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for pd, frags in by_pd.items():
        coverage[pd] = merge_intervals(frags, max_gap=merge_gap_s)
    return coverage


def _intersect_with_captions(
    audio_sessions: list[list[dict[str, Any]]],
    caption_coverage: list[tuple[float, float]],
) -> list[tuple[float, float, list[dict[str, Any]]]]:
    """For each audio session, intersect its wall-clock range with the
    caption-coverage intervals; each non-empty overlap becomes a session.

    Returns ``[(sub_start, sub_end, clips_in_sub), ...]`` in chronological
    order. Clips assigned to sub-sessions by midpoint rule; clips whose
    midpoint falls outside all coverage intervals are dropped.
    """
    out: list[tuple[float, float, list[dict[str, Any]]]] = []
    for clips in audio_sessions:
        if not clips:
            continue
        audio_start = clips[0]["wall_start_s"]
        audio_end = clips[-1]["wall_start_s"] + clips[-1]["duration_s"]
        sub_ranges = intersect_intervals(
            (audio_start, audio_end), caption_coverage
        )
        for sub_start, sub_end in sub_ranges:
            sub_clips = [
                c for c in clips
                if sub_start <= (c["wall_start_s"] + c["duration_s"] / 2) < sub_end
            ]
            if not sub_clips:
                continue
            out.append((sub_start, sub_end, sub_clips))
    return out


def _stitch_session(
    ffmpeg: str,
    ffprobe: str,
    clips: list[dict[str, Any]],
    out_path: Path,
) -> float:
    """ffmpeg concat-demuxer -> single mono 16 kHz FLAC. Returns actual
    duration in seconds. Raises on any failure — no soft-error return."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=out_path.parent, delete=False
    ) as tmpf:
        for c in clips:
            # Escape single quotes per ffmpeg concat protocol spec.
            escaped = c["path"].replace("'", "'\\''")
            tmpf.write(f"file '{escaped}'\n")
        list_path = tmpf.name

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-vn",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-c:a", "flac",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            out_path.unlink(missing_ok=True)
            msg = (proc.stderr or "").strip().splitlines()
            last = msg[-1] if msg else f"returncode={proc.returncode}"
            raise RuntimeError(f"ffmpeg concat failed for {out_path}: {last[:300]}")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced empty output at {out_path}")
    finally:
        Path(list_path).unlink()  # tempfile guaranteed to exist; raise if not

    return _probe_duration(ffprobe, out_path)  # raises on failure


def _process_participant_day(
    participant: str,
    day: str,
    clips: list[dict[str, Any]],
    caption_coverage: list[tuple[float, float]],
    output_root: Path,
    audio_gap_threshold_s: float,
    skip_existing: bool,
    ffmpeg: str,
    ffprobe: str,
) -> list[dict[str, Any]]:
    """Process one (participant, day):
      1. Split clips into audio sessions (audio-gap > threshold).
      2. Intersect each audio session with caption-coverage intervals.
      3. Stitch each resulting sub-session's clips into a FLAC.

    Sub-sessions are renumbered contiguously (S01, S02, ...) across the
    participant-day. Raises on any per-session failure."""
    audio_sessions = _split_into_audio_sessions(clips, audio_gap_threshold_s)
    subs = _intersect_with_captions(audio_sessions, caption_coverage)
    records = []
    for k, (sub_start, sub_end, sub_clips) in enumerate(subs, start=1):
        uid = f"{participant}_{day}_S{k:02d}"
        out_path = output_root / participant / day / f"S{k:02d}.flac"

        # Precompute session-relative audio offsets for the wall-clock map.
        # Sum of prior clips' actual durations (concat is contiguous audio).
        audio_offset = 0.0
        clip_records = []
        for c in sub_clips:
            clip_records.append({
                "filename": c["filename"],
                "wall_start_s": c["wall_start_s"],
                "duration_s": c["duration_s"],
                "audio_offset_s": audio_offset,
            })
            audio_offset += c["duration_s"]

        # Wall-clock bounds of the sub-session = intersection endpoints
        # (may be tighter than the clip range if the caption coverage
        # trims a partial clip at each edge).
        clip_wall_start = sub_clips[0]["wall_start_s"]
        clip_wall_end = sub_clips[-1]["wall_start_s"] + sub_clips[-1]["duration_s"]

        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            dur = _probe_duration(ffprobe, out_path)  # raises on unprobeable
        else:
            dur = _stitch_session(ffmpeg, ffprobe, sub_clips, out_path)

        records.append({
            "uid": uid,
            "participant": participant,
            "day": day,
            "session_idx": k,
            "audio_path": to_repo_relative(out_path, "[egolife.extract]"),
            "raw_audio_dur_s": dur,
            "wall_clock_start": _fmt_wall(clip_wall_start),
            "wall_clock_end": _fmt_wall(clip_wall_end),
            "wall_clock_start_s": clip_wall_start,
            "wall_clock_end_s": clip_wall_end,
            "caption_coverage_wall_start_s": sub_start,
            "caption_coverage_wall_end_s": sub_end,
            "clips": clip_records,
        })

    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW,
                   help="Root of the downloaded HF snapshot "
                        "(contains A?_*/DAY?/*.mp4).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audio-gap-threshold-s", type=float,
                   default=AUDIO_GAP_THRESHOLD_S,
                   help="Split audio sessions at any real inter-clip audio "
                        "gap > this many seconds. Default 1.0.")
    p.add_argument("--caption-merge-gap-s", type=float,
                   default=CAPTION_MERGE_GAP_S,
                   help="Merge DenseCaption fragments into disjoint coverage "
                        "intervals joining any two whose gap is <= this. "
                        "Default 60.0.")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel ffmpeg concat processes (one per participant-"
                        "day chunk).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N participant-days (smoke).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip sessions whose .flac already exists.")
    args = p.parse_args()

    try:
        ffmpeg = _resolve_ffmpeg()
        ffprobe = _resolve_ffprobe(ffmpeg)
    except RuntimeError as e:
        print(f"[extract] {e}", file=sys.stderr)
        return 2
    print(f"[extract] using ffmpeg: {ffmpeg}", flush=True)
    print(f"[extract] using ffprobe: {ffprobe}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)

    by_pd = _discover_clips(args.raw_root, ffprobe)
    if args.limit:
        by_pd = dict(list(by_pd.items())[: args.limit])

    print(f"[extract] loading DenseCaption coverage from {args.raw_root}/EgoLifeCap/DenseCaption",
          flush=True)
    caption_coverage = _load_caption_coverage(
        args.raw_root, merge_gap_s=args.caption_merge_gap_s
    )
    total_caption_h = sum(
        sum(e - s for s, e in intervals) / 3600 for intervals in caption_coverage.values()
    )
    print(f"[extract] caption coverage: {len(caption_coverage)} pds, "
          f"{total_caption_h:.1f} h total (merge_gap_s={args.caption_merge_gap_s})",
          flush=True)

    t0 = time.time()
    all_sessions: list[dict[str, Any]] = []
    n_clips_used = sum(len(v) for v in by_pd.values())

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _process_participant_day,
                participant, day, clips,
                caption_coverage.get((participant, day), []),
                args.output_root, args.audio_gap_threshold_s, args.skip_existing,
                ffmpeg, ffprobe,
            ): (participant, day)
            for (participant, day), clips in by_pd.items()
        }
        # fut.result() re-raises worker exceptions here; the whole job
        # aborts on the first per-day failure (fail-loudly discipline).
        for i, fut in enumerate(as_completed(futures), start=1):
            participant, day = futures[fut]
            records = fut.result()
            all_sessions.extend(records)
            elapsed = time.time() - t0
            print(
                f"[extract] {i}/{len(futures)} pd-groups done "
                f"({participant}/{day}: {len(records)} sessions) | "
                f"{elapsed:.0f}s elapsed",
                flush=True,
            )

    manifest = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "audio_gap_threshold_s": args.audio_gap_threshold_s,
        "caption_merge_gap_s": args.caption_merge_gap_s,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "n_participant_days": len(by_pd),
        "n_sessions": len(all_sessions),
        "n_clips_used": n_clips_used,
        "sessions": all_sessions,
    }
    manifest_path = args.output_root / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[extract] DONE in {time.time() - t0:.1f}s. "
        f"{len(all_sessions)} sessions extracted. "
        f"Manifest: {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
