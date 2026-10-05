# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 2 of the Ego4D data-prep pipeline: extract raw FLAC audio.

For each ``video_uid`` in ``ego4d.json`` that has an audio stream in its
mp4 metadata, transcodes the full audio track to mono 16 kHz FLAC and
records the real duration via ``ffprobe``. No narration-aware clipping
here — Stage 3 (``build_manifest.py``) owns that. Writes one FLAC per
extracted uid and an aggregate ``<output_root>/audio_manifest.json``
with per-uid extraction status and ``raw_audio_dur_s``. ffmpeg/ffprobe
binaries are auto-located from ``imageio_ffmpeg`` if not on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from long_audio.utils.paths import to_repo_relative  # noqa: E402

EGO4D_DIR = REPO_ROOT / "datasets" / "ego4d"
DEFAULT_EGO4D = EGO4D_DIR / "ego4d.json"
DEFAULT_VIDEO_ROOT = EGO4D_DIR / "v2"
DEFAULT_OUTPUT = EGO4D_DIR / "audio"

SAMPLE_RATE = 16000  # Hz, fixed per project decision
CHANNELS = 1         # mono, matches SINS pipeline


def _resolve_ffmpeg() -> str:
    """Return a usable ffmpeg path. PATH first, then imageio_ffmpeg's
    bundled static binary."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not on PATH and imageio_ffmpeg not installed. "
            "Install ffmpeg system-wide or `pip install imageio-ffmpeg`."
        ) from e


def _resolve_ffprobe(ffmpeg: str) -> str:
    """PATH first; else sibling ``ffprobe`` next to the resolved ffmpeg."""
    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError(
        f"ffprobe not on PATH and not found next to ffmpeg at {ffmpeg}"
    )


def _locate_mp4(uid: str, video_root: Path) -> Path | None:
    for sub in ("video_540ss", "full_scale"):
        p = video_root / sub / f"{uid}.mp4"
        if p.is_file():
            return p
    return None


def _probe_duration(ffprobe: str, path: Path, uid: str = "<unknown>") -> float | None:
    """Return the audio container/stream duration in seconds, or None on
    parse failure. Emits a per-uid WARN to stderr on any failure path so
    the issue isn't silently absorbed by the caller's bookkeeping."""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print(
            f"[extract] WARN uid={uid} ffprobe TimeoutExpired (>30s) on {path}",
            file=sys.stderr, flush=True,
        )
        return None
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:200]
        print(
            f"[extract] WARN uid={uid} ffprobe returncode={proc.returncode} stderr={err!r}",
            file=sys.stderr, flush=True,
        )
        return None
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError) as e:
        print(
            f"[extract] WARN uid={uid} ffprobe parse error {type(e).__name__}: {str(e)[:200]}",
            file=sys.stderr, flush=True,
        )
        return None


def _extract_flac(ffmpeg: str, mp4: Path, out_path: Path) -> tuple[bool, str | None]:
    """ffmpeg → mono 16 kHz FLAC, full mp4 audio (no -ss / -to). Returns
    (success, error)."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(mp4),
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "flac",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        out_path.unlink(missing_ok=True)
        return False, "ffmpeg timeout (>600s)"
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        msg = (proc.stderr or "").strip().splitlines()
        last = msg[-1] if msg else f"ffmpeg returncode={proc.returncode}"
        return False, last[:300]
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return False, "ffmpeg produced empty output"
    return True, None


def _process_one(
    uid: str,
    video_root: Path,
    output_root: Path,
    has_audio_in_metadata: bool,
    skip_existing: bool,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    """Process one uid; return its manifest entry."""
    entry: dict[str, Any] = {
        "uid": uid,
        "mp4_path": None,
        "has_audio_in_metadata": has_audio_in_metadata,
        "extracted_path": None,
        "raw_audio_dur_s": None,
        "error": None,
    }
    mp4 = _locate_mp4(uid, video_root)
    if mp4 is None:
        entry["error"] = "no_mp4"
        return entry
    entry["mp4_path"] = str(mp4)
    if not has_audio_in_metadata:
        entry["error"] = "no_audio_in_metadata"
        return entry

    out_path = output_root / f"{uid}.flac"
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        existing_dur = _probe_duration(ffprobe, out_path, uid=uid)
        entry["extracted_path"] = to_repo_relative(out_path, "[ego4d.extract]")
        entry["raw_audio_dur_s"] = (
            float(existing_dur) if existing_dur is not None else None
        )
        if existing_dur is None:
            entry["error"] = "existing_flac_unprobeable"
        return entry

    ok, err = _extract_flac(ffmpeg, mp4, out_path)
    if not ok:
        entry["error"] = err or "ffmpeg_failed"
        return entry
    actual_dur = _probe_duration(ffprobe, out_path, uid=uid)
    entry["extracted_path"] = to_repo_relative(out_path, "[ego4d.extract]")
    entry["raw_audio_dur_s"] = (
        float(actual_dur) if actual_dur is not None else None
    )
    if actual_dur is None:
        entry["error"] = "post_extract_probe_failed"
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ego4d", type=Path, default=DEFAULT_EGO4D)
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT,
                   help="Root containing video_540ss/ and full_scale/.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel ffmpeg processes (1 CPU core each).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N videos (smoke).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip uids whose <uid>.flac already exists. The "
                        "existing FLAC's duration is still probed + recorded.")
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

    print(f"[extract] loading {args.ego4d}", flush=True)
    ego = json.loads(args.ego4d.read_text())
    videos = ego["videos"]
    if args.limit:
        videos = videos[: args.limit]
    print(f"[extract] {len(videos)} videos in ego4d.json (after --limit)",
          flush=True)

    work: list[tuple[str, bool]] = []
    for v in videos:
        uid = v["video_uid"]
        dur = v["video_metadata"]["audio_duration_sec"]  # may be None for silent
        has_audio_md = bool(dur and dur > 0)
        work.append((uid, has_audio_md))

    t0 = time.time()
    entries: dict[str, dict[str, Any]] = {}
    n_done = n_extracted = n_no_mp4 = n_no_audio_md = n_errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_one, uid, args.video_root, args.output_root,
                      has_audio_md, args.skip_existing, ffmpeg, ffprobe): uid
            for uid, has_audio_md in work
        }
        for fut in as_completed(futures):
            entry = fut.result()
            uid = entry["uid"]
            entries[uid] = entry
            n_done += 1
            if entry["extracted_path"] and not entry["error"]:
                n_extracted += 1
            elif entry["error"] == "no_mp4":
                n_no_mp4 += 1
            elif entry["error"] == "no_audio_in_metadata":
                n_no_audio_md += 1
            else:
                n_errors += 1
            if n_done % 100 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed else 0
                eta_min = ((len(work) - n_done) / rate / 60) if rate else 0
                print(
                    f"[extract] {n_done}/{len(work)} | "
                    f"ok={n_extracted} no_mp4={n_no_mp4} "
                    f"no_audio_md={n_no_audio_md} err={n_errors} | "
                    f"{rate:.1f} vid/s | eta {eta_min:.1f} min",
                    flush=True,
                )

    manifest = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ego4d_path": str(args.ego4d),
        "video_root": str(args.video_root),
        "output_root": str(args.output_root),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "n_total": len(work),
        "n_extracted": n_extracted,
        "n_skipped_no_mp4": n_no_mp4,
        "n_skipped_no_audio_metadata": n_no_audio_md,
        "n_errors": n_errors,
        "videos": entries,
    }
    manifest_path = args.output_root / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[extract] DONE in {time.time()-t0:.1f}s. "
        f"{n_extracted}/{len(work)} extracted. Manifest: {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
