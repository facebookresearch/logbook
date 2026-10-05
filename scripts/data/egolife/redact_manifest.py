"""Redact EgoLife's ``manifest.json`` to a release-safe
``manifest.public.json`` that ships in the repo.

Strips EgoLife DenseCaption text + our NLLB translation + facts derived
from them (``summaries[*].text_en``, ``text_zh``, ``facts``). Also
drops the always-empty ``moments[]`` field for parity with the Ego4D
public manifest. Keeps the OLMo-derived segmentation (``actions[]``)
and the summary-tile coordinates needed to rehydrate the full manifest
later via ``build_manifest.py --public-manifest``.

Rewrites ``audio_path`` to repo-relative (POSIX) so the artifact works
on any machine. Raises if ``audio_path`` is absolute and not under
``REPO_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


_VIDEO_KEEP = (
    "uid",
    "audio_path",
    "audio_offset_s",
    "duration",
    "sample_rate",
    "scenarios",
    "participant",
    "day",
    "session_idx",
    "clip_offsets",
    "wall_clock_start",
    "wall_clock_end",
)

# Summaries ship only tile coordinates; text + facts must come from the
# user's own translator + Stage-4 OLMo run.
_SUMMARY_KEEP = ("start", "end")

_ACTION_KEEP = ("start", "end", "event")


def _relativize_audio_path(audio_path: str) -> str:
    p = Path(audio_path)
    if not p.is_absolute():
        return str(PurePosixPath(audio_path))
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        raise ValueError(
            f"audio_path {audio_path!r} is absolute and not under REPO_ROOT "
            f"({REPO_ROOT}). Re-run extract_audio.py / build_manifest.py "
            f"with --output-root under datasets/egolife/ (or symlink your "
            f"storage there: `ln -s /path/to/egolife-storage datasets/egolife`)."
        ) from None
    return str(PurePosixPath(rel))


def _redact_video(v: dict) -> dict:
    out: dict[str, Any] = {k: v[k] for k in _VIDEO_KEEP if k in v}
    out["audio_path"] = _relativize_audio_path(v["audio_path"])

    public_passes: dict[str, dict] = {}
    for pkey, pdata in v["passes"].items():
        public_summaries = [
            {k: s[k] for k in _SUMMARY_KEEP} for s in pdata["summaries"]
        ]
        public_actions = [
            {k: a[k] for k in _ACTION_KEEP} for a in pdata["actions"]
        ]
        public_passes[pkey] = {
            "actions": public_actions,
            "summaries": public_summaries,
        }
    out["passes"] = public_passes
    return out


def redact(manifest_path: Path, output_path: Path) -> dict:
    full = json.loads(Path(manifest_path).read_text())
    public_videos = [_redact_video(v) for v in full["videos"]]

    public = {
        "dataset": full["dataset"],
        "version": full.get("version"),
        "description": (
            "EgoLife segmentation-only PUBLIC manifest. Contains per-session "
            "timing + wall-clock metadata, DenseCaption-source identifiers "
            "(participant, day, session_idx, clip_offsets), summary-tile "
            "coordinates (`passes.1.summaries[{start, end}]`), and the "
            "OLMo-derived coalesced segmentation "
            "(`passes.1.actions[{start, end, event}]`). EgoLife "
            "DenseCaption text, our NLLB translations, and OLMo `facts` "
            "are NOT included; rehydrate them from your own EgoLife "
            "download + translator + Stage-4 OLMo run via "
            "`python scripts/data/egolife/build_manifest.py "
            "--public-manifest <path>` + `annotate_manifest.py`."
        ),
        "sample_rate": full["sample_rate"],
        "channels": full.get("channels"),
        "summary_window_s": full.get("summary_window_s"),
        "n_videos": len(public_videos),
        "built_at": full.get("date"),
        "redacted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": public_videos,
    }
    public = {k: v for k, v in public.items() if v is not None}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(public, indent=2))
    print(
        f"[redact] wrote {output_path} ({len(public_videos)} videos)",
        flush=True,
    )
    return public


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    EGOLIFE_DIR = REPO_ROOT / "datasets" / "egolife"
    p.add_argument(
        "--manifest",
        type=Path,
        default=EGOLIFE_DIR / "manifest.json",
        help="Full manifest to redact (input).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=EGOLIFE_DIR / "manifest.public.json",
        help="Public manifest path (output).",
    )
    args = p.parse_args()
    redact(args.manifest, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
