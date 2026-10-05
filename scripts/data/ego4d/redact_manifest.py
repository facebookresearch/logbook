"""Redact Ego4D's ``annotated_manifest.json`` to a release-safe
``manifest.public.json`` that ships in the repo.

Strips fields derived from Ego4D consortium annotations we can't
redistribute (per-slice ``raw_summary``, ``summary``, ``facts``,
``raw_classify``, per-pass ``facts[]``, ``moments[]``). Keeps the
OLMo-derived segmentation surface (``actions[]``) and the thin
coordinates needed to rehydrate the full manifest later via
``build_manifest.py --public-manifest``.

Rewrites ``audio_path`` to repo-relative (POSIX) so the artifact works
on any machine. Raises if ``audio_path`` is absolute and not under
``REPO_ROOT`` — tells the caller to re-extract with
``--output-root`` under ``datasets/ego4d/`` (or symlink their storage
there).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# Public per-video whitelist (top-level fields).
_VIDEO_KEEP = (
    "uid",
    "audio_path",
    "audio_offset_s",
    "duration",
    "sample_rate",
    "scenarios",
    "fb_participant_id",
    "video_source",
    "split",
)

# Public per-slice whitelist (passes.{1,2}.slices[*]).
# raw_summary / summary / facts / event / raw_classify all come from or
# depend on Ego4D narration text — dropped.
_SLICE_KEEP = ("start", "end", "annotation_uid")

# Public per-action whitelist (passes.{1,2}.actions[*]).
# OLMo-derived segmentation, releasable.
_ACTION_KEEP = ("start", "end", "event")


def _relativize_audio_path(audio_path: str) -> str:
    """Convert an absolute ``audio_path`` into a repo-relative POSIX string.

    Already-relative paths pass through unchanged (idempotent). Absolute
    paths must live under ``REPO_ROOT``; otherwise raise with a clear
    rehome instruction.
    """
    p = Path(audio_path)
    if not p.is_absolute():
        return str(PurePosixPath(audio_path))
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        raise ValueError(
            f"audio_path {audio_path!r} is absolute and not under REPO_ROOT "
            f"({REPO_ROOT}). Re-run extract_audio.py / build_manifest.py "
            f"with --output-root under datasets/ego4d/ (or symlink your "
            f"storage there: `ln -s /path/to/ego4d-storage datasets/ego4d`)."
        ) from None
    return str(PurePosixPath(rel))


def _redact_video(v: dict) -> dict:
    """Project one full-manifest video down to the public whitelist."""
    out: dict[str, Any] = {k: v[k] for k in _VIDEO_KEEP if k in v}
    out["audio_path"] = _relativize_audio_path(v["audio_path"])

    public_passes: dict[str, dict] = {}
    for pkey, pdata in v["passes"].items():
        public_slices = [
            {k: s[k] for k in _SLICE_KEEP} for s in pdata["slices"]
        ]
        public_actions = [
            {k: a[k] for k in _ACTION_KEEP} for a in pdata["actions"]
        ]
        public_passes[pkey] = {
            "slices": public_slices,
            "actions": public_actions,
        }
    out["passes"] = public_passes
    return out


def redact(manifest_path: Path, output_path: Path) -> dict:
    """Read full annotated manifest, write public-safe manifest."""
    full = json.loads(Path(manifest_path).read_text())
    public_videos = [_redact_video(v) for v in full["videos"]]

    public = {
        "dataset": full["dataset"],
        "description": (
            "Ego4D segmentation-only PUBLIC manifest. Contains per-video "
            "timing metadata, Ego4D-public identifiers (`fb_participant_id`, "
            "`video_source`), the deterministic train/val/test split, "
            "narration-slice coordinates (`passes.*.slices[{start, end, "
            "annotation_uid}]`), and the OLMo-derived coalesced "
            "segmentation (`passes.*.actions[{start, end, event}]`). "
            "Ego4D consortium annotations (narration text, moments) are "
            "NOT included; rehydrate them from your own Ego4D download "
            "via `python scripts/data/ego4d/build_manifest.py "
            "--public-manifest <path>` + `annotate_manifest.py` for the "
            "`facts` field used by description-quality eval."
        ),
        "sample_rate": full["sample_rate"],
        "n_videos": len(public_videos),
        "built_at": full.get("built_at"),
        "annotated_at": full.get("annotated_at"),
        "olmo_model_id": full.get("olmo_model_id"),
        "redacted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_config": full.get("split_config"),
        "videos": public_videos,
    }
    # Drop keys that were absent in the input (keep output tidy).
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
    EGO4D_DIR = REPO_ROOT / "datasets" / "ego4d"
    p.add_argument(
        "--manifest",
        type=Path,
        default=EGO4D_DIR / "annotated_manifest.json",
        help="Full annotated manifest to redact (input).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=EGO4D_DIR / "manifest.public.json",
        help="Public manifest path (output).",
    )
    args = p.parse_args()
    redact(args.manifest, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
