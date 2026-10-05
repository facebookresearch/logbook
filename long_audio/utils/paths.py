"""Path helpers shared across data-prep scripts and dataset loaders.

Centralizes repo-relative vs. absolute ``audio_path`` handling so the
same policy applies to extract, build, redact, and load.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def to_repo_relative(path: Path | str, warn_prefix: str = "[paths]") -> str:
    """Return a repo-relative POSIX string if ``path`` lives under
    :data:`REPO_ROOT`; otherwise return the absolute path as a string
    and print a WARN explaining that the resulting manifest won't be
    publishable without a re-extract under ``datasets/<ds>/``.

    Already-relative paths pass through as POSIX strings.
    """
    p = Path(path)
    if not p.is_absolute():
        return str(PurePosixPath(path))
    try:
        rel = p.relative_to(REPO_ROOT)
        return str(PurePosixPath(rel))
    except ValueError:
        print(
            f"{warn_prefix} WARN audio_path {p} is not under REPO_ROOT "
            f"({REPO_ROOT}); recording absolute path. "
            f"The resulting manifest cannot be redacted for public release "
            f"— re-extract with --output-root under datasets/<ds>/ (or "
            f"symlink your storage there) if you plan to publish.",
            flush=True,
        )
        return str(p)


def resolve_audio_path(path: Path | str) -> Path:
    """Resolve a manifest ``audio_path`` for reading.

    Absolute paths are returned as-is. Relative paths are resolved
    against :data:`REPO_ROOT` so repo-relative public manifests work
    out of the box as long as the user has set up ``datasets/<ds>/``
    (directory or symlink to storage).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
