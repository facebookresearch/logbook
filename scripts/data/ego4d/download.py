# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the Ego4D data-prep pipeline: thin wrapper around the
``ego4d`` CLI for project-default Ego4D downloads.

Two passes: first ``annotations + metadata`` (small; grabbed without
``--universities``), then per-university ``video_540ss`` (big; one
``ego4d`` invocation per university so resume is per-university).
Defaults match the long-audio project's canonical 13-university bundle;
override with ``--universities`` / ``--skip-videos`` / ``--skip-annotations``.
Requires the ``ego4d`` pip package on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


ALL_UNIVERSITIES = [
    "bristol", "cmu", "cmu_africa", "frl_track_1_public", "georgiatech",
    "iiith", "indiana", "kaust", "minnesota", "nus", "uniandes", "unict",
    "utokyo",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, help="Local destination for Ego4D files")
    p.add_argument(
        "--universities", nargs="+", default=ALL_UNIVERSITIES,
        help=f"Subset of universities for the per-university video pass "
             f"(default: all {len(ALL_UNIVERSITIES)}).",
    )
    p.add_argument(
        "--video-datasets", nargs="+", default=["video_540ss"],
        help="Ego4D --datasets value for the per-university pass "
             "(default: video_540ss).",
    )
    p.add_argument(
        "--skip-annotations", action="store_true",
        help="Skip the global annotations + metadata download pass.",
    )
    p.add_argument(
        "--skip-videos", action="store_true",
        help="Skip the per-university video download pass.",
    )
    p.add_argument(
        "--aws-profile", default="ego4d",
        help="AWS CLI profile name with Ego4D S3 read access (default: ego4d).",
    )
    p.add_argument(
        "--ego4d-bin", default=shutil.which("ego4d") or "ego4d",
        help="Path to the ego4d CLI (default: first on PATH).",
    )
    p.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=[],
        help="Anything after this is passed verbatim to every ego4d invocation.",
    )
    args = p.parse_args()

    def run_ego4d(extra: list[str]) -> int:
        cmd = [
            args.ego4d_bin,
            "--output_directory", args.output_dir,
            "--aws_profile_name", args.aws_profile,
            *extra,
            "-y",
            *args.extra_args,
        ]
        print(f"$ {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd).returncode

    # Pass 1: annotations + metadata, no --universities (one-shot global).
    if not args.skip_annotations:
        rc = run_ego4d(["--datasets", "annotations", "--metadata"])
        if rc != 0:
            return rc

    # Pass 2: per-university videos.
    if not args.skip_videos:
        for univ in args.universities:
            rc = run_ego4d([
                "--datasets", *args.video_datasets,
                "--universities", univ,
            ])
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
