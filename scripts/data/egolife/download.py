# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the EgoLife data-prep pipeline: thin wrapper around
``huggingface_hub.snapshot_download`` for the ``lmms-lab/EgoLife`` dataset.

Downloads the full 512 GB raw snapshot: 32,817 files across the six
participants' seven days (A{1..6}_{NAME}/DAY{1..7}/*.mp4, plus
EgoLifeCap/{DenseCaption,Transcript}/, EgoIT/, EgoLifeQA/).

Xet backend is disabled by default (per project standing rule: it stalls
on the AWS cluster; use vanilla hf-hub with parallel workers). Resume-able:
re-running with the same ``--local-dir`` skips already-downloaded files.

Usage:
    python scripts/data/egolife/download.py \\
        --output-dir ./datasets/egolife/raw \\
        --max-workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_REPO_ID = "lmms-lab/EgoLife"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Local destination for the EgoLife snapshot.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Parallel download workers (default: 16).")
    p.add_argument("--allow-patterns", nargs="+", default=None,
                   help="Optional glob-style filter (e.g., 'A1_JAKE/**' 'EgoLifeCap/**'). "
                        "Default: everything.")
    args = p.parse_args()

    # Standing rule: HF Xet backend stalls on the AWS Meta cluster. Force off.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Import late so the env var takes effect before the client initializes.
    from huggingface_hub import snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={args.repo_id} -> {args.output_dir}", flush=True)
    print(f"[download] max_workers={args.max_workers} "
          f"allow_patterns={args.allow_patterns}", flush=True)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        max_workers=args.max_workers,
        allow_patterns=args.allow_patterns,
    )
    print(f"[download] DONE. Snapshot at: {local_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
