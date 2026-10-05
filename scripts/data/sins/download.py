# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the SINS data-prep pipeline: download Zenodo zips +
SINS_database GitHub repo.

Per node 1-13 (no Node 5): fetch all per-node Zenodo zips (10 parts for
nodes 1-8, 9 parts for 9-13). Each zip ~400 MB; total ~50 GB. Also
fetch SINS_database-master.zip from GitHub (annotation CSVs + per-clip
``.mat`` timestamp metadata; small).

Stops after downloading — does NOT extract or concatenate. Pair with
``scripts/data/sins/build_flacs.py`` (extract + per-node WAV → FLAC
concat) and ``scripts/data/sins/build_manifest.py`` (mono.flac + final
manifest). The split exists because all-at-once extract would peak at
~440 GB of uncompressed WAV on a disk-constrained machine.

Usage:
    python scripts/data/sins/download.py                  # all 13 nodes
    python scripts/data/sins/download.py --nodes 1,2,3    # just these
    python scripts/data/sins/download.py --skip-metadata  # nodes only
    python scripts/data/sins/download.py --skip-nodes     # metadata only

Idempotent: HTTP Range resume; per-file `download_file` short-circuits
if already complete (server returns 416).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "SINS"


# Zenodo record IDs per node (no Node 5).
RECORDS = {
    "1": "2546677",
    "2": "2547307",
    "3": "2547309",
    "4": "2555084",
    "6": "2547313",
    "7": "2547315",
    "8": "2547319",
    "9": "2555080",
    "10": "2555137",
    "11": "2558362",
    "12": "2555141",
    "13": "2555143",
}

ALL_NODES = list(RECORDS.keys())


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file with HTTP-Range resume + tqdm progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    if resp.status_code == 416:
        # Already fully downloaded.
        return dest
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + existing_size
    mode = "ab" if existing_size and resp.status_code == 206 else "wb"

    with (
        open(dest, mode) as f,
        tqdm(
            total=total, initial=existing_size, unit="B", unit_scale=True,
            desc=dest.name, leave=False,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def download_node(node_id: str, output_dir: Path) -> list[Path]:
    """Download all zip files for a single SINS node from Zenodo.

    Always invokes ``download_file`` for every zip even if it exists on
    disk — `download_file` uses HTTP Range to either short-circuit
    (server returns 416 if already complete) or resume (server returns
    206). This is the only way truncated zips from a prior interrupted
    run get repaired; a prior "skip if exists" guard left partials
    broken forever.
    """
    record = RECORDS[node_id]
    node_int = int(node_id)

    # Nodes 1-8 have 10 zip parts; 9-13 have 9.
    n_parts = 10 if node_int < 9 else 9
    files = []
    for j in range(1, n_parts + 1):
        if node_int < 10:
            fname = f"Node{node_id}_audio_{j:02d}.zip"
        else:
            fname = f"Node{node_id}_audio_{j}.zip"
        url = f"https://zenodo.org/records/{record}/files/{fname}"
        dest = output_dir / fname
        download_file(url, dest)
        files.append(dest)

    # License PDF.
    license_url = f"https://zenodo.org/records/{record}/files/license.pdf"
    license_dest = output_dir / f"license_node{node_id}.pdf"
    download_file(license_url, license_dest)

    return files


def download_github_repo(output_dir: Path) -> Path:
    """Download SINS_database GitHub repo (annotations + .mat metadata)."""
    url = "https://github.com/KULeuvenADVISE/SINS_database/archive/master.zip"
    zip_path = output_dir / "SINS_database-master.zip"
    if not zip_path.exists():
        download_file(url, zip_path)
    return zip_path


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file into output_dir."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


# ---------------------------------------------------------------------------
# Phase orchestrators
# ---------------------------------------------------------------------------

def download_metadata(raw_dir: Path) -> Path:
    """Download + extract the SINS_database GitHub repo. Returns the
    extracted repo directory."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sins.download_metadata] fetching SINS_database repo → {raw_dir}")
    download_github_repo(raw_dir)
    repo_dir = raw_dir / "SINS_database-master"
    if not repo_dir.exists() and (raw_dir / "SINS_database-master.zip").exists():
        extract_zip(raw_dir / "SINS_database-master.zip", raw_dir)
    return repo_dir


def download_node_zips(
    node_ids: list[str],
    raw_dir: Path,
    n_workers: int = 4,
) -> dict[str, list[Path]]:
    """Download all Zenodo zips for the given node IDs in parallel.
    Returns {node_id: [zip_path, ...]}."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[sins.download_node_zips] downloading {len(node_ids)} nodes "
        f"({n_workers} workers) → {raw_dir}"
    )

    out: dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(download_node, node_id, raw_dir): node_id
            for node_id in node_ids
        }
        for future in as_completed(futures):
            node_id = futures[future]
            out[node_id] = future.result()
            print(f"  Node {node_id}: downloaded")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--nodes", default=None,
        help="Comma-separated node IDs (default: all 13).",
    )
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel Zenodo download workers (default: 4).")
    p.add_argument("--skip-metadata", action="store_true",
                   help="Skip the SINS_database GitHub repo download.")
    p.add_argument("--skip-nodes", action="store_true",
                   help="Skip the per-node Zenodo zip downloads.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    nodes = args.nodes.split(",") if args.nodes else list(ALL_NODES)

    if not args.skip_metadata:
        download_metadata(raw_dir)

    if not args.skip_nodes:
        download_node_zips(nodes, raw_dir, n_workers=args.workers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
