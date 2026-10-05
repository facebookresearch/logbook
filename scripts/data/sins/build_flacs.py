"""Stage 2 of the SINS data-prep pipeline: per-node WAV → FLAC concat.

Processes one node at a time (unzip → refine per-clip timestamps via
counter-reset pulses + per-node offsets → concat to a single mono 16 kHz
FLAC that is wall-clock aligned) and by default deletes the extracted
WAVs immediately, keeping peak disk to one node (~50 GB) rather than the
~440 GB the all-at-once flow needs. Writes ``<output_dir>/Node*.flac``
and ``<output_dir>/nodes.json`` (per-node audio + parsed annotations,
consumed by ``build_manifest.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
import soundfile
from natsort import natsorted
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "SINS"

# Global start time: timestamp of the first recording of Node1.
T0 = datetime.strptime("20170130_043836_435", "%Y%m%d_%H%M%S_%f")

# Per-node room assignments.
ROOMS = {
    "bathroom": [13],
    "bedroom": [9, 10],
    "hall": [12],
    "living": [1, 2, 3, 4, 6, 7, 8],
    "wcroom": [11],
}

NODE_TO_ROOM = {
    f"Node{node_id}": room for room, nodes in ROOMS.items() for node_id in nodes
}

# Cross-node sync offsets (seconds) — added to each node's per-clip filename
# timestamp inside refine_timestamps so cross-node sample-N alignment matches
# Node1's clock. Derived from 1Hz reference-clock pulses in
# Pulse_samples_NodeN.mat (~50ms precision; nodes fall into two startup batches
# ~415 ms apart). Replaces UPB Paderborn's wrong-by-up-to-1s legacy values
# (Nodes 2/6 had wrong sign + magnitude).
OFFSETS = {
    "Node1": +0.0000, "Node2": +0.4151, "Node3": +0.0020, "Node4": +0.0013,
    "Node6": +0.4152, "Node7": +0.0015, "Node8": +0.4157, "Node9": +0.4154,
    "Node10": +0.4156, "Node11": +0.4064, "Node12": +0.0015, "Node13": +0.4154,
}

ALL_NODES = [str(n) for room_nodes in ROOMS.values() for n in room_nodes]
ALL_NODES.sort(key=int)


# ---------------------------------------------------------------------------
# Tiny shared helper (also lives in download.py; duplicated per "no _*.py").
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


# ---------------------------------------------------------------------------
# Timestamp + audio primitives
# ---------------------------------------------------------------------------

def refine_timestamps(
    dataset: dict, counter_reset_idx: list, offset: float = 0.0,
) -> None:
    """Refine clip timestamps using .mat counter-reset data (in-place).

    Filename timestamps are inaccurate. Counter resets from the
    reference clock are used to compute accurate timestamps + audio lengths.
    """
    ds = sorted(dataset.values(), key=lambda x: x["timestamp"])
    if not ds:
        return

    samples_list, seconds_list = [], []
    for i, p in enumerate(counter_reset_idx):
        if not (ds[i]["sample_loss"] or ds[i]["counter_reset_loss"]):
            samples_list.append(p[-1] - p[0] - 1)
            seconds_list.append(len(p) - 1)

    sr = sum(samples_list) / sum(seconds_list) if samples_list else 16000.0

    latest_tail = 0
    ds[0]["timestamp_orig"] = ds[0]["timestamp"]
    for i, example in enumerate(ds):
        if i == 0 or (
            abs(example["timestamp"] - ds[i - 1]["timestamp"]
                - ds[i - 1]["audio_length"]) > 5
        ):
            audio_length = counter_reset_idx[i][0] / sr
        else:
            audio_length = counter_reset_idx[i][0] / (
                counter_reset_idx[i][0] + latest_tail
            )
        audio_length += len(counter_reset_idx[i]) - 1 + example["counter_reset_loss"]
        latest_tail = example["num_samples"] - counter_reset_idx[i][-1]
        if i == len(ds) - 1 or (
            abs(ds[i + 1]["timestamp"] - example["timestamp"] - audio_length) > 5
        ):
            audio_length += latest_tail / sr
        else:
            audio_length += latest_tail / (latest_tail + counter_reset_idx[i + 1][0])

        example["audio_length"] = audio_length
        timestamp = example["timestamp"]
        if i < len(ds) - 1:
            ds[i + 1]["timestamp_orig"] = ds[i + 1]["timestamp"]
            if abs(timestamp + audio_length - ds[i + 1]["timestamp"]) < 2.1:
                ds[i + 1]["timestamp"] = timestamp + audio_length

    if offset:
        for example in ds:
            example["timestamp"] += offset


def _downmix_to_mono(audio_f32: np.ndarray) -> np.ndarray:
    """Mean-downmix (n_samples, n_ch) float32 → (n_samples, 1) mono."""
    if audio_f32.ndim == 1:
        return audio_f32[:, None]
    if audio_f32.shape[1] == 1:
        return audio_f32
    return audio_f32.mean(axis=1, keepdims=True).astype(np.float32, copy=False)


def _resample(audio_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample float32 audio src_sr → dst_sr."""
    if src_sr == dst_sr:
        return audio_f32
    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    out = scipy.signal.resample_poly(audio_f32, up, down, axis=0)
    return out.astype(np.float32, copy=False)


def concatenate_wavs(
    clips: list[dict],
    output_path: Path,
    target_sample_rate: int = 16000,
    chunk_frames: int = 1_000_000,
    pad_to_t0: bool = True,
) -> float:
    """Concatenate per-clip WAVs into a wall-clock-aligned mono FLAC.

    Why FLAC: WAV's 4 GB cap; 7-day 16 kHz mono is ~9.7 GB.
    Why mono: downstream consumers only ever read mono.
    Why gap silence: SINS has small inter-clip gaps. Naive concat
    breaks sample N == T0 + N/sr; we pad zeros for detected gaps so
    the output's time axis matches T0 exactly.
    Sample-rate mismatch: warn once + scipy.signal.resample_poly.
    Overlaps: first-writer-wins. Pre-t=0 clips dropped if pad_to_t0.

    Args:
        clips: list of {"audio_path", "timestamp"} (timestamp = seconds
            from T0). Order doesn't matter — sorted internally.
        output_path: destination FLAC.
        target_sample_rate: output sr.
        chunk_frames: streaming write chunk (default 1M).
        pad_to_t0: if True, output starts at t=0; leading silence
            inserted before first clip.

    Returns total duration written, in seconds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not clips:
        return 0.0

    clips = sorted(clips, key=lambda c: c["timestamp"])

    if pad_to_t0:
        cursor = 0
    else:
        cursor = int(round(clips[0]["timestamp"] * target_sample_rate))
    start_cursor = cursor

    warned_sr: set[int] = set()

    def _write_silence(out_sf, n_frames: int) -> None:
        remaining = n_frames
        while remaining > 0:
            chunk = min(remaining, chunk_frames)
            out_sf.write(np.zeros((chunk, 1), dtype=np.float32))
            remaining -= chunk

    with soundfile.SoundFile(
        str(output_path), mode="w",
        samplerate=target_sample_rate, channels=1,
        subtype="PCM_16", format="FLAC",
    ) as out_sf:
        for clip in tqdm(clips, desc=f"Concatenating → {output_path.name}", leave=False):
            path = str(clip["audio_path"])
            timestamp = float(clip["timestamp"])

            audio, src_sr = soundfile.read(path, always_2d=True, dtype="float32")

            if src_sr != target_sample_rate:
                if src_sr not in warned_sr:
                    print(
                        f"  [WARN] {path}: sample rate {src_sr} ≠ target "
                        f"{target_sample_rate}; resampling (further mismatches "
                        f"of this rate suppressed).",
                        flush=True,
                    )
                    warned_sr.add(src_sr)
                audio = _resample(audio, src_sr, target_sample_rate)
            audio = _downmix_to_mono(audio)

            clip_frames = audio.shape[0]
            clip_start = int(round(timestamp * target_sample_rate))
            clip_end = clip_start + clip_frames

            if clip_end <= cursor:
                continue
            if clip_start > cursor:
                _write_silence(out_sf, clip_start - cursor)
                cursor = clip_start
            src_skip = 0
            if clip_start < cursor:
                src_skip = cursor - clip_start
            write_buf = audio[src_skip:]
            pos = 0
            while pos < write_buf.shape[0]:
                block = write_buf[pos : pos + chunk_frames]
                out_sf.write(block)
                cursor += block.shape[0]
                pos += block.shape[0]

    return (cursor - start_cursor) / target_sample_rate


# ---------------------------------------------------------------------------
# Annotation primitives
# ---------------------------------------------------------------------------

def parse_annotations(database_path: Path) -> list[dict]:
    """Parse annotation CSVs from SINS_database-master/annotation/*.csv,
    merge to apartment-level events, smooth tiny inter-annotation gaps.
    Returns [{"label": "<activity>:<room>", "start": float, "end": float}, ...]
    sorted by start."""
    database_path = Path(database_path)
    annotations: dict[str, list] = {}

    for room in ["living", "bedroom", "wcroom", "hall", "bathroom"]:
        annotations[room] = []
        csv_path = database_path / "annotation" / f"{room}_labels.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as fid:
            for row_idx, row in enumerate(csv.reader(fid, delimiter=";"), start=1):
                if len(row) < 3:
                    print(
                        f"[parse_annotations] WARN {database_path}:row {row_idx}: "
                        f"malformed CSV row, expected >=3 cols, got {len(row)}: {row}",
                        file=sys.stderr, flush=True,
                    )
                    continue
                label, start_time_str, stop_time_str = row[0], row[1], row[2]
                if label in ["Class", "dont use"]:
                    continue
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
                stop_time = datetime.strptime(stop_time_str, "%Y-%m-%d %H:%M:%S.%f")
                annotations[room].append((
                    label,
                    (start_time - T0).total_seconds(),
                    (stop_time - T0).total_seconds(),
                ))

    apartment_annotations = sorted(
        [
            (f"{scene}:{room}", start, stop)
            for room, annots in annotations.items()
            for scene, start, stop in annots
        ],
        key=lambda x: x[1],
    )

    # Smooth tiny (<3s) gaps between consecutive annotations.
    last_scene = None
    for i, cur in enumerate(apartment_annotations):
        cur = list(cur)
        apartment_annotations[i] = cur
        if last_scene is None:
            last_scene = cur
        elif 3.0 > (cur[1] - last_scene[2]) > 0.0:
            mid = int((last_scene[2] + cur[1]) / 2 * 1000) / 1000
            cur[1] = mid
            last_scene[2] = mid
        if cur[2] > (last_scene[2] if last_scene else 0):
            last_scene = cur

    return [
        {"label": label, "start": start, "end": end}
        for label, start, end in apartment_annotations
    ]


def build_node_dataset(node_id: str, database_path: Path) -> dict:
    """Build dataset dict for a single node using .mat metadata.

    ``database_path`` must point to the raw-download root that contains
    the ``SINS_database-master`` extraction; .mat metadata is read from
    ``{database_path}/SINS_database-master/example_code/other/``.
    """
    node_str = f"Node{node_id}"
    node_int = int(node_id)

    mat_dir = database_path / "SINS_database-master" / "example_code" / "other"
    if not mat_dir.exists():
        raise FileNotFoundError(
            f"build_node_dataset: SINS metadata dir not found at {mat_dir}. "
            "Pass the raw-download root (containing SINS_database-master/)."
        )

    wav_timestamps_path = mat_dir / f"WavTimestamps_Node{node_int}.mat"
    if not wav_timestamps_path.exists():
        raise FileNotFoundError(
            f"build_node_dataset: WavTimestamps_Node{node_int}.mat not found "
            f"at {wav_timestamps_path}"
        )

    wavfiles = scipy.io.loadmat(str(wav_timestamps_path))["WavFiles"].squeeze().tolist()
    wavfiles = [f.tolist()[0] for f in wavfiles]

    pulse_data = scipy.io.loadmat(str(mat_dir / f"Pulse_samples_Node{node_int}.mat"))
    num_samples = pulse_data["length_files"][..., 0]
    counter_reset_samples = pulse_data["pulses"].squeeze().tolist()
    counter_reset_samples = [resets.squeeze() - 1 for resets in counter_reset_samples]

    sample_loss = []
    counter_reset_loss = []
    pulse_loss_threshold = 1000
    sample_loss_threshold = 100
    for reset_samples in counter_reset_samples:
        sample_loss.append(bool(np.any(
            np.abs(reset_samples[1:] - reset_samples[:-1] - 16000) > sample_loss_threshold
        )))
        counter_reset_loss.append(bool(np.any(
            (reset_samples[1:] - reset_samples[:-1] - 16000) > pulse_loss_threshold
        )))

    dataset = {}
    for i, wavfile in enumerate(wavfiles):
        example_id = wavfile[:-10]  # strip "_audio.wav"
        timestamp_str = example_id.split("_", 1)[-1]
        timestamp = (
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f") - T0
        ).total_seconds()
        dataset[example_id] = {
            "timestamp": timestamp,
            "audio_path": str(database_path / "audio" / node_str / "audio" / wavfile),
            "num_samples": int(num_samples[i]),
            "sample_loss": sample_loss[i],
            "counter_reset_samples": counter_reset_samples[i].tolist(),
            "counter_reset_loss": counter_reset_loss[i],
            "node_id": node_int,
        }

    refine_timestamps(dataset, counter_reset_samples, OFFSETS[node_str])
    return dataset


# ---------------------------------------------------------------------------
# Phase orchestrators
# ---------------------------------------------------------------------------

def extract_node_zips(node_id: str, raw_dir: Path) -> Path:
    """Extract all zip parts for one node and reorganize Original/audio/NodeX
    → audio/NodeX so build_node_dataset can find the WAVs. Returns the
    directory containing the WAV clips."""
    raw_dir = Path(raw_dir)
    node_str = f"Node{node_id}"
    node_int = int(node_id)
    target = raw_dir / "audio" / node_str / "audio"
    if target.exists() and any(target.iterdir()):
        return target

    n_parts = 10 if node_int < 9 else 9
    for j in range(1, n_parts + 1):
        if node_int < 10:
            fname = f"Node{node_id}_audio_{j:02d}.zip"
        else:
            fname = f"Node{node_id}_audio_{j}.zip"
        zip_path = raw_dir / fname
        if not zip_path.exists():
            raise FileNotFoundError(
                f"extract_node_zips: missing zip part {fname} at {zip_path}. "
                "Run download.py to fetch missing parts before extracting."
            )
        extract_zip(zip_path, raw_dir)

    original_node = raw_dir / "Original" / "audio" / node_str
    target_parent = raw_dir / "audio" / node_str
    if original_node.exists() and not target_parent.exists():
        target_parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(original_node), str(target_parent))
    original_audio = raw_dir / "Original" / "audio"
    if original_audio.exists() and not any(original_audio.iterdir()):
        shutil.rmtree(raw_dir / "Original", ignore_errors=True)

    return target


def concat_node_flac(
    node_id: str,
    raw_dir: Path,
    output_dir: Path,
) -> dict | None:
    """Build one per-node FLAC from extracted WAVs (wall-clock aligned via
    .mat timestamp metadata). Returns a node-manifest dict, or None if
    skipped (missing .mat metadata or no WAVs found)."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_str = f"Node{node_id}"
    out_flac = output_dir / f"{node_str}.flac"

    print(f"[sins.concat_node_flac] {node_str}")

    try:
        dataset = build_node_dataset(node_id, raw_dir)
    except FileNotFoundError as e:
        print(f"    WARNING: {e} — skipping node {node_str}", flush=True)
        return None

    dropped = [ex for ex in dataset.values() if not Path(ex["audio_path"]).exists()]
    clips = [
        {"audio_path": ex["audio_path"], "timestamp": ex["timestamp"]}
        for ex in dataset.values() if Path(ex["audio_path"]).exists()
    ]
    if dropped:
        print(
            f"    WARNING: {len(dropped)}/{len(dataset)} clips missing on disk "
            f"for {node_str} (first 3: "
            f"{[str(ex['audio_path']) for ex in dropped[:3]]})",
            flush=True,
        )
    if len(dropped) > 0.1 * len(dataset):
        raise RuntimeError(
            f"concat_node_flac: {len(dropped)}/{len(dataset)} (>10%) clips "
            f"missing for {node_str} — refusing to build a holey FLAC. "
            "Re-extract zips first."
        )
    if not clips:
        print(f"    WARNING: No wav files found for {node_str}, skipping")
        return None

    if not out_flac.exists():
        duration = concatenate_wavs(clips, out_flac)
    else:
        info = soundfile.info(str(out_flac))
        duration = info.duration

    return {
        "id": f"sins_{node_str.lower()}",
        "audio_path": str(out_flac),
        "sample_rate": 16000,
        "duration": duration,
        "node_id": int(node_id),
        "room": NODE_TO_ROOM[node_str],
    }


# ---------------------------------------------------------------------------
# Build orchestration (per-node loop)
# ---------------------------------------------------------------------------

def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def folder_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total / 1e9


def verify_flac(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        info = soundfile.info(str(path))
    except Exception as e:
        print(
            f"[build_flacs] WARN verify_flac failed on {path}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )
        return False
    return info.format == "FLAC" and info.frames > 0


def process_node(
    node_id: str,
    raw_dir: Path,
    output_dir: Path,
    keep_extracted: bool,
) -> dict | None:
    node_str = f"Node{node_id}"
    out_flac = output_dir / f"{node_str}.flac"

    if verify_flac(out_flac):
        info = soundfile.info(str(out_flac))
        print(f"  OK {node_str}.flac already exists ({info.duration / 3600:.2f}h) — skipping")
        return {
            "id": f"sins_{node_str.lower()}",
            "audio_path": str(out_flac),
            "sample_rate": info.samplerate,
            "duration": info.duration,
            "node_id": int(node_id),
            "room": NODE_TO_ROOM[node_str],
        }

    print(f"  [{node_str}] extracting zips…")
    t0 = time.time()
    try:
        audio_dir = extract_node_zips(node_id, raw_dir)
    except FileNotFoundError as e:
        print(f"    ! {e} — skipping {node_str}", file=sys.stderr, flush=True)
        return None
    extracted_gb = folder_size_gb(audio_dir)
    print(f"    extracted {extracted_gb:.1f} GB in {time.time() - t0:.0f}s")

    print(f"  [{node_str}] building per-clip timestamps + concatenating to FLAC…")
    t0 = time.time()
    nm = concat_node_flac(node_id, raw_dir, output_dir)
    if nm is None:
        return None

    flac_gb = out_flac.stat().st_size / 1e9
    print(
        f"    wrote {out_flac.name}: {nm['duration'] / 3600:.2f}h, {flac_gb:.2f} GB "
        f"in {time.time() - t0:.0f}s (compression {extracted_gb / max(flac_gb, 0.01):.1f}x)"
    )

    if not verify_flac(out_flac):
        print(f"    ! verification failed for {out_flac}; keeping extracted clips")
        return None

    if not keep_extracted:
        print(f"  [{node_str}] removing extracted WAVs to free disk…")
        shutil.rmtree(raw_dir / "audio" / node_str, ignore_errors=True)
        audio_root = raw_dir / "audio"
        if audio_root.exists() and not any(audio_root.iterdir()):
            shutil.rmtree(audio_root, ignore_errors=True)

    return nm


def write_node_inputs(
    output_dir: Path,
    node_manifests: list[dict],
    annotations: list[dict],
) -> Path:
    """Write the intermediate ``nodes.json`` consumed by `build_manifest.py`.

    Each node entry carries the per-node FLAC path + the full raw
    apartment-level annotation list (`{"label": "activity:room",
    "start": float, "end": float}`). `build_manifest.py` then runs
    `clean_annotations` on these to produce the final cleaned
    `manifest.json`.
    """
    for nm in node_manifests:
        nm["annotations"] = annotations
    manifest = {
        "dataset": "SINS",
        "description": "SINS per-node FLAC + raw apartment-level annotations. "
                       "Intermediate input for build_manifest.py.",
        "nodes": node_manifests,
    }
    path = output_dir / "nodes.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--nodes", default=None,
                   help="Comma-separated subset (default: all)")
    p.add_argument("--keep-extracted", action="store_true",
                   help="Don't delete extracted per-clip WAVs after concat")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    nodes = args.nodes.split(",") if args.nodes else list(ALL_NODES)

    repo_dir = raw_dir / "SINS_database-master"
    if not repo_dir.exists():
        print(
            f"! GitHub metadata repo not found at {repo_dir}. "
            "Run `python scripts/data/sins/download.py` first.",
            file=sys.stderr,
        )
        return 1

    print(f"Processing {len(nodes)} nodes → {output_dir}")
    print(f"Disk free at start: {disk_free_gb(output_dir):.1f} GB\n")

    node_manifests: list[dict] = []
    nodes_path = output_dir / "nodes.json"
    if nodes_path.exists():
        prior = json.load(nodes_path.open())
        for n in prior["nodes"]:
            n.pop("annotations", None)
            node_manifests.append(n)
        print(f"  (resumed from existing nodes.json with {len(node_manifests)} nodes)")

    done_ids = {n["node_id"] for n in node_manifests}

    for node_id in natsorted(nodes, key=int):
        if int(node_id) in done_ids and verify_flac(
            Path(next(n["audio_path"] for n in node_manifests if n["node_id"] == int(node_id)))
        ):
            print(f"Node{node_id}: already in manifest with valid FLAC — skipping")
            continue
        nm = process_node(node_id, raw_dir, output_dir, args.keep_extracted)
        if nm is None:
            continue
        node_manifests = [n for n in node_manifests if n["node_id"] != int(node_id)]
        node_manifests.append(nm)
        annotations = parse_annotations(repo_dir)
        path = write_node_inputs(
            output_dir,
            sorted(node_manifests, key=lambda n: n["node_id"]),
            annotations,
        )
        print(f"    nodes.json updated: {path}")
        print(f"    disk free: {disk_free_gb(output_dir):.1f} GB\n")

    print(f"\nDone. {len(node_manifests)} nodes in manifest. "
          f"Disk free: {disk_free_gb(output_dir):.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
