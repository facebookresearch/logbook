"""Stage 3 of the EgoLife data-prep pipeline: translate DenseCaption SRTs
Chinese -> English via ``facebook/nllb-200-distilled-600M``.

Iterates over ``<raw_root>/EgoLifeCap/DenseCaption/A{i}_{NAME}/DAY{d}/*.srt``,
parses each subtitle block (index, timestamp, Chinese text), batches the
Chinese text through NLLB (``zho_Hans`` -> ``eng_Latn``, batch=64 by
default on GPU), and emits one JSONL per input SRT:

    <output_root>/DenseCaption/A{i}_{NAME}/DAY{d}/*.jsonl
      { "idx": <int>, "start_s": <float>, "end_s": <float>,
        "text_zh": "...", "text_en": "..." }

The output preserves the original Chinese text so downstream code can
re-translate later if we swap models.

GPU stage; submit to AWS SLURM h100 or a100.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EGOLIFE_DIR = REPO_ROOT / "datasets" / "egolife"
DEFAULT_RAW = EGOLIFE_DIR / "raw"
DEFAULT_OUTPUT = EGOLIFE_DIR / "translated"
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# HH:MM:SS,mmm --> HH:MM:SS,mmm
_TS_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


def _srt_time_to_s(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt(text: str, path: Path) -> list[dict]:
    """Return a list of ``{idx, start_s, end_s, text_zh}`` from an SRT file.

    Assumes text is a single line per entry (EgoLife's DenseCaption format).
    Raises on any structural malformation so upstream input problems surface."""
    entries = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        # skip blanks between entries
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        # index line
        try:
            idx = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(
                f"{path}: expected integer index at line {i + 1}, got "
                f"{lines[i]!r}"
            ) from e
        i += 1
        # timestamp line
        if i >= len(lines):
            raise ValueError(f"{path}: EOF after index {idx}, expected timestamp")
        m = _TS_RE.match(lines[i])
        if not m:
            raise ValueError(
                f"{path}: expected SRT timestamp at line {i + 1}, got {lines[i]!r}"
            )
        start_s = _srt_time_to_s(*m.group(1, 2, 3, 4))
        end_s = _srt_time_to_s(*m.group(5, 6, 7, 8))
        i += 1
        # text lines until blank
        buf = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        text_zh = " ".join(s.strip() for s in buf).strip()
        if not text_zh:
            raise ValueError(
                f"{path}: entry {idx} at line {i} has empty text"
            )
        entries.append({
            "idx": idx,
            "start_s": start_s,
            "end_s": end_s,
            "text_zh": text_zh,
        })
    return entries


def _iter_srts(raw_root: Path) -> Iterator[tuple[Path, str, str]]:
    """Yield (srt_path, participant, day) for every DenseCaption SRT."""
    dc_root = raw_root / "EgoLifeCap" / "DenseCaption"
    for participant_dir in sorted(dc_root.glob("A?_*")):
        if not participant_dir.is_dir():
            continue
        for day_dir in sorted(participant_dir.glob("DAY?")):
            for srt in sorted(day_dir.glob("*.srt")):
                yield srt, participant_dir.name, day_dir.name


def _load_nllb(model_id: str, device: str):
    """Return (tokenizer, model) ready for batch translation."""
    print(f"[translate] loading {model_id} on {device}", flush=True)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, src_lang="zho_Hans"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _translate_batch(tokenizer, model, texts, device, max_new_tokens=192):
    """Batch-translate a list of Chinese strings to English."""
    import torch
    with torch.inference_mode():
        enc = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)
        forced = tokenizer.convert_tokens_to_ids("eng_Latn")
        out = model.generate(
            **enc,
            forced_bos_token_id=forced,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        return tokenizer.batch_decode(out, skip_special_tokens=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda",
                   help="'cuda' (default) or 'cpu'.")
    p.add_argument("--limit-srts", type=int, default=None,
                   help="Process only the first N SRT files (smoke).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip SRTs whose output JSONL already exists.")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_nllb(args.model_id, args.device)

    srts = list(_iter_srts(args.raw_root))
    if args.limit_srts:
        srts = srts[: args.limit_srts]
    print(f"[translate] {len(srts)} SRT files to process", flush=True)

    t0 = time.time()
    total_fragments = 0
    for si, (srt_path, participant, day) in enumerate(srts, start=1):
        out_dir = args.output_root / "DenseCaption" / participant / day
        out_path = out_dir / (srt_path.stem + ".jsonl")
        if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"[translate] skip existing: {out_path}", flush=True)
            continue

        entries = _parse_srt(srt_path.read_text(encoding="utf-8"), srt_path)
        if not entries:
            raise ValueError(f"empty SRT (zero parseable entries): {srt_path}")

        # Batch-translate all Chinese texts in this SRT.
        texts = [e["text_zh"] for e in entries]
        translations: list[str] = []
        for batch in _batched(texts, args.batch_size):
            translations.extend(
                _translate_batch(tokenizer, model, batch, args.device)
            )
        assert len(translations) == len(entries)

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for entry, en in zip(entries, translations):
                entry["text_en"] = en
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total_fragments += len(entries)
        elapsed = time.time() - t0
        rate = total_fragments / elapsed if elapsed else 0
        print(
            f"[translate] {si}/{len(srts)} {srt_path.name} "
            f"({len(entries)} fragments) | total={total_fragments} "
            f"| {rate:.0f} frag/s",
            flush=True,
        )

    # Small run-summary manifest for provenance.
    summary = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "n_srts_processed": len(srts),
        "n_fragments_translated": total_fragments,
    }
    (args.output_root / "translate_manifest.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        f"[translate] DONE in {time.time() - t0:.1f}s. "
        f"{total_fragments} fragments translated.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
