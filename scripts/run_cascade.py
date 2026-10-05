"""Cascade inference: Stage-A audio captions → Stage-B text segmentation.

Stage A (audio LM) emits one short description per ``chunk_seconds``
window; Stage B (text LM) reads ``window_minutes`` of those descriptions
and produces JSON segmentation. The two stages can run end-to-end in one
process or be split (``--skip-segment`` / ``--skip-describe``) so Stage A
runs on GPU hosts and Stage B runs where the text-LM API is reachable.

Outputs land under ``--output-root/<tag>/`` where ``tag`` is
``cascade_<describe_model>``. Entry point: :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_audio.inference.cascade import run_cascade_segmentation
from long_audio.inference.describe import describe_audio_chunks


REPO_ROOT = Path(__file__).resolve().parent.parent
SINS_MONO_FLAC = REPO_ROOT / "datasets" / "SINS" / "mono.flac"
EGO4D_MANIFEST = REPO_ROOT / "datasets" / "ego4d" / "annotated_manifest.json"
EGOLIFE_MANIFEST = REPO_ROOT / "datasets" / "egolife" / "manifest.json"
RUNS_DIR = REPO_ROOT / "runs"


# Audio (Stage A) adapters — same registry as run_e2e.py for parity.
DESCRIBE_MODELS = {
    "qwen2.5-omni":         ("long_audio.inference.models.qwen2_5_omni",
                             "Qwen2_5OmniVLLMAdapter"),
    "qwen3-omni":           ("long_audio.inference.models.qwen3_omni",
                             "Qwen3OmniVLLMAdapter"),
    "qwen3-omni-captioner": ("long_audio.inference.models.qwen3_omni_captioner",
                             "Qwen3OmniCaptionerVLLMAdapter"),
    "af3-hf":               ("long_audio.inference.models.audio_flamingo3_hf",
                             "AudioFlamingo3HFAdapter"),
    "af-next":              ("long_audio.inference.models.audio_flamingo_next_hf",
                             "AudioFlamingoNextHFAdapter"),
    "af-next-captioner":    ("long_audio.inference.models.audio_flamingo_next_captioner_hf",
                             "AudioFlamingoNextCaptionerHFAdapter"),
    "enclap":               ("long_audio.inference.models.enclap",
                             "EnClapLargeAdapter"),
    "msclap-cap":           ("long_audio.inference.models.msclap",
                             "MSClapCapAdapter"),
    "fake":                 ("long_audio.inference.models.fake", "FakeAdapter"),
}


# Text-only (Stage B) adapters. Keep this small — pluggable downstream LLM.
TEXT_MODELS = {
    "gemini":       ("long_audio.inference.models.gemini", "GeminiTextAdapter"),
    "luna":         ("long_audio.inference.models.openai_compat", "OpenAICompatTextAdapter"),
    "qwen2.5-72b":  ("long_audio.inference.models.vllm_text", "Qwen2_5_72BTextAdapter"),
    "llama3.3-70b": ("long_audio.inference.models.vllm_text", "Llama3_3_70BTextAdapter"),
    "olmo3.1-32b":  ("long_audio.inference.models.vllm_text", "Olmo3_1_32BTextAdapter"),
    "k2-v2":        ("long_audio.inference.models.vllm_text", "K2_V2TextAdapter"),
    "qwen3-32b":    ("long_audio.inference.models.vllm_text", "Qwen3_32BTextAdapter"),
    "gemma4-31b":   ("long_audio.inference.models.vllm_text", "Gemma4_31BTextAdapter"),
    "fake":         ("long_audio.inference.models.fake", "FakeTextLLMAdapter"),
}


def _dataset_config(name: str) -> dict:
    if name == "sins":
        from long_audio.datasets.sins.schema import SINS_LABEL_HINTS, SINS_LABELS
        return {
            "labels": SINS_LABELS,
            "label_hints": SINS_LABEL_HINTS,
            "dataset_name": "SINS",
            "t0_iso": "2017-01-30T04:38:36.435Z",
            "default_audio": SINS_MONO_FLAC,
        }
    if name == "ego4d":
        from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
        return {
            "labels": ATUS_LABELS,
            "label_hints": ATUS_HINTS,
            "dataset_name": "Ego4D",
            "t0_iso": "1970-01-01T00:00:00Z",
            "default_manifest": EGO4D_MANIFEST,
        }
    if name == "egolife":
        # EgoLife re-exports Ego4D's ATUS-6 taxonomy verbatim for
        # cross-dataset comparability.
        from long_audio.datasets.egolife.schema import EGOLIFE_HINTS, EGOLIFE_LABELS
        return {
            "labels": EGOLIFE_LABELS,
            "label_hints": EGOLIFE_HINTS,
            "dataset_name": "EgoLife",
            "t0_iso": "1970-01-01T00:00:00Z",
            "default_manifest": EGOLIFE_MANIFEST,
        }
    raise ValueError(f"Unknown dataset: {name!r}")


# Stage A describe models that take vLLM-specific tuning kwargs
# (max_num_seqs / max_model_len). The HF-transformers and custom
# adapters (af3-hf, af-next, af-next-captioner, enclap, msclap-cap)
# do not — passing those kwargs to them is a user-intent mismatch.
_VLLM_DESCRIBE_ADAPTERS = {
    "qwen2.5-omni", "qwen3-omni", "qwen3-omni-captioner",
}


def _load(
    registry: dict,
    name: str,
    model_id: str | None = None,
    extra_kwargs: dict | None = None,
):
    """Instantiate an adapter from ``registry[name]``.

    No kwarg filtering — ``extra_kwargs`` must already be tailored to
    the target adapter's ``__init__`` (e.g. by ``_describe_kwargs`` which
    only packs vLLM kwargs when the describe model is vLLM-based). Any
    mismatch surfaces loudly as a TypeError from the constructor.
    """
    module_path, cls_name = registry[name]
    mod = __import__(module_path, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    kwargs: dict = dict(extra_kwargs or {})
    if model_id:
        kwargs["model_id"] = model_id
    return cls(**kwargs)


def _describe_kwargs(args, dcfg: dict) -> dict:
    """Build the extra-kwargs dict for the describe adapter.

    Only emits vLLM-flavored kwargs (max_num_seqs, max_model_len) when
    the describe model is in ``_VLLM_DESCRIBE_ADAPTERS``. For non-vLLM
    describe models, returns ``{}`` so the adapter constructor sees
    only the inputs it accepts.

    Special case: if the user explicitly passes ``--describe-max-model-len``
    (default None) with a non-vLLM describe model, we surface a loud error
    rather than silently ignoring the flag. ``--describe-batch-size`` has
    a default of 64, so we cannot tell ``user-typed`` from ``defaulted``;
    that flag is silently no-op for non-vLLM adapters, matching the
    pre-flip launcher behavior.
    """
    if args.describe_model in _VLLM_DESCRIBE_ADAPTERS:
        out: dict = {"max_num_seqs": args.describe_batch_size}
        if args.describe_max_model_len is not None:
            out["max_model_len"] = args.describe_max_model_len
        return out
    if args.describe_max_model_len is not None:
        raise SystemExit(
            f"--describe-max-model-len is a vLLM-only flag and describe "
            f"adapter {args.describe_model!r} is not vLLM-based. "
            f"vLLM describe adapters: {sorted(_VLLM_DESCRIBE_ADAPTERS)}."
        )
    return {}


def _run_one(
    args,
    audio_path: Path,
    run_dir: Path,
    dcfg: dict,
    *,
    describe_adapter,
    text_adapter,
    descriptions_path_override: Path | None,
    audio_duration_s: float | None,
    audio_offset_s: float = 0.0,
) -> tuple[dict | None, dict | None]:
    """Run Stage A and/or Stage B for ONE audio. Returns (describe_summary_dict,
    cascade_summary_dict); either may be None if the corresponding stage
    was skipped."""
    run_dir.mkdir(parents=True, exist_ok=True)

    a_summary_dict: dict | None = None
    descriptions_path: Path
    if args.skip_describe:
        if descriptions_path_override is None:
            raise SystemExit("--skip-describe requires --descriptions-path "
                             "(SINS) or annotated_manifest+rsync layout (Ego4D)")
        descriptions_path = descriptions_path_override
        if not descriptions_path.exists():
            # FileNotFoundError (not SystemExit) so _run_one_with_retry can
            # DROP this uid and let the cell continue instead of killing the
            # whole cascB job. Typical cause: an upstream Stage-A shard was
            # dropped.
            raise FileNotFoundError(f"descriptions not found: {descriptions_path}")
    else:
        a_summary = describe_audio_chunks(
            audio_path, describe_adapter, run_dir,
            chunk_seconds=args.describe_chunk_s,
            batch_size=args.describe_batch_size,
            max_chunks=args.max_describe_chunks if args.smoke else None,
            max_new_tokens=args.describe_max_new_tokens,
            temperature=args.temperature,
            audio_offset_s=audio_offset_s,
            audio_window_s=audio_duration_s,
        )
        descriptions_path = Path(a_summary.descriptions_path)
        from dataclasses import asdict as _asdict
        a_summary_dict = _asdict(a_summary)

    b_summary_dict: dict | None = None
    if not args.skip_segment:
        b_summary = run_cascade_segmentation(
            descriptions_path, text_adapter, run_dir, dcfg["labels"],
            label_hints=dcfg["label_hints"],
            window_minutes=args.window_min,
            decoder=args.decoder,
            max_windows=args.max_windows if args.smoke else None,
            max_new_tokens=args.segment_max_new_tokens,
            temperature=args.temperature,
            context_mode=args.context,
            time_unit=args.time_unit,
            audio_path=audio_path,
            audio_duration_s=audio_duration_s,
            dataset_name=dcfg["dataset_name"],
            t0_iso=dcfg["t0_iso"],
            with_description=args.emit_description,
        )
        from dataclasses import asdict as _asdict
        b_summary_dict = _asdict(b_summary)
    return a_summary_dict, b_summary_dict


def _run_one_with_retry(
    args,
    item: dict,
    out_dir: Path,
    dcfg: dict,
    *,
    describe_adapter,
    text_adapter,
    descriptions_override: Path | None,
    uid: str,
    max_retries: int = 3,
) -> tuple[dict | None, dict | None]:
    """Call ``_run_one`` with retry-on-transient-error semantics.

    Returns ``(None, None)`` on DROP (missing input file, or all retries
    exhausted). Cell continues in either case; the caller records dropped
    uids in ``run_index.json`` so eval can filter them.

    Exception taxonomy:
      * ``KeyboardInterrupt`` / ``SystemExit`` → propagate (clean shutdown +
        loud misconfig, e.g. "--skip-describe requires --descriptions-path").
      * ``FileNotFoundError`` → DROP immediately, no retry (typical case:
        upstream Stage-A shard dropped; retry can't create it).
      * everything else → retry up to ``max_retries`` with exponential
        backoff (60s, 120s, 240s), then DROP. Catches transient Gemini API
        errors (429, 5xx, ``httpx.RemoteProtocolError``) that used to kill
        the entire cell.
    """
    import time
    for attempt in range(1, max_retries + 1):
        try:
            return _run_one(
                args, item["audio_path"], out_dir, dcfg,
                describe_adapter=describe_adapter,
                text_adapter=text_adapter,
                descriptions_path_override=descriptions_override,
                audio_duration_s=float(item["duration"]),
                audio_offset_s=float(item["audio_offset_s"]),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except FileNotFoundError as e:
            print(
                f"[run_cascade] DROP uid={uid}: {e}. Cell continues.",
                flush=True,
            )
            return None, None
        except Exception as e:  # noqa: BLE001 — broad catch for transient network / API errors
            wait = min(60 * (2 ** (attempt - 1)), 300)
            print(
                f"[run_cascade] WARN uid={uid} attempt {attempt}/{max_retries} "
                f"failed: {type(e).__name__}: {e}",
                flush=True,
            )
            if attempt < max_retries:
                print(f"[run_cascade]   retrying in {wait}s", flush=True)
                time.sleep(wait)
    print(
        f"[run_cascade] DROP uid={uid} after {max_retries} attempts. "
        f"Cell continues with remaining uids.",
        flush=True,
    )
    return None, None


def _run_sins(
    args,
    dcfg: dict,
    *,
    describe_adapter,
    text_adapter,
    tag: str,
) -> int:
    audio_path = Path(args.audio) if args.audio else dcfg["default_audio"]
    run_dir = Path(args.output_root) / tag
    print(f"[run_cascade] SINS run dir: {run_dir}", flush=True)
    descriptions_override = (
        Path(args.descriptions_path) if args.descriptions_path else None
    )
    a, b = _run_one(
        args, audio_path, run_dir, dcfg,
        describe_adapter=describe_adapter,
        text_adapter=text_adapter,
        descriptions_path_override=descriptions_override,
        audio_duration_s=None,  # cascade.py infers from descriptions
    )
    if a is not None:
        print(f"[run_cascade] Stage A: {a['n_chunks_completed']} captions, "
              f"RT {a['rt_factor']:.1f}x → {a['descriptions_path']}", flush=True)
    if b is not None:
        print(f"[run_cascade] Stage B: {b['n_chunks_completed']}/{b['n_chunks']} "
              f"windows, {b['n_segments_total']} segments, "
              f"RT {b['rt_factor']:.1f}x. Summary: {run_dir / 'summary.json'}",
              flush=True)
    return 0


def _run_ego4d(
    args,
    dcfg: dict,
    *,
    describe_adapter,
    text_adapter,
    tag: str,
) -> int:
    from long_audio.datasets.ego4d import Ego4DDataset

    manifest_path = Path(args.manifest) if args.manifest else dcfg["default_manifest"]
    if args.eval_set:
        ds = Ego4DDataset.eval_set(manifest_path, split=args.split)
    else:
        ds = Ego4DDataset(
            manifest_path=manifest_path,
            min_events=args.min_events,
            min_duration_s=args.min_duration_s,
            split=args.split,
        )
    n_total = len(ds)
    cap = args.max_videos if args.max_videos else n_total
    n_run = min(cap, n_total)
    print(f"[run_cascade] Ego4D dataset: {n_total} videos pass filter "
          f"(min_events={args.min_events}, min_duration_s={args.min_duration_s}); "
          f"running {n_run}", flush=True)

    run_root = Path(args.output_root) / tag
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[run_cascade] Ego4D run root: {run_root}", flush=True)

    per_uid: list[dict] = []
    dropped_uids: list[str] = []
    skipped_uids: list[str] = []
    for idx in range(n_run):
        item = ds[idx]
        uid = item["uid"]
        out_dir = run_root / uid
        # Skip uids that already have a completed summary.json — enables
        # cheap resume after a wall-time TIMEOUT or wall-time-bounded rerun.
        # ``--force-rerun`` bypasses this (re-processes every uid).
        if (not getattr(args, "force_rerun", False)
                and (out_dir / "summary.json").exists()):
            skipped_uids.append(uid)
            if idx == 0 or (idx + 1) % 50 == 0 or (idx + 1) == n_run:
                print(f"[run_cascade] ({idx + 1}/{n_run}) uid={uid} SKIP (summary.json exists)", flush=True)
            continue
        print(f"[run_cascade] ({idx + 1}/{n_run}) uid={uid} "
              f"duration={item['duration']:.0f}s", flush=True)
        # For --skip-describe, the rsync'd descriptions live alongside the
        # per-uid output dir at <out_dir>/<describe_model>.descriptions.jsonl.
        # That's the same path describe_audio_chunks would have written, so
        # the convention is symmetric across Stage A / Stage B hosts.
        descriptions_override = (
            out_dir / f"{args.describe_model}.descriptions.jsonl"
            if args.skip_describe else None
        )
        a, b = _run_one_with_retry(
            args, item, out_dir, dcfg,
            describe_adapter=describe_adapter,
            text_adapter=text_adapter,
            descriptions_override=descriptions_override,
            uid=uid,
        )
        if a is None and b is None:
            dropped_uids.append(uid)
            continue
        per_uid.append({
            "uid": uid,
            "duration": item["duration"],
            "describe_summary": a,
            "cascade_summary": b,
        })

    # Top-level run_index.json so eval scripts can iterate per-uid.
    # Include SKIPPED uids in `videos` too — they have a completed summary.json
    # from a previous run, so seg-eval must be able to score them. Omitting
    # them here caused a silent "n_videos=0" bug when a whole cell was resumed
    # (all 170 uids skipped → videos:[] → eval crashed on None metrics).
    all_videos = per_uid + [{"uid": u, "skipped": True} for u in skipped_uids]
    (run_root / "run_index.json").write_text(json.dumps({
        "describe_model": args.describe_model,
        "text_model": args.text_model,
        "dataset": "ego4d",
        "n_videos": len(all_videos),
        "n_processed": len(per_uid),
        "n_dropped": len(dropped_uids),
        "dropped_uids": dropped_uids,
        "n_skipped": len(skipped_uids),
        "skipped_uids": skipped_uids,
        "videos": all_videos,
    }, indent=2))
    print(f"[run_cascade] Ego4D DONE. {len(per_uid)} videos processed "
          f"({len(dropped_uids)} dropped, {len(skipped_uids)} skipped as already-done). "
          f"Index: {run_root / 'run_index.json'}", flush=True)
    return 0


def _run_egolife(
    args,
    dcfg: dict,
    *,
    describe_adapter,
    text_adapter,
    tag: str,
) -> int:
    """EgoLife cascade — single-annotator variant of _run_ego4d.

    EgoLife has one annotator pass and no train/val/test split. The
    canonical eval-set filter reuses Ego4D's ``min_events`` /
    ``min_duration_s`` defaults (see EgoLifeDataset.EVAL_DEFAULTS)
    without a ``--split`` filter.
    """
    from long_audio.datasets.egolife.dataset import EgoLifeDataset

    manifest_path = Path(args.manifest) if args.manifest else dcfg["default_manifest"]
    if args.eval_set:
        ds = EgoLifeDataset.eval_set(manifest_path)
    else:
        ds = EgoLifeDataset(
            manifest_path=manifest_path,
            min_events=args.min_events,
            min_duration_s=args.min_duration_s,
        )
    n_total = len(ds)
    cap = args.max_videos if args.max_videos else n_total
    n_run = min(cap, n_total)
    print(f"[run_cascade] EgoLife dataset: {n_total} sessions pass filter "
          f"(min_events={args.min_events}, min_duration_s={args.min_duration_s}); "
          f"running {n_run}", flush=True)

    run_root = Path(args.output_root) / tag
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[run_cascade] EgoLife run root: {run_root}", flush=True)

    per_uid: list[dict] = []
    dropped_uids: list[str] = []
    skipped_uids: list[str] = []
    for idx in range(n_run):
        item = ds[idx]
        uid = item["uid"]
        out_dir = run_root / uid
        # Skip uids that already have a completed summary.json — enables
        # cheap resume after a wall-time TIMEOUT or wall-time-bounded rerun.
        # ``--force-rerun`` bypasses this (re-processes every uid).
        if (not getattr(args, "force_rerun", False)
                and (out_dir / "summary.json").exists()):
            skipped_uids.append(uid)
            if idx == 0 or (idx + 1) % 50 == 0 or (idx + 1) == n_run:
                print(f"[run_cascade] ({idx + 1}/{n_run}) uid={uid} SKIP (summary.json exists)", flush=True)
            continue
        print(f"[run_cascade] ({idx + 1}/{n_run}) uid={uid} "
              f"duration={item['duration']:.0f}s", flush=True)
        # Same skip-describe convention as _run_ego4d — descriptions live
        # per-uid at <out_dir>/<describe_model>.descriptions.jsonl.
        descriptions_override = (
            out_dir / f"{args.describe_model}.descriptions.jsonl"
            if args.skip_describe else None
        )
        a, b = _run_one_with_retry(
            args, item, out_dir, dcfg,
            describe_adapter=describe_adapter,
            text_adapter=text_adapter,
            descriptions_override=descriptions_override,
            uid=uid,
        )
        if a is None and b is None:
            dropped_uids.append(uid)
            continue
        per_uid.append({
            "uid": uid,
            "duration": item["duration"],
            "describe_summary": a,
            "cascade_summary": b,
        })

    # Same skipped-uids-must-be-in-videos rule as _run_ego4d — see comment there.
    all_videos = per_uid + [{"uid": u, "skipped": True} for u in skipped_uids]
    (run_root / "run_index.json").write_text(json.dumps({
        "describe_model": args.describe_model,
        "text_model": args.text_model,
        "dataset": "egolife",
        "n_videos": len(all_videos),
        "n_processed": len(per_uid),
        "n_dropped": len(dropped_uids),
        "dropped_uids": dropped_uids,
        "n_skipped": len(skipped_uids),
        "skipped_uids": skipped_uids,
        "videos": all_videos,
    }, indent=2))
    print(f"[run_cascade] EgoLife DONE. {len(per_uid)} sessions processed "
          f"({len(dropped_uids)} dropped, {len(skipped_uids)} skipped as already-done). "
          f"Index: {run_root / 'run_index.json'}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sins", "ego4d", "egolife"], default="sins")
    parser.add_argument("--audio", default=None,
                        help="SINS only: override the mono.flac path.")
    parser.add_argument("--manifest", default=None,
                        help="Ego4D only: override the annotated_manifest.json path.")
    parser.add_argument(
        "--describe-model", choices=list(DESCRIBE_MODELS.keys()), default="qwen3-omni",
        help="Stage-A audio LM. Default qwen3-omni.",
    )
    parser.add_argument(
        "--text-model", choices=list(TEXT_MODELS.keys()), default="gemini",
        help="Stage-B downstream text LM. Default gemini.",
    )
    parser.add_argument("--describe-model-id", default=None,
                        help="Override Stage-A HF repo / API id.")
    parser.add_argument("--text-model-id", default=None,
                        help="Override Stage-B model id (e.g. gemini-2.5-flash).")
    parser.add_argument("--text-adapter-dir", default=None,
                        help="Stage-B PEFT LoRA adapter dir (a cascade FT "
                             "checkpoint, e.g. runs/sft/cascaded_<model>-<TS>). "
                             "vLLM serves the base (--text-model / --text-model-id) "
                             "+ this LoRA via enable_lora + LoRARequest. Only valid "
                             "for vLLM text models (not gemini/fake).")
    parser.add_argument("--describe-chunk-s", type=float, default=10.0,
                        help="Stage-A caption window in seconds.")
    parser.add_argument("--describe-batch-size", type=int, default=64,
                        help="Stage-A chunks per generate_batch call (default 64). "
                             "For vLLM adapters, max_num_seqs is automatically "
                             "set to this value so the cart-and-checkout-lanes "
                             "always match (else vLLM silently serializes).")
    parser.add_argument("--describe-max-model-len", type=int, default=None,
                        help="vLLM-only: per-sequence max context length. "
                             "Stage A prompts are tiny (~200 tokens), so lower "
                             "values free a lot of KV cache for higher "
                             "max_num_seqs. Default (None) keeps the adapter's "
                             "own constructor default (32768).")
    parser.add_argument("--window-min", type=float, default=10.0,
                        help="Stage-B segmentation window in minutes.")
    parser.add_argument("--describe-max-new-tokens", type=int, default=2048)
    parser.add_argument("--segment-max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--decoder", choices=["structured", "freeform"], default="structured")
    parser.add_argument("--context", choices=["none", "prev"], default="none")
    parser.add_argument("--time-unit", choices=["second", "minute"], default="second")
    parser.add_argument(
        "--emit-description", action="store_true",
        help="Per-segment 1-2 sentence sub-activity narrative in addition to "
             "{label,start,end}. Default off keeps prompt + schema byte-identical "
             "to the seg-only baseline.",
    )
    parser.add_argument(
        "--skip-describe", action="store_true",
        help="Skip Stage A; consume an existing descriptions JSONL. "
             "For SINS: pass --descriptions-path. For Ego4D: the per-uid "
             "descriptions are read from <run_root>/<uid>/<model>.descriptions.jsonl "
             "(rsync them into place before this invocation).",
    )
    parser.add_argument("--descriptions-path", default=None,
                        help="SINS only, with --skip-describe: the existing JSONL.")
    parser.add_argument("--skip-segment", action="store_true",
                        help="Run Stage A only; skip Stage B.")
    parser.add_argument("--output-root", default=str(RUNS_DIR),
                        help="Output goes to <output_root>/<tag>/ (NO <UTC> subdir). "
                             "Pass e.g. runs/sins-e2e-v-cascade to match the canonical "
                             "weekend-pipeline layout.")
    parser.add_argument("--flat-output-tag", action="store_true",
                        help="Write directly under --output-root instead of appending "
                             "cascA_<describe-model>/<dataset>. Intended for SFT "
                             "caption materialization paths.")
    # Ego4D-specific filters (mirror scripts/run_e2e.py).
    parser.add_argument("--min-events", type=int, default=2,
                        help="Ego4D/EgoLife filter: min merged-runs in pass-1 to include. "
                             "Default 2 matches Ego4DDataset.EVAL_DEFAULTS / EgoLifeDataset.EVAL_DEFAULTS.")
    parser.add_argument("--min-duration-s", type=float, default=600.0,
                        help="Ego4D filter: min audio duration in seconds.")
    parser.add_argument("--eval-set", action="store_true",
                        help="Ego4D: use the canonical eval-set filter "
                             "(Ego4DDataset.EVAL_DEFAULTS).")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Ego4D: filter to one split. Default = all splits; pass "
             "`test` for inference / eval runs.",
    )
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Ego4D only: cap number of videos to run.")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Reprocess every uid even if summary.json exists. "
                             "Default is to skip already-completed uids for cheap "
                             "resume after wall-time TIMEOUTs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true",
                       help="Cap chunks/windows for quick verification.")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--max-describe-chunks", type=int, default=3,
                        help="With --smoke: max Stage-A captions per audio.")
    parser.add_argument("--max-windows", type=int, default=1,
                        help="With --smoke: max Stage-B windows per audio.")
    # ---- Stage-B reasoning-effort (Luna gpt-5-6 family) --------------------
    parser.add_argument(
        "--reasoning-effort", default="none",
        choices=["none", "low", "medium", "high"],
        help="Stage-B OpenAI-compat reasoning models only: sets "
             "reasoning_effort on the API call. 'none' fully skips hidden "
             "reasoning tokens; 'low'/'medium'/'high' turn reasoning on at "
             "the corresponding effort level. For Gemini native use "
             "--thinking-budget; for vLLM-served models (Gemma-4, "
             "Qwen 3-32B) use --thinking-enable + --thinking-budget-enforce.",
    )
    # ---- Stage-B thinking mode (Gemini native + Qwen/Gemma s1) --------
    parser.add_argument(
        "--thinking-budget", type=int, default=0,
        help="Gemini Stage-B only: sets thinking_config.thinking_budget "
             "on the google-genai config. 0 = off (default). Positive int "
             "allows up to N reasoning tokens.",
    )
    parser.add_argument(
        "--thinking-enable", action="store_true",
        help="vLLM Stage-B (Gemma-4, Qwen 3-32B, ...) only: enable the "
             "chat-template <think>...</think> prefix.",
    )
    parser.add_argument(
        "--thinking-budget-enforce", type=int, default=0,
        help="vLLM Stage-B only: s1-style forced-termination cap on "
             "reasoning tokens. Requires --thinking-enable.",
    )
    args = parser.parse_args()

    if args.thinking_budget_enforce > 0 and not args.thinking_enable:
        parser.error(
            "--thinking-budget-enforce requires --thinking-enable "
            "(the chat template must emit <think> for the LogitsProcessor "
            "to have anything to close)."
        )
    if args.thinking_budget > 0 and (args.thinking_enable or args.thinking_budget_enforce > 0):
        parser.error(
            "--thinking-budget is Gemini-only; combining it with "
            "--thinking-enable / --thinking-budget-enforce (vLLM-side) is "
            "a config error."
        )

    dcfg = _dataset_config(args.dataset)
    tag = "." if args.flat_output_tag else f"cascA_{args.describe_model}/{args.dataset}"

    # Load Stage A adapter only when we're actually running Stage A.
    # The skip-describe path used to construct it just to read .name, but
    # name == args.describe_model by registry convention — and CLAP
    # adapters (MS-CLAP, M2D-CLAP) require candidate_labels at __init__,
    # which Stage-B-only callers had no reason to supply.
    describe_adapter = None
    if not args.skip_describe:
        describe_adapter = _load(
            DESCRIBE_MODELS, args.describe_model,
            model_id=args.describe_model_id,
            extra_kwargs=_describe_kwargs(args, dcfg),
        )
        print(f"[run_cascade] Stage A: loading {describe_adapter.name} …", flush=True)
        describe_adapter.load()

    text_adapter = None
    if not args.skip_segment:
        text_extra: dict = {}
        if args.text_adapter_dir:
            if TEXT_MODELS[args.text_model][0] != "long_audio.inference.models.vllm_text":
                raise SystemExit(
                    f"--text-adapter-dir is a vLLM-only flag; text model "
                    f"{args.text_model!r} is not vLLM-based (LoRA serving requires "
                    "a VLLMTextAdapter)."
                )
            text_extra["adapter_dir"] = args.text_adapter_dir
        # Thinking-mode plumbing to the Stage-B text adapter. Gemini
        # (GeminiTextAdapter) accepts thinking_budget; vLLM-served
        # adapters (Gemma-4, Qwen 3-32B, ...) accept enable_thinking +
        # thinking_budget_enforce. Routing is by text-model registry
        # module -- mismatch surfaces as a TypeError from the ctor.
        if args.thinking_budget > 0:
            if args.text_model != "gemini":
                raise SystemExit(
                    f"--thinking-budget={args.thinking_budget} is Gemini-only; "
                    f"text model {args.text_model!r} does not accept it. For "
                    "vLLM-served text models use --thinking-enable + "
                    "--thinking-budget-enforce."
                )
            text_extra["thinking_budget"] = args.thinking_budget
        if args.reasoning_effort != "none":
            if args.text_model != "luna":
                raise SystemExit(
                    f"--reasoning-effort={args.reasoning_effort} is Luna-only; "
                    f"text model {args.text_model!r} does not accept it. "
                    "For Gemini use --thinking-budget; for vLLM-served text "
                    "models use --thinking-enable + --thinking-budget-enforce."
                )
            text_extra["reasoning_effort"] = args.reasoning_effort
        if args.thinking_enable or args.thinking_budget_enforce > 0:
            if TEXT_MODELS[args.text_model][0] != "long_audio.inference.models.vllm_text":
                raise SystemExit(
                    f"--thinking-enable / --thinking-budget-enforce are for "
                    f"vLLM-served text models; text model {args.text_model!r} "
                    "does not accept them."
                )
            text_extra["enable_thinking"] = args.thinking_enable
            if args.thinking_budget_enforce > 0:
                text_extra["thinking_budget_enforce"] = args.thinking_budget_enforce
        text_adapter = _load(
            TEXT_MODELS, args.text_model,
            model_id=args.text_model_id, extra_kwargs=text_extra,
        )
        print(f"[run_cascade] Stage B: loading {text_adapter.name} …", flush=True)
        text_adapter.load()

    if args.dataset == "sins":
        return _run_sins(args, dcfg,
                         describe_adapter=describe_adapter,
                         text_adapter=text_adapter, tag=tag)
    if args.dataset == "ego4d":
        return _run_ego4d(args, dcfg,
                          describe_adapter=describe_adapter,
                          text_adapter=text_adapter, tag=tag)
    if args.dataset == "egolife":
        return _run_egolife(args, dcfg,
                            describe_adapter=describe_adapter,
                            text_adapter=text_adapter, tag=tag)
    raise ValueError(f"Unknown dataset: {args.dataset!r}")


if __name__ == "__main__":
    raise SystemExit(main())
