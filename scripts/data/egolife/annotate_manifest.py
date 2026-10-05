# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 4 of the EgoLife data-prep pipeline: 3-stage OLMo enrichment
(clean, classify, afg) per 5-min window + coalesce into ``actions``.

For each session, bins translated DenseCaption fragments into
non-overlapping 5-min windows and runs OLMo-2-SFT (in one shared vLLM
instance) through CLEAN (rewrite first-person as "the wearer"),
CLASSIFY (vLLM guided-choice over the ATUS-6 labels), and AFG (BM25 +
FActScore-style atomic facts). Adjacent same-label windows are coalesced
into variable-length ``actions`` runs. Writes
``<output_root>/activity_labels.json``. GPU stage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from long_audio.datasets.egolife.schema import EGOLIFE_HINTS, EGOLIFE_LABELS
from long_audio.eval.description.afg import (
    afg_messages,
    BM25DemoRetriever,
    load_demons,
    parse_facts,
)
from long_audio.utils.events import coalesce_runs, tile_by_midpoint


DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-SFT"
DEFAULT_WINDOW_S = 300  # 5 minutes, matches Ego4D classifier granularity
DEFAULT_AUDIO_MANIFEST = (
    REPO_ROOT / "datasets" / "egolife" / "audio" / "audio_manifest.json"
)
DEFAULT_TRANSLATED_ROOT = (
    REPO_ROOT / "datasets" / "egolife" / "translated" / "DenseCaption"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "egolife"


# --------------------------------------------------------------------------
# Prompts — llm_clean, classify (ATUS-6), afg
# --------------------------------------------------------------------------

_CLEAN_SYSTEM_PROMPT = (
    "You are a text cleaner for EgoLife DenseCaption windows. The input is "
    "a machine translation of Chinese first-person narrations (~150 short "
    "clauses concatenated) captured by the recording wearer over ~5 minutes. "
    "Rewrite the text so that any first-person reference to the recording "
    "wearer becomes third-person 'the wearer' (matching verb agreement and "
    "possessive):\n"
    "  - 'I' (subject) -> 'the wearer'\n"
    "  - 'me' (object) -> 'the wearer'\n"
    "  - 'my' / 'mine' -> \"the wearer's\"\n"
    "  - 'myself' -> 'themself'\n"
    "  - 'we' / 'us' / 'our' when the group includes the wearer -> "
    "'the wearer and the others' (subject/object as needed)\n\n"
    "Keep all proper names (Jake, Alice, Tasha, Lucia, Katrina, Shure) and "
    "all third-person references unchanged. Do not add, remove, or reword "
    "any factual content. Do not editorialize. Output ONLY the rewritten "
    "text on a single line, no preamble, no explanation."
)


def _clean_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_cleaned(raw: str, fallback: str) -> str:
    """Extract the cleaned rewrite from the model's reply.

    OLMo occasionally prefixes with 'Output:' or wraps in quotes; strip
    those. If the model returns nothing usable, fall back to the raw
    input text (matches Ego4D's parse_cleaned behavior)."""
    stripped = raw.strip().strip("`\"' ").splitlines()
    if not stripped:
        return fallback
    line = stripped[0].strip()
    for prefix in ("Output:", "Cleaned:", "Rewritten:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    return line or fallback


_CLASSIFY_SYSTEM_PROMPT = (
    "You are an activity classifier. You're given the cleaned text of a "
    "5-minute segment from an egocentric daily-life recording (six people "
    "living together for a week to prepare an Earth Day party, from the "
    "EgoLife dataset). Label the segment with the single best class below "
    "using the ATUS taxonomy. If several activities appear, pick the "
    "dominant one.\n\n"
    "Classes:\n{class_block}"
)


def _format_class_block(hints: dict[str, str]) -> str:
    name_width = max(len(k) for k in hints)
    return "\n".join(
        f"- {name.ljust(name_width)} -- {desc}" for name, desc in hints.items()
    )


def _classify_messages(text: str) -> list[dict[str, str]]:
    system = _CLASSIFY_SYSTEM_PROMPT.format(
        class_block=_format_class_block(EGOLIFE_HINTS)
    )
    user = (
        f"Segment text (5 min): {text}\n\n"
        "Reply with exactly one class name from the list above."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_class(raw: str) -> str:
    """Return the model's reply as one of ``EGOLIFE_LABELS``.

    CLASSIFY is called with ``guided_choice=list(EGOLIFE_LABELS)`` so
    vLLM's structured-outputs constraint locks the output to exactly one
    of those strings. If ``raw`` is anything else, that's a vLLM
    constraint violation — raise, don't silently substring-match."""
    stripped = raw.strip()
    if stripped not in EGOLIFE_LABELS:
        raise ValueError(
            f"classify reply {raw!r} is not a valid ATUS-6 label "
            f"(guided_choice constraint violated) — expected one of "
            f"{sorted(EGOLIFE_LABELS)}"
        )
    return stripped


# --------------------------------------------------------------------------
# Fragment loading + windowing
# --------------------------------------------------------------------------


def _load_translated(root: Path) -> dict[tuple[str, str], list[dict]]:
    """Load all translated JSONL fragments keyed by (participant, day).

    Each fragment carries the original SRT-file's wall-clock offset baked
    into ``start_s`` / ``end_s`` (those are RELATIVE to the SRT file which
    is per-hour). We normalize to absolute-wall-clock seconds here by
    parsing the SRT filename's hour prefix.

    SRT filename format: ``A1_JAKE_DAY1_HHMMSS00.srt`` (from EgoLifeCap).
    """
    _FN = re.compile(r"^A\d_[A-Z]+_DAY\d+_(\d{2})(\d{2})(\d{2})(\d{2})$")
    by_pd: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for participant_dir in sorted(root.glob("A?_*")):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        for day_dir in sorted(participant_dir.glob("DAY?")):
            day = day_dir.name
            for jl in sorted(day_dir.glob("*.jsonl")):
                m = _FN.match(jl.stem)
                if not m:
                    raise ValueError(f"unparseable translated jsonl filename: {jl}")
                h, mi, se, ff = (int(x) for x in m.groups())
                hour_offset = h * 3600 + mi * 60 + se + ff / 100.0
                with jl.open(encoding="utf-8") as f:
                    for line in f:
                        e = json.loads(line)
                        e = dict(e)
                        e["abs_start_s"] = hour_offset + e["start_s"]
                        e["abs_end_s"] = hour_offset + e["end_s"]
                        by_pd[(participant, day)].append(e)
            by_pd[(participant, day)].sort(key=lambda x: x["abs_start_s"])
    return by_pd


def _fragments_in_session(
    fragments: list[dict],
    session_wall_start_s: float,
    session_dur_s: float,
) -> list[dict]:
    """Filter and re-time fragments to session-local ``[0, session_dur_s]``."""
    wall_end = session_wall_start_s + session_dur_s
    out = []
    for f in fragments:
        if f["abs_end_s"] <= session_wall_start_s:
            continue
        if f["abs_start_s"] >= wall_end:
            break
        # clip to session bounds; re-time to session-local
        s = max(0.0, f["abs_start_s"] - session_wall_start_s)
        e = min(session_dur_s, f["abs_end_s"] - session_wall_start_s)
        out.append(
            {
                "start": s,
                "end": e,
                "text_en": f["text_en"],
                "text_zh": f["text_zh"],
            }
        )
    return out


def _window_fragments(
    fragments: list[dict], window_s: float, session_dur_s: float
) -> list[dict]:
    """Bin session-local fragments into non-overlapping ``window_s`` tiles.
    Returns ``[{start, end, text}, ...]``.

    Every window must contain at least one non-empty fragment — DenseCaption
    covers each recorded hour at ~1 caption per 2 s, so a 5-min window
    should always have ~150 fragments. An empty window means an upstream
    alignment / coverage bug; raise so it can be investigated."""
    if not fragments:
        raise ValueError(
            f"no translated fragments provided (session_dur_s={session_dur_s})"
        )
    windows = []
    for w_start, w_end, tile_frags in tile_by_midpoint(
        fragments, session_dur_s, window_s
    ):
        text = " ".join(f["text_en"] for f in tile_frags)
        if not text:
            raise ValueError(
                f"empty window [{w_start:.1f}-{w_end:.1f}] in "
                f"session_dur_s={session_dur_s:.1f} — DenseCaption coverage gap "
                f"or wall-clock alignment drift"
            )
        windows.append({"start": w_start, "end": w_end, "text": text})
    return windows


# --------------------------------------------------------------------------
# vLLM batch inference — shared instance runs clean, classify, afg
# --------------------------------------------------------------------------


class _LLMBatch:
    """Thin wrapper around a single vLLM instance so all three enrichment
    stages (clean, classify, afg) share model residency + KV warmup."""

    def __init__(self, model_id: str) -> None:
        from vllm import LLM

        self._llm = LLM(model=model_id, dtype="bfloat16")

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        max_tokens: int,
        guided_choice: list[str] | None,
    ) -> list[str]:
        """Run a batch of chat prompts. ``guided_choice`` is required
        (pass ``None`` explicitly for free-form CLEAN / AFG). When a list
        is given, vLLM's ``StructuredOutputsParams(choice=...)`` locks
        the model output to exactly one of the provided strings — used
        by CLASSIFY so the parse_class_failed drop path can't fire."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        so = (
            StructuredOutputsParams(choice=list(guided_choice))
            if guided_choice is not None
            else None
        )
        sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, structured_outputs=so
        )
        outs = self._llm.chat(prompts, sp)
        return [o.outputs[0].text for o in outs]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audio-manifest",
        type=Path,
        default=DEFAULT_AUDIO_MANIFEST,
        help="Stage-2 output listing per-session FLACs + wall clocks.",
    )
    p.add_argument(
        "--translated-root",
        type=Path,
        default=DEFAULT_TRANSLATED_ROOT,
        help="Stage-3 output root containing translated DenseCaption " "JSONL files.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument(
        "--limit", type=int, default=None, help="Process only first N sessions (smoke)."
    )
    args = p.parse_args()

    audio_manifest = json.loads(args.audio_manifest.read_text())
    sessions = audio_manifest["sessions"]
    if args.limit:
        sessions = sessions[: args.limit]

    print(
        f"[annotate] loading translated fragments from {args.translated_root}",
        flush=True,
    )
    frags_by_pd = _load_translated(args.translated_root)

    # Load AFG demo pool (vendored FActScore biography demos) + BM25 retriever.
    demons = load_demons()
    afg_retriever = BM25DemoRetriever(demons)
    print(f"[annotate] AFG demo pool: {len(demons)} demos", flush=True)

    # Bin fragments into windows per session; build clean prompts.
    # Clean must run BEFORE classify + afg so those two consume the cleaned
    # (subject-aligned) text.
    #
    # Stage 2 (extract_audio) intersects audio sessions with DenseCaption
    # coverage, so every session here has caption coverage by construction.
    # If _fragments_in_session or _window_fragments come back empty, that
    # indicates a Stage-2 bug — let the raise inside those helpers surface
    # it (do not add a defensive drop path here).
    per_session_windows: dict[str, list[dict]] = {}
    clean_prompts: list[list[dict[str, str]]] = []
    prompt_index: list[tuple[str, int]] = []  # (uid, window_idx)

    t0 = time.time()
    for sess in sessions:
        uid = sess["uid"]
        pd_key = (sess["participant"], sess["day"])
        if pd_key not in frags_by_pd:
            raise RuntimeError(
                f"session {uid}: no translated fragments for {pd_key} "
                f"— Stage 3 output missing"
            )
        session_frags = _fragments_in_session(
            frags_by_pd[pd_key],
            sess["wall_clock_start_s"],
            sess["raw_audio_dur_s"],
        )
        windows = _window_fragments(
            session_frags, args.window_s, sess["raw_audio_dur_s"]
        )
        per_session_windows[uid] = windows
        for i, w in enumerate(windows):
            # Every window is guaranteed non-empty by _window_fragments.
            clean_prompts.append(_clean_messages(w["text"]))
            prompt_index.append((uid, i))

    print(
        f"[annotate] {len(clean_prompts)} windows across "
        f"{len(per_session_windows)} sessions",
        flush=True,
    )
    print(f"[annotate] loading vLLM ({args.model_id})", flush=True)
    batch = _LLMBatch(args.model_id)

    # Stage A: CLEAN (first-person -> "the wearer").
    print("[annotate] Stage A: llm_clean", flush=True)
    clean_replies = batch.chat(clean_prompts, max_tokens=512, guided_choice=None)
    assert len(clean_replies) == len(clean_prompts)
    cleaned_texts: list[str] = []
    for (uid, wi), raw in zip(prompt_index, clean_replies):
        raw_text = per_session_windows[uid][wi]["text"]
        cleaned = parse_cleaned(raw, fallback=raw_text)
        per_session_windows[uid][wi]["raw_clean"] = raw
        per_session_windows[uid][wi]["cleaned_text"] = cleaned
        cleaned_texts.append(cleaned)

    # Stage B: CLASSIFY (ATUS-6, vLLM structured output).
    # Model is forced to emit exactly one of EGOLIFE_LABELS via
    # StructuredOutputsParams(choice=...) — mirrors Ego4D's approach and
    # eliminates the parse_class_failed drop path entirely. Runs on the
    # CLEANED text (fragment list) — previous summarize-then-classify
    # experiment primed OLMo with boilerplate ("Earth Day party ->
    # cleaning and organizing") and made the classifier read hallucinated
    # verbs verbatim (44% parse_failure). Guided-choice + raw cleaned
    # text is the working combination.
    print("[annotate] Stage B: classify (ATUS-6, guided_choice)", flush=True)
    classify_prompts = [_classify_messages(t) for t in cleaned_texts]
    classify_replies = batch.chat(
        classify_prompts, max_tokens=32, guided_choice=list(EGOLIFE_LABELS)
    )
    assert len(classify_replies) == len(cleaned_texts)

    # Stage C: AFG on the CLEANED texts (full fragment content, granular).
    # Shares the LLM instance with classify (KV cache stays warm).
    print("[annotate] Stage C: afg (on cleaned text)", flush=True)
    afg_prompts_list = []
    for t in cleaned_texts:
        demo_sent, demo_facts = afg_retriever.top1(t)
        afg_prompts_list.append(afg_messages(t, demo_sent, demo_facts))
    # max_tokens=768: EgoLife cleaned_text is ~2000-2500 chars (~150
    # DenseCaption fragments per 5-min window). AFG emits ~14 bullets per
    # window and Ego4D's default 128 truncates mid-bullet, yielding
    # ~2% stub facts ("The.", "The wearer."). 768 tokens = ~50 bullets
    # of headroom.
    afg_replies = batch.chat(afg_prompts_list, max_tokens=768, guided_choice=None)
    assert len(afg_replies) == len(cleaned_texts)

    # Reattach classify + afg replies to their windows. parse_class raises
    # on any non-label reply, so w["label"] is always a valid ATUS-6 string.
    for (uid, wi), cls_raw, afg_raw in zip(prompt_index, classify_replies, afg_replies):
        w = per_session_windows[uid][wi]
        w["raw_classify"] = cls_raw
        w["label"] = parse_class(cls_raw)
        w["raw_afg"] = afg_raw
        w["facts"] = parse_facts(afg_raw)

    # Coalesce adjacent same-label windows per session into ``actions`` runs.
    out_sessions: dict[str, dict] = {}
    for uid, windows in per_session_windows.items():
        typed_windows = [
            {"start": w["start"], "end": w["end"], "event": w["label"]} for w in windows
        ]
        actions = coalesce_runs(typed_windows, key="event")
        out_sessions[uid] = {
            "windows": windows,
            "actions": actions,
            "error": None,
        }

    output = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "window_s": args.window_s,
        "n_sessions_input": len(sessions),
        "n_sessions_labeled": len(out_sessions),
        "sessions": out_sessions,
    }
    out_path = args.output_dir / "activity_labels.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(
        f"[annotate] DONE in {time.time() - t0:.1f}s. "
        f"labeled={len(out_sessions)} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
