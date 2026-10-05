# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 4 of the Ego4D data-prep pipeline: OLMo enrichment + per-pass
``actions`` / ``facts`` assembly.

Reads Stage 3's ``manifest.json`` and runs a 3-stage OLMo pipeline per
slice (``llm_clean`` → ``afg`` atomic facts → ATUS 6-way ``classify``).
CLASSIFY is vLLM-``choice``-constrained to the valid ATUS set; a
constraint violation (parse_class failure) or a Stage-3 coverage
violation inside ``_build_actions_for_pass`` raises — those indicate
infrastructure bugs, not per-uid data drops. Writes
``annotated_manifest.json`` with per-pass ``actions`` (midpoint-resolved
+ coalesced GT timeline) and per-pass ``facts`` (per-slice fact
records). GPU stage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from long_audio.eval.description.afg import (
    BM25DemoRetriever,
    afg_messages,
    load_demons,
    parse_facts,
)
from long_audio.utils.events import (
    assert_gt_contiguous,
    coalesce_runs,
    midpoint_boundaries,
)


# ============================================================================
# Inlined OLMo helpers (formerly classify.py / llm_clean.py / afg.py).
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EGO4D_DATA_DIR = REPO_ROOT / "long_audio" / "datasets" / "ego4d"

_DEFAULT_ATUS_PATH = EGO4D_DATA_DIR / "atus.json"
# NOTE: AFG helpers (load_demons, afg_messages, parse_facts, BM25DemoRetriever)
# live in long_audio.eval.description.afg — imported at the top of this file.
# The description-quality eval and this data-prep stage share them so the AFG
# behavior on GT summaries here matches AFG on predicted descriptions there.


# ---- ATUS classify ---------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = (
    "You are an activity classifier. You're given a recording scenario "
    "(the camera wearer's overall setting or role for the whole video) and "
    "a summary of one 5-minute segment from it. Label the segment with the "
    "single best class below, using the scenario as context for what the "
    "segment's activities mean (for example, whether an action is part of "
    "the person's job). If several activities appear, pick the dominant "
    "one. When the summary is ambiguous between multiple classes, prefer "
    "the class most consistent with the recording scenario.\n\n"
    "Classes:\n{class_block}"
)

_CLASS_TOKEN_RE = re.compile(r"[a-z][a-z_]*(?::[a-z][a-z_]*)?")


def load_atus(path: Path | str = _DEFAULT_ATUS_PATH) -> dict[str, str]:
    """Load the ATUS taxonomy as ``{class_name: description}``."""
    raw = json.loads(Path(path).read_text())
    return {k: v["description"] for k, v in raw.items()}


def _format_class_block(atus: dict[str, str]) -> str:
    """Render the class taxonomy as a `- name — description` bullet list."""
    name_width = max(len(k) for k in atus)
    return "\n".join(
        f"- {name.ljust(name_width)} — {desc}"
        for name, desc in atus.items()
    )


def _classify_messages(
    text: str,
    scenarios: list[str] | None,
    atus: dict[str, str],
) -> list[dict[str, str]]:
    """Build the chat-format messages list for OLMo-2-SFT classification."""
    system = _CLASSIFY_SYSTEM_PROMPT.format(class_block=_format_class_block(atus))
    if scenarios:
        scen_str = ", ".join(scenarios)
        user = (
            f"Recording scenario (whole video): {scen_str}\n"
            f"Segment summary (5 min): {text}\n"
            "Class label:"
        )
    else:
        user = f"Segment summary (5 min): {text}\nClass label:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_class(
    raw: str, valid_classes: Iterable[str],
) -> tuple[str, bool]:
    """Pick the ATUS class label from OLMo's free-text reply.

    Raises :class:`ValueError` on parse failure. CLASSIFY is called with
    vLLM ``StructuredOutputsParams(choice=valid_classes)`` so the model is
    constrained to emit exactly one of the valid class strings — a parse
    failure means the structured-outputs constraint was violated, which
    is an infrastructure bug worth killing on, not a per-slice drop.
    """
    valid = set(valid_classes)
    if not raw:
        raise ValueError(
            f"CLASSIFY produced empty output — vLLM structured-outputs "
            f"constraint violated (expected one of {sorted(valid)})."
        )
    s = raw.strip().strip("\"'`").rstrip(".;,:!?").strip().lower()
    if s in valid:
        return (s, True)
    for tok in _CLASS_TOKEN_RE.findall(s):
        if tok in valid:
            return (tok, True)
    raise ValueError(
        f"CLASSIFY reply {raw!r} is not a valid ATUS class "
        f"(structured-outputs constraint violated) — expected one of "
        f"{sorted(valid)}."
    )


# ---- LLM clean (summary rewrite) -------------------------------------------

_CLEAN_SYSTEM_PROMPT = """\
You are a text cleaner for Ego4D video summaries. Ego4D uses single \
uppercase letters to identify people in the recording: C is always the \
camera wearer; other single uppercase letters (X, Y, Z, A, B, O, W, D, \
etc.) denote other people in the scene. Your job is to rewrite the \
summary so it reads as natural English prose, removing the letter codes.

Rules:
1. Remove any leading "#Summary " prefix.
2. Rewrite person-letter codes based on context:
   - Standalone C → "the wearer".
   - Any other single uppercase letter used as a person identifier →
     "another person".
   - A list of letters (e.g. "X, Y and Z", "A, B, W and D") → "other
     people".
   - When a gender word ("man", "woman", "person", "boy", "girl",
     "lady", "kid", "child") is followed by a person letter, keep the
     gender word and drop the letter (e.g. "a man Y" → "a man",
     "person A" → "person").
3. Preserve single uppercase letters that are NOT person tags. Use
   context to tell. Examples that must stay intact:
   - Articles and pronouns: "A man entered", "I saw".
   - Technical compounds: "O-ring", "U-turn", "X-ray", "type O blood",
     "Vitamin C".
4. Fix obvious punctuation/typo artifacts from the annotation:
   - Stray space before commas/periods.
   - Missing space after commas.
   - Duplicated punctuation, trailing slashes/dashes/quotes that look
     like annotation cruft.
   - Repeated dangling "man, man, man" patterns left over from
     letter-stripping → collapse into natural phrasing
     ("with several men", "with two men", etc.).
   - Annotator hash-tag markers like `#unsure` or `#unknown` are
     placeholders for things the annotator could not identify. Replace
     them with "something" when they stand in for an object, or drop
     them entirely when they are noise. NEVER treat them as person
     identifiers.
   Keep all factual content intact.
5. Do not add information that was not in the original. Do not invent
   actions, objects, or people. Do not editorialize.
6. Output ONLY the cleaned summary text on a single line. No quotes,
   no preamble, no "Output:" prefix, no explanation."""


_CLEAN_DEMOS: list[tuple[str, str]] = [
    ("#Summary C was in a house, she interacted with a man Y and a woman X "
     "and C entered the bedroom to wear her boots.",
     "The wearer was in a house, she interacted with a man and a woman, "
     "and the wearer entered the bedroom to wear her boots."),
    ("#Summary C was in his house with person A,B, W and D, prepared for "
     "the journey and then left the house.",
     "The wearer was in his house with other people, prepared for the "
     "journey, and then left the house."),
    ("#Summary C was in a workshop, he removed and replaced some O-rings "
     "with a scriber in a cylinder and a bolt.",
     "The wearer was in a workshop, he removed and replaced some O-rings "
     "with a scriber in a cylinder and a bolt."),
    ("#Summary The person decorated the room while talking to O then O sat "
     "on the bed.",
     "The wearer decorated the room while talking to another person, then "
     "that other person sat on the bed."),
    ("#Summary C was in a car with man X, man Y, man A, man A drove the car",
     "The wearer was in a car with several men. One of the men drove the car."),
    ("#Summary C made a U-turn at the intersection.",
     "The wearer made a U-turn at the intersection."),
    ("#Summary C exercised and later on used his mobile phone",
     "The wearer exercised and later on used his mobile phone."),
    ("#Summary C scooped #unsure with a scrapper and put it in the bucket "
     "then took the container, #unsure and the scrapper and went downstairs "
     "to rinse them.",
     "The wearer scooped something with a scrapper and put it in the "
     "bucket, then took the container, something else, and the scrapper "
     "and went downstairs to rinse them."),
]


def _clean_messages(raw_summary: str) -> list[dict[str, str]]:
    parts = ["Examples:"]
    for raw, cleaned in _CLEAN_DEMOS:
        parts.append("")
        parts.append(f"Input: {raw}")
        parts.append(f"Output: {cleaned}")
    examples = "\n".join(parts)
    system = f"{_CLEAN_SYSTEM_PROMPT}\n\n{examples}"
    user = f"Input: {raw_summary}\nOutput:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_cleaned(generation: str, fallback: str) -> str:
    """Extract the cleaned text from the model's reply, or fall back."""
    if not generation:
        return fallback
    s = generation.strip()
    for prefix in ("Output:", "output:", "OUTPUT:"):
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
            break
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    else:
        return fallback
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'", "`"):
        s = s[1:-1].strip()
    return s if s else fallback


# ---- AFG helpers moved to long_audio/eval/description/afg.py (see top import).
# The description-quality eval and this data-prep stage share them so AFG on
# GT summaries here stays behavior-compatible with AFG on predicted descriptions.


# ============================================================================
# Timeline assembly (formerly _apply_midpoint_boundaries + _build_actions).
# ============================================================================

def _apply_midpoint_boundaries(
    sorted_slices: list[dict],
    duration: float,
) -> list[dict]:
    """Resolve adjacent overlapping/gapped slices via the midpoint rule.

    Thin wrapper over :func:`long_audio.utils.events.midpoint_boundaries` (the
    shared midpoint-rule core). Input slices must have ``start``, ``end``,
    ``event`` (post-OLMo, unified schema); output dicts have ``start``, ``end``,
    ``event``. By Stage 3 contract the input covers ``[0, duration]`` without
    interior gaps, so this only sets the boundaries between adjacent slices.
    Edges are clamped but NOT exact-snapped here (``snap_edges=False``);
    :func:`_build_actions_for_pass` snaps them before ``coalesce_runs``.
    """
    return midpoint_boundaries(
        sorted_slices,
        duration,
        carry={"event": "event"},
        snap_edges=False,
        name="_apply_midpoint_boundaries",
    )


def _build_actions_for_pass(
    enriched_slices: list[dict],
    duration: float,
    uid: str,
    pass_num: int,
) -> list[dict]:
    """Build per-pass `actions`: midpoint boundaries → coalesce by event →
    snap edges → assert_gt_contiguous. No gap-fill (Stage 3 enforced
    coverage)."""
    sorted_s = sorted(enriched_slices, key=lambda s: (s["start"], -s["end"]))
    boundaried = _apply_midpoint_boundaries(sorted_s, duration)
    # Snap edges exactly so float drift doesn't leak into contiguity assert.
    if boundaried:
        boundaried[0] = {**boundaried[0], "start": 0.0}
        boundaried[-1] = {**boundaried[-1], "end": float(duration)}
    actions = coalesce_runs(boundaried, key="event")
    assert_gt_contiguous(actions, duration, name=f"uid={uid} pass{pass_num} actions")
    if len(actions) < 1:
        raise ValueError(
            f"_build_actions_for_pass: uid={uid} pass{pass_num} produced "
            f"{len(actions)} events, expected >=1."
        )
    # NOTE: previously required >=2 events (rejected uniform-activity
    # passes, i.e. videos where OLMo classified every slice into the same
    # ATUS class). That cost ~2200 videos / ~728 h on a recent run, and
    # those drops aren't *classification failures* — single-activity
    # timelines are perfectly valid segmentations and the eval can
    # handle 1-event GT fine. Single-event passes are kept; the test-set
    # filter at consumption time (Ego4DDataset.EVAL_DEFAULTS.min_events)
    # is where we gate "interesting enough to eval on".
    return actions


# ============================================================================
# OLMo enrichment + main pipeline.
# ============================================================================

def _enrich_pass(
    slices: list[dict],
    scenarios: list[str],
    llm,
    tokenizer,
    afg_retriever: BM25DemoRetriever,
    atus: dict,
    clean_sp,
    afg_sp,
    cls_sp,
    *,
    run_classify: bool = True,
) -> tuple[list[dict], int, int]:
    """OLMo-enrich every slice in one pass. Returns
    ``(enriched_slices, n_clean_fallbacks, n_parse_class_failures)``.

    An enriched slice carries ``summary``, ``facts``, ``event``,
    ``raw_classify`` in addition to its original fields. (Internally
    the OLMo classifier variable is called ``activity``; the enriched
    slice dict renames it to ``event`` so downstream ``_build_actions``
    / ``coalesce_runs(..., key="event")`` finds it.) ``parse_class``
    raises on constraint violation; the caller does not catch, so one
    bad slice kills the run.

    When ``run_classify`` is False (idempotent mode), the CLASSIFY
    stage is skipped — ``event`` and ``raw_classify`` are left as None.
    Callers that supply pre-computed per-pass ``actions`` from a prior
    run use this to save GPU time on the classify pass.

    The third return value is kept at 0 for backward compatibility with
    callers that previously used it as a per-slice parse_class failure
    counter.
    """
    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    raws = [s["raw_summary"] for s in slices]

    # Stage A: CLEAN
    clean_prompts = [render(_clean_messages(r)) for r in raws]
    clean_outs = llm.generate(clean_prompts, clean_sp, use_tqdm=False)
    cleaned_texts: list[str] = []
    n_clean_fb = 0
    for raw, out in zip(raws, clean_outs):
        cleaned = parse_cleaned(out.outputs[0].text, raw)
        cleaned_texts.append(cleaned)
        if cleaned == raw:
            n_clean_fb += 1

    # Stage B: AFG (BM25 top-1 demo → OLMo)
    afg_prompts = []
    for c in cleaned_texts:
        demo_sent, demo_facts = afg_retriever.top1(c)
        afg_prompts.append(render(afg_messages(c, demo_sent, demo_facts)))
    afg_outs = llm.generate(afg_prompts, afg_sp, use_tqdm=False)
    facts_list = [parse_facts(o.outputs[0].text) for o in afg_outs]

    # Stage C: CLASSIFY (skipped in idempotent mode)
    if run_classify:
        cls_prompts = [
            render(_classify_messages(c, scenarios, atus)) for c in cleaned_texts
        ]
        cls_outs = llm.generate(cls_prompts, cls_sp, use_tqdm=False)
        raw_cls_texts = [o.outputs[0].text for o in cls_outs]
        valid_classes = list(atus.keys())
        # parse_class raises on constraint violation — do NOT catch; this
        # indicates vLLM structured-outputs is broken, which affects every
        # slice from here on. Fail loudly instead of silently dropping uids.
        activities = [parse_class(t, valid_classes)[0] for t in raw_cls_texts]
    else:
        raw_cls_texts = [None] * len(cleaned_texts)
        activities = [None] * len(cleaned_texts)

    enriched = [
        {
            **s,
            "summary": cleaned,
            "facts": facts,
            "event": activity,
            "raw_classify": raw_cls,
        }
        for s, cleaned, facts, activity, raw_cls
        in zip(slices, cleaned_texts, facts_list, activities, raw_cls_texts)
    ]
    return enriched, n_clean_fb, 0


def annotate_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    model_id: str,
    tensor_parallel_size: int = 8,
    gpu_memory_utilization: float = 0.90,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    clean_max_tokens: int = 200,
    afg_max_tokens: int = 128,
    classify_max_tokens: int = 32,
    limit_videos: int | None = None,
    force_reclassify: bool = False,
) -> dict:
    """Read Stage 3 manifest.json → OLMo-enrich every uid's slices →
    build per-pass actions + facts → write annotated_manifest.json. A
    constraint-violating CLASSIFY output or a Stage-3 coverage violation
    inside ``_build_actions_for_pass`` propagates — the run dies rather
    than silently dropping the offending uid.

    Idempotent mode: if the input manifest's ``passes.{1,2}`` already
    contain ``actions``, the CLASSIFY stage and the per-pass actions
    rebuild are skipped; existing actions are kept verbatim and only
    CLEAN + AFG run (populating ``summary`` + ``facts`` + per-pass
    ``facts`` records). Opt back into a full re-run with
    ``force_reclassify=True``."""

    # Lazy imports so --help works without GPU stack.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    manifest = json.loads(Path(manifest_path).read_text())
    videos = manifest["videos"]
    if limit_videos:
        videos = videos[:limit_videos]
    print(f"[annotate] manifest: {len(videos)} videos", flush=True)

    atus = load_atus()
    valid_classes = list(atus.keys())
    demons = load_demons()
    afg_retriever = BM25DemoRetriever(demons)
    print(f"[annotate] ATUS: {len(atus)} classes; FactScore demos: {len(demons)}",
          flush=True)

    print(f"[annotate] loading OLMo: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        seed=seed,
    )
    clean_sp = SamplingParams(temperature=temperature, top_p=top_p,
                              max_tokens=clean_max_tokens, seed=seed)
    afg_sp = SamplingParams(temperature=temperature, top_p=top_p,
                            max_tokens=afg_max_tokens, seed=seed)
    # Constrain classify output to exactly one of the 6 ATUS classes via
    # vLLM structured outputs. If parse_class fails anyway, that's a
    # constraint violation — annotate_manifest raises, not drops.
    cls_sp = SamplingParams(
        temperature=temperature, top_p=top_p,
        max_tokens=classify_max_tokens, seed=seed,
        structured_outputs=StructuredOutputsParams(choice=valid_classes),
    )
    print("[annotate] vLLM ready.", flush=True)

    annotated_videos: list[dict] = []
    n_clean_fb_total = 0
    t0 = time.time()

    for vi, v in enumerate(videos):
        uid = v["uid"]
        scenarios = v["scenarios"]
        annotated_passes: dict[str, dict] = {}

        for pass_num in ("1", "2"):
            pass_data = v["passes"][pass_num]
            slices = pass_data["slices"]
            existing_actions = pass_data.get("actions") if not force_reclassify else None
            run_classify = existing_actions is None

            enriched, n_clean_fb, _ = _enrich_pass(
                slices, scenarios, llm, tokenizer, afg_retriever,
                atus, clean_sp, afg_sp, cls_sp,
                run_classify=run_classify,
            )
            n_clean_fb_total += n_clean_fb

            # In idempotent mode we trust the existing per-pass actions
            # (already OLMo-derived upstream) and skip the midpoint-rebuild
            # to guarantee bit-identical output across invocations.
            # Non-idempotent mode: _build_actions_for_pass raises on
            # Stage-3 coverage violations, which propagate (bug, not drop).
            if existing_actions is not None:
                actions = existing_actions
            else:
                actions = _build_actions_for_pass(
                    enriched, float(v["duration"]), uid, int(pass_num),
                )
            facts_records = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "facts": s["facts"],
                }
                for s in enriched
            ]
            annotated_passes[pass_num] = {
                "slices": enriched,
                "actions": actions,
                "facts": facts_records,
            }

        new_video = {**v, "passes": annotated_passes}
        annotated_videos.append(new_video)

        if (vi + 1) % 10 == 0 or vi == len(videos) - 1:
            elapsed = time.time() - t0
            rate = (vi + 1) / elapsed if elapsed else 0
            eta_min = ((len(videos) - vi - 1) / rate / 60) if rate else 0
            print(
                f"[annotate] {vi+1}/{len(videos)} uids | "
                f"clean_fb={n_clean_fb_total} | "
                f"{rate:.2f} uid/s | eta {eta_min:.1f} min",
                flush=True,
            )

    out = {
        **{k: v for k, v in manifest.items() if k != "videos"},
        "n_videos": len(annotated_videos),
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "olmo_model_id": model_id,
        "videos": annotated_videos,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))

    # --- summary log ---
    print()
    print("=" * 80)
    print(f"[annotate] DONE in {(time.time()-t0)/60:.1f} min. "
          f"Kept all {len(annotated_videos)} input uids.")
    print(f"  output: {output_path}")
    print(f"  clean fallbacks (kept raw): {n_clean_fb_total}")
    print("=" * 80)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    EGO4D_DIR = REPO_ROOT / "datasets" / "ego4d"
    p.add_argument("--manifest", type=Path, default=EGO4D_DIR / "manifest.json")
    p.add_argument("--output", type=Path,
                   default=EGO4D_DIR / "annotated_manifest.json")
    p.add_argument("--model-id", default="allenai/OLMo-2-1124-7B-SFT")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean-max-tokens", type=int, default=200)
    p.add_argument("--afg-max-tokens", type=int, default=128)
    p.add_argument("--classify-max-tokens", type=int, default=32)
    p.add_argument("--limit-videos", type=int, default=None,
                   help="Smoke: cap to N input uids.")
    p.add_argument(
        "--force-reclassify",
        action="store_true",
        help="Re-run CLASSIFY + rebuild per-pass actions even when the "
        "input manifest already carries actions (e.g. after "
        "`build_manifest.py --public-manifest`). Default: skip "
        "CLASSIFY and keep input actions verbatim.",
    )
    args = p.parse_args()

    annotate_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        clean_max_tokens=args.clean_max_tokens,
        afg_max_tokens=args.afg_max_tokens,
        classify_max_tokens=args.classify_max_tokens,
        limit_videos=args.limit_videos,
        force_reclassify=args.force_reclassify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
