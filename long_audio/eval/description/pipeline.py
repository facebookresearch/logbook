"""AFG → NLI orchestration for description-quality eval.

Compute-side orchestration lives here so the CLI wiring in
``scripts/eval_descriptions.py`` stays thin and dataset-shape-specific I/O
stays out of the library.

Two things this module owns:

  1. ``run_afg`` — batched OLMo AFG over a flat list of description
     sentences, keyed by an opaque caller-supplied handle.
  2. ``evaluate_cell_uid`` / ``evaluate_human_baseline`` — build the pair
     batches, invoke NLI, fold to per-target counts + trace rows for one
     ``(cell, uid)`` (or human baseline over one uid).

The boundary type is ``GroundTruth``: a per-uid record of ``passes`` +
``moments`` already flattened to ``Atom`` lists. Callers convert their
dataset-specific manifest shape into ``GroundTruth`` at the CLI boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Hashable

from long_audio.eval.description.afg import (
    BM25DemoRetriever,
    afg_messages,
    load_demons,
    parse_facts,
)
from long_audio.eval.description.metrics import (
    Atom,
    DirectionResult,
    _build_pair_batch,
    score_direction,
)
from long_audio.eval.description.nli import BartMNLI


# ---- Boundary types -------------------------------------------------------


@dataclass
class GroundTruth:
    """Per-uid ground-truth summary + moments already flattened to ``Atom``s.

    ``passes``: ``{pass_key: [Atom, ...]}`` — GT summary atoms per pass.
    ``moments``: ``[Atom, ...]`` — moment labels (may be empty).
    """
    passes: dict[str, list[Atom]]
    moments: list[Atom] = field(default_factory=list)


@dataclass
class PredictedSegment:
    """One predicted segment carrying its raw ``description`` for AFG.

    ``key`` is any hashable — the caller uses it to map AFG outputs back
    to segments (typically ``(cell, uid, chunk_idx, seg_idx)``).
    """
    key: Hashable
    description: str
    start: float
    end: float


# ---- Atom adapters --------------------------------------------------------


def atoms_from_pass(pass_atoms: list[tuple[str, float, float]]) -> list[Atom]:
    """Flatten a per-pass ``[(fact, src_start, src_end), ...]`` list to ``Atom``s."""
    return [Atom(text=t, start=s, end=e) for (t, s, e) in pass_atoms]


def atoms_from_moments(moments: list[tuple[str, float, float]]) -> list[Atom]:
    """Flatten ``[(label, start, end), ...]`` moments to ``Atom``s."""
    return [Atom(text=t, start=s, end=e) for (t, s, e) in moments]


def pred_atoms_from_facts(pred_facts_doc: dict) -> list[Atom]:
    """Flatten per-segment ``facts`` (from ``predicted_facts.json``) into ``Atom``s."""
    atoms: list[Atom] = []
    for seg in pred_facts_doc["segments"]:
        for f in seg["facts"]:
            atoms.append(Atom(text=f, start=seg["start"], end=seg["end"]))
    return atoms


# ---- AFG runner -----------------------------------------------------------


@dataclass
class AFGConfig:
    model_id: str = "allenai/OLMo-2-1124-7B-SFT"
    tensor_parallel_size: int = 8
    max_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42
    gpu_memory_utilization: float = 0.90


def run_afg(
    descriptions: list[tuple[Hashable, str]],
    cfg: AFGConfig,
) -> dict[Hashable, list[str]]:
    """Batched OLMo AFG over ``descriptions``.

    Loads OLMo once via vLLM, runs generation in one shot, returns
    ``{key: [atomic_fact, ...]}``. Silent on empty input (returns ``{}``).
    """
    if not descriptions:
        return {}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    demons = load_demons()
    retriever = BM25DemoRetriever(demons)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    llm = LLM(
        model=cfg.model_id,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16",
        seed=cfg.seed,
    )
    sp = SamplingParams(
        temperature=cfg.temperature, top_p=1.0,
        max_tokens=cfg.max_tokens, seed=cfg.seed,
    )

    # OLMo-2-1124-7B-SFT has max_model_len=4096; leave headroom for the
    # 128-token output. Truncate over-length prompts token-side rather than
    # dropping, so all descriptions contribute a score. Observed on desc-v4
    # qwen2.5-72b × af-next-captioner × egolife where one atom's prompt hit
    # 4097 input tokens and crashed the whole batch (VLLMValidationError).
    MAX_INPUT_TOKENS = 4096 - cfg.max_tokens - 8  # 8-token safety margin
    prompt_token_ids: list[list[int]] = []
    n_truncated = 0
    for _key, desc in descriptions:
        demo_sent, demo_facts = retriever.top1(desc)
        msgs = afg_messages(desc, demo_sent, demo_facts)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_INPUT_TOKENS:
            ids = ids[:MAX_INPUT_TOKENS]
            n_truncated += 1
        prompt_token_ids.append(ids)
    if n_truncated:
        print(f"[afg] truncated {n_truncated}/{len(prompt_token_ids)} "
              f"prompts to {MAX_INPUT_TOKENS} input tokens (model max 4096)",
              flush=True)
    print(f"[afg] running vLLM on {len(prompt_token_ids)} prompts …", flush=True)
    from vllm import TokensPrompt
    t0 = time.time()
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids],
        sp, use_tqdm=False,
    )
    dt = time.time() - t0
    print(f"[afg] done in {dt:.1f}s ({len(prompt_token_ids) / max(dt, 1e-6):.1f} prompts/s)",
          flush=True)

    return {
        key: parse_facts(out.outputs[0].text)
        for (key, _desc), out in zip(descriptions, outs)
    }


# ---- Metric compute -------------------------------------------------------


def _run_direction(
    *,
    hypotheses: list[Atom],
    premises: list[Atom],
    nli: BartMNLI,
    side_label: str,
    pass_num: int | None,
    uid: str,
) -> DirectionResult:
    pairs, idx_map = _build_pair_batch(hypotheses, premises)
    probs = nli.entail_probs(pairs)
    return score_direction(
        hypotheses, premises, probs, idx_map,
        side_label=side_label, pass_num=pass_num, uid=uid,
    )


def evaluate_cell_uid(
    *,
    pred_atoms: list[Atom],
    gt: GroundTruth,
    nli: BartMNLI,
    uid: str,
) -> tuple[dict, list[dict]]:
    """Compute Summary Recall / Precision (per pass) + Moments Recall for one uid.

    Returns ``(counts_dict, trace_rows)`` — same layout the old script
    wrote to ``summary_facts.json`` / ``entailment_trace.jsonl``:

    ```
    {
      "pred_n_atoms": int,
      "passes": {"1": {...}, "2": {...}},
      "moments": {"n_moments": int, "moments_recall_covered": int},
    }
    ```
    """
    trace_rows: list[dict] = []
    counts: dict = {"pred_n_atoms": len(pred_atoms), "passes": {}, "moments": {}}

    for pkey, gt_atoms in gt.passes.items():
        r = _run_direction(
            hypotheses=gt_atoms, premises=pred_atoms, nli=nli,
            side_label=f"summary_recall_p{pkey}",
            pass_num=int(pkey), uid=uid,
        )
        p = _run_direction(
            hypotheses=pred_atoms, premises=gt_atoms, nli=nli,
            side_label=f"summary_precision_p{pkey}",
            pass_num=int(pkey), uid=uid,
        )
        counts["passes"][pkey] = {
            "gt_n_atoms": r.n_targets,
            "summary_recall_covered": r.n_covered,
            "summary_precision_supported": p.n_covered,
            "n_pred_atoms_for_precision_denom": p.n_targets,
        }
        trace_rows.extend(r.trace_rows)
        trace_rows.extend(p.trace_rows)

    if gt.moments:
        m = _run_direction(
            hypotheses=gt.moments, premises=pred_atoms, nli=nli,
            side_label="moments_recall",
            pass_num=None, uid=uid,
        )
        counts["moments"] = {
            "n_moments": m.n_targets,
            "moments_recall_covered": m.n_covered,
        }
        trace_rows.extend(m.trace_rows)
    else:
        counts["moments"] = {"n_moments": 0, "moments_recall_covered": 0}

    return counts, trace_rows


def evaluate_human_baseline(
    *,
    gt: GroundTruth,
    nli: BartMNLI,
    uid: str,
) -> tuple[dict, list[dict]]:
    """Human baseline: pass1 vs pass2 summary (both directions) + moments recall
    per pass, all micro-pooled.

    Summary directions:
      A: pass1 = GT, pass2 = pred
      B: pass2 = GT, pass1 = pred

    Moments recall (added): each pass's summary atoms treated as premises,
    moments as hypotheses. Micro-pooled across the two passes so the human
    baseline for moments_recall is directly comparable to the model-side
    moments_recall metric.
    """
    if set(gt.passes.keys()) != {"1", "2"}:
        raise ValueError(
            f"human baseline requires exactly two passes '1' and '2', "
            f"got {sorted(gt.passes.keys())}"
        )
    trace_rows: list[dict] = []
    counts = {"directions": {}, "combined": {}, "moments": {}}

    # ---- Summary Recall / Precision (both directions) ----
    for dir_label, gt_key, pred_key in [
        ("A_p1_gt_p2_pred", "1", "2"),
        ("B_p2_gt_p1_pred", "2", "1"),
    ]:
        gt_side = gt.passes[gt_key]
        pred_side = gt.passes[pred_key]
        r = _run_direction(
            hypotheses=gt_side, premises=pred_side, nli=nli,
            side_label=f"human_summary_recall_{dir_label}",
            pass_num=None, uid=uid,
        )
        p = _run_direction(
            hypotheses=pred_side, premises=gt_side, nli=nli,
            side_label=f"human_summary_precision_{dir_label}",
            pass_num=None, uid=uid,
        )
        counts["directions"][dir_label] = {
            "recall_covered": r.n_covered,
            "recall_denom": r.n_targets,
            "precision_covered": p.n_covered,
            "precision_denom": p.n_targets,
        }
        trace_rows.extend(r.trace_rows)
        trace_rows.extend(p.trace_rows)

    both = counts["directions"]
    a, b = both["A_p1_gt_p2_pred"], both["B_p2_gt_p1_pred"]
    counts["combined"] = {
        "recall_covered": a["recall_covered"] + b["recall_covered"],
        "recall_denom": a["recall_denom"] + b["recall_denom"],
        "precision_covered": a["precision_covered"] + b["precision_covered"],
        "precision_denom": a["precision_denom"] + b["precision_denom"],
    }

    # ---- Moments Recall (each pass's summary as pred, micro-pooled) ----
    if gt.moments:
        per_pass: dict[str, dict[str, int]] = {}
        for pkey in ("1", "2"):
            m = _run_direction(
                hypotheses=gt.moments, premises=gt.passes[pkey], nli=nli,
                side_label=f"human_moments_recall_p{pkey}",
                pass_num=int(pkey), uid=uid,
            )
            per_pass[pkey] = {
                "moments_recall_covered": m.n_covered,
                "n_moments": m.n_targets,
            }
            trace_rows.extend(m.trace_rows)
        counts["moments"] = {
            "per_pass": per_pass,
            "combined": {
                "moments_recall_covered": (
                    per_pass["1"]["moments_recall_covered"]
                    + per_pass["2"]["moments_recall_covered"]
                ),
                "n_moments": (
                    per_pass["1"]["n_moments"] + per_pass["2"]["n_moments"]
                ),
            },
        }
    else:
        counts["moments"] = {
            "per_pass": {},
            "combined": {"moments_recall_covered": 0, "n_moments": 0},
        }

    return counts, trace_rows
