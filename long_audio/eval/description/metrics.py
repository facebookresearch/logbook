"""Description-quality metric machinery.

Three metrics — all use temporal-overlap-restricted candidate pools and
micro-pool aggregation:

  - Summary Recall  = (# GT summary atoms covered) / (# GT summary atoms)
  - Summary Precision = (# pred atoms supported)  / (# pred atoms)
  - Moments Recall  = (# moment labels covered)   / (# moment labels)

Overlap rule (open-open): ``a`` overlaps ``b`` iff
``a.start < b.end AND b.start < a.end``.

NLI direction (BART-MNLI is asymmetric):
  - Recall metrics: premise = predicted atom, hypothesis = GT atom.
  - Precision metric: premise = GT atom, hypothesis = predicted atom.

Decision threshold: P(entailment) > 0.5 → atom is covered/supported.

This module is compute-agnostic: it builds the (premise, hypothesis)
pair lists + index maps, and later folds NLI scores back to per-target
counts. The caller (see ``pipeline.py``) owns the actual BART-MNLI batching.
"""

from __future__ import annotations

from dataclasses import dataclass, field


P_ENTAIL_THRESHOLD = 0.5


@dataclass
class Atom:
    """One atomic fact bound to a temporal window ``[start, end]``.

    Same shape for GT summary atoms (window = the source slice's
    ``[start, end]``), moment labels (window = the moment's ``[start, end]``,
    ``text`` is the raw label), and predicted atoms (window = the segment's
    absolute temporal bounds, inherited by every atom the segment produces).

    The GT/pred distinction is a role at metric time, not a type.
    """
    text: str
    start: float
    end: float


def overlaps(a: Atom, b: Atom) -> bool:
    """Positive intersection between ``a``'s and ``b``'s temporal windows."""
    return a.start < b.end and b.start < a.end


@dataclass
class DirectionResult:
    """Micro-poolable outcome for one direction of one metric on one instance.

    ``n_targets`` = the atoms whose coverage we're measuring (denominator).
    ``n_covered`` = subset with at least one candidate above threshold.
    ``trace_rows`` are the entailment_trace.jsonl lines (emit for
    covered AND uncovered).
    """
    n_targets: int
    n_covered: int
    trace_rows: list[dict] = field(default_factory=list)


def _build_pair_batch(
    hypotheses: list[Atom],
    premises: list[Atom],
) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    """Enumerate overlapping (premise, hypothesis) NLI pairs.

    The hypothesis side is the denominator side of the metric (the atoms
    we're computing coverage for): recall passes ``hypotheses=gt,
    premises=pred``; precision passes ``hypotheses=pred, premises=gt``.

    Returns ``(pairs, idx_map)`` where each ``pairs[i]`` is
    ``(premise_text, hypothesis_text)`` (BART-MNLI input order) and
    ``idx_map[i] = (hypothesis_idx, premise_idx)``.
    """
    pairs: list[tuple[str, str]] = []
    idx_map: list[tuple[int, int]] = []
    for hi, h in enumerate(hypotheses):
        for pi, p in enumerate(premises):
            if not overlaps(h, p):
                continue
            pairs.append((p.text, h.text))
            idx_map.append((hi, pi))
    return pairs, idx_map


def score_direction(
    hypotheses: list[Atom],
    premises: list[Atom],
    entail_probs: list[float],
    idx_map: list[tuple[int, int]],
    *,
    side_label: str,
    threshold: float = P_ENTAIL_THRESHOLD,
    pass_num: int | None = None,
    uid: str | None = None,
) -> DirectionResult:
    """Given NLI scores for enumerated pairs, fold to per-hypothesis verdicts.

    Emits one ``trace_rows`` entry per hypothesis — including uncovered
    ones with their best (below-threshold) premise for debugging.
    Hypotheses with no premises at all (empty overlap pool) get a trace
    row with ``premise=None, p_entail=None, is_covered=False``.
    """
    n = len(hypotheses)
    best_p: list[float | None] = [None] * n
    best_pi: list[int | None] = [None] * n
    for (hi, pi), p in zip(idx_map, entail_probs):
        if best_p[hi] is None or p > best_p[hi]:
            best_p[hi] = p
            best_pi[hi] = pi

    n_covered = 0
    rows: list[dict] = []
    for hi, h in enumerate(hypotheses):
        bp = best_p[hi]
        bpi = best_pi[hi]
        premise = premises[bpi] if bpi is not None else None
        is_covered = bp is not None and bp > threshold
        if is_covered:
            n_covered += 1
        rows.append({
            "side": side_label,
            "uid": uid,
            "pass": pass_num,
            "hypothesis": h.text,
            "hypothesis_window": [h.start, h.end],
            "premise": premise.text if premise is not None else None,
            "premise_window": [premise.start, premise.end] if premise is not None else None,
            "p_entail": bp,
            "is_covered": is_covered,
        })
    return DirectionResult(n_targets=n, n_covered=n_covered, trace_rows=rows)


@dataclass
class MicroAccumulator:
    """Sums numerators + denominators across (uid, pass) for one metric."""
    n_targets: int = 0
    n_covered: int = 0

    def add(self, r: DirectionResult) -> None:
        self.n_targets += r.n_targets
        self.n_covered += r.n_covered

    def add_from_raw(self, covered: int, targets: int) -> None:
        """Add a raw ``(covered, targets)`` pair — useful when the caller
        has already collapsed a ``DirectionResult`` into counts before
        micro-pooling (e.g. after reading them back from ``summary_facts.json``).
        """
        self.n_targets += int(targets)
        self.n_covered += int(covered)

    @property
    def ratio(self) -> float | None:
        return self.n_covered / self.n_targets if self.n_targets else None
