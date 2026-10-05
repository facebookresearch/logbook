# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Self-consistency check for atomic-fact lists.

For each fact ``F`` inside a per-segment ``facts[]`` list of size ``n``:
  - **redundant** iff there exists ``G != F`` in the same list with
    ``entail(G -> F) > entail_threshold`` — some more-specific / paraphrasing
    sibling ``G`` subsumes ``F``.

Runs ``n * (n - 1)`` ordered NLI comparisons per list. Both orderings are
naturally captured by the enumeration; no double count.

Micro aggregation across lists uses total fact count as the denominator:
  ``%_redundant = sum(n_redundant_i) / sum(n_facts_i)``

Note: the earlier version of this module also tracked a ``contradicted``
axis via ``contradict(G -> F) > contradict_threshold``. It was removed
because BART-MNLI at any threshold treated co-occurring activities
("wearer is coughing" vs "wearer is singing") as contradictions — the
signal was dominated by false positives on activity-narrative corpora.
Redundancy remains reliable because entailment correctly fires on
paraphrase / detail-bleaching / conjunction-subsumption pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from long_audio.eval.description.nli import BartMNLI


DEFAULT_ENTAIL_THRESHOLD = 0.5


@dataclass
class ListResult:
    """Self-consistency result for a single facts list.

    ``redundant_indices`` are 0-based positions into the input ``facts``
    list of facts subsumed by some sibling.
    """
    n_facts: int
    redundant_indices: list[int]
    n_redundant: int


def score_facts_list(
    facts: list[str],
    nli: BartMNLI,
    *,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
) -> ListResult:
    """Score self-consistency of ONE facts list.

    Returns a :class:`ListResult`. For ``n < 2`` the function short-circuits
    to zero-marks (a singleton or empty list can't be self-inconsistent).
    """
    n = len(facts)
    if n < 2:
        return ListResult(n_facts=n, redundant_indices=[], n_redundant=0)

    # Build n * (n - 1) ordered pairs (premise=i, hypothesis=j), i != j.
    pair_idx: list[tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(n) if i != j
    ]
    pairs = [(facts[i], facts[j]) for i, j in pair_idx]

    entails = nli.entail_probs(pairs)

    redundant: set[int] = set()
    for (_i, j), e in zip(pair_idx, entails):
        # If entail(i -> j) > threshold, fact j is subsumed by i => j is REDUNDANT.
        if e > entail_threshold:
            redundant.add(j)

    return ListResult(
        n_facts=n,
        redundant_indices=sorted(redundant),
        n_redundant=len(redundant),
    )


@dataclass
class UidResult:
    """Per-uid roll-up over multiple per-segment lists.

    ``per_list`` carries one entry per input list with counts + a stable
    caller-supplied ``key`` (segment idx, slice idx, tile idx, ...).
    ``total`` is the intra-uid micro-aggregate.
    """
    per_list: list[dict[str, Any]]
    total: dict[str, int]


def evaluate_self_consistency_uid(
    lists: list[dict[str, Any]],
    nli: BartMNLI,
    *,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
) -> UidResult:
    """Score each list in ``lists`` independently and roll up to per-uid totals.

    ``lists`` items must carry:
      - ``"facts"``: ``list[str]``
      - ``"key"``: any JSON-serialisable identifier (segment idx, slice idx, ...)
    Optional keys (``"start"``, ``"end"``, ``"pass"``, ``"seg_idx"``, ...) are
    passed through into ``per_list`` verbatim for downstream provenance.
    """
    per_list: list[dict[str, Any]] = []
    total = {"n_facts": 0, "n_redundant": 0}
    for item in lists:
        facts = item["facts"]
        res = score_facts_list(facts, nli, entail_threshold=entail_threshold)
        row = {
            **{k: v for k, v in item.items() if k != "facts"},
            "n_facts": res.n_facts,
            "n_redundant": res.n_redundant,
            "redundant_indices": res.redundant_indices,
        }
        per_list.append(row)
        total["n_facts"] += res.n_facts
        total["n_redundant"] += res.n_redundant
    return UidResult(per_list=per_list, total=total)


def micro_percents(total: dict[str, int]) -> dict[str, float]:
    """Compute micro percentage from a totals dict.

    Returns ``{pct_redundant}``; ``count / n_facts`` (0 when ``n_facts == 0``).
    """
    n = total["n_facts"]
    if n == 0:
        return {"pct_redundant": 0.0}
    return {"pct_redundant": total["n_redundant"] / n}
