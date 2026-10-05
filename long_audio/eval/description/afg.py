# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Atomic-fact generation for description-quality eval.

Mirrors the exact behavior of ``scripts/data/ego4d/annotate_manifest.py``'s
AFG stage — same system prompt, same 1-demo few-shot, same BM25 top-1
demo selection from the vendored FActScore biography ``demons.json``,
same bullet parser. ``annotate_manifest.py`` imports from here so the
two stay behavior-compatible.

Wording of ``AFG_INSTRUCT`` and the ``_afg_demo_block`` / ``afg_messages``
shape is verbatim from OpenFActScore (Lage & Couto, 2025). See
``long_audio/datasets/ego4d/factscore/README.md`` for demo provenance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DEMONS_PATH = (
    _REPO_ROOT / "long_audio" / "datasets" / "ego4d" / "factscore" / "demons.json"
)


AFG_INSTRUCT = (
    "You are an annotator that breaks down sentences into independent facts, "
    "short statements that each contain one piece of information contained "
    "in the given sentence.\n"
    "in the next paragraphs you have examples of sentences broken down in "
    "atomic facts.\n"
    "You have to complete the example given by the user.\n"
    "Do not add new entities, do not deviate from the subject of the sentence "
    "given by the user, do not hallucinate, do not repeat facts in the "
    "system prompt.\n"
    "List the sentences using -"
)

_BULLET_RE = re.compile(r"^\s*[-*•]\s*(.+?)\s*$")


def load_demons(path: Path | str = _DEFAULT_DEMONS_PATH) -> dict[str, list[str]]:
    """Load the ``{sentence: [facts]}`` demo pool from disk.

    Vendored bio-flavored demos from FActScore (Min et al., 2023).
    """
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(
            f"demons.json must be {{sentence: [facts]}}, got {type(d)}"
        )
    return d


class BM25DemoRetriever:
    """BM25 top-1 demo retriever over a ``{sentence: [facts]}`` pool.

    Wraps ``rank_bm25.BM25Okapi``. Constructor tokenises demo keys by
    whitespace; ``top1`` does the same on the query — matches
    ``annotate_manifest.py``'s prior inline usage exactly.
    """

    def __init__(self, demons: dict[str, list[str]]):
        from rank_bm25 import BM25Okapi
        self.demons = demons
        self._keys = list(demons.keys())
        self._bm25 = BM25Okapi([k.split(" ") for k in self._keys])

    def top1(self, target: str) -> tuple[str, list[str]]:
        """Return ``(demo_sentence, demo_facts)`` — closest demo to ``target``."""
        top = self._bm25.get_top_n(target.split(" "), self._keys, 1)[0]
        return top, self.demons[top]


def _afg_demo_block(demo_sentence: str, demo_facts: list[str]) -> str:
    lines = [f"Please breakdown the following sentence into independent facts: {demo_sentence}"]
    for fact in demo_facts:
        lines.append(f"- {fact}")
    return "\n".join(lines) + "\n"


def afg_messages(
    target_sentence: str,
    demo_sentence: str,
    demo_facts: list[str],
) -> list[dict[str, str]]:
    """Chat-format messages for atomic-fact generation.

    Single BM25-selected in-context demo is folded into the system
    message (matches OpenFActScore's ``HFmodel.chat_formatter`` behavior:
    demo goes to system channel, target to user channel).
    """
    system_text = f"{AFG_INSTRUCT}\n{_afg_demo_block(demo_sentence, demo_facts)}"
    user_text = (
        f"Please breakdown the following sentence into independent facts: "
        f"{target_sentence}"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def parse_facts(text: str) -> list[str]:
    """Extract bullet-prefixed facts from the model's free-text reply.

    Accepts ``-``, ``*``, or ``•`` bullets. Strips whitespace. Appends
    ``.`` if missing — keeps downstream NLI hypothesis formatting
    consistent. Order-preserving-dedupes exact-string repeats (OLMo's
    AFG stage under greedy decoding can enter repetition loops on
    hard-to-narrate windows; the pre-dedup manifest had 16% exact
    duplicates within-summary, with worst-case runs of 128 facts / 1
    unique — see memory).
    """
    facts: list[str] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if not m:
            continue
        s = m.group(1).strip()
        if not s:
            continue
        if not s.endswith("."):
            s = s + "."
        facts.append(s)
    return list(dict.fromkeys(facts))
