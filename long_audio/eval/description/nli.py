# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""BART-MNLI verifier for description-quality eval.

Wraps ``facebook/bart-large-mnli``. Batched entail-probability scoring
over ``(premise, hypothesis)`` pairs. Label-index resolution reads
``model.config.id2label`` so it also works if HF ever renumbers.
"""

from __future__ import annotations

from typing import Iterable


DEFAULT_MODEL_ID = "facebook/bart-large-mnli"


class BartMNLI:
    """Batched NLI scorer returning P(entailment) per (premise, hypothesis) pair."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)

        # Find the ENTAILMENT class index — models label them differently
        # (BART-MNLI: {0:contradiction, 1:neutral, 2:entailment}; DeBERTa may differ).
        entail_idx = None
        for i, lab in self.model.config.id2label.items():
            if lab.lower() == "entailment":
                entail_idx = int(i)
                break
        if entail_idx is None:
            raise RuntimeError(
                f"No ENTAILMENT class in id2label={self.model.config.id2label}"
            )
        self.entail_idx = entail_idx

    def entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return P(entailment) for each ``(premise, hypothesis)`` pair.

        Order-dependent: ``entail_probs([(A, B)])`` ≠ ``entail_probs([(B, A)])``.
        Callers wanting bidirectional should submit both orderings explicitly.
        """
        if not pairs:
            return []
        out: list[float] = []
        with self._torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                premises = [p for p, _ in batch]
                hypotheses = [h for _, h in batch]
                enc = self.tokenizer(
                    premises, hypotheses,
                    return_tensors="pt", truncation=True,
                    padding=True, max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                probs = self._torch.softmax(
                    self.model(**enc).logits, dim=-1
                ).cpu().tolist()
                out.extend(row[self.entail_idx] for row in probs)
        return out
