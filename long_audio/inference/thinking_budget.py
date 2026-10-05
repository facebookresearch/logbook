"""s1-style thinking-mode support for vLLM-backed adapters.

Provides :class:`ThinkingBudgetLogitsProcessor` (the s1 recipe as a
vLLM V1 LogitsProcessor — engine-registered, batch-aware, activated
per-request through ``SamplingParams.extra_args``) and
:class:`ThinkingMixin` (shared wiring for the Qwen 3-Omni and text-vLLM
adapters — owns THINK_START/END constants, engine-kwarg hooks,
per-request extra_args, and the output-splitter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig


# Key used in ``SamplingParams.extra_args`` to activate this processor
# for a single request. The value is a dict:
#   {"budget": int > 0, "end_token_ids": list[int] (non-empty)}
EXTRA_ARGS_KEY = "thinking_budget"


def resolve_end_token_ids(tokenizer, think_end: str) -> list[int]:
    """Encode the END-of-think marker into its (possibly multi-token) id
    sequence. add_special_tokens=False because the marker is inline
    inside the assistant turn, not a chat-turn boundary.
    """
    ids = tokenizer.encode(think_end, add_special_tokens=False)
    if not ids:
        raise ValueError(
            f"Tokenizer produced 0 tokens for think_end={think_end!r} -- "
            "cannot enforce budget without a stop sequence."
        )
    return list(ids)


class _RequestState:
    """Per-request mutable state for the s1 processor.

    Not a dataclass so we can drop instances cheaply from the state
    dict on request completion (nothing to release).
    """

    __slots__ = ("budget", "end_ids", "output_tok_ids", "forced_idx", "done")

    def __init__(
        self,
        budget: int,
        end_ids: Sequence[int],
        output_tok_ids: Sequence[int],
    ):
        self.budget = budget
        self.end_ids: list[int] = list(end_ids)
        # A LIVE reference to the request's running output list -- vLLM
        # appends to it as tokens are sampled, so len() is the current
        # generated-token count without any bookkeeping on our side.
        self.output_tok_ids = output_tok_ids
        # None: not currently forcing. int in [0, len(end_ids)): forcing
        # the k-th end token on THIS apply() call. done=True after the
        # full end sequence has been emitted (naturally or forced).
        self.forced_idx: int | None = None
        self.done: bool = False


def _saw_natural_end(
    output_tok_ids: Sequence[int], end_ids: Sequence[int]
) -> bool:
    """True if the full end-token sequence appears anywhere in the
    generated output so far. Sliding window (not just tail match) —
    the model might have emitted THINK_END and then generated
    post-answer content before this apply() fires.
    """
    n = len(end_ids)
    if len(output_tok_ids) < n:
        return False
    end_list = list(end_ids)
    for i in range(len(output_tok_ids) - n + 1):
        if list(output_tok_ids[i : i + n]) == end_list:
            return True
    return False


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """vLLM V1 logits processor implementing s1-style forced thinking
    termination.

    Registered once per engine via ``LLM(logits_processors=[...])``.
    Activated per-request via ``SamplingParams.extra_args`` — requests
    that don't set the extras entry are ignored (fast path in apply()).

    Algorithm per request:
      1. Watch generated tokens. If the full end-token sequence appears
         naturally, mark done and stay out of the way.
      2. Once budget is reached and no natural end has been seen, start
         forcing: on each subsequent apply() call, mask all logits to
         -inf except the next end-token id. Advance through end_ids
         until the whole sequence has been forced; then done.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        # No engine-level allocation needed -- state is a sparse dict.
        # Requests without the extra_args key are never added to it, so
        # workloads that don't use thinking-budget pay zero overhead.
        self.states: dict[int, _RequestState] = {}
        self.device = device

    def is_argmax_invariant(self) -> bool:
        # Forcing a specific end token DOES change argmax when the
        # model would otherwise have emitted something else.
        return False

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        cfg = (sampling_params.extra_args or {}).get(EXTRA_ARGS_KEY)
        if cfg is None:
            return
        budget = cfg.get("budget")
        end_ids = cfg.get("end_token_ids")
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"{EXTRA_ARGS_KEY}.budget must be a positive int, got {budget!r}"
            )
        if not end_ids or not all(isinstance(x, int) for x in end_ids):
            raise ValueError(
                f"{EXTRA_ARGS_KEY}.end_token_ids must be a non-empty list "
                f"of ints, got {end_ids!r}"
            )

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if not batch_update:
            return

        for index, params, _prompt, output_tok_ids in batch_update.added:
            cfg = (params.extra_args or {}).get(EXTRA_ARGS_KEY)
            if cfg is None:
                # No thinking-budget for this slot -- ensure any stale
                # entry from a previous occupant is cleared.
                self.states.pop(index, None)
                continue
            self.states[index] = _RequestState(
                budget=int(cfg["budget"]),
                end_ids=cfg["end_token_ids"],
                output_tok_ids=output_tok_ids,
            )

        if self.states:
            for index in batch_update.removed:
                self.states.pop(index, None)

            # Moves: rebind slot indices. UNIDIRECTIONAL a->b means
            # slot a is being overwritten by whatever is at b (or a's
            # request is moving to b and slot a is freed); SWAP means
            # exchange. Follow the same pop-then-place shape as V1's
            # process_dict_updates helper so we can't leak a stale
            # entry at the source index.
            for a_index, b_index, direct in batch_update.moved:
                from vllm.v1.sample.logits_processor.interface import (
                    MoveDirectionality,
                )

                a_state = self.states.pop(a_index, None)
                b_state = self.states.pop(b_index, None)
                if a_state is not None:
                    self.states[b_index] = a_state
                if b_state is not None:
                    if direct == MoveDirectionality.SWAP:
                        self.states[a_index] = b_state
                    # UNIDIRECTIONAL: b's state is discarded.

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Sparse fast-path: workloads without thinking-budget pay
        # one dict-emptiness check per token.
        if not self.states:
            return logits

        # Process each tracked slot. logits shape: (batch_size, vocab).
        for index, st in list(self.states.items()):
            if st.done:
                continue

            # Watch for natural termination BEFORE deciding to force.
            if st.forced_idx is None:
                if _saw_natural_end(st.output_tok_ids, st.end_ids):
                    st.done = True
                    continue
                if len(st.output_tok_ids) < st.budget:
                    continue
                # Budget hit -- start forcing on this token.
                st.forced_idx = 0

            forced_token = st.end_ids[st.forced_idx]
            # Mask all logits at this row to -inf except the forced token.
            row = logits[index]
            row.fill_(float("-inf"))
            row[forced_token] = 0.0

            st.forced_idx += 1
            if st.forced_idx >= len(st.end_ids):
                st.done = True
                st.forced_idx = None

        return logits


def split_thinking(
    full_text: str, think_start: str, think_end: str,
) -> tuple[str | None, str]:
    """Peel a leading ``<THINK_START>...<THINK_END>`` block off the
    model's raw output. Returns ``(thinking_trace, visible_text)``.

    * If THINK_START is absent -- returns ``(None, full_text)``.
    * If THINK_START is present but THINK_END is not (mid-reason
      truncation) -- everything after THINK_START becomes the trace
      and ``visible_text`` is empty.
    * If THINK_START and THINK_END both present -- the substring
      between them is the trace; everything after THINK_END (with a
      leading whitespace/newline stripped) is ``visible_text``.

    Shared by Qwen3OmniVLLMAdapter and VLLMTextAdapter via
    ``ThinkingMixin._split_thinking``.
    """
    if not full_text:
        return None, ""
    idx_start = full_text.find(think_start)
    if idx_start == -1:
        return None, full_text
    remainder = full_text[idx_start + len(think_start):]
    idx_end = remainder.find(think_end)
    if idx_end == -1:
        return remainder, ""
    trace = remainder[:idx_end]
    visible = remainder[idx_end + len(think_end):].lstrip("\n\r ")
    return trace, visible


class ThinkingMixin:
    """Shared thinking-mode wiring for vLLM-served adapters.

    Both audio (Qwen3OmniVLLMAdapter) and text (VLLMTextAdapter)
    inherit this. Subclasses whose chat template uses different
    boundary tokens (Gemma-4-thinking is the concrete case in-tree)
    override the two class constants.
    """

    # Boundary tokens for the model's chain-of-thought template.
    # Qwen 3 / Olmo 3.1 default. Gemma-4-thinking overrides in subclass
    # (verified against the model's chat template at implementation time).
    THINK_START: str = "<think>"
    THINK_END: str = "</think>"

    def _init_thinking(
        self, enable_thinking: bool, thinking_budget_enforce: int
    ) -> None:
        """Call from adapter ``__init__``. Establishes state fields."""
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget_enforce = int(thinking_budget_enforce)
        self._think_end_token_ids: list[int] | None = None

    def _resolve_think_end_tokens(self, tokenizer) -> None:
        """Call from adapter ``load()`` -- once the tokenizer/processor
        is available. Encodes THINK_END whenever thinking is either
        enabled (needed for the thoughts_tokens accounting done at
        generation time via _thoughts_tokens_from_ids) OR enforced
        (needed by the LogitsProcessor for s1-style hard-cap).
        """
        if self.enable_thinking or self.thinking_budget_enforce > 0:
            self._think_end_token_ids = resolve_end_token_ids(
                tokenizer, self.THINK_END,
            )

    def _thoughts_tokens_from_ids(self, token_ids) -> int | None:
        """Given the model's generated token IDs, find the first occurrence
        of the THINK_END marker subsequence and return its position — i.e.
        how many tokens the model spent on the <think>...</think> block
        (excludes the marker itself).

        Preserves the invariant thoughts_tokens + response_tokens ==
        completion_tokens (since we're slicing the actual generated stream
        rather than re-tokenizing decoded text — round-trip BPE could
        otherwise diverge by a few tokens).

        Returns None when: thinking wasn't enabled, no end-marker was
        resolved (adapter load() didn't run), or the marker never appears
        in the stream (model ran out of budget mid-think — the whole
        response is thoughts, no visible answer).
        """
        end = self._think_end_token_ids
        if not end or not token_ids:
            return None
        # Normalize to plain list of ints (vLLM sometimes hands back
        # tuples / numpy arrays / other sequences).
        token_ids = list(token_ids)
        if len(end) == 1:
            try:
                return token_ids.index(end[0])
            except ValueError:
                return None
        n = len(end)
        end_list = list(end)
        for i in range(len(token_ids) - n + 1):
            if token_ids[i:i + n] == end_list:
                return i
        return None

    def _thinking_llm_kwargs(self) -> dict:
        """Fragment to merge into ``LLM()`` kwargs so the processor
        class is registered at engine construction. Empty dict when
        enforcement is off (zero engine-level cost).
        """
        if self.thinking_budget_enforce > 0:
            return {"logits_processors": [ThinkingBudgetLogitsProcessor]}
        return {}

    def _thinking_extra_args(self) -> dict | None:
        """Return a ``SamplingParams.extra_args`` dict to activate the
        processor for a single request. None when enforcement is off
        or the end tokens haven't been resolved (load() not called yet).
        Callers merge the return value into ``sp_kwargs["extra_args"]``.
        """
        if (
            self.thinking_budget_enforce > 0
            and self._think_end_token_ids
        ):
            return {
                EXTRA_ARGS_KEY: {
                    "budget": self.thinking_budget_enforce,
                    "end_token_ids": self._think_end_token_ids,
                }
            }
        return None

    def _split_thinking(self, full_text: str) -> tuple[str | None, str]:
        """Adapter-facing wrapper around the module-level splitter."""
        return split_thinking(full_text, self.THINK_START, self.THINK_END)

    def _thinking_metadata(self) -> dict:
        """Fragment to merge into ``ModelOutput.metadata`` so downstream
        eval / notebook tooling can filter and label runs by their
        thinking-mode config.
        """
        return {
            "enable_thinking": self.enable_thinking,
            "thinking_budget_enforce": self.thinking_budget_enforce,
        }
