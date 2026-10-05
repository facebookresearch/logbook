# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Generation-based Ego4D eval for the fine-tuned model.

Runs the fine-tuned model over an Ego4D split through the SAME inference path as
the baselines (``chunk_runner.run_inference``), stitches the per-chunk
predictions, and scores them with the shared aggregation
(``long_audio.eval.ego4d_aggregate``) so the output matches
``scripts/eval_run_ego4d.py`` byte-for-byte.

Two consumers:
  * :func:`run_ego4d_eval` — the standalone driver (used by
    ``scripts/eval_finetuned.py`` and importable for ad-hoc eval).
  * :class:`Ego4DEvalCallback` — a HF ``TrainerCallback`` that runs the
    four-metric eval on a val subset during training and logs the headline
    numbers (which flow to ``report_to`` = wandb / tensorboard).

The driver is model-agnostic: it takes any
:class:`~long_audio.inference.models.base.ModelAdapter`, so it is exercised
end-to-end on CPU in tests with ``FakeAdapter`` (only the real Qwen3-Omni
``generate`` is GPU-only).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from long_audio.datasets.ego4d.dataset import Ego4DDataset
from long_audio.datasets.ego4d.schema import ATUS_HINTS, ATUS_LABELS
from long_audio.eval.segmentation import (
    aggregate,
    get_adapter,
    headline_metrics,
    score_video,
    stitch_chunk_predictions,
)
from long_audio.inference.chunk_runner import run_inference

_EGO4D_CFG = get_adapter("ego4d").metric_config()


def run_ego4d_eval(
    adapter: Any,
    manifest_path: str | Path,
    split: str | None,
    *,
    chunk_minutes: float = 10.0,
    time_unit: str = "minute",
    with_description: bool = True,
    max_videos: int | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    model_name: str = "qwen3-omni-ft",
    min_events: int = 1,
    min_duration_s: float = 0.0,
    require_both_passes: bool = True,
    pass_id: str = "1",
    work_dir: str | Path | None = None,
) -> dict:
    """Run inference over an Ego4D split and return the ``eval_aggregate`` dict.

    ``adapter`` is any ``ModelAdapter``; the decoder is ``"structured"`` when the
    adapter supports it, else ``"freeform"`` (the fine-tuned Qwen path). Each
    video is chunked with the same ``audio_offset_s`` seek + ``audio_window_s`` =
    manifest ``duration`` as the baseline runner. ``max_videos`` caps the set
    (use a small value for in-training eval). ``work_dir`` defaults to a temp dir
    that is removed afterwards.
    """
    ds = Ego4DDataset(
        manifest_path=manifest_path,
        split=split,
        min_events=min_events,
        min_duration_s=min_duration_s,
        require_both_passes=require_both_passes,
        pass_filter=pass_id,
    )
    videos = ds.videos if max_videos is None else ds.videos[:max_videos]
    decoder = "structured" if adapter.supports_structured() else "freeform"
    adapter.load()

    cleanup = work_dir is None
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ego4d_eval_"))
    per_datapoint: list[dict] = []
    try:
        for v in videos:
            uid = v["uid"]
            uid_dir = work / uid
            summary = run_inference(
                v["audio_path"],
                adapter,
                uid_dir,
                ATUS_LABELS,
                label_hints=ATUS_HINTS,
                chunk_minutes=chunk_minutes,
                decoder=decoder,
                time_unit=time_unit,
                with_description=with_description,
                dataset_name="ego4d",
                audio_offset_s=float(v["audio_offset_s"]),
                audio_window_s=float(v["duration"]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                write_raw_text=False,
            )
            preds = stitch_chunk_predictions(uid_dir)
            per_datapoint.extend(
                score_video(
                    uid=uid,
                    predictions=preds,
                    video=v,
                    audio_duration_s=float(summary.audio_duration_s),
                    n_chunks=int(summary.n_chunks),
                    config=_EGO4D_CFG,
                )
            )
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
    return aggregate(
        per_datapoint, model_name,
        boundary_tolerance_s=_EGO4D_CFG.boundary_tolerance_s,
    )


try:  # TrainerCallback is only needed when transformers is installed.
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - keeps the module importable w/o transformers
    _TrainerCallback = object


class Ego4DEvalCallback(_TrainerCallback):
    """Run the four-metric Ego4D eval on a val subset during training.

    Piggybacks on ``on_evaluate`` (fires with ``eval_strategy="steps"`` /
    ``"epoch"``). Generates predictions with the live model via ``adapter``,
    scores them, and logs ``{log_prefix}{metric}`` for the four headline metrics
    (boundary_f1, frame_accuracy, event_f1, event_error_rate). Logging goes
    through the bound Trainer so it reaches ``report_to`` (wandb / tensorboard)
    and ``state.log_history``.

    Generation uses the same in-memory model the Trainer is optimizing, so the
    callback toggles ``eval()`` + ``use_cache=True`` around the run and restores
    training state afterwards.
    """

    def __init__(
        self,
        adapter: Any,
        manifest_path: str | Path,
        split: str,
        *,
        chunk_minutes: float = 10.0,
        time_unit: str = "minute",
        with_description: bool = True,
        max_videos: int | None = 8,
        max_new_tokens: int = 2048,
        model_name: str = "qwen3-omni-ft",
        log_prefix: str = "eval_seg/",
        **eval_kwargs: Any,
    ):
        self.adapter = adapter
        self.manifest_path = manifest_path
        self.split = split
        self.chunk_minutes = chunk_minutes
        self.time_unit = time_unit
        self.with_description = with_description
        self.max_videos = max_videos
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.log_prefix = log_prefix
        self.eval_kwargs = eval_kwargs
        self._trainer = None

    def bind_trainer(self, trainer: Any) -> None:
        """Give the callback the Trainer so it can ``trainer.log(...)``."""
        self._trainer = trainer

    def compute(self) -> dict:
        """Run the eval and return the flat headline-metric dict (no logging)."""
        agg = run_ego4d_eval(
            self.adapter,
            self.manifest_path,
            self.split,
            chunk_minutes=self.chunk_minutes,
            time_unit=self.time_unit,
            with_description=self.with_description,
            max_videos=self.max_videos,
            max_new_tokens=self.max_new_tokens,
            model_name=self.model_name,
            **self.eval_kwargs,
        )
        return {f"{self.log_prefix}{k}": v for k, v in headline_metrics(agg).items()}

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D102 - HF hook
        # NB: ``kwargs["model"]`` is the PEFT-wrapped thinker Trainer optimizes;
        # it does NOT have a ``.thinker`` attribute (that's on the top-level
        # ``Qwen3OmniMoeForConditionalGeneration``). Reach the actual top via
        # the adapter — same object graph as the one ``self.compute()``
        # generates through, so toggling its config actually affects
        # generation. See training code review, finding #4.
        top = getattr(self.adapter, "model", None)
        thinker = getattr(top, "thinker", top) if top is not None else None
        prev_training = getattr(top, "training", None)
        prev_cache = getattr(getattr(thinker, "config", None), "use_cache", None)
        try:
            if top is not None and hasattr(top, "eval"):
                top.eval()
            if thinker is not None and hasattr(thinker, "config"):
                thinker.config.use_cache = True
            metrics = self.compute()
        finally:
            if thinker is not None and hasattr(thinker, "config") and prev_cache is not None:
                thinker.config.use_cache = prev_cache
            # Restore training mode unless we know the model was previously in
            # eval mode. ``prev_training is not False`` covers both True (was
            # training) and None (``.training`` attr missing on some peft
            # versions) — safer default is to leave it in training mode rather
            # than silently leave it in eval. See training code review #12.
            if top is not None and prev_training is not False and hasattr(top, "train"):
                top.train()

        if self._trainer is not None:
            self._trainer.log(metrics)
        else:
            print(f"[ego4d-eval] {metrics}", flush=True)
        # Surface into the metrics dict Trainer may be assembling.
        if isinstance(kwargs.get("metrics"), dict):
            kwargs["metrics"].update(metrics)
        return control
