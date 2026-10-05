"""Shared chunking policy for long-audio inference.

Sub-1s trailing chunks are dropped (not fed to the model). Downstream
eval scripts fill the resulting <1s coverage gap with
``PRED_MISSING_LABEL`` via ``_preprocess_pred``, so the honest coverage
story surfaces in the ``pred_missing_rate`` metric.

Rationale: uniform ``chunk_samples``-sized inputs across all adapters
avoid audio-encoder minimum-input crashes (Music-Flamingo
``_build_audio_timestamps`` IndexError on 0-frame post_lengths,
Qwen-Omni ``audio (len=N) is too short to be represented inside the
model``, etc.). Prior ``merge_tail`` in ``describe.py`` extended the
last chunk's read range to absorb sub-1s tails — that fixed cascade
Stage A but never got ported to E2E ``chunk_runner``, and produced
adapter-specific edge cases (models had to tolerate up to
``chunk_seconds + 1s`` inputs). Drop-only replaces it: uniform
inputs, single policy across Stage A / E2E, coverage gap handled by
the existing eval-time gap-fill.

Threshold: 1s (= ``_MIN_INFERENCE_SAMPLES_S``). Above every audio
adapter's minimum encoder-input length (~250 ms floor). Change in
lockstep with the adapter family's floor.
"""

from __future__ import annotations

_MIN_INFERENCE_SAMPLES_S = 1.0


def compute_n_chunks(usable_frames: int, chunk_samples: int, sr: int) -> int:
    """Return the number of chunks to feed to the model.

    Contract:
      * Every chunk fed to the model has exactly ``chunk_samples`` samples,
        except the sole-chunk case where ``usable_frames < chunk_samples``
        (still guaranteed ``>= _MIN_INFERENCE_SAMPLES_S`` seconds).
      * A trailing remainder shorter than ``_MIN_INFERENCE_SAMPLES_S`` is
        dropped — no inference is run on it; eval fills the gap with
        ``PRED_MISSING_LABEL``.
      * Raises ``ValueError`` when the entire session is shorter than
        ``_MIN_INFERENCE_SAMPLES_S`` (the dataset filter's
        ``min_duration_s`` should prevent this from occurring).
    """
    n_full = usable_frames // chunk_samples
    remainder = usable_frames - n_full * chunk_samples
    min_inference_samples = int(round(_MIN_INFERENCE_SAMPLES_S * sr))
    if n_full == 0:
        if remainder < min_inference_samples:
            raise ValueError(
                f"usable_frames={usable_frames} shorter than min inference "
                f"length {min_inference_samples} samples "
                f"({_MIN_INFERENCE_SAMPLES_S}s at sr={sr}). Raise the "
                "dataset filter's min_duration_s to prevent this."
            )
        return 1
    if remainder >= min_inference_samples:
        return n_full + 1
    return n_full
