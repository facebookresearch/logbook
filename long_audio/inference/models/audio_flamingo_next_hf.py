"""Adapter for nvidia/audio-flamingo-next-hf.

The model is internally named ``MusicFlamingo`` in transformers — config
``model_type: musicflamingo``, architecture
``MusicFlamingoForConditionalGeneration``. Loading uses the transformers
built-in classes (no ``trust_remote_code`` needed; the classes are
registered in transformers ≥ 5.9).

CONSTRAINTS:
- Same audio-frontend story as AudioFlamingo-3-hf: the processor
  natively handles long audio via internal Whisper chunking; pass full
  10-min chunks.
- We use a free-text single-label prompt; the chunk runner runs the
  raw reply through ``json_repair``. If AF-Next handles JSON well we
  can add a structured path later. Non-JSON output yields zero segments
  (no chunk-wide-guess fallback).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from long_audio.inference.models.base import Decoder, ModelAdapter, ModelOutput


# HuggingFace repo ID — pass --model-id <local-path> to use a local symlink.
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-hf"

class AudioFlamingoNextHFAdapter(ModelAdapter):
    """nvidia/audio-flamingo-next-hf (internal name: MusicFlamingo).

    Args:
        model_id: HF repo id or local path.
        device: cuda device pin. Same caveat as AF3-hf — ``device_map=
            "auto"`` can produce cross-device tensor errors at generate
            time, so pin to one GPU.
    """

    name = "af-next"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            MusicFlamingoForConditionalGeneration,
            MusicFlamingoProcessor,
        )

        self._torch = torch
        self._processor = MusicFlamingoProcessor.from_pretrained(self.model_id)
        self._model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass

    def supports_structured(self) -> bool:
        # Same conservative default as AF3-hf; revisit if AF-Next proves
        # to honor JSON-schema prompts.
        return False

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelOutput:
        return self.generate_batch(
            [audio], sample_rate, [prompt],
            schema=schema, decoder=decoder,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )[0]

    def generate_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int,
        prompts: list[str],
        *,
        schema: dict | None = None,
        decoder: Decoder = "structured",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[ModelOutput]:
        if self._model is None:
            raise RuntimeError("Call load() first.")
        if len(audios) != len(prompts):
            raise ValueError(
                f"generate_batch: len(audios)={len(audios)} != len(prompts)={len(prompts)}"
            )
        prepared: list[np.ndarray] = []
        for a in audios:
            if a.ndim != 1:
                raise ValueError(f"Expected mono audio, got shape {a.shape}")
            prepared.append(a.astype(np.float32, copy=False))

        # Truncate short trailing sub-windows to avoid a CUDA IndexKernel
        # assert in MusicFlamingoForConditionalGeneration._build_audio_timestamps.
        # AF-Next's processor internally splits each audio into 30s Whisper
        # windows. When the trailing window carries fewer than ~30ms of
        # audio (e.g., a 60.006s chunk → windows [30s, 30s, 6ms]), the
        # Whisper feature extractor emits 1 mel frame for it, which
        # collapses to post_length=0 after conv+avg-pool. Then in
        # ``_build_audio_timestamps``:
        #     cumsum_post   = [0, 750, 1500]
        #     cumsum_samples = [1500]
        #     sample_indices = searchsorted(cumsum_samples, cumsum_post,
        #                                   right=True)
        #                    = [0, 0, 1]        # right=True → overshoot
        #     sample_start_rows[sample_indices]  # index 1 into size-1 tensor
        # triggers ``Assertion `-sizes[i] <= index && index < sizes[i]`
        # failed.`` (see IndexKernel.cu:111). Reproduced on egolife
        # A3_TASHA_DAY2_S03 (dur=4860.006s) chunk 8.
        #
        # Fix: if the trailing window would have less than ``_MIN_TAIL_S``
        # of real audio, drop it — truncate the input to the last clean
        # 30s boundary. Symmetric with the chunk_runner's sub-1s tail-drop
        # policy (``chunking.py:_MIN_INFERENCE_SAMPLES_S``); at most 999ms
        # of audio is lost per chunk, and the model output is free of
        # spurious silence-derived segments. Only applies when the audio
        # is > 1 full window; sole-window (audio ≤ 30s) case doesn't
        # trigger the searchsorted overshoot and is left alone.
        _WHISPER_WINDOW_S = 30.0
        _MIN_TAIL_S = 1.0
        window_samples = int(sample_rate * _WHISPER_WINDOW_S)
        min_tail_samples = int(sample_rate * _MIN_TAIL_S)
        for i, a in enumerate(prepared):
            tail = a.shape[0] % window_samples
            if 0 < tail < min_tail_samples and a.shape[0] > window_samples:
                prepared[i] = a[:-tail]

        # No system message — same rationale as audio_flamingo3_hf.py:
        # the stale "answer with one short label" was actively pushing the
        # model toward narration when the user prompt asks for JSON segments
        # (E2E) or a one-sentence caption (cascade A).
        texts: list[str] = []
        for p in prompts:
            messages = [
                {"role": "user",
                 "content": [
                     {"type": "audio", "audio": "placeholder"},
                     {"type": "text", "text": p},
                 ]},
            ]
            texts.append(self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        # IMPORTANT: do NOT pass a top-level `padding=...` here. The
        # MusicFlamingoProcessor's _defaults already do the right thing:
        #   text_kwargs  = {"padding": True}           # pad text to longest-in-batch
        #   audio_kwargs = {"padding": "max_length"}   # pad audio to 30s, set
        #                                                 input_features_mask to
        #                                                 reflect TRUE per-item
        #                                                 length. AF-Next's
        #                                                 MusicFlamingoRotaryEmbedding
        #                                                 computes absolute
        #                                                 timestamps from this
        #                                                 mask — getting it wrong
        #                                                 corrupts the per-window
        #                                                 time encoding.
        # Top-level `padding=True` overrides both and was the source of an
        # earlier crash (500 vs 1500 frame mismatch when batched 10s clips
        # weren't padded to the encoder's required 30s).
        inputs = self._processor(
            text=texts,
            audio=prepared,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device).to(self._model.dtype)

        # AF-Next's max_position_embeddings is 1200 (per HF config); rotary
        # positional embeddings index-out-of-bounds when input+generation
        # exceeds that. Cap max_new_tokens dynamically to fit. Observed
        # crash: egolife E2E job 130399 at uid A3_TASHA_DAY2_S03 —
        #   CUDA error: device-side assert triggered
        #   /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111 ...
        #     Assertion `-sizes[i] <= index && index < sizes[i]` failed.
        # Ego4D E2E happened to hit EOS before overflowing, but any
        # long-generation session (EgoLife's activity-verbose sessions,
        # any dataset's chatty output) can trigger it. 50-token safety
        # margin below the 1200 cap.
        _AF_NEXT_MAX_POS = 1200
        in_len = inputs["input_ids"].shape[1]
        safe_max_new = max(50, _AF_NEXT_MAX_POS - in_len - 50)
        max_new_tokens = min(max_new_tokens, safe_max_new)

        t0 = time.time()
        with self._torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-3),
            )
        wall = time.time() - t0
        per_call_latency = wall / max(1, len(audios))

        in_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[:, in_len:]
        raw_texts = self._processor.batch_decode(
            new_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: list[ModelOutput] = []
        for i, raw_text in enumerate(raw_texts):
            results.append(ModelOutput(
                raw_text=raw_text,
                raw_json=None,
                latency_s=per_call_latency,
                metadata={
                    "model_id": self.model_id,
                    "decoder": "freetext_simple",
                    "structured_supported": False,
                    "audio_seconds": audios[i].shape[0] / sample_rate,
                    "completion_tokens": int((new_ids[i] != self._processor.tokenizer.pad_token_id).sum()),
                    "batch_size": len(audios),
                    "batch_wall_s": wall,
                },
            ))
        return results
