"""Adapter for ``nvidia/audio-flamingo-next-captioner-hf``.

Same MusicFlamingo architecture as ``nvidia/audio-flamingo-next-hf``;
only the HF model_id differs. Subclass the base adapter; override the
default model_id and ``name`` slug.

Intended slot: cascade Stage A (per-clip captioning). The captioner
variant was post-trained for short-clip dense audio captioning.
"""

from __future__ import annotations

from long_audio.inference.models.audio_flamingo_next_hf import (
    AudioFlamingoNextHFAdapter,
)


DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-captioner-hf"


class AudioFlamingoNextCaptionerHFAdapter(AudioFlamingoNextHFAdapter):
    """nvidia/audio-flamingo-next-captioner-hf via transformers.

    Same constructor surface as :class:`AudioFlamingoNextHFAdapter`;
    only the default ``model_id`` differs.
    """

    name = "af-next-captioner"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
