"""Model adapters for chunk-level audio LLM inference."""

from long_audio.inference.models.base import ModelAdapter, ModelOutput
from long_audio.inference.models.fake import FakeAdapter

__all__ = ["ModelAdapter", "ModelOutput", "FakeAdapter"]
