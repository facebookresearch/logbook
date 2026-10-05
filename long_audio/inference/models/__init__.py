# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Model adapters for chunk-level audio LLM inference."""

from long_audio.inference.models.base import ModelAdapter, ModelOutput
from long_audio.inference.models.fake import FakeAdapter

__all__ = ["ModelAdapter", "ModelOutput", "FakeAdapter"]
