# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""EgoLife shares Ego4D's ATUS-based prompt renderer.

EgoLife uses the same 6-class ATUS taxonomy as Ego4D (see ``schema.py``),
so it can use Ego4D's prompt template unchanged. This module just aliases
the Ego4D renderer under an EgoLife-namespaced name so callers importing
``render_egolife_prompt`` keep working.
"""

from __future__ import annotations

from long_audio.datasets.ego4d.prompt import render_ego4d_prompt as render_egolife_prompt

__all__ = ["render_egolife_prompt"]
