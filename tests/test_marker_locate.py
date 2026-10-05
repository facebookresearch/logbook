# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``_locate_marker_ends`` in ``long_audio.training.data``.

Guards the response-span locator used by both the E2E audio collator
(``Qwen3OmniSFTCollator``) and the cascade text collator
(``Stage2TextSFTCollator``). BPE-boundary correctness of the *marker-locate*
approach (versus a two-tokenization prefix-length approach) is the whole point
of the rewrite — see training code review, finding #2.
"""

from __future__ import annotations

import unittest

import torch

from long_audio.training.data import _locate_marker_ends


class LocateMarkerEndsTest(unittest.TestCase):
    def test_returns_index_just_past_last_occurrence(self):
        # marker = (7, 8, 9); two matches per row -> return the LAST.
        ids = torch.tensor([
            [1, 2, 7, 8, 9, 4, 5, 6, 0, 0],  # single match at 2..4 -> end=5
            [7, 8, 9, 1, 2, 7, 8, 9, 3, 4],  # two matches; last at 5..7 -> end=8
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],  # match at 6..8 -> end=9
        ])
        self.assertEqual(_locate_marker_ends(ids, (7, 8, 9)), [5, 8, 9])

    def test_returns_minus_one_when_marker_absent(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.assertEqual(_locate_marker_ends(ids, (7, 8, 9)), [-1])

    def test_returns_minus_one_when_marker_longer_than_row(self):
        ids = torch.tensor([[1, 2]])
        self.assertEqual(_locate_marker_ends(ids, (1, 2, 3)), [-1])

    def test_empty_marker_returns_minus_one(self):
        ids = torch.tensor([[1, 2, 3]])
        self.assertEqual(_locate_marker_ends(ids, ()), [-1])


if __name__ == "__main__":
    unittest.main()
