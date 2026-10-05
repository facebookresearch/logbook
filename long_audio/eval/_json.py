# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared JSON encoder for eval-script outputs.

Eval scripts emit dicts that may contain numpy arrays / scalars
(confusion matrices, per-class arrays). Standard ``json`` chokes on
those, so we register a small custom encoder in one place rather than
copy-pasting it into every eval CLI.
"""

from __future__ import annotations

import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy arrays / scalars as nested lists / native Python numbers."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)
