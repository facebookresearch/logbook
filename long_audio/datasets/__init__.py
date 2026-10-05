"""Dataset abstractions for long-audio understanding.

Two datasets are supported:
  - SINS — stationary multi-mic home recording (test benchmark)
  - Ego4D — egocentric daily activity videos

Each lives in its own subpackage with dataset class, label set, prompt
helpers, and data-prep utilities. The generic ``BaseDataset`` interface
is here at the top.
"""

from long_audio.datasets.base import BaseDataset

__all__ = ["BaseDataset"]
