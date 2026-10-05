<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the CC-BY-NC-4.0 license found in the
LICENSE file in the root directory of this source tree.
-->

# Notebooks

Four analysis notebooks. All expect the `runs/` tree to be populated (see the top-level README for how to reproduce it) and consume the same directory pointed at by the `LONG_AUDIO_REPO` env var — the setup cell auto-detects the repo root either from the env var or from the notebook's working directory.

| Notebook | Produces | Input |
|----------|----------|-------|
| `paper_summary.ipynb` | Paper tables (II–IV) + Figures 2–3, comparing E2E vs cascade arms on Ego4D / EgoLife / SINS, plus the windowsize and thinking-budget ablations. | `runs/desc-v4-*` (main matrix), `runs/ablation-windowsize/`, `runs/thinking-*`, plus per-dataset `datasets/*/annotated_manifest.json`. |
| `ego_viz.ipynb` | Per-uid interactive viewer for Ego4D + EgoLife predictions. Timeline plots of each arm's `segments_abs`, side-by-side with GT. Dev/debug only. | `runs/desc-v4-*/{cascB_*,e2e_*}/{ego4d,egolife}/`. |
| `sins_viz.ipynb` | Per-arm SINS timeline viewer (single-file dataset). Same shape as `ego_viz` but for SINS. | `runs/desc-v4-*/{cascB_*,e2e_*}/sins/`. |
| `desc_viz.ipynb` | Per-uid description-side inspector: shows the `descriptions.jsonl` chunks vs. the description-quality eval (`eval_descriptions.py`) output. Dev/debug only. | `runs/desc-v4-*/cascA_*/{ego4d,egolife}/` + `runs/desc-v4-*/eval_descriptions/`. |

## Running

Notebooks assume Python 3.10+ with the `[gpu]` extras installed (see the top-level README). Recommended kernel: the same venv you used for inference and eval.

```bash
# From the repo root:
LONG_AUDIO_REPO=$(pwd) jupyter lab notebooks/
```

Or execute headlessly:

```bash
LONG_AUDIO_REPO=$(pwd) jupyter nbconvert --to notebook --execute --inplace \
    notebooks/paper_summary.ipynb --ExecutePreprocessor.timeout=600
```

## Output policy

- `paper_summary.ipynb` — outputs are committed (they are the published figures / tables); re-execute after any change to `runs/` or metrics to refresh them.
- `desc_viz.ipynb`, `ego_viz.ipynb`, `sins_viz.ipynb` — outputs are stripped by default. They are per-uid inspection tools; re-execute locally to view the current run.

## `notebooks/data/`

Untracked; populated on demand from `runs/` by the notebooks themselves. Safe to delete and regenerate.
