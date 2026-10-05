#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC-4.0 license found in the
# LICENSE file in the root directory of this source tree.

# Canonical install of the consolidated `las-all` env — covers every
# E2E + Cascade Stage A model family on the H100 cluster:
#
#   vLLM    — Qwen2.5-Omni, Qwen3-Omni, Qwen3-Omni-Captioner
#   HF      — AudioFlamingo-3
#
# NOTE: cascade Stage A captioner arms `enclap` and `msclap-cap` live in
# a SEPARATE las-enclap venv (transformers 4.x), bootstrapped via
# scripts/install_las_enclap.sh. They can't share las-all because
# their wrapped libraries depend on transformers < 5 internals that
# vLLM/Qwen-Omni require us to be on transformers 5.9 to avoid.
#
# Why this isn't just `pip install -e ".[gpu,clap]"`:
#   * vllm 0.21.0 needs the cu129-tagged wheel
#     (https://wheels.vllm.ai/0.21.0/cu129); PyPI's default is cu13 which
#     fails at import on the cluster's CUDA 12.9 driver
#     (libcudart.so.13: not found).
#   * torch 2.11.0 needs the cu129 wheel from PyTorch's index; default
#     PyPI torch is also cu13.
#   * EnCLAP + MS-CLAP captioners require transformers 4.x internals
#     (BartDecoder embed_tokens arg, tokenizer.encode_plus); they live
#     in the separate las-enclap venv built by install_las_enclap.sh
#     so they never collide with this env's transformers 5.9 pin.
#
# Usage:  bash scripts/install_las_all.sh [<venv_path>]
# Default venv_path: envs/las-all
#
# Run from the repo root with an existing Python 3.10 interpreter
# available (this script bootstraps a fresh venv).

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-envs/las-all}"
SEED_PY="${SEED_PY:-python3.10}"
PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

echo "==> creating venv at ${VENV}"
"${SEED_PY}" -m venv "${VENV}" --prompt "$(basename "${VENV}")"
"${PIP}" install -U pip setuptools wheel

echo "==> installing the long-audio package itself (no deps)"
"${PIP}" install -e . --no-deps

echo "==> installing torch + vllm with cu129 wheels"
"${PIP}" install \
    --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.11.0+cu129" "torchaudio==2.11.0+cu129" "torchvision==0.26.0+cu129" \
    "vllm==0.21.0" "transformers==5.9.0" \
    accelerate "huggingface-hub[cli]" hf_transfer pillow av librosa \
    google-genai soundfile numpy scipy tqdm natsort requests \
    "json-repair>=0.50" "rapidfuzz>=3" "tenacity>=8"

echo "==> sanity import"
"${PY}" -c "
import torch, vllm, transformers
print(f'torch {torch.__version__} | vllm {vllm.__version__} | transformers {transformers.__version__}')
print('all imports ok')
"

cat <<EOF

las-all bootstrapped at ${VENV}.

Verify a GPU forward pass with:
  ${PY} scripts/run_e2e.py qwen3-omni --dataset sins --smoke --max-chunks 1 \\
      --time-unit minute --chunk-min 10 --decoder freeform --output-root /tmp

The 13-arm smoke matrix is at scripts/smoke_all_arms.sh.
EOF
