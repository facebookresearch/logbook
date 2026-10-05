#!/usr/bin/env bash
# Canonical install of the captioner-only las-enclap venv.
#
# Covers cascade Stage A captioner arms:
#   - enclap     : LAION-CLAP + EnCodec + BART (github.com/jaeyeonkim99/EnCLAP)
#   - msclap-cap : MS-CLAP + GPT-2 prefix decoder (PyPI: msclap)
#
# These cannot live in las-all because both wrap models that depend on
# transformers < 5.x internals:
#   - EnCLAP's enclap_bart.py imports _expand_mask + calls
#     BartDecoder(config, embed_tokens) — both removed in transformers 5.
#   - msclap calls tokenizer.encode_plus(pad_to_max_length=True) — same.
# Las-all is pinned to transformers 5.9 for vLLM / Qwen-Omni, so we
# isolate the captioners here on transformers 4.36.
#
# Usage:  bash scripts/install_las_enclap.sh [<venv_path>]
# Default venv_path: envs/las-enclap
#
# Run from the repo root with a Python 3.10 interpreter available
# (this script bootstraps a fresh venv).

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-envs/las-enclap}"
SEED_PY="${SEED_PY:-python3.10}"
PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

echo "==> creating venv at ${VENV}"
"${SEED_PY}" -m venv "${VENV}" --prompt "$(basename "${VENV}")"
"${PIP}" install -U pip setuptools wheel

echo "==> installing the long-audio package itself (no deps)"
"${PIP}" install -e . --no-deps

echo "==> installing torch + transformers 4.x stack with cu129 wheels"
"${PIP}" install \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.11.0+cu129" "torchaudio==2.11.0+cu129" "torchvision==0.26.0+cu129" \
    "transformers==4.29.0" "tokenizers<0.14" \
    accelerate "huggingface-hub<1.0" hf_transfer librosa soundfile \
    numpy scipy tqdm natsort requests rapidfuzz tenacity json-repair

echo "==> layering EnCLAP + msclap with --no-deps + their runtime deps"
"${PIP}" install --no-deps laion-clap encodec msclap
"${PIP}" install einops julius ftfy regex progressbar wget braceexpand \
    webdataset h5py torchlibrosa pandas

echo "==> sanity import"
"${PY}" -c "
import os, sys
enclap_dir = os.environ.get('ENCLAP_REPO_DIR')
if enclap_dir:
    sys.path.insert(0, enclap_dir)
import torch, transformers
print(f'torch {torch.__version__} | transformers {transformers.__version__}')
if enclap_dir:
    from inference import EnClap
    print('EnCLAP import OK')
else:
    print('WARN: ENCLAP_REPO_DIR not set; skipped EnCLAP import check.')
from msclap import CLAP
print('MS-CLAP import OK')
"

cat <<EOF

las-enclap bootstrapped at ${VENV}.

Required external artifacts (point env vars at your local copies):
  - EnCLAP repo clone         -> \$ENCLAP_REPO_DIR
    (clone from github.com/jaeyeonkim99/EnCLAP)
  - EnCLAP-BART checkpoint    -> \$ENCLAP_CKPT_DIR
    (contains config.json + pytorch_model.bin; see
     long_audio/inference/models/enclap.py for the drive.google URL)
  - LAION-CLAP checkpoint     -> \$LAION_CLAP_CKPT
    (630k-audioset-fusion-best.pt from huggingface.co/lukewys/laion_clap)
  - MS-CLAP clapcap weights   -> auto-downloaded by msclap on first
                                CLAP(version='clapcap') call
EOF
