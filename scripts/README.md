# Scripts

End-to-end recipes for reproducing the paper's numbers using plain
`python scripts/<name>.py` invocations — no Slurm, no cluster-specific
wrappers required.

## Layout

- **`run_e2e.py`** — end-to-end audio-LM inference. One entry point
  for `qwen2.5-omni` / `qwen3-omni` / `af3-hf` / `af-next` / `gemini`
  / `fake`.
- **`run_cascade.py`** — cascaded inference. Stage A: audio-LM
  captioner (`af3-hf`, `af-next`, `af-next-captioner`, `qwen2.5-omni`,
  `qwen3-omni`, `qwen3-omni-captioner`, `enclap`, `msclap-cap`). Stage
  B: text-only LLM (public: `gpt-4o` / `claude-*` via OpenAI-compat;
  local: `Gemma-4-31B`, `Qwen3-32B`, `Qwen2.5-72B`, `Llama-3.3-70B`,
  `Olmo-3.1-32B`, `K2-V2` via vLLM).
- **`eval_segmentation.py`** — segmentation eval (bF1, evER, evF1,
  fAcc) — one CLI, `--dataset {sins,ego4d,egolife}`.
- **`eval_descriptions.py`** — description-quality eval (OLMo AFG →
  BART-MNLI entailment → summary R/P + moments R + self-consistency).
- **`eval_finetuned.py`** — same segmentation eval, but reloads a
  merged LoRA adapter first (Table IV FT rows).
- **`human_baseline.py`** — pass-1 vs pass-2 human ceiling (Ego4D
  only; single-annotator datasets don't have one).
- **`train_e2e.py`** / **`train_cascade.py`** — LoRA fine-tuning for
  the Table IV FT rows.
- **`merge_lora.py`** — merge trained LoRA weights back into a base
  checkpoint for inference.
- **`compute_flops_desc_v4.py`** / **`probe_af_and_clap_flops.py`** —
  measure per-arm PFLOPs used in Table IV.
- **`install_las_all.sh`** / **`install_las_enclap.sh`** — env
  bootstrap for the two venvs (`las-all` = transformers 5.9 + vLLM
  0.21+cu129; `las-enclap` = transformers 4.29 for the CLAP arms).
- **`data/`** — per-dataset preparation stages. See the top-level
  README for the stage-by-stage layout of each.

## Reproducing the paper

All runs land under `runs/<RUN_TS>/` and are consumed by the eval
scripts. `RUN_TS` is any string you like; the paper used
`desc-v4-20260715T160821Z`.

Prereqs: complete the data-prep stages from the top-level README
(sections **SINS**, **Ego4D**, **EgoLife**), and export the
machine-local env vars listed in the top-level README's *Storage
layout* section.

For segmentation-only reproduction, the shipped `manifest.public.json`
under `datasets/{ego4d,egolife}/` is sufficient — `eval_segmentation.py`
defaults to it. Description-quality eval (`eval_descriptions.py`) needs
the rehydrated + annotated manifest; see the top-level README's
*Public manifests* subsection for the `build_manifest.py
--public-manifest` + `annotate_manifest.py` recipe.

### 1. Main results — Tables II and III

**E2E** (single audio-LM per chunk):

```bash
for MODEL in qwen2.5-omni qwen3-omni af3-hf af-next; do
  for DATASET in sins ego4d egolife; do
    python scripts/run_e2e.py $MODEL \
      --dataset $DATASET --eval-set --full \
      --time-unit minute --chunk-min 10 --decoder freeform \
      --output-root runs/desc-v4/e2e_${MODEL}
  done
done

# Gemini (needs GOOGLE_API_KEY):
for DATASET in sins ego4d egolife; do
  python scripts/run_e2e.py gemini \
    --dataset $DATASET --eval-set --full \
    --time-unit minute --chunk-min 10 --decoder freeform \
    --output-root runs/desc-v4/e2e_gemini
done
```

**Cascade** — Stage A (per-arm audio captions) followed by Stage B
(text-LLM segmentation over the captions):

```bash
# Stage A: audio-LM captioner over 10-s sub-chunks
for ARM in af3-hf af-next qwen3-omni-captioner; do
  for DATASET in sins ego4d egolife; do
    python scripts/run_cascade.py \
      --dataset $DATASET --eval-set --full \
      --describe-model $ARM --skip-segment \
      --describe-chunk-s 10 \
      --output-root runs/desc-v4/cascA_${ARM}
  done
done

# Stage B: text-LLM reads Stage-A descriptions.jsonl and emits segments
for TEXT_MODEL in gemma4-31b qwen3-32b qwen2.5-72b llama3.3-70b olmo3.1-32b k2-v2; do
  for ARM in af3-hf af-next qwen3-omni-captioner; do
    for DATASET in sins ego4d egolife; do
      python scripts/run_cascade.py \
        --dataset $DATASET --eval-set --full \
        --describe-model $ARM --text-model $TEXT_MODEL \
        --skip-describe \
        --window-min 10 --time-unit minute --decoder freeform \
        --output-root runs/desc-v4/cascB_${TEXT_MODEL}
    done
  done
done
```

### 2. Segmentation + description eval

```bash
# Segmentation metrics (bF1, evER, evF1, fAcc):
for CELL in runs/desc-v4/*/{sins,ego4d,egolife}; do
  DS=$(basename $CELL)
  python scripts/eval_segmentation.py --dataset $DS $CELL
done

# Description-quality metrics (Ego4D + EgoLife only; SINS has no
# description ground truth):
for CELL in runs/desc-v4/*/{ego4d,egolife}; do
  DS=$(basename $CELL)
  python scripts/eval_descriptions.py eval \
    --dataset $DS \
    --cells-config cells.json \
    --output-root runs/desc-v4/eval_descriptions/$DS/$(dirname $CELL)
done
```

### 3. Human baseline (Ego4D only)

```bash
python scripts/human_baseline.py \
  --dataset ego4d --output-root runs/desc-v4/human_baseline
```

### 4. Fine-tuning + Table IV SFT rows

```bash
# Train (LoRA on cascade Stage B text-LM)
python scripts/train_cascade.py \
  --model gemma4-31b \
  --cascA_root runs/desc-v4/cascA_af3-hf \
  --output-dir runs/ft/cascB_gemma4-31b_ft

# Merge LoRA back for inference:
python scripts/merge_lora.py \
  --adapter runs/ft/cascB_gemma4-31b_ft \
  --base google/gemma-4-31B-it \
  --out runs/ft/cascB_gemma4-31b_ft/merged

# Rerun cascade with merged FT model:
python scripts/run_cascade.py \
  --dataset ego4d --eval-set --full \
  --describe-model af3-hf --text-model gemma4-31b \
  --text-model-path runs/ft/cascB_gemma4-31b_ft/merged \
  --skip-describe \
  --output-root runs/desc-v4/cascB_gemma4-31b_ft

# Re-eval:
python scripts/eval_finetuned.py \
  --dataset ego4d runs/desc-v4/cascB_gemma4-31b_ft/cascA_af3-hf/ego4d
```

E2E SFT (Qwen3-Omni) uses `scripts/train_e2e.py` with the same
merge → rerun → re-eval flow; see the script's `--help`.

### 5. Ablations

**Windowsize** (Figure 2) — same E2E command as (1) but sweep
`--chunk-min`:

```bash
for LEN in 1 2 5 10 15 30 60; do
  python scripts/run_e2e.py qwen3-omni \
    --dataset sins --eval-set --full \
    --time-unit minute --chunk-min $LEN --decoder freeform \
    --output-root runs/ablation-windowsize/${LEN}min/e2e_qwen3-omni
done
```

**Thinking-budget** (Figure 3) — for models that expose an explicit
reasoning budget, pass `--thinking-budget-tokens N`. See
`long_audio/inference/thinking_budget.py` for supported adapters.

### 6. FLOPs (Table IV cost column)

```bash
python scripts/compute_flops_desc_v4.py \
  --runs-root runs/desc-v4 \
  --out runs/desc-v4/flops.json
```

### 7. Assemble figures + tables

```bash
LONG_AUDIO_REPO=$(pwd) jupyter nbconvert --to notebook --execute --inplace \
  notebooks/paper_summary.ipynb --ExecutePreprocessor.timeout=600
```

See `notebooks/README.md` for what each notebook produces.

## Troubleshooting

- **CUDA wheel mismatch** (cu13 vs cu129): the H100 driver only loads
  CUDA 12.x. Use `scripts/install_las_all.sh`, not a bare
  `pip install -e ".[gpu]"`.
- **EnCLAP / MS-CLAP arms** live in a separate `las-enclap` venv
  (transformers 4.29). Bootstrap with `scripts/install_las_enclap.sh`
  and point `ENCLAP_REPO_DIR` / `ENCLAP_CKPT_DIR` / `LAION_CLAP_CKPT`
  at your local weight copies (see `long_audio/inference/models/enclap.py`).
- **Qwen3-Omni-30B-A3B** hits a vLLM V1 KV-cache bug at BS > 64 with
  verbose outputs — drop to BS=64 (instruct) / BS=32 (captioner) if
  you see preemption.
- **Gemini** — requires `GOOGLE_API_KEY`. Uses `gemini-2.5-pro` by
  default; override with `--model-id`.
