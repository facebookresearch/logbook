# Logbook: Extremely Long-form Audio Event Understanding

Slices each recording into fixed-minute chunks, prompts a model with a structured-output JSON
schema over a small activity-label vocabulary, stitches per-chunk
predictions at eval time.

Two inference paths:

- **E2E** — one audio LLM per chunk emits the full segment list
  directly (`scripts/run_e2e.py`).
- **Cascade** — Stage A: an audio LLM captions 10-second sub-chunks;
  Stage B: a text-only LLM consumes the caption stream and emits the
  segment list (`scripts/run_cascade.py`).

Datasets: **SINS** (DCASE 2017 apartment recordings), **Ego4D**
(per-video egocentric, two annotator passes), **EgoLife** (CVPR 2025,
6 co-living participants × 7 days).

## Installation

Linux + CUDA 12.8/12.9 (tested on H100). CPU-only is fine for data
prep.

```bash
git clone https://github.com/facebookresearch/logbook.git
cd logbook

# Pick one of the following extras depending on what you're doing:
pip install -e ".[gpu]"          # full vLLM 0.21 / torch / transformers 5.9 stack
pip install -e ".[cpu]"          # data-prep only, no GPU deps
pip install -e ".[train]"        # HF Trainer + peft + wandb for LoRA fine-tuning
pip install -e ".[enclap]"       # EnCLAP / MS-CLAP captioners (separate las-enclap venv)
```

The `[gpu]` and `[enclap]` extras alone don't pin CUDA-tagged wheels
(vLLM 0.21+cu129 and torch+cu129 live at
`https://wheels.vllm.ai/0.21.0/cu129` and
`https://download.pytorch.org/whl/cu129`, not PyPI). For an exact
reproduction of the H100 environments used in the paper, use the
bootstrap scripts:

```bash
bash scripts/install_las_all.sh     # transformers 5.9 + vLLM 0.21+cu129
bash scripts/install_las_enclap.sh  # transformers 4.29 for the CLAP arms
```

The two venvs are separate because EnCLAP + MS-CLAP depend on
transformers < 5.x internals (`_expand_mask`, the old `BartDecoder`
signature) that were removed in transformers 5.9.

## Storage layout

Datasets, model weights, and run outputs live outside the source tree
by convention. The repo ships empty `.gitkeep` placeholders that you
replace with symlinks to your actual storage:

```bash
rm datasets/SINS/.gitkeep    && ln -s /path/to/sins-storage      datasets/SINS
rm datasets/ego4d/.gitkeep   && ln -s /path/to/ego4d-storage     datasets/ego4d
rm datasets/egolife/.gitkeep && ln -s /path/to/egolife-storage   datasets/egolife
rm models/.gitkeep           && ln -s /path/to/model-weights     models
rm runs/.gitkeep             && ln -s /path/to/run-outputs       runs
```

Scripts resolve all paths repo-relative; CLI args (`--audio`,
`--manifest`, `--output-root`, `--model-id`) still override
per-invocation. Machine-local paths (EnCLAP checkpoints, LAION-CLAP
weights, the FLOPs runs root) are read from env vars —
`ENCLAP_REPO_DIR`, `ENCLAP_CKPT_DIR`, `LAION_CLAP_CKPT`, `RUNS_DIR`,
`FLOPS_OUT`. Set them in your shell profile or a `.env` file.

`audio_path` in every manifest ships as a POSIX repo-relative string
(e.g. `datasets/ego4d/audio/<uid>.flac`); dataset loaders resolve it
against the repo root at read time via
`long_audio.utils.paths.resolve_audio_path`.

## Repo layout

`long_audio/datasets/<DATASET>/` carries only the loading-side surface
(dataset loader + label schema + prompt). All data-prep code lives in
`scripts/data/<DATASET>/` as self-contained CLI stages so each prep
step can be developed, tested, and run independently.

```
long_audio/
  datasets/
    sins/           dataset.py, schema.py, prompt.py (loading only)
    ego4d/          dataset.py, schema.py, prompt.py, atus.json,
                    factscore/demons.json
    egolife/        dataset.py, schema.py, prompt.py — thin re-exports
                    over Ego4D's ATUS-6 taxonomy for cross-dataset parity
  inference/
    chunk_runner.py    chunk iter + raw per-chunk output
    chunking.py        chunk boundary math + sub-1 s tail drop
    prompt.py          structured / simple / lenient prompts
    schema.py          {segments:[{start,end,event}]} JSON-Schema generators
    lenient_parser.py  free-text fallback parser (json-repair backed)
    describe.py        Stage-A audio captioning driver (cascade)
    thinking_budget.py adapter-side reasoning-budget wiring
    models/
      qwen2_5_omni.py         Qwen2.5-Omni-7B via vLLM 0.21 V1
      qwen3_omni.py           Qwen3-Omni-30B-A3B-Instruct via vLLM 0.21 V0
      qwen3_omni_captioner.py Qwen3-Omni caption-only variant
      audio_flamingo3_hf.py   AudioFlamingo-3 via transformers
      audio_flamingo_next_hf.py           AudioFlamingo-Next via transformers
      audio_flamingo_next_captioner_hf.py caption-only variant
      gemini.py               Google Gemini via google-genai SDK
      openai_compat.py        Generic OpenAI-compat client (text-LLM Stage B)
      vllm_text.py            Local text-LLM adapters (Gemma-4, Qwen3-32B, ...)
      enclap.py               EnCLAP-large audio captioner
      msclap.py               MS-CLAP + GPT-2 caption decoder
      fake.py                 FakeAdapter for tests / smoke runs
  eval/
    metrics.py         boundary_f1, frame_level_accuracy, event_based_f1,
                       event_error_rate + summarize() + MetricConfig
    segmentation/      per-dataset shape adapters + aggregate.py scoring core
    description/       AFG + BART-MNLI entailment pipeline
  training/            LoRA fine-tuning scaffolding (E2E + cascade)
  utils/
    events.py          coalesce_runs / fill_gaps / resolve_overlaps
scripts/
  run_e2e.py                    E2E inference — see scripts/README.md
  run_cascade.py                Cascade inference
  eval_segmentation.py          Segmentation metrics
  eval_descriptions.py          Description-quality metrics (Ego4D + EgoLife)
  eval_finetuned.py             Segmentation eval for merged LoRA checkpoints
  human_baseline.py             Ego4D pass1↔pass2 human ceiling
  train_{e2e,cascade}.py        LoRA fine-tuning
  merge_lora.py                 Merge LoRA adapters into a base checkpoint
  compute_flops_desc_v4.py      Per-arm PFLOPs for the paper's cost column
  probe_af_and_clap_flops.py    calflops probes for AF / EnCLAP / MS-CLAP
  install_las_all.sh            Bootstrap the transformers-5.9 venv
  install_las_enclap.sh         Bootstrap the transformers-4.29 CLAP venv
  data/                         Per-dataset prep stages (see Datasets below)
notebooks/                      Analysis + paper figures (see notebooks/README.md)
tests/                          unittest suite
```

See [`scripts/README.md`](scripts/README.md) for concrete reproduction
recipes and [`notebooks/README.md`](notebooks/README.md) for what each
notebook produces.

## Datasets

### SINS

Build a single wall-clock-aligned `mono.flac` + `manifest.json` from a
SINS-format multi-node home recording (DCASE 2017 home-recording
corpus). Three self-contained per-stage scripts:

```bash
# Stage 1 — fetch SINS_database GitHub repo + per-node Zenodo zips
python scripts/data/sins/download.py --output-dir datasets/SINS

# Stage 2 — per-node extract → WAV concat → FLAC, with immediate WAV cleanup
python scripts/data/sins/build_flacs.py --output-dir datasets/SINS

# Stage 3 — clean annotations, stream per-node FLACs into one mono.flac
#          with 2 s linear crossfades, write final manifest.json
python scripts/data/sins/build_manifest.py --data-dir datasets/SINS
```

Stage 3 follows a *following-listener* policy: at each interval, pick
the node closest to the active room (kitchen mic for
cooking/eating/dishwashing, central mic for everything else in
living/hall, etc.), with a 2 s linear crossfade on every transition.
See `scripts/data/sins/build_manifest.py:clean_annotations` for the
sweep-line algorithm; `tests/test_mono_builder.py` for the algorithm
tests.

Outputs `datasets/SINS/mono.flac` + `datasets/SINS/manifest.json`:

```json
{
  "dataset": "SINS",
  "audio_path": ".../mono.flac",
  "sample_rate": 16000,
  "duration": 536026.1,
  "crossfade_s": 2.0,
  "annotations": [{"start": 0.0, "end": 100.0, "activity": "other"}, ...],
  "node_trace": [{"start": 0.0, "end": 100.0, "node": "Node1"}, ...]
}
```

### Ego4D

Four self-contained per-stage scripts. Stages 1-3 are CPU-only (Stage
1 is network-bound; Stages 2-3 are fast metadata work); Stage 4 is
GPU-bound (OLMo enrichment via vLLM).

```bash
# Stage 1 — pull per-university video shards via the ego4d SDK
python scripts/data/ego4d/download.py

# Stage 2 — raw mono 16 kHz FLAC per uid + audio_manifest.json
python scripts/data/ego4d/extract_audio.py

# Stage 3 — apply filters, compute cross-pass coverage intersection,
#          assign per-participant deterministic train/val/test splits
python scripts/data/ego4d/build_manifest.py

# Stage 4 — OLMo 3-stage enrichment per uid+pass+slice (clean → AFG →
#          classify). Writes annotated_manifest.json. GPU-bound.
python scripts/data/ego4d/annotate_manifest.py \
    --model-id /path/to/OLMo-2-1124-7B-SFT \
    --tensor-parallel-size 8
```

The OLMo classifier targets the 6-class ATUS taxonomy
(`long_audio/datasets/ego4d/atus.json`):

```
productive, food, leisure, purchasing, traveling, other
```

Per-uid `manifest.json` record (Stage 3 output):

```json
{
  "uid": "...",
  "audio_path": "datasets/ego4d/audio/<uid>.flac",
  "audio_offset_s": 5.0,
  "duration": 587.4,
  "sample_rate": 16000,
  "scenarios": [...],
  "fb_participant_id": "...",
  "video_source": "cmu",
  "split": "train",
  "passes": {
    "1": {"slices": [{"start_sec", "end_sec", "raw_summary", "annotation_uid"}, ...]},
    "2": {"slices": [...]}
  },
  "moments": [{"start", "end", "label"}, ...]
}
```

All `start_sec` / `end_sec` in `passes.*.slices` and `moments` live in
manifest-local `[0, duration]` coordinates. Inference + eval read
audio as
`soundfile.read(audio_path, start=int(audio_offset_s * sr), frames=int(duration * sr))`.

**Train/val/test splits** are baked into each per-uid record at Stage 3
via `sha256("{video_source}::{fb_participant_id}")[:8] % 100` → 0–59
train, 60–79 val, 80–99 test. Deterministic + stateless: a
participant's split survives filter changes as long as the participant
still survives. Filter by split via `Ego4DDataset(..., split="train")`.

**University filter**: Stage 3 restricts to four universities with
enough test-split audio for the paper's evaluation
(`frl_track_1_public`, `iiith`, `kaust`, `minnesota`); see
`scripts/data/ego4d/build_manifest.py:INCLUDED_UNIVERSITIES`.

### EgoLife

[EgoLife](https://arxiv.org/abs/2503.03803) (CVPR 2025,
`lmms-lab/EgoLife`): 6 participants co-living for 7 days, ~265 h of
Aria-glasses audio, Chinese DenseCaption narrations. Five
self-contained per-stage scripts. Stages 2 (CPU) and 3 (GPU) are
independent and run in parallel; Stage 4 (GPU) waits for both; Stage 5
(CPU) waits for 4.

```bash
# Stage 1 — hf snapshot_download lmms-lab/EgoLife
python scripts/data/egolife/download.py --output-dir datasets/egolife/raw

# Stage 2 — intersect audio-gap sessions with DenseCaption coverage,
#          per-session mono 16 kHz FLAC + audio_manifest.json
python scripts/data/egolife/extract_audio.py

# Stage 3 — NLLB-200-distilled-600M zho_Hans→eng_Latn on DenseCaption
#          SRTs → bilingual JSONL. Independent of Stage 2.
python scripts/data/egolife/translate_captions.py

# Stage 4 — 3-stage OLMo per 5-min tile: llm_clean → classify (ATUS-6)
#          → AFG. Coalesces adjacent same-label tiles → actions[].
python scripts/data/egolife/annotate_manifest.py \
    --audio-manifest datasets/egolife/audio/audio_manifest.json \
    --translated-root datasets/egolife/translated/DenseCaption \
    --output-dir datasets/egolife

# Stage 5 — assemble Ego4D-shape manifest.json (passes.1.actions[] +
#          passes.1.summaries[] with facts). CPU only.
python scripts/data/egolife/build_manifest.py
```

**Session definition = audio ∩ DenseCaption coverage.** Audio-gap
threshold 1 s (bimodal — nothing in [30 s, 60 min] band);
DenseCaption merge threshold 60 s (also bimodal). Every session has
continuous audio AND continuous narration by construction. Expected
outcome: ~265 h audio → ~193 h retained across ~300-500 sessions.

**Taxonomy reuses Ego4D's ATUS-6** (`long_audio/datasets/ego4d/atus.json`),
not EgoLife paper's 14-category labels — enables direct cross-dataset
comparison of segmentation numbers.

Per-uid `manifest.json` record (Stage 5 output):

```json
{
  "uid": "A1_JAKE_DAY1_S01",
  "audio_path": ".../egolife/audio/A1_JAKE/DAY1/S01.flac",
  "sample_rate": 16000, "duration": 5432.10,
  "audio_offset_s": 0.0,
  "wall_clock_start": "11:09:42.08",
  "wall_clock_end": "12:39:54.15",
  "scenarios": ["egolife_home"],
  "participant": "A1_JAKE", "day": "DAY1", "session_idx": 1,
  "clip_offsets": [{"filename": "...mp4", "wall_start_s": ..., "duration_s": 30.04, "audio_offset_s": 0.0}, ...],
  "passes": {
    "1": {
      "actions":   [{"start": 0.0, "end": 900.0, "event": "leisure"}, ...],
      "summaries": [{"start": 0.0, "end": 300.0,
                     "text_en": "the wearer picked up their phone...",
                     "text_zh": "我拿着手机...",
                     "facts": ["the wearer held a phone.", ...]}, ...]
    }
  },
  "moments": []
}
```

### Public manifests

Ego4D and EgoLife both ship a redacted `manifest.public.json` under
`datasets/<ds>/` that external users can consume directly for
segmentation evaluation without downloading the raw dataset annotations.
These files are version-controlled; the full annotated manifests
(`datasets/ego4d/annotated_manifest.json`,
`datasets/egolife/manifest.json`) are not — they contain content
derived from the upstream dataset annotations (Ego4D narration summaries
and moments; EgoLife DenseCaption Chinese text + our NLLB English
translations) which we cannot redistribute.

The public manifest contains:

- **Ego4D**: timing metadata, Ego4D-public identifiers
  (`fb_participant_id`, `video_source`), deterministic
  train/val/test `split`, narration slice coordinates (`start`, `end`,
  `annotation_uid`), and OLMo-derived segmentation (`passes.*.actions`).
- **EgoLife**: session metadata (`participant`, `day`, `session_idx`,
  `clip_offsets`, wall-clock), summary tile coordinates (`start`,
  `end`), and OLMo-derived segmentation (`passes.1.actions`).

What's dropped from the public artifact: Ego4D `raw_summary` / `summary`
/ per-slice `facts` / per-pass `facts[]` / `moments`; EgoLife
`text_en` / `text_zh` / `facts`.

Description rehydration recipes (run from your own Ego4D / EgoLife download):

```bash
# Ego4D
python scripts/data/ego4d/build_manifest.py \
    --public-manifest datasets/ego4d/manifest.public.json \
    --narration <your-ego4d>/v2/annotations/narration.json \
    --moments  <your-ego4d>/v2/annotations/moments_train.json \
               <your-ego4d>/v2/annotations/moments_val.json \
    --audio-root datasets/ego4d/audio \
    --output datasets/ego4d/manifest.json
python scripts/data/ego4d/annotate_manifest.py \
    --manifest datasets/ego4d/manifest.json \
    --output   datasets/ego4d/annotated_manifest.json \
    --model-id /path/to/OLMo-2-1124-7B-SFT

# EgoLife (requires Stage 3 `translate_captions.py` + Stage 4
# `annotate_manifest.py` on your own data first; --public-manifest
# then re-runs Stage 5 using the public uid set + actions)
python scripts/data/egolife/build_manifest.py \
    --public-manifest datasets/egolife/manifest.public.json \
    --audio-manifest  datasets/egolife/audio/audio_manifest.json \
    --translated-root datasets/egolife/translated/DenseCaption \
    --activity-labels datasets/egolife/activity_labels.json \
    --output          datasets/egolife/manifest.json
```

Regenerate the public manifests after any pipeline change:

```bash
python scripts/data/ego4d/redact_manifest.py
python scripts/data/egolife/redact_manifest.py
```

## Inference

E2E — one audio LLM per chunk:

```bash
python scripts/run_e2e.py {qwen2.5-omni|qwen3-omni|af3-hf|af-next|gemini|fake} \
    --dataset {sins|ego4d|egolife} {--smoke|--full} \
    [--chunk-min 10] [--decoder {structured|lenient|freeform}] \
    [--time-unit {second|minute}] \
    [--model-id <hf-repo-or-local-path>] \
    [--max-new-tokens N] [--temperature T]
```

Cascade — audio-LM captioner (Stage A) + text-LLM segmenter (Stage
B):

```bash
python scripts/run_cascade.py \
    --dataset {sins|ego4d|egolife} --eval-set --full \
    --describe-model {af3-hf|af-next|qwen3-omni-captioner|enclap|msclap-cap|...} \
    --text-model {gemma4-31b|qwen3-32b|qwen2.5-72b|llama3.3-70b|olmo3.1-32b|k2-v2|gemini|...} \
    [--skip-describe|--skip-segment] \
    [--window-min 10] [--time-unit minute] [--decoder freeform]
```

Per-chunk model output lands at
`<runs>/<UTC>/<model_name>/(<dataset>/<uid>/)?chunk_NNNNN.json` — raw,
unstitched. The `summary.json` carries run metadata + per-chunk index;
the eval scripts stitch at scoring time so you can re-eval with
different gap-fill / coalesce policies without re-running inference.

Adapter notes:

- **Qwen2.5-Omni-7B** — vLLM 0.21 V1, `Qwen/Qwen2.5-Omni-7B`, ~17 GB
  on a single H100, supports JSON-schema-guided decoding via xgrammar.
- **Qwen3-Omni-30B-A3B-Instruct** — vLLM 0.21 V0 (adapter sets
  `VLLM_USE_V1=0` at load time per the Qwen3-Omni cookbook workaround),
  ~59 GB weights + ~12 GB KV cache on a single 80 GB H100.
- **AudioFlamingo-3-hf** — `nvidia/audio-flamingo-3-hf` via
  transformers; returns placeholder text under structured prompts, so
  the adapter uses the simple single-label prompt and relies on the
  chunk runner's lenient parser.
- **AudioFlamingo-Next-hf** — `nvidia/audio-flamingo-next-hf`, same
  single-label-per-chunk pattern.
- **Gemini** — `gemini-2.5-pro` via google-genai SDK; requires
  `GOOGLE_API_KEY`. JSON-schema-guided via `response_schema` +
  `response_mime_type="application/json"`.
- **Text-LLM Stage B** — local: `gemma-4-31B`, `qwen3-32B`,
  `qwen2.5-72B`, `llama-3.3-70B`, `olmo-3.1-32B`, `k2-v2` via vLLM.
  Remote: any OpenAI-compatible endpoint via `openai_compat.py`.
- **fake** — `FakeAdapter` for tests / smoke runs without a GPU or
  remote API.

By default, structured decoding aborts with an error if the adapter
doesn't support it. Pass `--allow-decoder-fallback` to silently fall
back to lenient parsing instead.

## Evaluation

The eval scripts read raw per-chunk predictions (concatenated into one
stream) and call the four-metric suite via `metrics.summarize`. The
pipeline does gap-fill + coalesce internally inside `_preprocess_pred`
— there is no top-level `--gap-fill-label` flag.

Single unified CLI with `--dataset` dispatch. For Ego4D and EgoLife,
`eval_segmentation.py` works directly against the shipped
`manifest.public.json` — no rehydration needed. Description-quality eval
(`eval_descriptions.py`) requires the rehydrated + annotated manifest
(see **Public manifests** above) because it needs the licensed
narration / caption-derived `facts`.

```bash
# Ego4D (per-video, 2 annotator passes → 2 datapoints/uid):
python scripts/eval_segmentation.py --dataset ego4d   <run_dir> [--manifest datasets/ego4d/manifest.public.json]

# EgoLife (per-video, 1 pass → 1 datapoint/uid):
python scripts/eval_segmentation.py --dataset egolife <run_dir> [--manifest datasets/egolife/manifest.public.json]

# SINS (single-file dataset, whole timeline):
python scripts/eval_segmentation.py --dataset sins    <run_dir> [--manifest datasets/SINS/manifest.json]
```

Per-video datasets (Ego4D, EgoLife) write
`<run_dir>/eval_segmentation.json` with per-video macro + corpus-pooled
micro metrics, plus per-data-point detail. SINS writes scores into
`<run_dir>/summary.json["eval_scores"]`.

The generic scoring core lives in
`long_audio.eval.segmentation.aggregate`; per-dataset shape (manifest
layout, uid extraction, `MetricConfig`) is provided by adapters under
`long_audio.eval.segmentation.datasets.{ego4d,egolife,sins}`. Same core
is reused by `scripts/eval_finetuned.py` (fine-tuned checkpoint eval)
and `long_audio.training.evaluate` (in-training val eval).

Description-quality eval for Ego4D + EgoLife (SINS has no description
ground truth):

```bash
python scripts/eval_descriptions.py eval \
    --dataset {ego4d|egolife} \
    --cells-config cells.json \
    --manifest datasets/<ds>/annotated_manifest.json \
    --output-root runs/<RUN_TS>/eval_descriptions/<ds>
```

## Metric definitions

Four metrics computed by `long_audio.eval.metrics.summarize`
(parameterized by `MetricConfig`):

- **boundary_f1** — precision/recall/F1 over event-endpoint sets with
  a configurable timing tolerance (Ego4D: 150 s = 2.5 min, half the
  5-min annotation quantization). Lenient bilateral within-tolerance
  match. Adjacent same-label runs are coalesced on both sides before
  deriving boundaries.
- **frame_level_accuracy** — per-frame label match at 1 s
  rasterization. Reports overall accuracy, per-class precision/recall,
  macro recall (catches majority-class collapse), and a sentinel column
  for missing-pred frames.
- **event_based_f1** — per-class greedy IoU matching with `IoU >= 0.5`
  default. Same coalescing on both sides. `other` is excluded from
  event matching by default.
- **event_error_rate** — ASR-style Levenshtein edit distance over the
  dedupe'd label sequences, decomposed into substitutions / deletions
  / insertions; `ER = (S + D + I) / max(N, 1)`.

Symmetric human ceiling (`scripts/human_baseline.py`) computes the
same four metrics treating each Ego4D narration pass in turn as ground
truth, then combines the two directions (per-video average for macro;
pool tp/fp/fn for micro). Diagnostic invariant: pooled `micro_precision
== micro_recall == micro_f1` because pred boundaries in one direction
are GT boundaries in the other.

## Reproducing the published results

See [`scripts/README.md`](scripts/README.md) for end-to-end
reproduction recipes (E2E arms, cascade arms, SFT training, ablations,
FLOPs). Paper figures and tables are assembled in
`notebooks/paper_summary.ipynb` — see
[`notebooks/README.md`](notebooks/README.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the process and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community standards.

Tests (unittest, runs in ~2 s):

```bash
python -m unittest discover tests
```

Baseline: 155 tests passing (heavy-GPU tests requiring `vllm` or
`torch.cuda` skip cleanly on CPU-only installs).

Linting / formatting: `ruff check` + `ruff format` (or `black` +
`flake8` if you prefer). PR conventions: any change to
`clean_annotations`, `single_stream_from_events`, or any metric needs
a paired test.

## License

CC BY-NC 4.0 — see [LICENSE](./LICENSE). Non-commercial use only;
commercial licensing not offered.

## Citation

```bibtex
@misc{longaudio2026,
  title  = {Long-Form Egocentric Audio Activity Segmentation with Audio LLMs},
  author = {TBD},
  year   = {2026}
}
```

An arXiv link will be added on release.

## Acknowledgments

- **SINS** — Dekkers et al., DCASE 2017 home-recording corpus.
- **Ego4D** — Grauman et al., CVPR 2022, Ego4D consortium.
- **EgoLife** — Yang et al., CVPR 2025, S-Lab NTU. Dataset +
  DenseCaption via `lmms-lab/EgoLife`.
- **NLLB** — Costa-jussà et al., 2022.
  `facebook/nllb-200-distilled-600M` for CN→EN translation.
- **Qwen2.5-Omni / Qwen3-Omni** — Alibaba Cloud / Qwen team.
- **AudioFlamingo-3 / -Next** — NVIDIA.
- **OLMo-2-1124-7B-SFT** — Allen Institute for AI (AI2).
- **EnCLAP** — Kim et al., ICASSP 2024
  (`github.com/jaeyeonkim99/EnCLAP`).
- **MS-CLAP** — Elizalde et al., ICASSP 2023.
- **OpenFActScore** — atomic-fact generation demonstrations
  (`long_audio/eval/description/afg.py`).
- **DCASE Sound Event Detection** working group — boundary F1 /
  event-based F1 / event error rate metric lineage.
- **vLLM** — Kwon et al., serving stack.
