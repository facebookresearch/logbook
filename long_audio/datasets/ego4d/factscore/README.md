# Vendored FActScore demos

## Contents

- `demons.json` — 21 hand-annotated `(biography sentence → list of atomic facts)`
  pairs used as the BM25 retrieval pool for atomic-fact-generation (AFG)
  few-shot prompting.

## Provenance

The file originates from the FActScore project (Min et al., 2023):
https://github.com/shmsw25/FActScore — distributed via a Google Drive
zip (gdrive id `1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt`) referenced from
their `factscore/download_data.py`.

We vendor it here unchanged so that `scripts/data/decompose_ego4d_summaries.py`
runs without an external download step. Vendoring is permitted by the
project's MIT license.

The same `demons.json` is re-shipped by the OpenFActScore project
(Lage & Couto, 2025; https://github.com/lflage/OpenFActScore), which
also uses the identical MIT copyright notice attributing Sewon Min.

## License

See [`LICENSE-third-party/FActScore-MIT`](../../../LICENSE-third-party/FActScore-MIT)
at the repo root. The MIT notice is reproduced verbatim from the
upstream `LICENSE` file.
