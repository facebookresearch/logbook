"""Description-quality eval: AFG on predicted `description` fields, then
NLI-based recall/precision against per-pass GT summary facts + moments
(Ego4D only).

Modules:
  - afg: atomic-fact generation for a sentence (BM25 top-1 demo + OLMo,
    exact behavior mirrors annotate_manifest.py which imports from here).
  - nli: BART-MNLI wrapper, batched entail-probability scoring.
  - metrics: overlap-filtered candidate-pool enumeration + micro-pool
    aggregation for Summary Recall / Summary Precision / Moments Recall.
  - pipeline: AFG → NLI orchestration; boundary types (GroundTruth,
    PredictedSegment) so dataset-shape-specific I/O stays in the CLI.
"""

from long_audio.eval.description.afg import (
    AFG_INSTRUCT,
    BM25DemoRetriever,
    afg_messages,
    load_demons,
    parse_facts,
)
from long_audio.eval.description.metrics import (
    Atom,
    DirectionResult,
    MicroAccumulator,
    P_ENTAIL_THRESHOLD,
    _build_pair_batch,
    overlaps,
    score_direction,
)
from long_audio.eval.description.nli import DEFAULT_MODEL_ID as NLI_MODEL_ID
from long_audio.eval.description.nli import BartMNLI
from long_audio.eval.description.pipeline import (
    AFGConfig,
    GroundTruth,
    PredictedSegment,
    atoms_from_moments,
    atoms_from_pass,
    evaluate_cell_uid,
    evaluate_human_baseline,
    pred_atoms_from_facts,
    run_afg,
)
from long_audio.eval.description.self_consistency import (
    DEFAULT_ENTAIL_THRESHOLD,
    ListResult,
    UidResult,
    evaluate_self_consistency_uid,
    micro_percents,
    score_facts_list,
)

__all__ = [
    "AFGConfig",
    "AFG_INSTRUCT",
    "Atom",
    "BM25DemoRetriever",
    "BartMNLI",
    "DEFAULT_ENTAIL_THRESHOLD",
    "DirectionResult",
    "GroundTruth",
    "ListResult",
    "MicroAccumulator",
    "NLI_MODEL_ID",
    "P_ENTAIL_THRESHOLD",
    "PredictedSegment",
    "UidResult",
    "_build_pair_batch",
    "afg_messages",
    "atoms_from_moments",
    "atoms_from_pass",
    "evaluate_cell_uid",
    "evaluate_human_baseline",
    "evaluate_self_consistency_uid",
    "load_demons",
    "micro_percents",
    "overlaps",
    "parse_facts",
    "pred_atoms_from_facts",
    "run_afg",
    "score_direction",
    "score_facts_list",
]
