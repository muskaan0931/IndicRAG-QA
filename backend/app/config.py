"""Central configuration. Every path can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path, otherwise the first candidate (used for error messages)."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DATA_DIR = Path(os.environ.get("INDICRAG_DATA_DIR", ROOT / "data"))
REPORTS_DIR = Path(os.environ.get("INDICRAG_REPORTS_DIR", ROOT / "reports"))
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
PAGES_PATH = DATA_DIR / "pages.jsonl"

PRETRAINED_MODEL_NAME = os.environ.get("PRETRAINED_MODEL_NAME", "intfloat/multilingual-e5-base")

# Where the pretrained-E5 passage embeddings (dense_embeddings.npy + dense_chunks.jsonl) live.
PRETRAINED_INDEX_DIR = (
    Path(os.environ["PRETRAINED_INDEX_DIR"])
    if os.environ.get("PRETRAINED_INDEX_DIR")
    else _first_existing([
        ROOT / "phase4_outputs" / "dense_index",
        ROOT / "models" / "dense_index",
        ROOT / "dense_index",
    ])
)

# Fine-tuned model directory (a SentenceTransformer folder) and its persisted index.
FINETUNED_MODEL_DIR = (
    Path(os.environ["FINETUNED_MODEL_DIR"])
    if os.environ.get("FINETUNED_MODEL_DIR")
    else _first_existing([
        ROOT / "models" / "finetuned_e5",
        ROOT / "models" / "indicrag-e5-finetuned",
    ])
)
FINETUNED_INDEX_DIR = Path(
    os.environ.get("FINETUNED_INDEX_DIR", ROOT / "models" / "finetuned_e5_index")
)

CORS_ORIGIN_REGEX = os.environ.get(
    "CORS_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
)
