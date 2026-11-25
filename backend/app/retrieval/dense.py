from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Ensure Hugging Face runs offline from local cache to avoid network timeouts on sandboxed/offline machines
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

RESULT_FIELDS = ["chunk_id", "doc_id", "document_name", "category", "version", "page", "text"]


class DenseRetriever:
    """Multilingual SentenceTransformer retriever with persisted, L2-normalised NumPy embeddings.

    E5 usage convention (required by the model card): queries are prefixed with "query: " and
    passages with "passage: ". Embeddings are L2-normalised, so the dot product equals cosine similarity.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.index_model: str | None = None

    # ---- indexing -----------------------------------------------------------------------
    def build(self, chunks: list[dict[str, Any]], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        if not chunks:
            raise ValueError("Cannot embed an empty corpus.")
        self.chunks = chunks
        texts = ["passage: " + c.get("text", "") for c in chunks]
        self.embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        self.index_model = self.model_name
        return self.embeddings

    def save(self, directory: str | Path) -> None:
        if self.embeddings is None:
            raise RuntimeError("Build embeddings before saving.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "dense_embeddings.npy", self.embeddings)
        with open(directory / "dense_chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (directory / "dense_model.txt").write_text(self.model_name, encoding="utf-8")

    def load(self, directory: str | Path, expected_chunk_ids: list[str] | None = None) -> None:
        """Load a persisted index and verify it matches the corpus (count, IDs and order)."""
        directory = Path(directory)
        emb_path = directory / "dense_embeddings.npy"
        chunk_path = directory / "dense_chunks.jsonl"
        if not emb_path.exists() or not chunk_path.exists():
            raise FileNotFoundError(
                f"Dense index not found in '{directory}'. Expected dense_embeddings.npy and dense_chunks.jsonl."
            )
        embeddings = np.load(emb_path).astype("float32")
        with open(chunk_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        if embeddings.ndim != 2 or len(chunks) != len(embeddings):
            raise ValueError(f"Embedding/chunk count mismatch ({len(embeddings)} vs {len(chunks)}).")
        if expected_chunk_ids is not None and [c["chunk_id"] for c in chunks] != list(expected_chunk_ids):
            raise ValueError(
                "Dense index was built from a different corpus than data/chunks.jsonl "
                "(chunk IDs or order differ). Rebuild it with backend/build_dense_index.py."
            )
        model_file = directory / "dense_model.txt"
        self.index_model = model_file.read_text(encoding="utf-8").strip() if model_file.exists() else None
        self.chunks, self.embeddings = chunks, embeddings

    # ---- retrieval ----------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.embeddings is None:
            raise RuntimeError("No index loaded. Call build() or load().")
        q = self.model.encode(
            ["query: " + query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")[0]
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension {q.shape[0]} != index dimension {self.embeddings.shape[1]}: "
                "the index was built with a different model."
            )
        scores = self.embeddings @ q
        order = np.argsort(-scores)[: max(1, min(top_k, len(scores)))]
        return [
            {"rank": i + 1, "score": round(float(scores[idx]), 6),
             **{k: self.chunks[idx].get(k) for k in RESULT_FIELDS}}
            for i, idx in enumerate(order)
        ]
