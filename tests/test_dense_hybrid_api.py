"""Plumbing tests that need no model download: a deterministic fake encoder stands in for E5."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import conftest  # noqa: F401,E402  (selects the real corpus or the synthetic fixture)
from app import config  # noqa: E402

from app.language import detect_language  # noqa: E402
from app.retrieval.bm25 import BM25Index, load_chunks  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402


class FakeModel:
    """Hashes tokens into a 64-d bag-of-words vector, L2-normalised."""
    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), 64), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, hash(tok) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


def make_dense(chunks):
    d = DenseRetriever.__new__(DenseRetriever)
    d.model, d.model_name, d.chunks, d.embeddings, d.index_model = FakeModel(), "fake", [], None, None
    d.build(chunks, show_progress=False)
    return d


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(config.CHUNKS_PATH)


def test_dense_search_is_cosine_ranked(chunks):
    d = make_dense(chunks)
    res = d.search("minimum attendance end semester examination", 5)
    assert [r["rank"] for r in res] == [1, 2, 3, 4, 5]
    assert all(res[i]["score"] >= res[i + 1]["score"] for i in range(4))
    assert np.allclose(np.linalg.norm(d.embeddings, axis=1), 1.0, atol=1e-4)


def test_index_roundtrip_and_corpus_mismatch_detected(chunks, tmp_path):
    d = make_dense(chunks)
    d.save(tmp_path)
    d2 = make_dense(chunks[:3])
    d2.load(tmp_path, expected_chunk_ids=[c["chunk_id"] for c in chunks])
    assert len(d2.chunks) == len(chunks)
    with pytest.raises(ValueError):
        d2.load(tmp_path, expected_chunk_ids=[c["chunk_id"] for c in chunks][::-1])


def test_rrf_dedupes_and_keeps_metadata(chunks):
    bm = BM25Index(chunks).search("scholarship renewal", 10)
    de = make_dense(chunks).search("scholarship renewal", 10)
    fused = reciprocal_rank_fusion(bm, de, top_k=5)
    ids = [f["chunk_id"] for f in fused]
    assert len(ids) == len(set(ids)) == 5
    assert all(f["document_name"] and f["page"] for f in fused)
    assert [f["rank"] for f in fused] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("q,label", [
    ("What is the minimum attendance required for end semester exams?", "English"),
    ("अंतिम सेमेस्टर परीक्षा देने के लिए न्यूनतम उपस्थिति कितनी होनी चाहिए?", "Hindi"),
    ("Agar student end-term exam miss kar de toh kya process hai?", "Hinglish"),
    ("Sports scholarship renew karne ke liye minimum CGPA kitna chahiye?", "Hinglish"),
    ("scholarship के लिए minimum eligibility क्या है?", "Code-mixed"),
])
def test_language_heuristic(q, label):
    assert detect_language(q)["label"] == label


def test_api_contract():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/ask", json={"query": "minimum attendance", "method": "bm25", "top_k": 3}).json()
        r = c.post("/ask", json={"query": "minimum attendance", "method": "bm25", "top_k": 3, "generator": "none"}).json()
        assert r["answer_mode"] == "evidence_only" and r["answer"] is None
        assert r["answerability"]["status"] in ("not_calibrated", "answerable", "unanswerable")
        assert len(r["evidence"]) == len(r["citations"]) == 3
        for e in r["evidence"]:
            for k in ("rank", "score", "chunk_id", "doc_id", "document_name", "page", "version", "category", "text"):
                assert k in e
        assert c.post("/retrieve", json={"query": "  ", "method": "bm25"}).status_code == 422
        assert c.post("/retrieve", json={"query": "x", "method": "bogus"}).status_code == 422
        assert c.get("/documents/nope").status_code == 404
        cmp = c.post("/compare", json={"query": "scholarship renewal", "top_k": 3}).json()
        assert cmp["methods"]["bm25"]["status"] == "ok"
