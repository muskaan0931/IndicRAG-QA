"""Regression tests for Bugs A, B, C, D, E, F."""
import conftest  # noqa: F401
from fastapi.testclient import TestClient

from app.main import app, engine, generate_answer
from app import config


def test_bug_a_attendance_query_consistency():
    """Bug A: Semantically equivalent questions must produce consistent grounded answers."""
    q1 = "What is the minimum attendance requirement?"
    q2 = "How much attendance does a student need to maintain to satisfy the university requirement?"
    
    with TestClient(app) as client:
        r1 = client.post("/ask", json={"query": q1, "method": "pretrained", "top_k": 5, "generator": "extractive"}).json()
        r2 = client.post("/ask", json={"query": q2, "method": "pretrained", "top_k": 5, "generator": "extractive"}).json()
    
    assert not r1["abstained"], f"Q1 abstained: {r1}"
    assert not r2["abstained"], f"Q2 abstained: {r2}"
    assert "75%" in r1["answer"], f"Expected 75% in Q1: {r1['answer']}"
    assert "75%" in r2["answer"], f"Expected 75% in Q2: {r2['answer']}"
    assert r1["cited_passages"] and r2["cited_passages"]


def test_bug_b_answerability_abstention_agreement():
    """Bug B: When generator abstains, answerability status must agree with abstention."""
    q_ood = "Who won the 2022 FIFA World Cup?"
    with TestClient(app) as client:
        r = client.post("/ask", json={"query": q_ood, "method": "pretrained", "top_k": 5, "generator": "extractive"}).json()
    
    assert r["abstained"]
    assert r["answer"] == "Insufficient information available in the provided documents."
    assert r["answerability"]["status"] == "unanswerable"
    assert not r["cited_passages"]


def test_bug_c_finetuned_checkpoint_graceful_unavailable():
    """Bug C: Fine-tuned E5 is excluded from production methods; requesting it via engine raises MethodUnavailable."""
    from app.main import MethodUnavailable
    import pytest
    av = engine.availability()
    assert set(av.keys()) == {"bm25", "pretrained", "hybrid"}
    
    with pytest.raises(MethodUnavailable, match="future research work"):
        engine.retrieve("attendance", "finetuned", 5)
    
    with TestClient(app) as client:
        # Pydantic schema validation rejects 'finetuned' as an invalid method in /compare
        r = client.post("/compare", json={"query": "attendance", "methods": ["bm25", "finetuned"]})
        assert r.status_code == 422


def test_bug_d_and_e_batch_runs_and_evaluation_diagnostics():
    """Bug D & E: /evaluation reports real batch run metrics and exact non-zero query counts."""
    with TestClient(app) as client:
        res = client.get("/evaluation").json()
    
    pilot = res["bm25_pilot"]
    summary = pilot["summary"]
    assert summary is not None
    assert summary["evaluated_queries"] > 0
    assert summary["excluded_queries"] > 0
    assert summary["total_queries_loaded"] == 93
    
    # Check per_method
    assert "bm25" in res["per_method"]
    assert "dense" in res["per_method"]
    assert "hybrid" in res["per_method"]
    assert "finetuned" not in res["per_method"]  # Fine-tuned honestly not run
    
    for m in ("bm25", "dense", "hybrid"):
        sm = res["per_method"][m]["summary"]
        assert sm["evaluated_queries"] > 0
        assert 0.0 < sm["mean_recall@5"] <= 1.0
        assert 0.0 < sm["mean_reciprocal_rank"] <= 1.0


def test_bug_f_honest_module_status():
    """Bug F: Module status strictly reports true state."""
    with TestClient(app) as client:
        res = client.get("/evaluation").json()
    
    mods = {m["id"]: m for m in res["modules"]}
    assert mods[1]["status"] == "measured"
    assert mods[2]["status"] == "measured"
    assert mods[3]["status"] == "measured"
