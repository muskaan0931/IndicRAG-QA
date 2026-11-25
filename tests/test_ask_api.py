import json

import conftest  # noqa: F401
from fastapi.testclient import TestClient

from app import answerability, config
from app.main import app

Q_ANS = "What is the minimum attendance required to appear in the end semester examination?"
Q_UNANS = "What is the average placement salary package at Bennett University?"


def test_generators_endpoint(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(app) as c:
        g = c.get("/generators").json()
        assert g["default"] == "extractive" and not g["generators"]["llm"]["available"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert c.get("/generators").json()["default"] == "llm"


def test_ask_returns_cited_extractive_answer_without_calibration(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(answerability, "load_thresholds", lambda *a, **k: {})
    with TestClient(app) as c:
        r = c.post("/ask", json={"query": Q_ANS, "method": "bm25", "top_k": 5}).json()
    assert r["answer_mode"] == "extractive" and not r["abstained"] and "75%" in r["answer"]
    assert r["cited_passages"] and all(1 <= n <= len(r["evidence"]) for n in r["cited_passages"])
    assert r["answerability"]["status"] == "not_calibrated"       # honest: no threshold fitted yet


def test_ask_abstains_below_calibrated_threshold(monkeypatch):
    monkeypatch.setattr(answerability, "load_thresholds", lambda *a, **k: {"bm25": {"threshold": 1000.0, "fit_n": 5}})
    with TestClient(app) as c:
        r = c.post("/ask", json={"query": Q_UNANS, "method": "bm25"}).json()
    assert r["abstained"] and r["answer"] == "Insufficient information available in the provided documents."
    assert r["answer_mode"] == "abstained" and r["generator"] == "answerability-gate" and r["answerability"]["status"] == "unanswerable"
    assert r["evidence"]                                           # evidence is still shown for inspection


def test_ask_extractive_abstains_when_evidence_is_off_topic(monkeypatch):
    monkeypatch.setattr(answerability, "load_thresholds", lambda *a, **k: {})
    with TestClient(app) as c:
        r = c.post("/ask", json={"query": "zebra giraffe safari", "method": "bm25"}).json()
    assert r["abstained"] and r["answer"].startswith("Insufficient")


def test_ask_with_llm_stub_and_error_isolation(monkeypatch):
    from app import generation, main
    monkeypatch.setattr(answerability, "load_thresholds", lambda *a, **k: {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(generation.LLMGenerator, "_http_post",
                        lambda self, p: {"content": [{"type": "text", "text": "The minimum attendance is 75% [1]."}]})
    with TestClient(app) as c:
        r = c.post("/ask", json={"query": Q_ANS, "method": "bm25", "generator": "llm"}).json()
        assert r["answer_mode"] == "llm" and r["cited_passages"] == [1] and not r["answer_warnings"]

        def boom(self, p):
            raise RuntimeError("network down")
        monkeypatch.setattr(generation.LLMGenerator, "_http_post", boom)
        r = c.post("/ask", json={"query": Q_ANS, "method": "bm25", "generator": "llm"}).json()
        assert r["answer"] is None and "network down" in r["generation_error"] and r["evidence"]


def test_evaluation_endpoint_reports_status(monkeypatch):
    with TestClient(app) as c:
        mods = {m["id"]: m for m in c.get("/evaluation").json()["modules"]}
    assert mods[5]["status"] in ("ready_not_run", "measured") and mods[4]["status"] in ("ready_not_run", "measured")
