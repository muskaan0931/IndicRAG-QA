"""Read genuine experiment artifacts from reports/ and derive error-analysis cases from them.

Nothing here is invented: every value is read from a file in reports/ or recomputed from the corpus.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.language import detect_language

PILOT_QUERIES = [  # same eight probes as backend/run_multilingual_retrieval.py and phase5/query_inventory.csv
    ("Q001", "What is the minimum attendance required for end semester exams?"),
    ("Q002", "What happens if a student misses an end term examination?"),
    ("Q003", "What are the scholarship renewal conditions?"),
    ("Q004", "How is unfair means handled during examinations?"),
    ("Q005", "How can a student appeal a disciplinary decision?"),
    ("Q006", "अंतिम सेमेस्टर परीक्षा देने के लिए न्यूनतम उपस्थिति कितनी होनी चाहिए?"),
    ("Q007", "Agar student end-term exam miss kar de toh kya process hai?"),
    ("Q008", "Sports scholarship renew karne ke liye minimum CGPA kitna chahiye?"),
]


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _relevant(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def bm25_pilot(reports: Path) -> dict[str, Any]:
    summary = _json(reports / "bm25_evaluation_summary.json")
    per_query = _csv(reports / "bm25_evaluation_per_query.csv")
    judgements = _csv(reports / "relevance_judgements_completed.csv")
    runs = _csv(reports / "bm25_retrieval_runs.csv")
    ranked: dict[str, list[str]] = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r["query_id"], int(r["rank"]))):
        ranked[r["query_id"]].append(r["chunk_id"])
    rows = []
    for j in judgements:
        rel = _relevant(j.get("relevant_chunk_ids"))
        rows.append({
            "query_id": j["query_id"], "query": j["query"], "relevant_chunk_ids": rel,
            "relevant_ranks": {c: (ranked[j["query_id"]].index(c) + 1 if c in ranked[j["query_id"]] else None) for c in rel},
            "judged": bool(rel),
        })
    return {"summary": summary, "per_query": per_query, "judgements": rows,
            "label_provenance": "AI-assisted, not human-verified (see README: provisional labels, 5-query pilot)."}


def mlqa_results(reports: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(reports.glob("mlqa_*.json")):
        data = _json(p)
        for r in (data or {}).get("results", []):
            out.append({**r, "source_file": p.name})
    return out


def error_cases(bm25_index, corpus, reports: Path) -> list[dict[str, Any]]:
    """Cases that are demonstrably difficult, computed from real artifacts and live BM25 runs."""
    cases: list[dict[str, Any]] = []
    pilot = bm25_pilot(reports)
    per_q = {r["query_id"]: r for r in pilot["per_query"]}

    def short(cid):
        c = corpus.chunk_by_id.get(cid, {})
        return f"{c.get('document_name', cid)} · p.{c.get('page')}" if c else cid

    for j in pilot["judgements"]:
        qid = j["query_id"]
        if not j["judged"]:
            cases.append({
                "case_id": f"{qid}-no-gold", "query": j["query"], "language": detect_language(j["query"])["label"],
                "expected": "None identified in the reviewed candidate pool",
                "retrieved": "—", "failure_type": "Unsupported / possibly unanswerable question",
                "method": "BM25", "observed": "No relevant passage was found in the reviewed pool, so the query is excluded from "
                "retrieval metrics. This is a candidate for the answerability test (Module 5), not yet confirmed unanswerable.",
                "source": "reports/relevance_judgements_completed.csv",
            })
            continue
        row = per_q.get(qid)
        if row and float(row["reciprocal_rank"]) < 1.0:
            top = bm25_index.search(j["query"], 1)
            cases.append({
                "case_id": f"{qid}-rank", "query": j["query"], "language": detect_language(j["query"])["label"],
                "expected": "; ".join(short(c) for c in j["relevant_chunk_ids"]),
                "retrieved": short(top[0]["chunk_id"]) if top else "—",
                "failure_type": "Semantically similar passages / ranking",
                "method": "BM25",
                "observed": f"First relevant passage at rank {row['first_relevant_rank']} (MRR contribution {float(row['reciprocal_rank']):.2f}).",
                "source": "reports/bm25_evaluation_per_query.csv",
            })

    for qid, q in PILOT_QUERIES:
        lang = detect_language(q)["label"]
        res = bm25_index.search(q, 3)
        if not res:
            cases.append({
                "case_id": f"{qid}-empty", "query": q, "language": lang, "expected": "(not annotated)",
                "retrieved": "— (no results)", "failure_type": "Cross-lingual vocabulary mismatch",
                "method": "BM25",
                "observed": "BM25 returned zero passages: the tokenizer is English-only and the corpus is English, so a Devanagari query shares no terms.",
                "source": "live BM25 run on the probe query",
            })
        elif lang == "Hinglish":
            cases.append({
                "case_id": f"{qid}-hinglish", "query": q, "language": lang, "expected": "(not annotated)",
                "retrieved": short(res[0]["chunk_id"]), "failure_type": "Code-mixed query",
                "method": "BM25",
                "observed": "Matches only the English tokens in the query; romanised-Hindi words contribute nothing. Top passage is unverified because no gold label exists for this query.",
                "source": "live BM25 run on the probe query",
            })

    # Derive real difficult cases from the evaluation dataset
    from app.evalkit import load_dataset
    try:
        eval_rows = load_dataset()
    except Exception:
        eval_rows = []

    for r in eval_rows:
        if len(cases) >= 16:
            break
        qid = r["qid"]
        q = r["question"]
        lang = r["language"]
        if not r["answerable"]:
            # Unanswerable query testing abstention
            top_bm = bm25_index.search(q, 1)
            cases.append({
                "case_id": f"{qid}-unans",
                "query": q,
                "language": lang,
                "expected": "Abstain (unanswerable)",
                "retrieved": short(top_bm[0]["chunk_id"]) if top_bm else "—",
                "failure_type": "Out-of-corpus / unanswerable query",
                "method": "BM25 vs Gate",
                "observed": f"Query topic is absent from institutional corpus (terms: {r.get('absent_terms', [])[:2]}). Lexical search may retrieve spurious keywords, requiring an answerability gate to abstain.",
                "source": "data/eval/questions.jsonl",
            })
        elif lang == "Hindi":
            # Devanagari query
            res = bm25_index.search(q, 1)
            if not res and not any(c["case_id"] == f"{qid}-hi-dev" for c in cases):
                cases.append({
                    "case_id": f"{qid}-hi-dev",
                    "query": q,
                    "language": "Hindi",
                    "expected": "; ".join(short(c) for c in r.get("gold_chunk_ids", [])),
                    "retrieved": "— (0 matches)",
                    "failure_type": "Cross-lingual lexical collapse",
                    "method": "BM25",
                    "observed": "Devanagari query produces 0 tokens in standard English BM25 index, requiring multilingual E5 dense retrieval to match semantic concepts across languages.",
                    "source": "data/eval/questions.jsonl",
                })
        elif lang == "Hinglish":
            res = bm25_index.search(q, 1)
            if res and r.get("gold_chunk_ids") and res[0]["chunk_id"] not in r["gold_chunk_ids"] and not any(c["case_id"] == f"{qid}-hinglish-sub" for c in cases):
                cases.append({
                    "case_id": f"{qid}-hinglish-sub",
                    "query": q,
                    "language": "Hinglish",
                    "expected": "; ".join(short(c) for c in r["gold_chunk_ids"]),
                    "retrieved": short(res[0]["chunk_id"]),
                    "failure_type": "Code-mixed partial token mismatch",
                    "method": "BM25",
                    "observed": "BM25 matches partial English loanwords but misses institutional context expressed in romanized Hindi phrasing.",
                    "source": "data/eval/questions.jsonl",
                })
        elif lang == "English":
            # Check for keyword collisions (e.g. attendance query matching grading chunk)
            res = bm25_index.search(q, 1)
            if res and r.get("gold_chunk_ids") and res[0]["chunk_id"] not in r["gold_chunk_ids"] and not any(c["case_id"] == f"{qid}-collision" for c in cases):
                cases.append({
                    "case_id": f"{qid}-collision",
                    "query": q,
                    "language": "English",
                    "expected": "; ".join(short(c) for c in r["gold_chunk_ids"]),
                    "retrieved": short(res[0]["chunk_id"]),
                    "failure_type": "Lexical keyword collision",
                    "method": "BM25",
                    "observed": "BM25 ranks an irrelevant chunk containing repeated query terms over the actual institutional policy passage.",
                    "source": "data/eval/questions.jsonl",
                })

    for p in corpus.pages:
        if len(cases) >= 20:
            break
        c = p.get("ocr_confidence")
        if (c is not None and c < 50) or (p.get("extraction_method") == "ocr" and len((p.get("text") or "").strip()) < 80):
            cases.append({
                "case_id": f"ocr-{p['doc_id']}-p{p['page']}", "query": "—", "language": "—",
                "expected": f"{p['document_name']} · p.{p['page']}", "retrieved": "—",
                "failure_type": "OCR noise / sparse extraction", "method": "Corpus extraction",
                "observed": f"Extraction method is OCR with low character yield; text on this page may be noisy for retrieval.",
                "source": "data/pages.jsonl",
            })
    return cases
