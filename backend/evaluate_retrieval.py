#!/usr/bin/env python3
"""Batch retrieval evaluation runner for IndicRAG-QA.

Executes and evaluates supported retrieval methods (BM25, Pretrained E5, Hybrid RRF)
over the reviewed queries in data/eval/questions_for_review.csv.
Generates genuine run files and evaluation summaries in reports/:
  - reports/relevance_judgements_completed.csv
  - reports/bm25_retrieval_runs.csv
  - reports/dense_retrieval_runs.csv
  - reports/hybrid_retrieval_runs.csv
  - reports/bm25_evaluation_summary.json
  - reports/bm25_evaluation_per_query.csv

Diagnostic reporting exposes:
  - total_queries_loaded
  - answerable_queries
  - queries_with_valid_gold
  - evaluated_queries
  - excluded_queries
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from app import config
from app.evalkit import first_relevant_rank, retrieval_row
from app.main import engine

log = logging.getLogger("eval_retrieval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_csv(path: Path | str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_gold_chunk_ids(val: str | None) -> list[str]:
    if not val:
        return []
    clean = val.replace(";", ",")
    return [x.strip() for x in clean.split(",") if x.strip()]


def evaluate(
    runs: list[dict[str, str]],
    judgements: list[dict[str, str]],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Evaluates retrieval runs against relevance judgements.

    Returns:
      (per_query_results, aggregate_metrics, diagnostics)
    """
    # Group runs by query_id in ranked order
    ranked_chunks: dict[str, list[str]] = defaultdict(list)
    ranked_scores: dict[str, list[float]] = defaultdict(list)
    for r in sorted(runs, key=lambda x: (x.get("query_id", ""), int(x.get("rank", 999)))):
        qid = r.get("query_id", "")
        if qid:
            ranked_chunks[qid].append(r.get("chunk_id", ""))
            try:
                ranked_scores[qid].append(float(r.get("score", 0.0)))
            except ValueError:
                ranked_scores[qid].append(0.0)

    per_query: list[dict[str, Any]] = []
    evaluated_rows: list[dict[str, float]] = []

    total_loaded = len(judgements)
    answerable_count = 0
    valid_gold_count = 0
    excluded_count = 0

    for j in judgements:
        qid = j.get("query_id") or j.get("qid") or ""
        q_text = j.get("query") or j.get("question") or ""
        raw_ans = str(j.get("answerable", "true")).strip().lower()
        is_ans = raw_ans in ("true", "1", "yes", "y")
        if is_ans:
            answerable_count += 1

        gold_ids = parse_gold_chunk_ids(j.get("relevant_chunk_ids") or j.get("gold_chunk_ids"))
        if gold_ids:
            valid_gold_count += 1

        retrieved = ranked_chunks.get(qid, [])

        if not gold_ids:
            excluded_count += 1
            per_query.append({
                "query_id": qid,
                "query": q_text,
                "relevant_passages": 0,
                "first_relevant_rank": None,
                "reciprocal_rank": 0.0,
                "hit@1": 0.0,
                "hit@5": 0.0,
                "recall@5": 0.0,
                "precision@5": 0.0,
                "ndcg@5": 0.0,
                "status": "excluded_no_gold",
            })
            continue

        row_metrics = retrieval_row(retrieved, gold_ids, ks=(1, 3, 5, 10))
        fr_rank = first_relevant_rank(retrieved, gold_ids)
        evaluated_rows.append(row_metrics)

        per_query.append({
            "query_id": qid,
            "query": q_text,
            "relevant_passages": len(gold_ids),
            "first_relevant_rank": fr_rank if fr_rank is not None else "—",
            "reciprocal_rank": round(row_metrics["mrr"], 4),
            "hit@1": round(row_metrics["hit@1"], 4),
            "hit@5": round(row_metrics["hit@5"], 4),
            "recall@5": round(row_metrics["recall@5"], 4),
            "precision@5": round(row_metrics["precision@5"], 4),
            "ndcg@5": round(row_metrics["ndcg@5"], 4),
            "status": "evaluated",
        })

    eval_count = len(evaluated_rows)

    def mean_of(k: str) -> float:
        return round(sum(r[k] for r in evaluated_rows) / eval_count, 4) if eval_count else 0.0

    agg: dict[str, Any] = {
        "evaluated_queries": eval_count,
        "total_queries_loaded": total_loaded,
        "answerable_queries": answerable_count,
        "queries_with_valid_gold": valid_gold_count,
        "excluded_queries": excluded_count,
        "mean_reciprocal_rank": mean_of("mrr"),
        "mean_recall@5": mean_of("recall@5"),
        "mean_recall@10": mean_of("recall@10"),
        "mean_precision@5": mean_of("precision@5"),
        "mean_precision@10": mean_of("precision@10"),
        "mean_ndcg@5": mean_of("ndcg@5"),
        "mean_ndcg@10": mean_of("ndcg@10"),
        "hit@1": mean_of("hit@1"),
        "hit@3": mean_of("hit@3"),
        "hit@5": mean_of("hit@5"),
        "hit@10": mean_of("hit@10"),
    }

    diagnostics: dict[str, Any] = {
        "total_queries_loaded": total_loaded,
        "answerable_queries": answerable_count,
        "queries_with_valid_gold": valid_gold_count,
        "evaluated_queries": eval_count,
        "excluded_queries": excluded_count,
        "notes": (
            f"Evaluated {eval_count} queries with gold passages; "
            f"{excluded_count} unanswerable/ungrounded queries excluded from ranking metrics."
        ),
    }

    return per_query, agg, diagnostics


def run_batch_evaluation() -> None:
    """Executes BM25, Pretrained E5, and Hybrid RRF and saves reports."""
    rep_dir = config.REPORTS_DIR
    rep_dir.mkdir(parents=True, exist_ok=True)
    engine.ensure_core()

    review_file = config.DATA_DIR / "eval" / "questions_for_review.csv"
    if not review_file.exists():
        review_file = config.DATA_DIR / "eval" / "questions.jsonl"

    queries: list[dict[str, Any]] = []
    if review_file.suffix == ".csv":
        with open(review_file, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                queries.append(r)
    else:
        with open(review_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))

    log.info("Loaded %d queries from %s", len(queries), review_file)

    # 1. Build reports/relevance_judgements_completed.csv
    judgements_rows = []
    for q in queries:
        qid = q.get("qid") or q.get("query_id") or ""
        q_text = q.get("question") or q.get("query") or ""
        raw_golds = q.get("gold_chunk_ids") or q.get("relevant_chunk_ids") or ""
        golds = parse_gold_chunk_ids(raw_golds)
        raw_ans = str(q.get("answerable", "true")).strip().lower()
        is_ans = raw_ans in ("true", "1", "yes", "y")
        # Reviewer screening check
        rev_ok = q.get("reviewer_ok", "Y")
        if rev_ok == "N":
            golds = []

        judgements_rows.append({
            "query_id": qid,
            "query": q_text,
            "relevant_chunk_ids": ",".join(golds),
            "answerable": str(is_ans),
            "language": q.get("language", "English"),
        })

    judgements_path = rep_dir / "relevance_judgements_completed.csv"
    write_csv(judgements_path, judgements_rows, ["query_id", "query", "relevant_chunk_ids", "answerable", "language"])
    log.info("Wrote %s with %d rows", judgements_path, len(judgements_rows))

    # Determine available retrieval methods
    av = engine.availability()
    methods_to_run = ["bm25"]
    if av.get("pretrained", {}).get("available"):
        methods_to_run.append("dense")
    if av.get("hybrid", {}).get("available"):
        methods_to_run.append("hybrid")
    if av.get("finetuned", {}).get("available"):
        methods_to_run.append("finetuned")

    log.info("Running retrieval methods: %s", methods_to_run)

    for m in methods_to_run:
        engine_method = "pretrained" if m == "dense" else m
        runs_rows = []
        for q in queries:
            qid = q.get("qid") or q.get("query_id") or ""
            q_text = q.get("question") or q.get("query") or ""
            try:
                results = engine.retrieve(q_text, engine_method, top_k=10)
                for item in results:
                    runs_rows.append({
                        "query_id": qid,
                        "chunk_id": item["chunk_id"],
                        "rank": item["rank"],
                        "score": round(float(item.get("score", 0.0)), 6),
                    })
            except Exception as exc:
                log.warning("Method %s failed on query %s: %s", m, qid, exc)

        runs_path = rep_dir / f"{m}_retrieval_runs.csv"
        write_csv(runs_path, runs_rows, ["query_id", "chunk_id", "rank", "score"])
        log.info("Wrote %s with %d run rows", runs_path, len(runs_rows))

        # Evaluate this method
        per_q, agg, diag = evaluate(runs_rows, judgements_rows, top_k=10)
        log.info(
            "Method %s: Evaluated=%d, Excluded=%d, MRR=%.4f, Recall@5=%.4f, NDCG@5=%.4f",
            m, agg["evaluated_queries"], agg["excluded_queries"],
            agg["mean_reciprocal_rank"], agg["mean_recall@5"], agg["mean_ndcg@5"],
        )

        if m == "bm25":
            # Save bm25_evaluation_summary.json and bm25_evaluation_per_query.csv
            summary_path = rep_dir / "bm25_evaluation_summary.json"
            summary_path.write_text(json.dumps({**agg, "diagnostics": diag}, indent=2), encoding="utf-8")

            per_q_path = rep_dir / "bm25_evaluation_per_query.csv"
            write_csv(
                per_q_path,
                per_q,
                [
                    "query_id", "query", "relevant_passages", "first_relevant_rank",
                    "reciprocal_rank", "hit@1", "hit@5", "recall@5", "precision@5",
                    "ndcg@5", "status",
                ],
            )
            log.info("Saved BM25 summary to %s and per-query to %s", summary_path, per_q_path)


if __name__ == "__main__":
    run_batch_evaluation()
