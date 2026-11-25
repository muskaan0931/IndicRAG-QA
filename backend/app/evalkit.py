"""Shared evaluation utilities: dataset loading, answer normalisation, QA metrics, retrieval metrics,
bootstrap confidence intervals and answerability (binary classification) metrics."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app import config

DATASET_PATH = config.DATA_DIR / "eval" / "questions.jsonl"
LANGUAGES = ["English", "Hindi", "Hinglish"]


def load_dataset(path: Path | None = None, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path or DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if split in (None, "all") or r["split"] == split]


# ---------------- answer normalisation and QA metrics ----------------
def normalize(text: str) -> str:
    """Lowercase and collapse every non-word run to one space (so '8.5' and '8 5' compare equal)."""
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def contains_gold(pred: str, gold: str) -> float:
    """1 if the normalised gold answer occurs inside the normalised prediction (word-bounded)."""
    p, g = f" {normalize(pred)} ", normalize(gold)
    return float(bool(g) and f" {g} " in p)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------- retrieval metrics (per question) ----------------
def first_relevant_rank(retrieved_ids: list[str], gold: Iterable[str]) -> int | None:
    gold = set(gold)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return i
    return None


def retrieval_row(retrieved_ids: list[str], gold: list[str], ks=(1, 3, 5, 10)) -> dict[str, float]:
    gold_set = set(gold)
    rank = first_relevant_rank(retrieved_ids, gold_set)
    row: dict[str, float] = {"mrr": 1 / rank if rank else 0.0}
    for k in ks:
        top = retrieved_ids[:k]
        hits = sum(c in gold_set for c in top)
        row[f"hit@{k}"] = float(hits > 0)                 # Hit Rate@K
        row[f"recall@{k}"] = hits / len(gold_set)         # share of judged-relevant passages found
        row[f"precision@{k}"] = hits / k
        dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(top, start=1) if c in gold_set)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
        row[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return row


def mean_ci(values: list[float], n_boot: int = 1000, seed: int = 13) -> dict[str, float]:
    """Mean with a 95% percentile-bootstrap interval. Questions are resampled, so with ~30 questions
    the interval is wide: that width is the honest statement of how little the sample can tell us."""
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(values) / n, "lo": means[int(0.025 * n_boot)], "hi": means[int(0.975 * n_boot) - 1], "n": n}


def aggregate(rows: list[dict[str, float]], metrics: list[str]) -> dict[str, dict[str, float]]:
    return {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in mean_ci([r[m] for r in rows]).items()}
            for m in metrics}


# ---------------- answerability: binary classification metrics ----------------
def confusion(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    """Positive class = ANSWERABLE."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
            "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
