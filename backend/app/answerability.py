
"""Answerability detection for IndicRAG-QA.

Uses retrieval signals and calibrated thresholds to decide whether
the retrieved evidence is sufficiently strong to answer a question.
This module must not import eval_common or app.main.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app import config
from app.evalkit import confusion

THRESHOLDS_PATH = config.REPORTS_DIR / "answerability_thresholds.json"

SIGNAL_TYPES = {
    "bm25": "bm25_top1",
    "pretrained": "cosine_top1",
    "hybrid": "dense_cosine_top1",
}


def evidence_signal(method: str, results: list[dict[str, Any]]) -> float | None:
    """Return the method-appropriate relevance signal."""

    if method == "bm25":
        return float(results[0]["score"]) if results else 0.0

    if method == "pretrained":
        return float(results[0]["score"]) if results else None

    if method == "hybrid":
        # main.py stores the strongest dense cosine in dense_top1.
        dense_scores = [
            float(r["dense_top1"])
            for r in results
            if r.get("dense_top1") is not None
        ]

        dense_scores.extend(
            float(r["dense_score"])
            for r in results
            if r.get("dense_score") is not None
        )

        return max(dense_scores) if dense_scores else None

    raise ValueError(f"Unknown method {method}")


def fit_threshold(
    signals: list[float],
    answerable: list[bool]
) -> dict[str, Any]:
    """Choose the threshold that maximises answerable-class F1."""

    if not signals or len(signals) != len(answerable):
        raise ValueError(
            "signals and labels must be non-empty and the same length"
        )

    values = sorted(set(float(value) for value in signals))

    candidates = (
        [values[0] - 1e-6]
        + [
            (left + right) / 2
            for left, right in zip(values, values[1:])
        ]
        + [values[-1] + 1e-6]
    )

    best: tuple[float, float, float] | None = None

    for threshold in candidates:
        metrics = confusion(
            answerable,
            [score >= threshold for score in signals]
        )

        candidate = (
            metrics["f1"],
            metrics["accuracy"],
            threshold
        )

        if best is None or candidate > best:
            best = candidate

    assert best is not None

    threshold = best[2]

    fitted_metrics = confusion(
        answerable,
        [score >= threshold for score in signals]
    )

    return {
        "threshold": round(threshold, 6),
        "fit_n": len(signals),
        **{
            f"fit_{key}": fitted_metrics[key]
            for key in ("f1", "accuracy", "precision", "recall")
        },
    }


def auc(
    signals: list[float],
    answerable: list[bool]
) -> float | None:
    """Calculate ROC-AUC."""

    positive = [
        score for score, label in zip(signals, answerable) if label
    ]

    negative = [
        score for score, label in zip(signals, answerable) if not label
    ]

    if not positive or not negative:
        return None

    wins = sum(
        (p > n) + 0.5 * (p == n)
        for p in positive
        for n in negative
    )

    return round(
        wins / (len(positive) * len(negative)),
        4
    )


_cache: dict[str, Any] = {
    "mtime": None,
    "path": None,
    "data": {}
}


def load_thresholds(
    path: Path | None = None
) -> dict[str, Any]:
    """Load calibrated thresholds and refresh cache when changed."""

    path = path or THRESHOLDS_PATH

    if not path.exists():
        return {}

    modified_time = path.stat().st_mtime

    if (
        _cache["path"] != str(path)
        or _cache["mtime"] != modified_time
    ):
        _cache.update(
            path=str(path),
            mtime=modified_time,
            data=json.loads(
                path.read_text(encoding="utf-8")
            ),
        )

    return _cache["data"]


def assess(
    method: str,
    results: list[dict[str, Any]],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return answerable, unanswerable, or not_calibrated."""

    if method not in SIGNAL_TYPES:
        raise ValueError(f"Unknown method {method}")

    thresholds = (
        load_thresholds()
        if thresholds is None
        else thresholds
    )

    signal = evidence_signal(method, results)
    entry = thresholds.get(method)

    base = {
        "signal": (
            None if signal is None
            else round(signal, 6)
        ),
        "signal_type": SIGNAL_TYPES[method],
        "threshold": (
            None if not entry
            else entry["threshold"]
        ),
    }

    if not entry:
        return {
            **base,
            "status": "not_calibrated",
            "reason": (
                f"No answerability threshold has been fitted for '{method}'. "
                "Run backend/evaluate_answerability.py to calibrate it; "
                "until then no question is rejected."
            ),
        }

    if signal is None:
        return {
            **base,
            "status": "unanswerable",
            "reason": (
                "No usable relevance signal was produced for this question."
            ),
        }

    is_answerable = signal >= entry["threshold"]

    return {
        **base,
        "status": (
            "answerable" if is_answerable
            else "unanswerable"
        ),
        "reason": (
            f"Top evidence signal {signal:.3f} "
            f"{'meets' if is_answerable else 'is below'} "
            f"the calibrated threshold "
            f"{entry['threshold']:.3f} "
            f"({SIGNAL_TYPES[method]}). "
            f"Fitted on {entry.get('fit_n', '?')} "
            f"{entry.get('fit_split', 'dev')} questions; "
            "a heuristic, not a probability."
        ),
    }
