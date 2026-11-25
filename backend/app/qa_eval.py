"""Pure evaluation logic for features 2 (answerability), 4 (results by language) and 5 (direct LLM vs RAG).

Nothing here touches the network or loads a model: callers pass in `retrieve(question, method, k)` and generator
objects, which keeps every function unit-testable with stand-ins.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from app import answerability
from app.evalkit import LANGUAGES, aggregate, confusion, contains_gold, mean_ci, retrieval_row, token_f1

Retrieve = Callable[[str, str, int], list[dict[str, Any]]]
RETRIEVAL_METRICS = ["hit@1", "hit@3", "hit@5", "recall@5", "recall@10", "mrr", "ndcg@10"]
CROSS = "Cross-lingual (Hindi+Hinglish)"
GROUPS = ["English", "Hindi", "Hinglish", CROSS, "All"]


def groups_of(language: str) -> list[str]:
    """English questions over an English corpus are monolingual; Hindi/Hinglish questions are cross-lingual
    (query language differs from document language)."""
    g = [language, "All"]
    if language in ("Hindi", "Hinglish"):
        g.append(CROSS)
    return g


# ------------------------------------------------------------------ feature 4: retrieval by language
def retrieval_by_language(rows: list[dict[str, Any]], retrieve: Retrieve, method: str, k: int = 10) -> dict[str, Any]:
    """Retrieval metrics (answerable questions only) per language group, with 95% bootstrap intervals."""
    per_row: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        if not r["answerable"]:
            continue
        ids = [x["chunk_id"] for x in retrieve(r["question"], method, k)]
        row = retrieval_row(ids, r["gold_chunk_ids"])
        for g in groups_of(r["language"]):
            per_row[g].append(row)
    return {g: aggregate(per_row[g], RETRIEVAL_METRICS) for g in GROUPS if per_row.get(g)}


def markdown_table(results: dict[str, Any], metric: str = "hit@5") -> str:
    """Methods x language groups table for one metric: 'mean [lo, hi] (n)'. Unavailable methods are listed as such."""
    groups = [g for g in GROUPS if any(isinstance(v, dict) and g in v for v in results.values())]
    lines = ["| Method | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for method, res in results.items():
        if "unavailable" in res:
            lines.append(f"| {method} | " + " | ".join("n/a" for _ in groups) + " |")
            continue
        cells = []
        for g in groups:
            m = res.get(g, {}).get(metric)
            cells.append(f"{m['mean']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}] (n={m['n']})" if m else "–")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ feature 2: answerability
def collect_signals(rows: list[dict[str, Any]], retrieve: Retrieve, method: str, k: int = 10) -> list[float | None]:
    return [answerability.evidence_signal(method, retrieve(r["question"], method, k)) for r in rows]


def answerability_report(dev_rows, dev_sig, test_rows, test_sig, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the threshold on dev, report on test. Returns (threshold entry, evaluation block)."""
    dev_pairs = [(s, r["answerable"]) for s, r in zip(dev_sig, dev_rows) if s is not None]
    if not dev_pairs or len({a for _, a in dev_pairs}) < 2:
        raise ValueError(f"{method}: need dev questions of both classes with a usable signal to fit a threshold")
    entry = answerability.fit_threshold([s for s, _ in dev_pairs], [a for _, a in dev_pairs])
    entry["fit_split"] = "dev"
    t = entry["threshold"]

    def block(rows, sig):
        pred = [(s is not None and s >= t) for s in sig]
        truth = [r["answerable"] for r in rows]
        out = {"overall": confusion(truth, pred),
               "auc": answerability.auc([s if s is not None else float("-inf") for s in sig], truth)}
        out["by_language"] = {lang: confusion([r["answerable"] for r in rows if r["language"] == lang],
                                              [p for p, r in zip(pred, rows) if r["language"] == lang])
                              for lang in LANGUAGES}
        return out

    return entry, {"threshold": t, "fit": {k: v for k, v in entry.items() if k != "threshold"},
                   "dev": block(dev_rows, dev_sig), "test": block(test_rows, test_sig)}


# ------------------------------------------------------------------ feature 5: direct LLM vs RAG
def classify_outcome(row: dict[str, Any], answer: str, abstained: bool) -> str:
    """answerable   : correct | wrong (answered, gold string absent) | abstained (over-abstention)
       unanswerable : correct_abstain | unsupported (answered although the corpus cannot answer)"""
    if row["answerable"]:
        if abstained:
            return "abstained"
        return "correct" if contains_gold(answer, row["answer_text"]) else "wrong"
    return "correct_abstain" if abstained else "unsupported"


def summarize_outcomes(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(records)
    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    c = Counter(r["outcome"] for r in recs)

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    answered = [r for r in recs if not r["abstained"]]
    return {
        "n": len(recs), "n_answerable": len(ans), "n_unanswerable": len(una),
        "outcomes": dict(c),
        "answerable_accuracy": rate(c["correct"], len(ans)),
        "answerable_accuracy_ci": {k: round(v, 4) for k, v in mean_ci([float(r["outcome"] == "correct") for r in ans]).items()} if ans else None,
        "answerable_mean_token_f1": round(sum(r["f1"] for r in ans) / len(ans), 4) if ans else None,
        "wrong_answer_rate_on_answerable": rate(c["wrong"], len(ans)),
        "over_abstention_rate": rate(c["abstained"], len(ans)),
        "correct_abstention_rate_on_unanswerable": rate(c["correct_abstain"], len(una)),
        # An "unsupported answer" is an answer a reader cannot trust: wrong on an answerable question, or any
        # answer to an unanswerable one.
        "unsupported_answers": c["wrong"] + c["unsupported"],
        "unsupported_rate_of_answered": rate(c["wrong"] + c["unsupported"], len(answered)),
        # Answers that carry no verifiable citation at all (every non-abstained direct-LLM answer by construction).
        "uncited_answers": sum(1 for r in answered if not r.get("cited")),
        "answers_with_verifier_warnings": sum(1 for r in answered if r.get("warnings")),
    }


def run_system(rows: list[dict[str, Any]], answer_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """answer_fn(row) -> {answer, abstained, cited, warnings}. Adds outcome and token-F1 and keeps the raw answer."""
    out = []
    for r in rows:
        a = answer_fn(r)
        out.append({"qid": r["qid"], "language": r["language"], "answerable": r["answerable"],
                    "question": r["question"], "gold": r["answer_text"], "answer": a["answer"],
                    "abstained": a["abstained"], "cited": a.get("cited", []), "warnings": a.get("warnings", []),
                    "outcome": classify_outcome(r, a["answer"], a["abstained"]),
                    "f1": token_f1(a["answer"], r["answer_text"]) if r["answerable"] and not a["abstained"] else 0.0})
    return out
