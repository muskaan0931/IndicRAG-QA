"""Feature 4 - retrieval results table for every method, split by language.

Languages: English (monolingual), Hindi, Hinglish, and the cross-lingual group (Hindi + Hinglish questions over the
English corpus). Writes reports/retrieval_by_language.json and reports/retrieval_by_language.md.
Methods that cannot run on this machine (no E5 index / no sentence-transformers) are listed as unavailable, never faked.

Usage:  python backend/evaluate_methods.py [--split test|dev|all] [--methods ...]
"""
import argparse

from eval_common import ALL_METHODS, config, make_retrieve, method_status, write_json
from app.evalkit import load_dataset
from app.qa_eval import markdown_table, retrieval_by_language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    a = ap.parse_args()
    rows = load_dataset(split=a.split)
    retrieve = make_retrieve()
    usable, skipped = method_status(a.methods)
    results = {m: retrieval_by_language(rows, retrieve, m) for m in usable}
    for m, why in skipped.items():
        results[m] = {"unavailable": why}
    write_json("retrieval_by_language.json", {
        "split": a.split, "n_answerable_questions": sum(r["answerable"] for r in rows),
        "metrics": "mean [95% bootstrap CI] over questions; Hit@k = any gold passage in top k",
        "caveat": "AI-drafted, not yet human-verified labels; gold lists may be incomplete (recall understated).",
        "results": results})
    md = [f"# Retrieval by language ({a.split} split)\n"]
    for metric in ("hit@1", "hit@5", "mrr", "ndcg@10"):
        md += [f"## {metric}\n", markdown_table(results, metric), ""]
    (config.REPORTS_DIR / "retrieval_by_language.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
