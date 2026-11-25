"""Feature 5 - direct LLM versus retrieval-augmented QA.

The SAME questions are answered (a) by the LLM from memory (no evidence) and (b) by the full RAG pipeline
(retrieve -> answerability gate -> LLM with cited passages). We count correct answers, wrong answers, abstentions
and UNSUPPORTED answers (wrong, or any answer to a question the corpus cannot answer), plus answers with no citation.
Answers are forced to English so that string matching against the gold answer is meaningful; every raw answer is saved
to reports/rag_vs_llm_answers.jsonl so a human can re-judge them (string matching is only a proxy).

Needs ANTHROPIC_API_KEY. Usage:  python backend/evaluate_rag_vs_llm.py [--split test] [--method hybrid] [--limit N]
"""
import argparse
import os
import sys

from eval_common import ALL_METHODS, config, make_retrieve, method_status, write_json
from app import answerability
from app.evalkit import load_dataset
from app.generation import ABSTAIN_MESSAGE, LLMGenerator
from app.qa_eval import run_system, summarize_outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--method", default="hybrid", choices=ALL_METHODS, help="retrieval method used by the RAG arm")
    ap.add_argument("--generator", default="auto", choices=["auto", "llm", "extractive"], help="generator mode")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (cheap smoke run)")
    a = ap.parse_args()
    gen_mode = a.generator
    if gen_mode == "auto":
        gen_mode = "llm" if LLMGenerator.available() else "extractive"
    if gen_mode == "llm" and not LLMGenerator.available():
        sys.exit("ANTHROPIC_API_KEY is not set; set the key or pass --generator extractive.")
    usable, skipped = method_status([a.method])
    if skipped:
        sys.exit(f"Retrieval method unavailable: {skipped[a.method]}")
    rows = load_dataset(split=a.split)[: a.limit]
    retrieve = make_retrieve()
    thresholds = answerability.load_thresholds()
    gated = a.method in thresholds

    if gen_mode == "llm":
        llm = LLMGenerator(answer_language="en")
        model_name = llm.model
        def direct(row):
            g = llm.direct(row["question"])
            return {"answer": g.answer, "abstained": g.abstained, "cited": [], "warnings": g.warnings}

        def rag(row):
            ev = retrieve(row["question"], a.method, a.top_k)
            if not ev or answerability.assess(a.method, ev, thresholds)["status"] == "unanswerable":
                return {"answer": ABSTAIN_MESSAGE, "abstained": True}
            g = llm.generate(row["question"], ev)
            return {"answer": g.answer, "abstained": g.abstained, "cited": g.cited, "warnings": g.warnings}
    else:
        from app.generation import ExtractiveGenerator
        from app.main import _sentence_scorer
        model_name = "extractive-offline"
        ext_gen = ExtractiveGenerator()
        def direct(row):
            # Direct generation without evidence cannot answer factoid institutional questions
            return {"answer": ABSTAIN_MESSAGE, "abstained": True, "cited": [], "warnings": ["Direct offline without retrieval"]}

        def rag(row):
            ev = retrieve(row["question"], a.method, a.top_k)
            if not ev or answerability.assess(a.method, ev, thresholds)["status"] == "unanswerable":
                return {"answer": ABSTAIN_MESSAGE, "abstained": True, "cited": []}
            g = ext_gen.generate(row["question"], ev)
            return {"answer": g.answer, "abstained": g.abstained, "cited": g.cited, "warnings": g.warnings}

    direct_recs, rag_recs = run_system(rows, direct), run_system(rows, rag)
    report = {
        "split": a.split, "model": model_name, "rag_retrieval": a.method, "answerability_gate_used": gated,
        "generator_mode": gen_mode,
        "caveat": "Correctness is a string-containment proxy against AI-drafted gold answers; re-judge "
                  "reports/rag_vs_llm_answers.jsonl by hand before quoting numbers. Small n: see CIs.",
        "direct_llm": summarize_outcomes(direct_recs), "rag": summarize_outcomes(rag_recs),
    }
    write_json("rag_vs_llm.json", report)
    import json
    with open(config.REPORTS_DIR / "rag_vs_llm_answers.jsonl", "w", encoding="utf-8") as f:
        for system, recs in (("direct_llm", direct_recs), ("rag", rag_recs)):
            for r in recs:
                f.write(json.dumps({"system": system, **r}, ensure_ascii=False) + "\n")
    for name in ("direct_llm", "rag"):
        s = report[name]
        print(f"{name:10} acc={s['answerable_accuracy']} wrong={s['wrong_answer_rate_on_answerable']} "
              f"abstain-correct={s['correct_abstention_rate_on_unanswerable']} unsupported={s['unsupported_answers']} "
              f"uncited={s['uncited_answers']}")


if __name__ == "__main__":
    main()
