"""Feature 2 - answerability detection.

Fits one threshold per retrieval method on the DEV split, then reports accuracy / precision / recall / F1 and the
confusion matrix on the untouched TEST split (positive class = ANSWERABLE). Writes
  reports/answerability_thresholds.json  (read by /ask)   and   reports/answerability_eval.json

Usage:  python backend/evaluate_answerability.py [--methods bm25 hybrid ...]
"""
import argparse

from eval_common import ALL_METHODS, make_retrieve, method_status, write_json
from app.evalkit import load_dataset
from app.qa_eval import answerability_report, collect_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    a = ap.parse_args()
    dev, test = load_dataset(split="dev"), load_dataset(split="test")
    retrieve = make_retrieve()
    usable, skipped = method_status(a.methods)
    thresholds, evals = {}, {}
    for m in usable:
        entry, ev = answerability_report(dev, collect_signals(dev, retrieve, m), test, collect_signals(test, retrieve, m), m)
        thresholds[m], evals[m] = entry, ev
        t = ev["test"]["overall"]
        print(f"{m:11} thr={entry['threshold']:.4f}  test acc={t['accuracy']:.2f} P={t['precision']:.2f} "
              f"R={t['recall']:.2f} F1={t['f1']:.2f} AUC={ev['test']['auc']}  confusion tp/fp/tn/fn="
              f"{t['tp']}/{t['fp']}/{t['tn']}/{t['fn']}")
    for m, why in skipped.items():
        print(f"{m:11} SKIPPED: {why}")
    write_json("answerability_thresholds.json", thresholds)
    write_json("answerability_eval.json", {
        "protocol": "threshold fitted on dev, scored on test; positive class = answerable",
        "dev_questions": len(dev), "test_questions": len(test),
        "caveat": "Questions are AI-drafted and not yet human-verified; with ~30 test questions per class the "
                  "numbers are indicative only.",
        "methods": evals, "skipped": skipped})


if __name__ == "__main__":
    main()
