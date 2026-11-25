import conftest  # noqa: F401
from app import qa_eval as q
from app.evalkit import LANGUAGES, confusion, contains_gold, exact_match, retrieval_row, token_f1


def row(qid, lang, answerable, gold_ids=("g1",), ans="75%", question="q"):
    return {"qid": qid, "language": lang, "answerable": answerable, "gold_chunk_ids": list(gold_ids) if answerable else [],
            "answer_text": ans if answerable else "", "question": question, "split": "test"}


def test_text_metrics():
    assert contains_gold("The minimum is 75% of classes", "75%") == 1.0
    assert contains_gold("175% is wrong", "75%") == 0.0
    assert exact_match("X grade", "x  grade.") == 1.0
    assert 0 < token_f1("an X grade", "X grade") < 1 and token_f1("", "") == 1.0


def test_retrieval_row_metrics():
    r = retrieval_row(["a", "g1", "b"], ["g1", "g2"])
    assert r["mrr"] == 0.5 and r["hit@1"] == 0 and r["hit@3"] == 1 and r["recall@3"] == 0.5 and r["precision@3"] == 1 / 3
    assert retrieval_row(["x"], ["g1"])["mrr"] == 0.0


def test_confusion_matrix():
    m = confusion([True, True, False, False], [True, False, False, True])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1) and m["accuracy"] == 0.5 and m["f1"] == 0.5


def test_retrieval_by_language_groups_and_cross_lingual():
    rows = [row("a-EN", "English", True), row("a-HI", "Hindi", True), row("a-HG", "Hinglish", True), row("u-EN", "English", False)]

    def retrieve(question, method, k):  # English finds gold at rank 1, everything else misses
        return [{"chunk_id": "g1" if question == "en" else "zz"}]
    rows[0]["question"] = "en"
    res = q.retrieval_by_language(rows, retrieve, "bm25")
    assert res["English"]["hit@1"]["mean"] == 1.0 and res["English"]["hit@1"]["n"] == 1   # unanswerable excluded
    assert res["Hindi"]["hit@1"]["mean"] == 0.0
    assert res[q.CROSS]["hit@1"]["n"] == 2 and res["All"]["hit@1"]["n"] == 3
    table = q.markdown_table({"bm25": res, "pretrained": {"unavailable": "no index"}}, "hit@1")
    assert "| bm25 |" in table and "| pretrained | n/a" in table and "Cross-lingual" in table


def test_answerability_report_fits_on_dev_scores_on_test():
    dev = [row("d1", "English", True), row("d2", "Hindi", True), row("d3", "English", False), row("d4", "Hindi", False)]
    test = [row("t1", "English", True), row("t2", "English", False), row("t3", "Hinglish", False)]
    entry, ev = q.answerability_report(dev, [5, 4, 1, 0], test, [6, 0.5, 4.5], "bm25")
    assert entry["fit_split"] == "dev" and 1 < entry["threshold"] <= 4
    c = ev["test"]["overall"]
    assert (c["tp"], c["tn"], c["fp"], c["fn"]) == (1, 1, 1, 0)          # t3 scores 4.5 -> false positive
    assert set(ev["test"]["by_language"]) == set(LANGUAGES) and ev["test"]["auc"] == 1.0


def test_answerability_report_requires_both_classes():
    import pytest
    with pytest.raises(ValueError):
        q.answerability_report([row("d1", "English", True)], [1.0], [], [], "bm25")


def test_classify_and_summarise_outcomes():
    a, u = row("a", "English", True), row("u", "English", False)
    assert q.classify_outcome(a, "It is 75% [1]", False) == "correct"
    assert q.classify_outcome(a, "It is 60%", False) == "wrong"
    assert q.classify_outcome(a, "x", True) == "abstained"
    assert q.classify_outcome(u, "Rs 5 lakh", False) == "unsupported"
    assert q.classify_outcome(u, "x", True) == "correct_abstain"

    def system(correct_on_a, abstain_on_u):
        def fn(r):
            if r["answerable"]:
                return {"answer": "75% [1]" if correct_on_a else "60%", "abstained": False, "cited": [1] if correct_on_a else []}
            return {"answer": "Insufficient" if abstain_on_u else "made up", "abstained": abstain_on_u}
        return fn
    rows = [a, row("a2", "Hindi", True), u, row("u2", "Hindi", False)]
    good = q.summarize_outcomes(q.run_system(rows, system(True, True)))
    bad = q.summarize_outcomes(q.run_system(rows, system(False, False)))
    assert good["answerable_accuracy"] == 1.0 and good["unsupported_answers"] == 0 and good["correct_abstention_rate_on_unanswerable"] == 1.0
    assert bad["answerable_accuracy"] == 0.0 and bad["unsupported_answers"] == 4 and bad["uncited_answers"] == 4
    assert bad["unsupported_rate_of_answered"] == 1.0
