import json

import pytest

import conftest  # noqa: F401
from app import answerability as ans


def test_fit_threshold_separates_and_prefers_conservative_tie():
    sig = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
    lab = [True, True, True, False, False, False]
    e = ans.fit_threshold(sig, lab)
    assert 0.2 < e["threshold"] <= 0.7 and e["fit_f1"] == 1.0 and e["fit_accuracy"] == 1.0
    with pytest.raises(ValueError):
        ans.fit_threshold([], [])


def test_auc_values():
    assert ans.auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert ans.auc([1, 1], [True, False]) == 0.5
    assert ans.auc([1, 2], [True, True]) is None


def test_evidence_signal_per_method():
    res = [{"score": 4.2}, {"score": 1.0}]
    assert ans.evidence_signal("bm25", res) == 4.2 and ans.evidence_signal("bm25", []) == 0.0
    assert ans.evidence_signal("pretrained", res) == 4.2 and ans.evidence_signal("pretrained", []) is None
    hy = [{"score": 0.03, "dense_score": 0.7, "dense_top1": 0.8}, {"score": 0.02, "dense_score": 0.5}]
    assert ans.evidence_signal("hybrid", hy) == 0.8
    assert ans.evidence_signal("hybrid", [{"score": 0.03}]) is None
    with pytest.raises(ValueError):
        ans.evidence_signal("nope", res)


def test_assess_statuses():
    res = [{"score": 3.0}]
    assert ans.assess("bm25", res, {})["status"] == "not_calibrated"
    th = {"bm25": {"threshold": 2.0, "fit_n": 10}}
    assert ans.assess("bm25", res, th)["status"] == "answerable"
    low = ans.assess("bm25", [{"score": 1.0}], th)
    assert low["status"] == "unanswerable" and "below" in low["reason"] and low["threshold"] == 2.0
    assert ans.assess("pretrained", [], {"pretrained": {"threshold": 0.8}})["status"] == "unanswerable"


def test_load_thresholds_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    assert ans.load_thresholds(p) == {}
    p.write_text(json.dumps({"bm25": {"threshold": 1.5}}))
    assert ans.load_thresholds(p)["bm25"]["threshold"] == 1.5
