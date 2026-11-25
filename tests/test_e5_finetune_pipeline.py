"""Tests for the E5 fine-tuning / evaluation pipeline.

Most tests are torch-free. The fake encoders used here are TEST DOUBLES that exercise the plumbing; their outputs are
not retrieval results and must never be reported as such. The training-loop test needs PyTorch and is skipped without it.
"""
import csv
import json
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

import e5_ft_common as C

ROOT = C.ROOT
HAVE_REAL = (ROOT / "data" / "chunks.jsonl").exists() and (ROOT / "data" / "eval" / "questions_for_review.csv").exists()
CSV_FIELDS = ["qid", "fact_id", "split", "language", "question", "answerable", "gold_chunk_ids", "answer_text",
              "reviewer_ok", "reviewer_notes"]


def _chunks(n=8):
    return {f"c{i}": {"chunk_id": f"c{i}", "text": " ".join(f"w{i}x{j}" for j in range(30))} for i in range(n)}


def _write_csv(rows):
    d = Path(tempfile.mkdtemp())
    p = d / "q.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return p


def _row(qid, fact, gold, ok="Y", answerable="True", q=None):
    return {"qid": qid, "fact_id": fact, "split": "dev", "language": "English", "question": q or f"question {qid}",
            "answerable": answerable, "gold_chunk_ids": gold, "answer_text": "a", "reviewer_ok": ok}


def _q(qid, fact, gold, lang="English"):
    return C.ReviewedQuestion(qid, fact, lang, f"question {qid} {lang}", tuple(gold), "a", "dev")


# ----------------------------------------------------------------------------------------- selection
def test_selection_uses_only_answerable_Y_with_valid_gold():
    rows = [
        _row("ok1", "F1", "c1"),
        _row("ok2", "F2", "c2;c3", ok="y"),                       # case-insensitive Y
        _row("rej", "F3", "c1", ok="N"),                          # rejected -> never
        _row("unrev", "F4", "c1", ok=""),                         # unreviewed -> never auto-accepted
        _row("unans", "F5", "", ok="Y", answerable="False"),      # not answerable -> never
        _row("badgold", "F6", "c1;nope"),                         # unknown gold id -> excluded, NOT trimmed
        _row("nogold", "F7", ""),
        _row("other", "F8", "c1", ok="maybe"),
    ]
    sel, rep = C.load_reviewed_questions(_write_csv(rows), _chunks())
    assert [q.qid for q in sel] == ["ok1", "ok2"]
    assert sel[1].gold == ("c2", "c3")
    reasons = {e["qid"]: e["reason"] for e in rep["excluded_after_Y"]}
    assert reasons == {"badgold": "gold_chunk_id_not_in_corpus", "nogold": "no_gold_chunk_ids"}
    assert rep["reviewer_N"] == 1 and rep["reviewer_blank"] == 1 and rep["reviewer_other"] == 1


def test_selection_rejects_duplicate_qids_and_empty_gold_text():
    with pytest.raises(ValueError):
        C.load_reviewed_questions(_write_csv([_row("a", "F", "c1"), _row("a", "G", "c2")]), _chunks())
    ch = _chunks()
    ch["c1"]["text"] = "  "
    sel, rep = C.load_reviewed_questions(_write_csv([_row("a", "F", "c1")]), ch)
    assert sel == [] and rep["excluded_after_Y"][0]["reason"] == "gold_chunk_text_empty"


@pytest.mark.skipif(not HAVE_REAL, reason="real project data not present")
def test_real_data_never_selects_rejected_or_unreviewed():
    chunks = C.load_chunks()
    sel, rep = C.load_reviewed_questions(C.DEFAULT_REVIEW_CSV, chunks)
    with open(C.DEFAULT_REVIEW_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    bad = {r["qid"] for r in rows if (r["reviewer_ok"] or "").strip().upper() != "Y" or r["answerable"].lower() != "true"}
    assert not ({q.qid for q in sel} & bad)
    assert all(g in chunks for q in sel for g in q.gold)
    assert rep["answerable_and_Y"] == len(sel) + len(rep["excluded_after_Y"])


# ----------------------------------------------------------------------------------------- groups / folds
def _toy_questions():
    return [
        _q("a-en", "F1", ["c1"]), _q("a-hi", "F1", ["c1"], "Hindi"),
        _q("b-en", "F2", ["c1", "c2"]),                      # shares c1 with F1 -> same group
        _q("c-en", "F3", ["c3"]), _q("c-hi", "F3", ["c3"], "Hindi"),
        _q("d-en", "F4", ["c4"]), _q("e-en", "F5", ["c5"]),
    ]


def test_groups_merge_shared_facts_and_gold_chunks():
    g = C.assign_groups(_toy_questions())
    assert g["a-en"] == g["a-hi"] == g["b-en"] == "F1+F2"
    assert len(set(g.values())) == 4


def test_folds_are_leak_free_and_cover_every_question_once():
    qs = _toy_questions()
    folds = C.make_folds(qs, 3)
    tests = [q.qid for f in folds for q in f.test]
    assert sorted(tests) == sorted(q.qid for q in qs)
    for f in folds:
        C.assert_no_leakage(f.train, f.test)
        assert {q.qid for q in f.train} | {q.qid for q in f.test} == {q.qid for q in qs}
    assert [f.test_groups for f in C.make_folds(qs, 3)] == [f.test_groups for f in folds]   # deterministic


def test_leakage_detector_fires():
    a, b = _q("x", "F1", ["c1"]), _q("y", "F9", ["c1"])      # different fact, same gold passage
    with pytest.raises(ValueError):
        C.assert_no_leakage([a], [b])
    with pytest.raises(ValueError):
        C.assert_no_leakage([a], [_q("z", "F1", ["c7"])])    # same fact, different language/row


@pytest.mark.skipif(not HAVE_REAL, reason="real project data not present")
def test_real_data_folds_have_no_leakage():
    sel, _ = C.load_reviewed_questions()
    for k in (2, 4, 5):
        for f in C.make_folds(sel, k):
            C.assert_no_leakage(f.train, f.test)
            assert f.test and f.train


# ----------------------------------------------------------------------------------------- batch layout
def test_batch_layout_masks_other_gold_but_keeps_unrelated_negatives():
    both = frozenset({"c1", "c2"})
    batch = [C.Pair("q1", "F1", "x", "c1", both), C.Pair("q1", "F1", "x", "c2", both),
             C.Pair("q2", "F2", "y", "c1", frozenset({"c1"}))]
    cols, target, mask = C.batch_layout(batch, ["c9", "c1"])      # c1 extra is a duplicate and must be deduped
    assert cols == ["c1", "c2", "c9"] and target == [0, 1, 0]
    assert mask == [[False, True, False], [True, False, False], [False, False, False]]


def test_negative_pool_excludes_all_gold_and_junk_chunks():
    ch = _chunks(6)
    ch["c5"]["text"] = "tiny"
    qs = [_q("a", "F1", ["c0"]), _q("b", "F2", ["c1"])]
    assert C.negative_pool(ch, qs) == ["c2", "c3", "c4"]


# ----------------------------------------------------------------------------------------- metrics
def test_multi_gold_metrics():
    ids = ["a", "b", "c", "d"]
    P = np.eye(4, dtype=np.float32)
    q = np.array([[0.9, 0.0, 0.8, 0.1]], dtype=np.float32)       # ranking: a, c, d, b
    row = C.score_questions("t", [_q("q", "F", ["c", "d"])], q, P, ids, {"q": "G"})[0]
    assert row["gold_best_rank"] == 2 and row["mrr"] == 0.5
    assert row["hit@1"] == 0.0 and row["hit@3"] == 1.0
    assert row["recall@1"] == 0.0 and row["recall@3"] == 1.0 and row["recall@5"] == 1.0
    assert row["top10_chunk_ids"] == ["a", "c", "d", "b"]


def test_score_questions_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        C.score_questions("t", [_q("q", "F", ["a"])], np.zeros((1, 3)), np.zeros((4, 4)), list("abcd"), {"q": "G"})


def test_summary_and_paired_difference_are_group_aware():
    def rows(vals):
        return [{"qid": f"q{i}", "group": f"G{i // 2}", "language": "English", **{m: v for m in C.METRICS}} for i, v in enumerate(vals)]
    a, b = rows([0, 0, 1, 1, 0, 0]), rows([1, 1, 1, 1, 0, 0])
    s = C.summarize(a)
    assert s["n_questions"] == 6 and s["n_groups"] == 3 and s["metrics"]["mrr"]["mean"] == round(2 / 6, 4)
    d = C.paired_difference(a, b)
    assert d["metrics"]["mrr"]["mean"] == round(2 / 6, 4)
    assert d["per_group_mrr"]["G0"] == {"n_questions": 2, "a_mrr": 0.0, "b_mrr": 1.0}


# ----------------------------------------------------------------------------------------- index checks
@pytest.mark.skipif(not (HAVE_REAL and (C.DEFAULT_PRETRAINED_INDEX / "dense_embeddings.npy").exists()), reason="pretrained index not present")
def test_pretrained_index_is_aligned_with_corpus():
    ids = list(C.load_chunks())
    emb, info = C.verify_index(C.DEFAULT_PRETRAINED_INDEX, ids, expected_model=C.BASE_MODEL)
    assert emb.shape == (len(ids), 768) and info["n_chunks"] == len(ids)
    with pytest.raises(ValueError):
        C.verify_index(C.DEFAULT_PRETRAINED_INDEX, ids[::-1])    # wrong order must be caught


# ----------------------------------------------------------------------------------------- CV driver (fake encoder)
def _hash_embed(texts, dim=64):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_cv_driver_trains_only_on_train_groups_and_scores_only_heldout():
    import evaluate_e5_retrieval as E

    chunks = {f"c{i}": {"chunk_id": f"c{i}", "text": f"topic{i} " * 20} for i in range(1, 7)}
    qs = [_q(f"q{i}-{l}", f"F{i}", [f"c{i}"], l) for i in range(1, 6) for l in ("English", "Hindi")]
    group_of = C.assign_groups(qs)
    folds = C.make_folds(qs, 3)
    ids = list(chunks)
    seen = []

    def fit(train_qs, fold_idx):
        seen.append((fold_idx, {q.qid for q in train_qs}))
        return E.FoldSystem(lambda texts: _hash_embed(texts), _hash_embed([chunks[c]["text"] for c in ids]))

    for q in qs:   # make the fake encoder able to match: question text must share the topic token
        object.__setattr__(q, "question", f"topic{q.fact_id[1:]} please")
    out = Path(tempfile.mkdtemp())
    rows = E.run_cv(qs, ids, folds, fit, group_of, out, "sig", resume=False)
    assert sorted(r["qid"] for r in rows) == sorted(q.qid for q in qs)
    assert all(r["system"] == "finetuned_cv_heldout" for r in rows)
    for (fi, train_ids), f in zip(seen, folds):
        assert train_ids == {q.qid for q in f.train} and not (train_ids & {q.qid for q in f.test})
    assert all(r["mrr"] == 1.0 for r in rows)   # plumbing check only (fake encoder), NOT a result
    # resume: cached folds are reused, fit is not called again
    before = len(seen)
    E.run_cv(qs, ids, folds, fit, group_of, out, "sig", resume=True)
    assert len(seen) == before


# ----------------------------------------------------------------------------------------- real training loop (needs torch)
def test_fine_tune_loop_reduces_loss_on_toy_model():
    torch = pytest.importorskip("torch")
    import train_e5_from_reviewed_data as T

    class _Auto:
        pass

    class TinyST(torch.nn.Module):
        """Minimal stand-in exposing the SentenceTransformer surface that fine_tune() uses."""
        def __init__(self, dim=16, vocab=512):
            super().__init__()
            self.vocab = vocab
            self.emb = torch.nn.Embedding(vocab, dim)
            self.lin = torch.nn.Linear(dim, dim)
            self.max_seq_length = 64
            first = _Auto()
            first.auto_model = _Auto()
            first.auto_model.embeddings = _Auto()
            first.auto_model.embeddings.word_embeddings = self.emb
            self._first = first

        def __getitem__(self, i):
            return self._first

        def tokenize(self, texts):
            ids = [[zlib.crc32(w.encode()) % self.vocab for w in t.split()][: self.max_seq_length] or [0] for t in texts]
            L = max(map(len, ids))
            return {"input_ids": torch.tensor([r + [0] * (L - len(r)) for r in ids]),
                    "attention_mask": torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in ids])}

        def forward(self, feats):
            e, m = self.emb(feats["input_ids"]), feats["attention_mask"].unsqueeze(-1).float()
            return {"sentence_embedding": self.lin((e * m).sum(1) / m.sum(1))}

    chunks = {f"c{i}": {"chunk_id": f"c{i}", "text": f"alpha{i} beta{i} gamma{i} shared words here"} for i in range(30)}
    qs = [_q(f"q{i}", f"F{i}", [f"c{i}"]) for i in range(6)]
    for q in qs:
        object.__setattr__(q, "question", f"alpha{q.fact_id[1:]} question")
    pairs = C.build_pairs(qs)
    pool = C.negative_pool(chunks, qs)
    cfg = {"device": "cpu", "seed": 1, "max_seq_length": 64, "freeze_embeddings": False, "batch_size": 3, "epochs": 25,
           "lr": 5e-3, "temperature": 0.05, "weight_decay": 0.0, "warmup_ratio": 0.1, "max_grad_norm": 1.0, "extra_negatives": 4}
    torch.manual_seed(0)
    hist = T.fine_tune(TinyST(), pairs, chunks, pool, cfg)
    assert len(hist) == 25 * 2 and all(np.isfinite(h["loss"]) for h in hist)
    assert np.mean([h["loss"] for h in hist[-6:]]) < np.mean([h["loss"] for h in hist[:6]])
