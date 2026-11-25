"""Torch-free helpers shared by the E5 fine-tuning, index-building and retrieval-evaluation scripts.

Everything here is plain Python / NumPy so that data selection, leakage-safe splitting, batch layout and the
retrieval metrics can be unit-tested without PyTorch or a model download.

LABEL PROVENANCE: the reviewer_ok=Y/N values in data/eval/questions_for_review.csv are ASSISTANT-ASSISTED
screening decisions. They are not independent human verification and must never be described as such.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
from app.evalkit import first_relevant_rank, retrieval_row  # noqa: E402  (project's own metric definitions)

BASE_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "      # E5 convention (model card); identical to app/retrieval/dense.py
PASSAGE_PREFIX = "passage: "

DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_REVIEW_CSV = ROOT / "data" / "eval" / "questions_for_review.csv"
DEFAULT_PRETRAINED_INDEX = ROOT / "phase4_outputs" / "dense_index"
DEFAULT_FT_MODEL = ROOT / "models" / "finetuned_e5"
DEFAULT_FT_INDEX = ROOT / "models" / "finetuned_e5_index"
DEFAULT_EVAL_OUT = ROOT / "reports" / "e5_finetune_eval"

LABEL_PROVENANCE = (
    "reviewer_ok=Y/N decisions are ASSISTANT-ASSISTED screening of AI-drafted questions. They are NOT "
    "independent human verification. Unreviewed rows (blank reviewer_ok) are never treated as accepted."
)

METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]


# --------------------------------------------------------------------------------------- paths / io
def resolve_path(p: str | Path) -> Path:
    """Relative paths are resolved against the PROJECT ROOT, never against the current working directory."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def load_chunks(path: str | Path = DEFAULT_CHUNKS) -> dict[str, dict[str, Any]]:
    """chunk_id -> chunk dict, in file order (which is the order every dense index must follow)."""
    chunks: dict[str, dict[str, Any]] = {}
    with open(resolve_path(path), encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["chunk_id"] in chunks:
                raise ValueError(f"Duplicate chunk_id in corpus: {c['chunk_id']}")
            chunks[c["chunk_id"]] = c
    if not chunks:
        raise ValueError("Corpus is empty.")
    return chunks


# --------------------------------------------------------------------------------------- reviewed data
@dataclass(frozen=True)
class ReviewedQuestion:
    qid: str
    fact_id: str
    language: str
    question: str
    gold: tuple[str, ...]
    answer_text: str
    split: str


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def load_reviewed_questions(
    csv_path: str | Path = DEFAULT_REVIEW_CSV, chunks: dict[str, dict] | None = None
) -> tuple[list[ReviewedQuestion], dict[str, Any]]:
    """Select ONLY rows with answerable=true AND reviewer_ok=Y AND every gold chunk id present in the corpus.

    Nothing is repaired silently: a row whose gold list contains an unknown chunk id is EXCLUDED (and reported),
    not trimmed. Rows marked N, blank (unreviewed) or any other value are never selected.
    """
    chunks = chunks if chunks is not None else load_chunks()
    csv_path = resolve_path(csv_path)
    rep: dict[str, Any] = {
        "csv": str(csv_path), "rows_total": 0, "answerable_rows": 0, "not_answerable_rows": 0,
        "reviewer_Y": 0, "reviewer_N": 0, "reviewer_blank": 0, "reviewer_other": 0,
        "answerable_and_Y": 0, "excluded_after_Y": [], "label_provenance": LABEL_PROVENANCE,
    }
    selected: list[ReviewedQuestion] = []
    seen: set[str] = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rep["rows_total"] += 1
            qid = (row.get("qid") or "").strip()
            if qid in seen:
                raise ValueError(f"Duplicate qid in review CSV: {qid}")
            seen.add(qid)
            ok = (row.get("reviewer_ok") or "").strip().upper()
            answerable = _truthy(row.get("answerable", ""))
            rep["answerable_rows" if answerable else "not_answerable_rows"] += 1
            key = {"Y": "reviewer_Y", "N": "reviewer_N", "": "reviewer_blank"}.get(ok, "reviewer_other")
            rep[key] += 1
            if not answerable or ok != "Y":
                continue
            rep["answerable_and_Y"] += 1
            question = (row.get("question") or "").strip()
            gold = tuple(dict.fromkeys(x.strip() for x in (row.get("gold_chunk_ids") or "").split(";") if x.strip()))
            reason = detail = None
            if not question:
                reason = "empty_question"
            elif not gold:
                reason = "no_gold_chunk_ids"
            else:
                missing = [g for g in gold if g not in chunks]
                empty = [g for g in gold if g in chunks and not (chunks[g].get("text") or "").strip()]
                if missing:
                    reason, detail = "gold_chunk_id_not_in_corpus", missing
                elif empty:
                    reason, detail = "gold_chunk_text_empty", empty
            if reason:
                rep["excluded_after_Y"].append({"qid": qid, "fact_id": row.get("fact_id"), "reason": reason, "detail": detail})
                continue
            selected.append(ReviewedQuestion(
                qid=qid, fact_id=(row.get("fact_id") or qid).strip(), language=(row.get("language") or "").strip(),
                question=question, gold=gold, answer_text=(row.get("answer_text") or "").strip(),
                split=(row.get("split") or "").strip()))
    rep["eligible_questions"] = len(selected)
    rep["eligible_facts"] = len({q.fact_id for q in selected})
    rep["eligible_by_language"] = dict(sorted(_count(q.language for q in selected).items()))
    return selected, rep


def _count(it: Iterable[Any]) -> dict[Any, int]:
    d: dict[Any, int] = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


# --------------------------------------------------------------------------------------- grouping / folds
def assign_groups(questions: Sequence[ReviewedQuestion]) -> dict[str, str]:
    """qid -> group id. Questions are in the same group when they share a fact_id OR any gold chunk
    (connected components). Different languages of one fact, and different facts answered by the same passage,
    are therefore never separated across train / held-out. Splitting any finer would leak."""
    parent: dict[Any, Any] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for q in questions:
        for g in q.gold:
            union(("fact", q.fact_id), ("chunk", g))
        find(("fact", q.fact_id))
    members: dict[Any, set[str]] = defaultdict(set)
    for q in questions:
        members[find(("fact", q.fact_id))].add(q.fact_id)
    name = {root: "+".join(sorted(facts)) for root, facts in members.items()}
    return {q.qid: name[find(("fact", q.fact_id))] for q in questions}


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[ReviewedQuestion, ...]
    test: tuple[ReviewedQuestion, ...]
    test_groups: tuple[str, ...]


def make_folds(questions: Sequence[ReviewedQuestion], k: int) -> list[Fold]:
    """Deterministic grouped K-fold (no randomness): whole groups go to the currently lightest fold."""
    gid = assign_groups(questions)
    groups: dict[str, list[ReviewedQuestion]] = defaultdict(list)
    for q in questions:
        groups[gid[q.qid]].append(q)
    if len(groups) < 2:
        raise ValueError("Need at least 2 independent groups for a held-out evaluation.")
    k = max(2, min(k, len(groups)))
    folds: list[list[str]] = [[] for _ in range(k)]
    loads = [0] * k
    for g, qs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        i = min(range(k), key=lambda j: (loads[j], j))
        folds[i].append(g)
        loads[i] += len(qs)
    out = []
    for i, gs in enumerate(folds):
        test = tuple(sorted((q for g in gs for q in groups[g]), key=lambda q: q.qid))
        test_ids = {q.qid for q in test}
        train = tuple(q for q in questions if q.qid not in test_ids)
        out.append(Fold(i, train, test, tuple(sorted(gs))))
    for f in out:
        assert_no_leakage(f.train, f.test)
    return out


def assert_no_leakage(train: Sequence[ReviewedQuestion], test: Sequence[ReviewedQuestion]) -> None:
    """Raise if train and held-out share a question, fact, or gold passage."""
    for what, a, b in (
        ("qid", {q.qid for q in train}, {q.qid for q in test}),
        ("fact_id", {q.fact_id for q in train}, {q.fact_id for q in test}),
        ("question text", {q.question for q in train}, {q.question for q in test}),
        ("gold chunk", {g for q in train for g in q.gold}, {g for q in test for g in q.gold}),
    ):
        if a & b:
            raise ValueError(f"Train/held-out leakage via {what}: {sorted(a & b)[:5]}")


# --------------------------------------------------------------------------------------- training layout
@dataclass(frozen=True)
class Pair:
    qid: str
    fact_id: str
    question: str
    positive_id: str
    gold: frozenset


def build_pairs(questions: Sequence[ReviewedQuestion]) -> list[Pair]:
    """One (query, positive passage) pair per gold chunk. `gold` keeps the FULL gold set so the loss never
    treats another gold passage of the same question as a negative."""
    return [Pair(q.qid, q.fact_id, q.question, g, frozenset(q.gold)) for q in questions for g in q.gold]


def negative_pool(chunks: dict[str, dict], eligible: Sequence[ReviewedQuestion], min_words: int = 10) -> list[str]:
    """Extra random negatives are drawn ONLY from chunks that are gold for no eligible question (train or
    held-out alike), so held-out gold passages are never used as negatives and CV stays clean."""
    gold = {g for q in eligible for g in q.gold}
    return [cid for cid, c in chunks.items() if cid not in gold and len((c.get("text") or "").split()) >= min_words]


def batch_layout(batch: Sequence[Pair], extra_ids: Sequence[str]) -> tuple[list[str], list[int], list[list[bool]]]:
    """Columns = unique positive passages of the batch + extra negatives (deduplicated).
    Returns (column_chunk_ids, target_column_per_row, mask[row][col]) where mask=True means the column is a
    gold passage of that row's query other than its target, i.e. must be excluded from the softmax."""
    cols: list[str] = []
    index: dict[str, int] = {}
    for p in batch:
        if p.positive_id not in index:
            index[p.positive_id] = len(cols)
            cols.append(p.positive_id)
    for e in extra_ids:
        if e not in index:
            index[e] = len(cols)
            cols.append(e)
    target = [index[p.positive_id] for p in batch]
    mask = [[(cols[j] in p.gold and j != target[i]) for j in range(len(cols))] for i, p in enumerate(batch)]
    return cols, target, mask


# --------------------------------------------------------------------------------------- shared CLI args
def add_training_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--review-csv", default=str(DEFAULT_REVIEW_CSV))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.02,
                    help="InfoNCE temperature. E5 cosine scores are compressed (~0.7-1.0); E5 was pretrained at 0.01.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--extra-negatives", type=int, default=8,
                    help="random unlabeled corpus chunks added as (gradient-free) negatives to every batch")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True,
                    help="freeze the 250k-token word-embedding matrix (saves ~3 GB of optimizer state)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="default cpu; MPS is never selected automatically")


TRAIN_KEYS = ["base_model", "epochs", "batch_size", "lr", "temperature", "weight_decay", "warmup_ratio",
              "max_grad_norm", "extra_negatives", "max_seq_length", "freeze_embeddings", "seed", "device"]


def train_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {k: getattr(args, k) for k in TRAIN_KEYS}


# --------------------------------------------------------------------------------------- index checks
def verify_index(index_dir: str | Path, chunk_ids: Sequence[str], expected_model: str | None = None
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """Check a persisted dense index is aligned with the corpus (count, ids, order), finite and L2-normalised."""
    d = resolve_path(index_dir)
    emb_p, ch_p, model_p = d / "dense_embeddings.npy", d / "dense_chunks.jsonl", d / "dense_model.txt"
    for p in (emb_p, ch_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
    emb = np.load(emb_p).astype("float32")
    with open(ch_p, encoding="utf-8") as f:
        ids = [json.loads(line)["chunk_id"] for line in f if line.strip()]
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        raise ValueError(f"{d}: embedding rows {emb.shape} != chunk records {len(ids)}")
    if ids != list(chunk_ids):
        raise ValueError(f"{d}: chunk ids/order differ from data/chunks.jsonl")
    if not np.isfinite(emb).all():
        raise ValueError(f"{d}: non-finite embeddings")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{d}: embeddings are not L2-normalised (norm range {norms.min():.4f}-{norms.max():.4f})")
    model = model_p.read_text(encoding="utf-8").strip() if model_p.exists() else None
    if expected_model is not None and model != expected_model:
        raise ValueError(f"{d}: dense_model.txt says {model!r}, expected {expected_model!r}")
    return emb, {"index_dir": str(d), "n_chunks": len(ids), "dim": int(emb.shape[1]), "model": model,
                 "embeddings_sha256": sha256_file(emb_p)}


# --------------------------------------------------------------------------------------- retrieval scoring
def score_questions(system: str, questions: Sequence[ReviewedQuestion], query_emb: np.ndarray,
                    passage_emb: np.ndarray, chunk_ids: Sequence[str], group_of: dict[str, str],
                    fold_of: dict[str, int] | None = None, top_k: int = 10) -> list[dict[str, Any]]:
    """Rank ALL corpus chunks per question by cosine and compute metrics against the FULL gold set.

    hit@k    = 1 if any gold chunk is in the top k
    recall@k = |gold in top k| / |gold|        (fractional when a question has several gold chunks)
    mrr      = 1 / rank of the best-ranked gold chunk, over the full ranking (not truncated at 10)
    """
    q = np.asarray(query_emb, dtype=np.float64)
    p = np.asarray(passage_emb, dtype=np.float64)
    if q.shape[0] != len(questions) or q.shape[1] != p.shape[1] or p.shape[0] != len(chunk_ids):
        raise ValueError(f"Shape mismatch: queries {q.shape}, passages {p.shape}, chunks {len(chunk_ids)}")
    scores = q @ p.T
    rows = []
    for i, qq in enumerate(questions):
        order = np.argsort(-scores[i], kind="stable")
        ranked = [chunk_ids[j] for j in order]
        m = retrieval_row(ranked, list(qq.gold))
        rows.append({
            "system": system, "qid": qq.qid, "fact_id": qq.fact_id, "group": group_of[qq.qid],
            "language": qq.language, "split_original": qq.split,
            "fold": None if fold_of is None else fold_of.get(qq.qid),
            "question": qq.question, "gold_chunk_ids": list(qq.gold),
            "gold_best_rank": first_relevant_rank(ranked, qq.gold),
            "top10_chunk_ids": ranked[:top_k], "top10_scores": [round(float(scores[i, j]), 6) for j in order[:top_k]],
            **{k: round(float(m[k]), 6) for k in METRICS},
        })
    return rows


# --------------------------------------------------------------------------------------- statistics
def summarize(rows: Sequence[dict[str, Any]], n_boot: int = 2000, seed: int = 13) -> dict[str, Any]:
    """Means plus a GROUP-level cluster bootstrap (resampling whole groups). Languages of one fact and
    facts that share a passage are not independent, so resampling single questions would be over-confident.
    With only a handful of groups the interval is extremely wide - that is the honest answer."""
    out: dict[str, Any] = {"n_questions": len(rows), "n_groups": len({r["group"] for r in rows}), "metrics": {}}
    if not rows:
        return out
    gids = sorted({r["group"] for r in rows})
    gi = np.array([gids.index(r["group"]) for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gids), size=(n_boot, len(gids)))
    cnt = np.bincount(gi, minlength=len(gids)).astype(float)
    for m in METRICS:
        v = np.array([r[m] for r in rows], dtype=float)
        s = np.bincount(gi, weights=v, minlength=len(gids))
        boot = s[draws].sum(1) / cnt[draws].sum(1)
        out["metrics"][m] = {"mean": round(float(v.mean()), 4), "ci95_lo": round(float(np.percentile(boot, 2.5)), 4),
                             "ci95_hi": round(float(np.percentile(boot, 97.5)), 4)}
    return out


def paired_difference(a_rows: Sequence[dict], b_rows: Sequence[dict], n_boot: int = 2000, seed: int = 13) -> dict[str, Any]:
    """b minus a, per question (same qids required), with the same group-level bootstrap."""
    amap = {r["qid"]: r for r in a_rows}
    bmap = {r["qid"]: r for r in b_rows}
    if set(amap) != set(bmap):
        raise ValueError("Paired comparison needs identical question sets.")
    qids = sorted(amap)
    diff_rows = [{"group": amap[q]["group"], **{m: bmap[q][m] - amap[q][m] for m in METRICS}} for q in qids]
    res = summarize(diff_rows, n_boot, seed)
    res["per_group_mrr"] = {}
    for g in sorted({r["group"] for r in diff_rows}):
        qs = [q for q in qids if amap[q]["group"] == g]
        res["per_group_mrr"][g] = {
            "n_questions": len(qs),
            "a_mrr": round(float(np.mean([amap[q]["mrr"] for q in qs])), 4),
            "b_mrr": round(float(np.mean([bmap[q]["mrr"] for q in qs])), 4)}
    return res


def by_language(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {lang: summarize([r for r in rows if r["language"] == lang]) for lang in sorted({r["language"] for r in rows})}
