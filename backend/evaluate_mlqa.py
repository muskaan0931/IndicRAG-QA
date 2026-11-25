"""Passage-retrieval evaluation on MLQA (Lewis et al., 2020).

WHAT THIS MEASURES
  For every MLQA question, the retrieval corpus is the set of all *unique context paragraphs* of the
  chosen file (e.g. test-context-hi-question-hi.json). The gold passage is the paragraph the question
  was written for. We report Recall@K, MRR@10 and nDCG@10.

WHAT THIS DOES NOT MEASURE
  * Answer quality (Exact Match / F1): that needs a reader/generator, which this script does not run.
  * Retrieval over the Bennett University corpus: MLQA is Wikipedia text, used here only as an
    external multilingual benchmark.
  * With exactly one gold paragraph per question, Hit Rate@K equals Recall@K, and binary nDCG@K is
    1/log2(rank+1) when the gold paragraph is in the top K.
  * MLQA paragraphs are long and topically distinct; absolute numbers are NOT comparable with the
    small Bennett pilot.

Methods: bm25 (language-agnostic Unicode tokenizer, no stemming), dense (SentenceTransformer, e.g. E5),
hybrid (Reciprocal Rank Fusion of bm25 + dense). Dense/hybrid need sentence-transformers and a model.

Usage:
  python backend/evaluate_mlqa.py --split dev  --pairs hi-hi en-en hi-en en-hi --method bm25
  python backend/evaluate_mlqa.py --split test --pairs hi-hi --method dense --model intfloat/multilingual-e5-base
  python backend/evaluate_mlqa.py --split test --pairs hi-hi --method dense --model models/finetuned_e5
Pair naming is <context language>-<question language>, matching MLQA file names.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLQA_DIR = ROOT / "external_data" / "MLQA_V1" / "MLQA_V1"
KS = (1, 3, 5, 10)


def tokenize(text: str) -> list[str]:
    """Language-agnostic tokenizer: NFC, lowercase, keep runs of letters/marks/digits (so Devanagari
    vowel signs stay attached to their consonants). No stopword removal, no stemming."""
    text = unicodedata.normalize("NFC", text).lower()
    toks, cur = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            cur.append(ch)
        elif cur:
            toks.append("".join(cur)); cur = []
    if cur:
        toks.append("".join(cur))
    return toks


def load_mlqa(split: str, ctx_lang: str, q_lang: str):
    path = MLQA_DIR / split / f"{split}-context-{ctx_lang}-question-{q_lang}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    contexts, questions = [], []
    for art in data:
        for para in art["paragraphs"]:
            cid = len(contexts)
            contexts.append(para["context"])
            for qa in para["qas"]:
                questions.append({"id": qa["id"], "question": qa["question"], "gold": cid})
    return contexts, questions


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.dl = [len(t) for t in toks]
        self.avgdl = sum(self.dl) / max(len(self.dl), 1)
        self.post = defaultdict(list)
        for i, t in enumerate(toks):
            for term, f in Counter(t).items():
                self.post[term].append((i, f))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def rank(self, query: str, k: int) -> list[int]:
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.post[term]:
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        # deterministic tie-break on document index
        return [i for i, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def rrf(lists: list[list[int]], k=60, top=10) -> list[int]:
    sc = defaultdict(float)
    for lst in lists:
        for r, i in enumerate(lst, 1):
            sc[i] += 1.0 / (k + r)
    return [i for i, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:top]]


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"questions": n}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    out["mrr@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    out["ndcg@10"] = sum(1 / math.log2(r + 1) for r in ranks if r and r <= 10) / n
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--pairs", nargs="+", default=["en-en", "hi-hi", "hi-en", "en-hi"])
    p.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model = None
    if a.method in ("dense", "hybrid"):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model)
        model.max_seq_length = a.max_seq_length
        e5 = "e5" in a.model.lower() or "finetuned" in a.model.lower()

    results = []
    for pair in a.pairs:
        ctx_lang, q_lang = pair.split("-")
        contexts, questions = load_mlqa(a.split, ctx_lang, q_lang)
        t0 = time.time()
        bm = BM25(contexts) if a.method in ("bm25", "hybrid") else None
        if model is not None:
            emb = model.encode([("passage: " if e5 else "") + c for c in contexts], batch_size=a.batch_size,
                               convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            qe = model.encode([("query: " if e5 else "") + q["question"] for q in questions], batch_size=64,
                              convert_to_numpy=True, normalize_embeddings=True)
        ranks = []
        for qi, q in enumerate(questions):
            if a.method == "bm25":
                top = bm.rank(q["question"], 10)
            else:
                d_top = list(np.argsort(-(emb @ qe[qi]))[:50])
                top = d_top[:10] if a.method == "dense" else rrf([bm.rank(q["question"], 50), d_top], top=10)
            ranks.append(top.index(q["gold"]) + 1 if q["gold"] in top else None)
        res = {"split": a.split, "context_language": ctx_lang, "question_language": q_lang,
               "setting": "monolingual" if ctx_lang == q_lang else "cross-lingual",
               "method": a.method, "model": a.model if model else None,
               "corpus_passages": len(contexts), **metrics(ranks), "seconds": round(time.time() - t0, 1)}
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    out = Path(a.out) if a.out else ROOT / "reports" / f"mlqa_{a.method}_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "MLQA passage retrieval. Corpus = unique contexts of each file; gold = the question's own paragraph.",
        "not_measured": ["answer exact match / F1", "retrieval on the Bennett corpus"],
        "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
