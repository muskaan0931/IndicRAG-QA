"""Self-contained BM25 baseline (Lucene-style IDF, k1=1.5, b=0.75, deterministic tie-breaks).

The tokenizer keeps only [a-z0-9]+ tokens, as in the original baseline: a Devanagari query produces ZERO tokens and
therefore zero results, and a Hinglish query only matches its English words. That is correct baseline behaviour
(the corpus is English) and is itself a finding for the cross-lingual comparison.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESULT_FIELDS = ["chunk_id", "doc_id", "document_name", "category", "version", "page", "text"]
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{Path(path).name}: invalid JSON on line {n}: {exc}") from exc
    return rows


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        toks = [tokenize(c.get("text", "")) for c in chunks]
        self.dl = [len(t) for t in toks]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for term, f in Counter(t).items():
                self.postings[term].append((i, f))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.postings[term]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[: max(1, top_k)]
        return [{"rank": r, "score": round(s, 6), **{k: self.chunks[i].get(k) for k in RESULT_FIELDS}}
                for r, (i, s) in enumerate(ranked, start=1)]
