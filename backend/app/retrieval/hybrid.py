"""Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank)."""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(*rankings: list[dict[str, Any]], k: int = 60, top_k: int = 5) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}
    source_ranks: dict[str, dict[int, int]] = {}
    for n, ranking in enumerate(rankings):
        for pos, item in enumerate(ranking, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos)
            meta.setdefault(cid, {f: v for f, v in item.items() if f not in ("rank", "score")})
            source_ranks.setdefault(cid, {})[n] = pos
    order = sorted(scores, key=lambda c: (-scores[c], c))[: max(1, top_k)]
    return [{"rank": i, "rrf_score": round(scores[c], 8), "source_ranks": source_ranks[c], **meta[c]}
            for i, c in enumerate(order, start=1)]
