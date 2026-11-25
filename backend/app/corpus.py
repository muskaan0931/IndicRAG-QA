"""Corpus access: documents, pages, chunks and quality information, all derived from the real data files."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Caveats that are stated in the project's own README / corpus_validation report, or that can be read
# directly from the extracted text. They are surfaced in the UI next to the evidence.
DOC_NOTES: dict[str, str] = {
    "studenthandbookbennett": "OCR-derived and noisy on some pages; treat as historical/contextual until manually verified (version 2019-20).",
    "ord_rulesandreg_ug_btechf": "B.Tech Ordinances dated 2016-17; must not silently override newer policies.",
    "exam_manual": "Version/year not confirmed from the cover (OCR of the cover is unreliable).",
    "notification_for_minimum_attendance_for_students": "Notification dated 8 September 2022 (AY 2022-23 onwards).",
    "scholarship_policy_2024_25_final": "Scholarship policy for AY 2024-25.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}: invalid JSON on line {n}: {exc}") from exc
    return rows


class Corpus:
    LOW_CONFIDENCE = 70.0

    def __init__(self, chunks_path: Path, pages_path: Path):
        self.chunks = read_jsonl(chunks_path)
        self.pages = read_jsonl(pages_path) if pages_path.exists() else []
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.page_by_key = {(p["doc_id"], p["page"]): p for p in self.pages}

    # ---- page / chunk level ------------------------------------------------------------
    def page_info(self, doc_id: str, page: int) -> dict[str, Any]:
        p = self.page_by_key.get((doc_id, page), {})
        return {
            "extraction_method": p.get("extraction_method"),
            "ocr_confidence": p.get("ocr_confidence"),
        }

    def enrich(self, item: dict[str, Any]) -> dict[str, Any]:
        """Add page-quality and source-version information to a retrieval result."""
        out = dict(item)
        chunk = self.chunk_by_id.get(item.get("chunk_id"), {})
        out["language"] = chunk.get("language")
        info = self.page_info(item.get("doc_id"), item.get("page"))
        out.update(info)
        warnings = []
        conf = info.get("ocr_confidence")
        if conf is not None and conf < self.LOW_CONFIDENCE:
            warnings.append(f"Low OCR confidence on this page ({conf:.1f}); text may contain errors.")
        note = DOC_NOTES.get(item.get("doc_id"))
        if note and item.get("doc_id") in {"studenthandbookbennett", "ord_rulesandreg_ug_btechf", "exam_manual"}:
            warnings.append(note)
        out["source_warnings"] = warnings
        return out

    # ---- document level ----------------------------------------------------------------
    def documents(self) -> list[dict[str, Any]]:
        by_doc_chunks: dict[str, list] = defaultdict(list)
        for c in self.chunks:
            by_doc_chunks[c["doc_id"]].append(c)
        by_doc_pages: dict[str, list] = defaultdict(list)
        for p in self.pages:
            by_doc_pages[p["doc_id"]].append(p)
        docs = []
        for doc_id, chunks in by_doc_chunks.items():
            pages = by_doc_pages.get(doc_id, [])
            conf = [p["ocr_confidence"] for p in pages if p.get("ocr_confidence") is not None]
            methods = sorted({p.get("extraction_method") for p in pages if p.get("extraction_method")})
            first = chunks[0]
            docs.append({
                "doc_id": doc_id,
                "document_name": first["document_name"],
                "category": first.get("category"),
                "version": first.get("version"),
                "language": first.get("language"),
                "pages": len(pages) or len({c["page"] for c in chunks}),
                "chunks": len(chunks),
                "extraction_methods": methods,
                "ocr_confidence_mean": round(statistics.mean(conf), 1) if conf else None,
                "ocr_confidence_min": round(min(conf), 1) if conf else None,
                "low_confidence_pages": sorted(
                    p["page"] for p in pages
                    if p.get("ocr_confidence") is not None and p["ocr_confidence"] < self.LOW_CONFIDENCE
                ),
                "note": DOC_NOTES.get(doc_id),
            })
        return sorted(docs, key=lambda d: d["document_name"].lower())

    def overview(self) -> dict[str, Any]:
        docs = self.documents()
        conf = [p["ocr_confidence"] for p in self.pages if p.get("ocr_confidence") is not None]
        lens = [len(c["text"]) for c in self.chunks]
        return {
            "documents": len(docs),
            "pages": len(self.pages),
            "chunks": len(self.chunks),
            "ocr_pages": sum(1 for p in self.pages if p.get("extraction_method") == "ocr"),
            "native_pages": sum(1 for p in self.pages if p.get("extraction_method") == "native"),
            "ocr_confidence_mean": round(statistics.mean(conf), 1) if conf else None,
            "low_confidence_pages": sum(1 for c in conf if c < self.LOW_CONFIDENCE),
            "chunk_chars_median": int(statistics.median(lens)) if lens else 0,
            "chunk_chars_max": max(lens) if lens else 0,
            "categories": sorted({d["category"] for d in docs if d["category"]}),
        }

    def search_chunks(self, q: str = "", doc_id: str | None = None, page: int | None = None,
                      limit: int = 20, offset: int = 0) -> dict[str, Any]:
        q_norm = q.strip().lower()
        rows = [
            c for c in self.chunks
            if (not doc_id or c["doc_id"] == doc_id)
            and (page is None or c["page"] == page)
            and (not q_norm or q_norm in c["text"].lower())
        ]
        total = len(rows)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "chunks": [self.enrich({k: c.get(k) for k in
                       ("chunk_id", "doc_id", "document_name", "category", "version", "page", "text")})
                       for c in rows[offset: offset + limit]],
        }
