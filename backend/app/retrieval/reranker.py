"""Optional cross-encoder reranker. No model is downloaded unless explicitly enabled.
Set INDICRAG_RERANKER_MODEL (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) and install sentence-transformers.
"""
import os
from typing import Any

class CrossEncoderReranker:
    def __init__(self, model_name=None):
        self.model_name=model_name or os.getenv("INDICRAG_RERANKER_MODEL")
        if not self.model_name: raise RuntimeError("INDICRAG_RERANKER_MODEL is not set")
        try: from sentence_transformers import CrossEncoder
        except ImportError as e: raise RuntimeError("Install sentence-transformers to use reranking") from e
        self.model=CrossEncoder(self.model_name)
    def rerank(self, query: str, results: list[dict[str,Any]], top_k: int|None=None):
        if not results:return results
        scores=self.model.predict([[query,r.get("text","")] for r in results])
        ranked=[]
        for r,score in zip(results,scores):
            item=dict(r);item["reranker_score"]=float(score);ranked.append(item)
        ranked.sort(key=lambda x:-x["reranker_score"])
        for i,r in enumerate(ranked,1):r["rank"]=i
        return ranked[:top_k or len(ranked)]
