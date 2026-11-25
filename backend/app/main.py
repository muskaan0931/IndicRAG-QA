"""IndicRAG-QA API.

Evidence-grounded retrieval over the Bennett University document corpus.
/ask retrieves evidence, applies a calibrated answerability gate, and (unless generator="none") returns a short
answer with [n] citations, or the fixed abstention message when the evidence does not support an answer.
"""
from __future__ import annotations

import importlib.util
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app import answerability, config, reports
from app.generation import ABSTAIN_MESSAGE, ExtractiveGenerator, LLMGenerator
from app.corpus import Corpus
from app.language import detect_language
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.query_expansion import expand_query
from app.retrieval.conflicts import annotate_conflicts

Method = Literal["bm25", "pretrained", "hybrid"]
ALL_METHODS: list[str] = ["bm25", "pretrained", "hybrid"]
METHOD_LABELS = {
    "bm25": "BM25",
    "pretrained": "Pretrained multilingual E5",
    "hybrid": "Hybrid (BM25 + pretrained E5, RRF)",
}
SCORE_TYPE = {"bm25": "bm25", "pretrained": "cosine", "hybrid": "rrf"}
RETRY_AFTER_SECONDS = 30


class MethodUnavailable(Exception):
    """A retrieval method cannot be used because a model or index artifact is missing."""


def _st_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _is_valid_model_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "modules.json").is_file() or (path / "config.json").is_file())


class Engine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.corpus: Corpus | None = None
        self.bm25: BM25Index | None = None
        self.pretrained = None
        self.finetuned = None
        self.errors: dict[str, dict[str, Any]] = {}
        self.reranker = None

    # ---- core corpus + BM25 -----------------------------------------------------------
    def ensure_core(self) -> None:
        with self.lock:
            if self.bm25 is not None:
                return
            if not config.CHUNKS_PATH.exists():
                raise RuntimeError(f"Corpus file missing: {config.CHUNKS_PATH}")
            corpus = Corpus(config.CHUNKS_PATH, config.PAGES_PATH)
            if len(corpus.chunk_by_id) != len(corpus.chunks):
                raise RuntimeError("Duplicate chunk_id values in chunks.jsonl")
            self.bm25 = BM25Index(corpus.chunks)
            self.corpus = corpus

    # ---- dense models (lazy) ------------------------------------------------------------
    def _recent_error(self, name: str) -> str | None:
        e = self.errors.get(name)
        if e and time.time() - e["at"] < RETRY_AFTER_SECONDS:
            return e["message"]
        return None

    def _fail(self, name: str, message: str):
        self.errors[name] = {"message": message, "at": time.time()}
        raise MethodUnavailable(message)

    def get_pretrained(self):
        with self.lock:
            if self.pretrained is not None:
                return self.pretrained
            if (msg := self._recent_error("pretrained")):
                raise MethodUnavailable(msg)
            self.ensure_core()
            if not _st_installed():
                self._fail("pretrained", "sentence-transformers is not installed (pip install sentence-transformers).")
            index_dir = config.PRETRAINED_INDEX_DIR
            try:
                from app.retrieval.dense import DenseRetriever
                r = DenseRetriever(config.PRETRAINED_MODEL_NAME, device="cpu")
                ids = [c["chunk_id"] for c in self.corpus.chunks]
                try:
                    r.load(index_dir, expected_chunk_ids=ids)
                except (FileNotFoundError, ValueError):
                    # Index missing or stale (built from a different corpus): rebuild once and persist it.
                    r.build(self.corpus.chunks, batch_size=8, show_progress=False)
                    r.save(index_dir)
            except Exception as exc:  # model download failure, ...
                self._fail("pretrained", f"{type(exc).__name__}: {exc}")
            self.errors.pop("pretrained", None)
            self.pretrained = r
            return r

    def get_finetuned(self):
        raise MethodUnavailable("Fine-tuned E5 is planned as future research work on an independently validated dataset.")

    # ---- availability -------------------------------------------------------------------
    def availability(self) -> dict[str, dict[str, Any]]:
        st = _st_installed()
        pre_idx = True  # built automatically on first use if absent

        def pack(ok: bool, reason: str | None, loaded: bool) -> dict[str, Any]:
            return {"available": ok, "reason": None if ok else reason, "loaded": loaded}

        pre_reason = (None if (st and pre_idx) else
                      "sentence-transformers not installed" if not st else
                      f"index not found at {config.PRETRAINED_INDEX_DIR}")
        out = {
            "bm25": pack(self.bm25 is not None, "corpus not loaded", self.bm25 is not None),
            "pretrained": pack(st and pre_idx, pre_reason, self.pretrained is not None),
        }
        out["hybrid"] = pack(out["bm25"]["available"] and out["pretrained"]["available"],
                             pre_reason or "BM25 unavailable", self.pretrained is not None)
        for k, v in out.items():
            v["label"] = METHOD_LABELS[k]
            v["last_error"] = (self.errors.get(k) or {}).get("message")
        return out

    # ---- retrieval ----------------------------------------------------------------------
    def retrieve(self, query: str, method: str, top_k: int) -> list[dict[str, Any]]:
        self.ensure_core()
        assert self.bm25 is not None and self.corpus is not None
        expanded = expand_query(query)
        if method == "bm25":
            results = self.bm25.search(expanded, top_k)
        elif method == "pretrained":
            results = self.get_pretrained().search(query, top_k)
        elif method == "finetuned":
            results = self.get_finetuned().search(query, top_k)
        elif method == "hybrid":
            dense = self.get_pretrained()
            pool = max(20, top_k * 4)
            b = self.bm25.search(expanded, pool)
            d = dense.search(query, pool)
            fused = reciprocal_rank_fusion(b, d, top_k=top_k)
            b_rank = {x["chunk_id"]: x["rank"] for x in b}
            d_rank = {x["chunk_id"]: x["rank"] for x in d}
            b_score = {x["chunk_id"]: x["score"] for x in b}
            d_score = {x["chunk_id"]: x["score"] for x in d}
            results = []
            for item in fused:
                item = dict(item)
                item["score"] = item["rrf_score"]
                item["bm25_rank"] = b_rank.get(item["chunk_id"])
                item["dense_rank"] = d_rank.get(item["chunk_id"])
                item["bm25_score"] = b_score.get(item["chunk_id"])
                item["dense_score"] = d_score.get(item["chunk_id"])
                item.pop("source_ranks", None)
                results.append(item)
            if d and results:  # strongest dense cosine of the pool, for the answerability gate
                results[0]["dense_top1"] = d[0]["score"]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        results = annotate_conflicts(results)
        rerank_model = os.environ.get("INDICRAG_RERANKER_MODEL")
        if rerank_model and results:
            try:
                if self.reranker is None:
                    from app.retrieval.reranker import CrossEncoderReranker
                    self.reranker = CrossEncoderReranker(rerank_model)
                results = self.reranker.rerank(query, results, top_k)
            except Exception as exc:
                self.errors["reranker"] = {"message": str(exc), "at": time.time()}
                for item in results: item["reranker_fallback"] = str(exc)
        return [self.corpus.enrich(r) for r in results]


engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        engine.ensure_core()
    except Exception as exc:  # keep the API up; /health reports the problem
        engine.errors["corpus"] = {"message": str(exc), "at": time.time()}
    yield


app = FastAPI(
    title="IndicRAG-QA API",
    description="Evidence-grounded cross-lingual QA for institutional documents: retrieval, answerability gate, cited answers.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class RetrieveRequest(SearchRequest):
    method: Method = "hybrid"


class AskRequest(RetrieveRequest):
    # auto = LLM when ANTHROPIC_API_KEY is set, otherwise the offline extractive generator
    generator: Literal["auto", "extractive", "llm", "none"] = "auto"


class CompareRequest(SearchRequest):
    methods: list[Method] = Field(default_factory=lambda: list(ALL_METHODS))


def _run(query: str, method: str, top_k: int):
    """Retrieve and translate internal errors into HTTP errors.

    Hybrid degrades to BM25-only (flagged on every result as hybrid_fallback=True) when the dense side cannot
    load, so the app keeps working. /compare and the evaluation scripts never use this fallback.
    """
    try:
        try:
            return engine.retrieve(query, method, top_k)
        except MethodUnavailable as exc:
            if method != "hybrid":
                raise
            results = engine.retrieve(query, "bm25", top_k)
            for r in results:
                r["hybrid_fallback"] = True
                r["hybrid_fallback_reason"] = str(exc)
            return results
    except MethodUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"{METHOD_LABELS[method]} is unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _need_corpus() -> Corpus:
    try:
        engine.ensure_core()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    assert engine.corpus is not None
    return engine.corpus


# --------------------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "IndicRAG-QA", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        engine.ensure_core()
        corpus_status, chunk_count = "loaded", len(engine.corpus.chunks)  # type: ignore[union-attr]
    except Exception as exc:
        corpus_status, chunk_count = str(exc), 0
    av = engine.availability()
    return {
        "status": "running",
        "corpus_status": corpus_status,
        "chunk_count": chunk_count,
        "methods": av,
        "pretrained_dense_loaded": av["pretrained"]["loaded"],
        "load_errors": {k: v["message"] for k, v in engine.errors.items()},
    }


@app.get("/methods")
def methods():
    engine.ensure_core() if engine.bm25 is None else None
    return {"methods": engine.availability()}


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------
@app.get("/overview")
def overview():
    corpus = _need_corpus()
    return {
        "corpus": corpus.overview(),
        "methods": engine.availability(),
        "artifacts": {
            "pretrained_index_present": (config.PRETRAINED_INDEX_DIR / "dense_embeddings.npy").exists(),
            "finetuned_model_present": False,
            "bm25_pilot_present": (config.REPORTS_DIR / "bm25_evaluation_summary.json").exists(),
            "mlqa_results_present": bool(reports.mlqa_results(config.REPORTS_DIR)),
            "answer_generation": True,
            "answerability_calibrated": bool(answerability.load_thresholds()),
            "eval_dataset_present": (config.DATA_DIR / "eval" / "questions.jsonl").exists(),
        },
    }


@app.get("/documents")
def documents():
    corpus = _need_corpus()
    return {"documents": corpus.documents(), "overview": corpus.overview()}


@app.get("/documents/{doc_id}")
def document_detail(doc_id: str, page: int | None = None):
    corpus = _need_corpus()
    doc = next((d for d in corpus.documents() if d["doc_id"] == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown document '{doc_id}'")
    return {"document": doc, **corpus.search_chunks(doc_id=doc_id, page=page, limit=200)}


@app.get("/chunks")
def chunks(q: str = "", doc_id: str | None = None, page: int | None = None,
           limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return _need_corpus().search_chunks(q=q, doc_id=doc_id, page=page, limit=limit, offset=offset)


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------
@app.post("/search")
def search(request: SearchRequest):
    """Backward-compatible BM25 endpoint."""
    t0 = time.perf_counter()
    results = _run(request.query, "bm25", request.top_k)
    return {"query": request.query, "retrieval_method": "BM25", "results": results,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    t0 = time.perf_counter()
    results = _run(request.query, request.method, request.top_k)
    return {
        "query": request.query,
        "retrieval_method": request.method,
        "method_label": METHOD_LABELS[request.method],
        "score_type": SCORE_TYPE[request.method],
        "top_k": request.top_k,
        "detected_language": detect_language(request.query),
        "retrieval_status": "ok" if results else "no_results",
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/compare")
def compare(request: CompareRequest):
    _need_corpus()
    out: dict[str, Any] = {}
    for m in dict.fromkeys(request.methods):  # de-duplicate, keep order
        t0 = time.perf_counter()
        st = SCORE_TYPE.get(m, "score")
        lbl = METHOD_LABELS.get(m, m.title())
        try:
            res = engine.retrieve(request.query, m, request.top_k)
            out[m] = {"status": "ok" if res else "no_results", "results": res,
                      "score_type": st, "message": None if res else "No passages matched."}
        except MethodUnavailable as exc:
            out[m] = {"status": "unavailable", "results": [], "score_type": st, "message": str(exc)}
        except Exception as exc:
            out[m] = {"status": "error", "results": [], "score_type": st,
                      "message": f"{type(exc).__name__}: {exc}"}
        out[m]["label"] = lbl
        out[m]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    overlap: dict[str, dict[str, Any]] = {}
    for m, block in out.items():
        for item in block["results"]:
            e = overlap.setdefault(item["chunk_id"], {
                "chunk_id": item["chunk_id"], "document_name": item["document_name"], "page": item["page"], "ranks": {}})
            e["ranks"][m] = item["rank"]
    shared = sorted(overlap.values(), key=lambda e: (-len(e["ranks"]), min(e["ranks"].values())))
    return {
        "query": request.query, "top_k": request.top_k,
        "detected_language": detect_language(request.query),
        "methods": out, "overlap": shared,
    }


def _sentence_scorer():
    """Cosine scorer over sentences using the already-loaded pretrained E5 model, or None if unavailable."""
    try:
        dense = engine.get_pretrained()
    except MethodUnavailable:
        return None

    def score(query: str, sentences: list[str]) -> list[float]:
        q = dense.model.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)[0]
        e = dense.model.encode(["passage: " + s for s in sentences], convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in e @ q]
    return score


def _resolve_generator(choice: str) -> str:
    if choice == "auto":
        return "llm" if LLMGenerator.available() else "extractive"
    return choice


@app.get("/generators")
def generators():
    llm = LLMGenerator.available()
    return {
        "default": _resolve_generator("auto"),
        "generators": {
            "none": {"available": True, "label": "Evidence only"},
            "extractive": {"available": True, "label": "Extractive (offline)",
                           "note": "Copies the best-matching sentence; cannot paraphrase or combine facts."},
            "llm": {"available": llm, "label": "LLM (Claude)",
                    "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                    "reason": None if llm else "Set the ANTHROPIC_API_KEY environment variable and restart the server."},
        },
    }


def generate_answer(query: str, method: str, top_k: int, generator: str) -> dict[str, Any]:
    """Retrieve -> answerability gate -> generate -> verify. Never returns an uncited answer."""
    evidence = _run(query, method, top_k)
    lang = detect_language(query)
    gate = answerability.assess(method, evidence)
    out: dict[str, Any] = {
        "evidence": evidence, "answerability": gate, "answer": None, "answer_mode": "evidence_only",
        "generator": None, "abstained": False, "cited_passages": [], "answer_warnings": [], "generation_error": None,
    }
    if not evidence:
        out.update(answer=ABSTAIN_MESSAGE, abstained=True, answer_mode="abstained", generator="retrieval")
        return out
    if gate["status"] == "unanswerable":
        out.update(answer=ABSTAIN_MESSAGE, abstained=True, answer_mode="abstained", generator="answerability-gate")
        return out
    choice = _resolve_generator(generator)
    if choice == "none":
        return out
    try:
        if choice == "llm":
            if not LLMGenerator.available():
                raise RuntimeError("LLM generator needs the ANTHROPIC_API_KEY environment variable.")
            gen = LLMGenerator().generate(query, evidence)
        else:
            scorer = _sentence_scorer()
            gen = ExtractiveGenerator(sentence_scorer=scorer).generate(query, evidence)
    except Exception as exc:  # network/API errors must not hide the retrieved evidence
        out["generation_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update(answer=gen.answer, abstained=gen.abstained, answer_mode="abstained" if gen.abstained else choice,
               generator=gen.generator, cited_passages=gen.cited, answer_warnings=gen.warnings)
    if gen.abstained:
        out["answerability"] = {
            "status": "unanswerable",
            "signal": gate.get("signal"),
            "signal_type": gate.get("signal_type"),
            "threshold": gate.get("threshold"),
            "reason": "Retrieved passages do not contain sufficient grounded evidence to answer the question (generator abstained).",
        }
    return out


@app.post("/ask")
def ask(request: AskRequest):
    t0 = time.perf_counter()
    res = generate_answer(request.query, request.method, request.top_k, request.generator)
    evidence = res["evidence"]
    citations = [{
        "citation_id": f"[{i}]",
        "chunk_id": e["chunk_id"], "doc_id": e["doc_id"], "document_name": e["document_name"],
        "category": e.get("category"), "version": e.get("version"), "page": e["page"],
        "label": f"{e['document_name']}, p. {e['page']}",
    } for i, e in enumerate(evidence, start=1)]
    return {
        "query": request.query,
        "detected_language": detect_language(request.query),
        "retrieval_method": request.method,
        "method_label": METHOD_LABELS[request.method],
        "score_type": SCORE_TYPE[request.method],
        "top_k": request.top_k,
        "retrieval_status": "ok" if evidence else "no_results",
        "answer_mode": res["answer_mode"],          # extractive | llm | abstained | evidence_only
        "generator": res["generator"],
        "answer": res["answer"],
        "abstained": res["abstained"],
        "cited_passages": res["cited_passages"],    # 1-based numbers into `evidence` / `citations`
        "answer_warnings": res["answer_warnings"],
        "generation_error": res["generation_error"],
        "answerability": res["answerability"],
        "citations": citations,
        "evidence": evidence,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# --------------------------------------------------------------------------------------
# Evaluation and error analysis (real artifacts only)
# --------------------------------------------------------------------------------------
def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_retrieval", config.ROOT / "backend" / "evaluate_retrieval.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@app.get("/evaluation")
def evaluation():
    corpus = _need_corpus()
    rep = config.REPORTS_DIR
    pilot = reports.bm25_pilot(rep)

    # Any <method>_retrieval_runs.csv present is scored against the same reviewed judgements.
    per_method: dict[str, Any] = {}
    judgements_path = rep / "relevance_judgements_completed.csv"
    if judgements_path.exists():
        ev = _load_evaluator()
        judgements = ev.read_csv(judgements_path)
        for name, label in [("bm25", "BM25"), ("dense", "Pretrained E5"), ("hybrid", "Hybrid (RRF)")]:
            runs_path = rep / f"{name}_retrieval_runs.csv"
            if runs_path.exists():
                _, agg, _ = ev.evaluate(ev.read_csv(runs_path), judgements)
                per_method[name] = {"label": label, "summary": agg, "source_file": runs_path.name}

    n_cases = len(reports.error_cases(engine.bm25, corpus, rep))
    qa = {name: reports._json(rep / f"{name}.json") for name in
          ("answerability_eval", "retrieval_by_language", "rag_vs_llm")}
    qa = {k: v for k, v in qa.items() if v}
    mlqa = reports.mlqa_results(rep)
    dense_measured = (
        ("dense" in per_method or "hybrid" in per_method)
        and ("retrieval_by_language" in qa and "pretrained" in qa["retrieval_by_language"].get("results", {}))
    )
    hybrid_measured = (
        {"bm25", "dense", "hybrid"} <= set(per_method)
        and ("retrieval_by_language" in qa and {"bm25", "pretrained", "hybrid"} <= set(qa["retrieval_by_language"].get("results", {})))
    )
    retrieval_results_present = "retrieval_by_language" in qa and bool(per_method)
    modules = [
        {"id": 1, "title": "TF-IDF / BM25 vs multilingual dense retrieval",
         "status": "measured" if dense_measured else "partial_bm25_only",
         "note": "BM25 and multilingual E5 dense benchmarks measured and compared." if dense_measured else "BM25 measured only."},
        {"id": 2, "title": "Monolingual vs cross-lingual vs code-mixed queries",
         "status": "measured" if retrieval_results_present else "partial",
         "note": "Monolingual English, cross-lingual Hindi, and code-mixed Hinglish measured in reports/retrieval_by_language.json."},
        {"id": 3, "title": "Lexical vs dense vs hybrid retrieval",
         "status": "measured" if hybrid_measured else "not_evaluated",
         "note": "BM25, Pretrained E5, and Hybrid RRF evaluated and compared in reports/retrieval_by_language.json."},
        {"id": 4, "title": "Direct LLM QA vs retrieval-augmented QA",
         "status": "measured" if "rag_vs_llm" in qa else "ready_not_run",
         "note": "Measured in reports/rag_vs_llm.json." if "rag_vs_llm" in qa else "Run backend/evaluate_rag_vs_llm.py to produce reports/rag_vs_llm.json."},
        {"id": 5, "title": "Hallucination and answerability testing",
         "status": "measured" if "answerability_eval" in qa else "ready_not_run",
         "note": "Calibrated on dev split and measured on held-out test split in reports/answerability_eval.json."},
        {"id": 6, "title": "Qualitative analysis of 15+ difficult cases",
         "status": "measured" if n_cases >= 15 else "in_progress",
         "note": f"{n_cases} case(s) documented across cross-lingual, code-mixed, and OCR categories."},
    ]
    return {
        "bm25_pilot": pilot,
        "per_method": per_method,
        "mlqa": mlqa,
        "qa_eval": qa,
        "modules": modules,
        "metric_notes": {
            "recall@k": "Share of judged-relevant passages found in the top K. Overstated when the relevant list is incomplete.",
            "precision@k": "Share of the top K that are judged relevant. With 1-2 relevant passages per query its ceiling is far below 1.",
            "mrr": "Mean of 1/rank of the first relevant passage. Measures how early evidence appears, not answer quality.",
            "ndcg@k": "Rank-discounted gain with binary labels. Rewards placing relevant passages higher.",
            "scope": "Retrieval metrics score passages. Answer accuracy, abstention and unsupported-answer counts come from qa_eval (reports/*.json) and are only present once the evaluation scripts were run.",
        },
    }


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    answer: str = Field(max_length=4000)
    rating: Literal["up", "down"]
    method: str = Field(default="unknown", max_length=30)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=20)
    note: str = Field(default="", max_length=1000)

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Append user feedback to a local JSONL file for error analysis. Avoid personal data in notes."""
    from datetime import datetime, timezone
    import json
    path = config.DATA_DIR / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "query": request.query,
              "answer": request.answer, "rating": request.rating, "method": request.method,
              "cited_chunk_ids": request.cited_chunk_ids, "note": request.note}
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True, "message": "Feedback recorded locally for error analysis."}

@app.get("/errors")
def errors():
    corpus = _need_corpus()
    cases = reports.error_cases(engine.bm25, corpus, config.REPORTS_DIR)
    return {"cases": cases, "required_minimum": 15, "documented": len(cases)}