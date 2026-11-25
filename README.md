# IndicRAG-QA: Evidence-Grounded Cross-Lingual Question Answering for Indic and Code-Mixed Languages

IndicRAG-QA is an evidence-grounded, multilingual Question Answering (QA) system designed for institutional document corpora. It bridges linguistic divides across **English (monolingual)**, **Hindi (cross-lingual)**, and **Hinglish (code-mixed)** queries using hybrid retrieval (BM25 + Pretrained Multilingual E5 with Reciprocal Rank Fusion), calibrated answerability gating to prevent hallucinations, and passage-level citation verification.

Developed for Bennett University institutional policies, academic regulations, examination manuals, and scholarship guidelines.

---

## 1. System Architecture & Features

```text
                                     [User Query]
                              (English / Hindi / Hinglish)
                                          |
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
               [BM25 Lexical]                     [Multilingual E5 Dense]
           (Sparse Token Matching)             (768-d Cosine Embeddings)
                       |                                     |
                       └──────────────────┬──────────────────┘
                                          ▼
                           [Reciprocal Rank Fusion (RRF)]
                           (Deep Candidate Pool Merging)
                                          |
                                          ▼
                             [Answerability Detector]
                           (Signal Threshold Evaluation)
                                          |
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
            [Signal >= Threshold]                 [Signal < Threshold]
                       |                                     |
                       ▼                                     ▼
          [Grounded Answer Generation]              [Abstention Message]
            • Extractive (Offline)               "Insufficient information
            • Optional LLM integration           available in the provided
            • Citation Verification [n]                  documents"
```

### Key Technical Highlights

- **Document Ingestion:** 434 chunks across 6 institutional documents (168 pages total), with page-aware boundaries and metadata preservation.
- **Lexical Retrieval (BM25):** Lucene-style BM25 (\(k_1=1.5, b=0.75\)) with transparent lexical expansion.
- **Multilingual Dense Retrieval:** `intfloat/multilingual-e5-base` (768-dimensional L2-normalised embeddings, `passage:` and `query:` prefix conventions).
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF, \(k=60\)) over deep candidate pools (`max(20, top_k * 4)`), mitigating sparse keyword collisions and dense semantic drift.
- **Calibrated Answerability Gate:** Threshold calibration on the development (DEV) split to maximise F1, followed by evaluation on the held-out TEST split. Previously documented results include ROC-AUC = 0.9444 and accuracy = 84.2%.
- **Evidence-Grounded Generation:** Offline extractive sentence selection, with optional configured LLM integration. Generated answers undergo citation association and factual-number checks where implemented.
- **Citation Verification:** Retrieved passage references are checked against the evidence used to construct the answer.
- **User Feedback:** Feedback capture endpoint (`/feedback`) with thumbs-up/down ratings and comments logged to `data/feedback.jsonl`.

---

## 2. Benchmark Results & Comparative Evaluation

All metrics below are reported for the held-out TEST split of the Bennett University QA dataset (`data/eval/questions.jsonl`).

### Evaluation Dataset and Human Validation

The evaluation questions used in IndicRAG-QA underwent manual review by the project team. The questions were checked for correctness, relevance, and suitability for evaluating institutional information retrieval and cross-lingual question answering.

The review covered the English (EN), Hindi (HI), and Hinglish (HG) query categories. The team completed the manual validation process before using the prepared evaluation questions for experimental assessment.

The evaluation dataset and its associated review artefacts are maintained under `data/eval/`.

Human review was performed during dataset preparation, while retrieval and answerability metrics were calculated programmatically using the project's evaluation scripts.

The benchmark file `questions.jsonl` contains 93 evaluation queries. This benchmark count should be distinguished from any larger question pool used during dataset preparation and review.

### A. Retrieval Performance by Language Group (Hit@5)

| Method | English (Monolingual) | Hindi (Devanagari) | Hinglish (Code-Mixed) | Cross-Lingual Overall | All Queries | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BM25 Baseline** | 0.85 [0.62, 1.00] | 0.23 [0.00, 0.46] | 0.69 [0.46, 0.92] | 0.46 [0.27, 0.65] | **0.59** | 0.51 | 0.55 |
| **Pretrained E5 Dense** | 1.00 [1.00, 1.00] | 0.62 [0.38, 0.85] | 0.62 [0.38, 0.85] | 0.62 [0.42, 0.81] | **0.74** | 0.65 | 0.70 |
| **Hybrid (BM25 + E5, RRF)** | **1.00 [1.00, 1.00]** | **0.62 [0.38, 0.85]** | **0.85 [0.62, 1.00]** | **0.73 [0.54, 0.88]** | **0.82** | **0.69** | **0.71** |

*Key Findings:*

1. Pure BM25 performs poorly on Devanagari Hindi queries (Hit@5 = 0.23), reflecting lexical mismatch between Hindi queries and predominantly English institutional documents.
2. Pretrained E5 achieves 1.00 Hit@5 on English and 0.62 on Hindi, demonstrating cross-lingual retrieval capability without requiring a separate machine-translation step.
3. Hybrid RRF achieves the highest overall retrieval quality (**Hit@5 = 0.82, MRR = 0.69, nDCG@10 = 0.71**), outperforming BM25 by 23 percentage points in overall Hit@5.

### B. Answerability Detection & Abstention Gate

| Method | Signal Metric | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| **BM25** | `bm25_top1` | -0.0000 | 68.4% | 68.4% | 100.0% | 0.81 | 0.8319 |
| **Pretrained E5** | `cosine_top1` | 0.8176 | **84.2%** | **91.7%** | 84.6% | **0.88** | **0.9444** |
| **Hybrid (RRF)** | `dense_cosine_top1` | 0.8176 | **84.2%** | **91.7%** | 84.6% | **0.88** | **0.9444** |

*Note:* These values represent the previously documented answerability evaluation results. They should be interpreted in the context of their associated evaluation configuration and split. Do not assume that results from separate evaluation artefacts or configurations are interchangeable.

---

## 3. Quick Start Guide

### Prerequisites

- macOS, Linux, or Windows
- Python 3.11+
- Node.js 18+ and npm

### Backend Setup & Execution

From the project root:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run backend server
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API Documentation: `http://127.0.0.1:8000/docs`

### Frontend Setup & Execution

In a separate terminal:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Start Vite development server
npm run dev
```

UI Application: `http://localhost:5173`

---

## 4. Running Experiments & Evaluations

All evaluation scripts operate on the project's prepared evaluation artefacts and calculate results using the implemented evaluation methodology.

### Evaluation Dataset and Human Validation

The evaluation questions were manually reviewed by the project team for correctness, relevance, and suitability across English, Hindi, and Hinglish query categories.

The manual review process was completed during dataset preparation. The prepared evaluation data is subsequently used by the automated retrieval and answerability evaluation scripts.

The review process and automated scoring process are distinct:

- **Human validation:** Manual checking of evaluation-question correctness, relevance, and suitability.
- **Automated evaluation:** Programmatic calculation of retrieval and answerability metrics using the prepared evaluation data.

### Execute Evaluation Commands

```bash
source .venv/bin/activate

# 1. Verify dataset consistency (gold passage mappings and unanswerable regexes)
PYTHONPATH=backend python backend/verify_dataset.py

# 2. Calibrate and evaluate answerability detection
PYTHONPATH=backend python backend/evaluate_answerability.py --methods bm25 pretrained hybrid

# 3. Benchmark retrieval by language
# English, Hindi, Hinglish, and cross-lingual groups
PYTHONPATH=backend python backend/evaluate_methods.py --split test --methods bm25 pretrained hybrid

# 4. Direct LLM vs RAG comparison using the extractive generator
PYTHONPATH=backend python backend/evaluate_rag_vs_llm.py --method hybrid --generator extractive

# 5. Execute automated test suite
pytest -v
```

**Evaluation methods included in the final system:**

- BM25 lexical retrieval.
- Pretrained multilingual E5 dense retrieval.
- Hybrid BM25 + E5 retrieval using RRF.

Fine-tuned E5 is not part of the final active retrieval implementation.

---

## 5. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System status, loaded chunks, and availability of retrieval methods |
| `/overview` | GET | Corpus statistics (documents, pages, median chunk length) and experiment artefacts |
| `/search` | POST | Backward-compatible BM25 sparse search |
| `/retrieve` | POST | Single-method retrieval (`bm25`, `pretrained`, `hybrid`) with language detection |
| `/ask` | POST | Full RAG pipeline: retrieval, answerability gate, answer generation, and citations |
| `/compare` | POST | Multi-method side-by-side comparison with passage overlap matrix |
| `/documents` | GET | Document-level metadata and extraction summaries |
| `/documents/{id}` | GET | Chunks and text for a specific document |
| `/evaluation` | GET | Results for implemented evaluation modules and available benchmarks |
| `/errors` | GET | Qualitative register of difficult error cases across failure modes |
| `/feedback` | POST | Logs user ratings (thumbs-up/down), cited chunk IDs, and notes |

Refer to the FastAPI interactive documentation at `http://127.0.0.1:8000/docs` for request and response schemas.

---

## 6. Project Directory Structure

```text
IndicRAG_QA_FINAL/
├── backend/
│   ├── app/
│   │   ├── retrieval/
│   │   │   ├── bm25.py             # Sparse BM25 index and tokenizer
│   │   │   ├── dense.py            # Pretrained multilingual E5 dense retriever
│   │   │   ├── hybrid.py           # Reciprocal Rank Fusion (RRF, k=60)
│   │   │   ├── query_expansion.py  # Transparent bilingual lexical expansion
│   │   │   ├── reranker.py         # Optional CrossEncoder reranking, if enabled
│   │   │   └── conflicts.py        # Policy version precedence hints
│   │   ├── answerability.py        # Signal extraction and threshold evaluation
│   │   ├── config.py               # Central environment and path configuration
│   │   ├── corpus.py               # Corpus loader and document enricher
│   │   ├── evalkit.py              # Metric calculation (Hit@K, MRR, nDCG, EM, F1)
│   │   ├── generation.py           # Extractive and configured LLM generators, citation verifier
│   │   ├── language.py             # Script and keyword language heuristic
│   │   ├── main.py                 # FastAPI application and lifespan events
│   │   ├── qa_eval.py              # Cross-lingual and RAG evaluation runner
│   │   └── reports.py              # Qualitative error case synthesizer
│   ├── build_dense_index.py        # Persists dense passage embeddings (.npy)
│   ├── evaluate_answerability.py   # Fits DEV thresholds and scores TEST split
│   ├── evaluate_methods.py         # Evaluates retrieval by language split
│   ├── evaluate_rag_vs_llm.py      # Direct LLM vs RAG answer generation evaluation
│   └── verify_dataset.py           # Integrity assertions for gold chunk mappings
├── data/
│   ├── chunks.jsonl                # 434 institutional document chunks
│   ├── pages.jsonl                 # 168 extracted page records
│   ├── documents.jsonl             # Corpus document-level metadata
│   ├── feedback.jsonl              # User rating logs
│   ├── eval/
│   │   ├── questions.jsonl         # 93 evaluation queries (EN, HI, HG)
│   │   └── questions_for_review.csv # Evaluation question review records
│   └── source_pdfs/                # Original institutional PDF source files
├── frontend/
│   ├── src/
│   │   ├── api.ts                  # Typed backend API client
│   │   ├── App.tsx                 # Navigation, tab switching, and layout
│   │   ├── views.tsx               # UI views (Ask, Search, Compare, Eval, Errors)
│   │   └── styles.css              # Custom styling
│   ├── package.json
│   └── vite.config.ts
├── phase4_outputs/
│   └── dense_index/                # Persisted pretrained E5 embeddings and index configuration
├── reports/                        # Generated benchmark JSON/MD artefacts
├── tests/                          # Automated pytest unit and integration tests
├── DEMO_SCRIPT.md                  # Viva and demonstration walkthrough
└── README.md                       # Main documentation
```

---

## 7. Implementation Notes and Limitations

### Multilingual Query Handling

IndicRAG-QA supports English, Hindi, and Hinglish queries using pretrained multilingual E5 dense retrieval and BM25 lexical retrieval.

Language detection is based on implemented script and keyword heuristics. It should not be interpreted as a comprehensive language-identification model.

### Evidence-Grounded Answer Generation

The system retrieves institutional passages before preparing an answer. Its offline extractive generation mechanism uses retrieved evidence to construct responses.

Optional external LLM integration should be treated as configured functionality unless successful execution and integration tests are available.

### Citation Verification

Citation verification associates answer references with retrieved passage identifiers. This improves evidence traceability but does not independently prove the complete semantic correctness of every generated statement.

### Conflict Detection

Conflict detection provides heuristic hints about potentially conflicting institutional policies or document versions. It does not constitute authoritative conflict resolution.

### Evaluation Limitations

The reported benchmark metrics depend on the prepared evaluation dataset, selected splits, retrieval configurations, and evaluation scripts. Results from different evaluation runs or configurations should not be directly combined without verification.

### Retrieval Methods

The final active retrieval architecture uses BM25, pretrained multilingual E5, and Hybrid RRF. Fine-tuned E5 was explored during earlier development but was removed from the final implementation.

---

## 8. Project Information

**University:** Bennett University

**School:** School of Computer Science Engineering and Technology

**Course:** CSET 346 – Natural Language Processing

**Project ID:** PJ-09

**Team Name:** ARTHAI

**Project Instructor:** Dr. Ankur Kumar Singhal

### Team Members

| Name | University ID | Role |
|---|---|---|
| Muskan Singh | S24CSEU2095 | Team Lead |
| Arisha Ali | S24CSEU2060 | Team Member |
| Tulsi Sanskriti Tomar | S24CSEU0084 | Team Member |
| Shermin Reaz | S24CSEU1730 | Team Member |
| Tanisha Gupta | S24CSEU0578 | Team Member |

---

**IndicRAG-QA – Evidence-Grounded Cross-Lingual Question Answering for Indic and Code-Mixed Languages.**