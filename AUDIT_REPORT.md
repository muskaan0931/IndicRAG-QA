# IndicRAG-QA — Project Audit (honest status)

**Scope and method.** I audited the files in `IndicRAG_QA_SLIM.zip`: code, data, reports, annotation files, frontend. Where the zip lacked something, I say so instead of assuming it exists. The last part of your brief (Part 10 onward) was cut off, so backend-performance requirements beyond what is covered here were not seen.

**What the zip did NOT contain** (so these claims could not be verified): any model weights, the dense index (`phase4_outputs/dense_index`), the fine-tuned model (`models/finetuned_e5`), training triplets, training logs, the Phase 4/6 notebooks, any dense/hybrid run files, and any MLQA evaluation script or result. I also could not download E5 here (huggingface.co is not reachable from my sandbox), so **I have never run dense, fine-tuned or hybrid retrieval.** Their code paths are unit-tested with a stand-in encoder only.

---
## 1. Original requirement (from the course brief you pasted)
Evidence-grounded cross-lingual QA over a public-information domain (you chose Bennett University documents). Required: 300–500 QA instances; English + Indic + code-mixed queries; BM25 baseline; multilingual dense retrieval; hybrid; RAG with cited passage; ANSWERABLE/UNANSWERABLE; six modules; separate retrieval vs QA evaluation; ≥15 difficult cases; IEEE paper, slides, GitHub README, demo.

## 2. Phase-by-phase reconstruction

| Phase | Intended | Evidence in zip | Verdict |
|---|---|---|---|
| 1–2 Corpus | OCR/extract, page-aware chunks | `data/pages.jsonl` (140 pages), `chunks.jsonl` (429), `extraction_report.csv`, `corpus_validation.json` | **Done, with defects** (below) |
| 3 BM25 | Lexical baseline + eval | `bm25.py`, tests, `reports/bm25_*` | **Implementation correct; evaluation is only a 4-query pilot** |
| 4 Dense/hybrid | multilingual-e5-base index, RRF | `dense.py`, `hybrid.py`; **no index, no run files** | **Code correct; results unverifiable** |
| 5 Annotation | Relevance labels | XLSX/CSV in zip: **all 77 labels blank** | **Labels not in the project files** |
| 6 Fine-tuning | Triplet fine-tune of E5 | `train_dense.py`, `build_training_triplets.py`; **no model/data/logs** | **Cannot be verified; method has flaws** |
| 7 MLQA | External benchmark | Raw MLQA data only; **no script/results** | **Did not exist; I wrote BM25 version** |
| Frontend | React UI | `App.tsx`, all pages were placeholders; Ask called `/retrieve` and printed results inline | **Now integrated (this release)** |
| QA / answerability / paper / slides | Modules 4–5, deliverables | none | **Not started** |

## 3. Critical evaluation

### A. Corpus and chunking
- 5 documents, 140 pages (92 OCR, 48 native), 429 chunks, median 952 chars, no overlap, page-bound.
- **Defects found:** `data/documents.jsonl` was corrupted (literal `\n` between records, so one unparsable line), listed 6 documents (the brochure is not in the corpus), used hyphenated IDs that do not match chunk IDs, and gave Exam Manual version `2024-25` while chunks say `unspecified`. *Fixed:* regenerated from the real corpus (original kept as `.corrupted_original`).
- `chunks.jsonl.backup` and `chunks.damaged_backup` are identical one-line damaged files; the brochure text in them shows character loss ("Grea er", "Compu er"), which is presumably why it was dropped.
- `needs_review` is `False` for all 140 pages although 10 pages have OCR confidence below 70 and 3 below 50 (e.g. Handbook p.28 = 33%). The flag is not doing its job; the UI now shows confidence directly.
- 27 of 429 chunks end in an orphan list marker ("b.", "e."); 14 contain "Page N of M" footer noise. Exam Manual p.2–3 (table of contents) is OCR garbage. Fixed-size, page-bound chunking splits clauses and cannot span pages.
- Version risk: 2016-17 ordinances vs 2022 attendance notification vs 2024-25 scholarship policy. Retrieval does not consider recency; the UI now warns.

### B. BM25
Correct Lucene-style IDF, k1=1.5, b=0.75, deterministic tie-breaks; I reproduced your reported pilot metrics exactly. **Limitation:** the tokenizer keeps only `[a-z0-9]+`, so a Devanagari query yields **zero** tokens and zero results, and Hinglish queries match only their English words. This is correct baseline behaviour (the corpus is English), and it is itself a finding for Modules 1–2. No stemming.

### C. Dense retrieval (code review only)
Model choice (multilingual-e5-base, 768-d, 100 languages incl. Hindi) is sound. `query: ` / `passage: ` prefixes, L2-normalised embeddings and dot-product-as-cosine are all correct. Weaknesses: no check that a loaded index matched the corpus (fixed: IDs and order are now verified), paths disagreed between scripts (fixed: one config), and the fine-tuned model re-embedded the corpus on first request (fixed: persisted index).

### D. Fine-tuning
I cannot confirm any model was fine-tuned: no weights, config, logs or data were provided. A claim of fine-tuning would need (a) the model folder, (b) a training log, (c) embeddings differing from the base model. Design problems visible in the code and README:
- README: 38 triplets from **6 queries** — about 6 distinct anchors and positives repeated against many negatives. This is a smoke test, not training.
- `TripletLoss` defaults to Euclidean distance with margin 5, while retrieval uses cosine. `MultipleNegativesRankingLoss` is the standard choice.
- Batch 4, 2 epochs on 38 rows is ~20 optimisation steps.
- No validation split; the training queries are the same eight used for evaluation, so any gain is **leakage, not evidence**.
- `model.fit` is the legacy API.
Verdict: acceptable only as a clearly labelled pilot. Do not retrain until labels are human-verified and there are ≥150 triplets with queries split train/val/test. Resource need is small (CPU is enough for a pilot; a free Colab GPU for 300 triplets, minutes).

### E. Hybrid
RRF (k=60) is implemented correctly: de-duplicates by chunk ID, keeps metadata, deterministic ties. I now fetch a candidate pool of max(10, 2K) per retriever and record each passage's BM25 and dense rank. Whether hybrid helps is **unknown** — no measurements exist.

### F. Annotation
In the zip, every label is blank. The README's labels were AI-assisted and are not in the repository, so provenance cannot be checked. Four queries are judged, one has no relevant passage. Concerns: AI labels must not be called gold; the Phase 3 judgement pool came from BM25 only, which favours BM25; with 1–2 relevant passages per query, Recall@10 = 1.0 is close to automatic.

### G. MLQA
No evaluation existed in the files. I wrote `backend/evaluate_mlqa.py`: corpus = unique contexts of each file, gold = the question's own paragraph, metrics Recall@1/3/5/10, MRR@10, nDCG@10. Results are reproducible (`reports/mlqa_bm25_*.json`). BM25 only, test split:

| context / question | Recall@10 | MRR@10 |
|---|---|---|
| en / en | 0.814 | 0.674 |
| hi / hi | 0.651 | 0.508 |
| hi / en (cross-lingual) | 0.077 | 0.042 |
| en / hi (cross-lingual) | 0.057 | 0.036 |

What this shows: lexical retrieval works within a language and collapses across languages. What it does **not** show: dense performance (not run), answer accuracy (EM/F1 need a reader), or anything about the Bennett corpus. This is our own retrieval protocol on MLQA, not comparable to published MLQA numbers. With one gold paragraph, Hit Rate@K equals Recall@K.

### H. Research validity
The central question (dense vs lexical, hybrid, RAG vs direct LLM, code-mixed difficulty) is **not yet answerable** from the data. Defensible today: BM25 baseline, corpus construction, the MLQA lexical vs cross-lingual contrast, the identified failure modes. Not defensible: any claim that dense/hybrid/fine-tuned improves anything, any RAG or hallucination claim, any "gold" labels.

## 4. Status and roadmap

**Completed and verified:** corpus files; BM25 and its pilot metrics (reproduced); MLQA BM25 benchmark; full-stack integration (14 Python tests, 7 UI tests against the live backend, production build passes, CORS checked).
**Partial:** dense/hybrid/fine-tuned (code only); annotation (blank in zip); error analysis (8 cases of 15); Evaluation Lab (BM25 only).
**Incorrect and fixed:** corrupted `documents.jsonl`; path mismatches; missing `@types/react` (your original `npm run build` failed); fake/placeholder UI numbers; UI claims of an "answerability decision".
**Remaining, in priority order:**
1. Build the pretrained index (`backend/build_dense_index.py`) and export dense/hybrid run files; run `evaluate_mlqa.py --method dense` on a GPU.
2. Create the real evaluation set: 300–500 questions (English, Hindi, Hinglish, cross-lingual), ANSWERABLE and UNANSWERABLE, human-verified, with a held-out test split.
3. Re-score all methods per language; add Hit Rate and confidence intervals.
4. Redo fine-tuning properly (or present it explicitly as a pilot).
5. Add answer generation with citations and an answerability threshold; then Modules 4–5 and EM/F1.
6. Write up ≥15 cases, the IEEE paper and the slides.

## 5. Integration summary (what changed)
- **Backend:** new `/ask`, `/retrieve`, `/compare`, `/documents`, `/chunks`, `/overview`, `/evaluation`, `/errors`, `/methods`; language heuristic; page-quality and version warnings on every passage; clear 503 for missing models; CORS for localhost.
- **Frontend:** original design kept. Analyse Question now opens a dedicated results view (query, language, method, top-K, status, ranked passages with document, ID, page, chunk, version, category, score, OCR confidence, citations, loading/error/empty states, back navigation). All other pages read real data; placeholders and invented numbers were removed.
- **`/ask` is evidence-only.** No answer generator or answerability model exists, and the UI says so.
