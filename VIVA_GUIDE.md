# IndicRAG-QA — Viva Guide

## 60-second summary
We built a retrieval system over five Bennett University documents (429 passages). A user asks in English, Hindi or Hinglish; the system retrieves ranked passages with citations. We compare BM25 (exact words) with multilingual dense retrieval (E5 embeddings) and a hybrid of both. It then answers in one or two sentences with [n] citations, or says the documents do not contain enough information. An answerability gate (threshold on the top evidence score, fitted on a dev split) rejects questions the corpus cannot answer.

## Concepts you must be able to explain
- **RAG:** retrieve evidence first, then generate an answer from it only. We have an offline extractive generator and an LLM generator; both are post-checked for valid citations and unsupported numbers.
- **BM25:** scores a passage by query-word frequency, weighted by word rarity (IDF) and normalised by passage length (k1, b). It cannot match words in different languages.
- **Dense retrieval:** a model maps text to vectors; similar meaning means close vectors, even across languages. E5 needs `query:` and `passage:` prefixes; normalised vectors make dot product equal cosine similarity.
- **RRF (hybrid):** score = sum of 1/(60 + rank) over each ranked list. It needs no score calibration, which matters because BM25 and cosine scores are on different scales.
- **Fine-tuning:** continue training the embedding model on (query, positive, negative) triplets so relevant passages move closer. Ours is a tiny pilot.
- **Metrics:** Recall@K (relevant found in top K), Precision@K, MRR (1/rank of first relevant), nDCG (rank-discounted gain). All measure retrieval, not answer quality.

## Questions to expect, and honest answers
1. **Does dense beat BM25?** Not measured yet on our data. On MLQA, BM25 gets Recall@10 0.81 (English) and 0.65 (Hindi) but only about 0.06–0.08 across languages; that gap is the motivation for dense retrieval.
2. **Why do Hindi queries return nothing with BM25?** The corpus is English and BM25 matches exact tokens; a Devanagari query shares none.
3. **Is the model fine-tuned?** We ran a pilot with a few dozen AI-assisted triplets from six queries. It is not validated, and the training queries overlap the evaluation queries, so we do not claim improvement.
4. **Are your labels gold?** No. They are AI-assisted, cover four queries, and need human verification.
5. **Why is Recall@10 = 1.0?** One or two relevant passages per query and a small judged pool make it nearly automatic. It cannot rank methods.
6. **What does MLQA show?** A separate Wikipedia benchmark for retrieval across Hindi and English. It says nothing about answer accuracy or about the Bennett corpus.
7. **How do you stop hallucination?** Three layers: the prompt allows only cited passages and an INSUFFICIENT reply; the answerability gate rejects weak evidence; a verifier flags missing citations and numbers absent from the cited passages. None is proof. We measure unsupported answers in the direct-LLM vs RAG experiment (once run).
7b. **How was the answerability threshold chosen?** Fitted on the dev split (max F1 of the answerable class), scored once on test. It measures closeness of the best passage to the question, not whether it holds the answer; related-but-wrong passages can pass.
8. **Why chunk by page at about 1000 characters?** Page numbers give citations. Cost: clauses can be split mid-list or across pages. Overlap or structure-aware chunking is an improvement.
9. **What are the data risks?** Noisy OCR (Handbook, Exam Manual), mixed document years (2016-17 to 2024-25), and no recency handling.
10. **What would you do next?** Human-verify and grow the question set to 300-500, run all scripts on the full corpus, and report the tables with their confidence intervals.
11. **Is your dataset big enough?** No: 93 AI-drafted questions (31 facts x 3 languages, 30 unanswerable). Bootstrap intervals are wide; that width is the honest result.

## Demo script (3 minutes)
1. Overview: show real corpus counts and which methods are ready.
2. Ask: "What is the minimum attendance required for end semester exams?" with BM25. Open the results page; point to the 75% notification, page, chunk ID, version.
3. Ask the Hindi question with BM25: empty state explains why. Then repeat with Hybrid/Dense (once the index is built).
4. Retrieval Comparison: run a pilot query and show the methods side by side.
5. Evaluation Lab: show MLQA monolingual vs cross-lingual, and the caveats.
6. Error Analysis and About: show the limitations openly.
