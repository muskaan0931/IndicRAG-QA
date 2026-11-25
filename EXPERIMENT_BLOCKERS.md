# Experiment blockers observed during this build

- Python test run: 34 passed, 4 failed (21 Nov 2026 build).
- Failures: two retrieval/generation expectations involving attendance-policy text, and the dataset verifier reports stale/missing gold passage IDs plus draft unanswerable questions that overlap source terms.
- `sentence-transformers` installation attempted but failed due unavailable DNS/network. No pretrained E5 model/index or fine-tuning could be run in this environment.
- Questions remain 0 human-verified; no evaluation scores are presented as completed research results.
- A corpus restore command was attempted but timed out; existing `pages.jsonl` and `chunks.jsonl` are bundled, but full OCR completeness is not verified.

These are reported explicitly rather than hidden or fabricated.

## November 21 update
The source build now includes optional reranking, bilingual lexical expansion, heuristic conflict hints, feedback collection, answer-level metrics and commands to build the E5 index/train the fine-tuned model. However, package-network access is unavailable in this build environment and no model cache exists, so pretrained/fine-tuned model artifacts and their measured results cannot honestly be bundled yet. See `RESEARCH_FEATURES_IMPLEMENTED.md`.
