# IndicRAG-QA research completion status (updated)

## Implemented in this source build
- Anthropic grounded generator wiring already in API: automatic selection if environment key exists; citation verifier; language instruction for Hindi/Hinglish.
- Selected bilingual BM25 query expansion (lexicon baseline).
- Optional CrossEncoder reranking, controlled by `INDICRAG_RERANKER_MODEL`.
- Heuristic policy-version conflict hint annotations.
- Local thumbs-up/thumbs-down API and UI; JSONL logging.
- EM, token F1, contains-gold and citation validity metric runner.
- Pretrained multilingual E5 index build command and fine-tuned E5 training/index command.

## Artifacts/results that are not present
- No Anthropic key supplied, so live LLM generation/API response was not tested.
- No SentenceTransformer model cache and package-network access was unavailable; pretrained E5 dense index was not generated.
- Fine-tuned E5 model weights/index are not trained/generated. This is gated on a model download plus a human-reviewed dataset.
- The evaluation set has 93 review-workflow questions; 0 are marked human verified. The dataset verifier currently reports label/chunk and unanswerability overlap issues.
- Four failing tests are not asserted here as a stable result; the latest full run had 37 passing and one failing dataset verifier test. Other backend tests run separately passed.
- Frontend feedback UI was added, but frontend build/test was not run because node_modules are absent and package network was unavailable.

See `RESEARCH_FEATURES_IMPLEMENTED.md` and `EXPERIMENT_BLOCKERS.md` for reproducible next steps and caveats.
