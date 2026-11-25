# Research-feature implementation notes

## 1. LLM generation
The API already supports automatic Anthropic-backed generation when `ANTHROPIC_API_KEY` is configured, otherwise it uses the offline extractive generator. Set the key in the server environment or `.env` (load it with your preferred dotenv/shell mechanism) and restart. Do not commit the key. LLM responses are passage-grounded, citation-checked, and follow the detected question language. This was not live-tested here because no API key was provided.

## 2. Dense and hybrid retrieval
Run `backend/build_pretrained_index.sh`. This builds an index using `intfloat/multilingual-e5-base` over the current corpus. Dense/hybrid evaluation should then be run with `backend/evaluate_methods.py` and reported by language/split. The first model download requires network access and sufficient RAM/disk. The index artifact is not falsely marked as built in this offline environment.

## 3. Human review
`data/eval/questions_for_review.csv` remains a review workflow: reviewers must check question wording, answerability, gold answer, gold chunk IDs and split leakage; mark `reviewer_ok=Y` only after inspection. Fine-tuning only consumes reviewer-approved answerable positive pairs. Do not use test labels for training or threshold tuning.

## 4. Optional reranker
Set `INDICRAG_RERANKER_MODEL` to a cross-encoder model (for example `cross-encoder/ms-marco-MiniLM-L-6-v2`) and install `sentence-transformers`. Retrieval reranks the candidate set and adds `reranker_score`. A model download is required and no reranker scores are claimed until it is run.

## 5. Multilingual query expansion
The BM25 lexical leg uses a small transparent bilingual expansion lexicon in `backend/app/retrieval/query_expansion.py`. This is a rule/lexicon baseline, not a translation model. It does not replace E5 or a validated translation experiment. Record the original query and expanded query when evaluating the effect.

## 6. Answer metrics
`backend/evaluate_answer_metrics.py` computes Exact Match, token F1, contains-gold and citation-presence/validity from a predictions JSONL file. Inputs must be human-reviewed and contain prediction, gold_answer and retrieved_passages.

## 7. Version conflict hints
The retrieval results annotate heuristic potential-version conflict hints when multiple retrieved sources expose year metadata. These are warnings only, not a policy determination or authoritative precedence rule. Verify effective dates and repeal/supersession clauses manually before asserting precedence.

## 8. Feedback
The `/feedback` endpoint appends 👍/👎 records to `data/feedback.jsonl`; the Ask-results screen includes controls. Avoid submitting personal or sensitive information in feedback notes. This local file is user feedback, not a human-verified evaluation label.

## Offline limitation in this build environment
`sentence-transformers` could not be installed because package-network DNS was unavailable and no cached model was present. Consequently the pretrained E5 index and fine-tuned model artifacts are NOT included, and their evaluation numbers are NOT reported. Use the scripts above in a network-enabled environment.
