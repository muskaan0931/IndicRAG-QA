# E5 fine-tuning, fine-tuned index and retrieval evaluation

**Status: implemented and unit-tested; NOT yet executed end-to-end.** The build sandbox had no PyTorch, no
`sentence-transformers`, no network and no E5 weights, so no model has been trained and no retrieval metric exists yet.
This file contains no results; they will appear in `reports/e5_finetune_eval/` only after you run the commands below.

## Pipeline

```
questions_for_review.csv ──filter──► reviewed pairs ──train──► models/finetuned_e5/
                                                         │
data/chunks.jsonl (434) ──build_dense_index.py───────────┴──► models/finetuned_e5_index/   (pretrained index untouched)
                                                                      │
phase4_outputs/dense_index (pretrained) ──────────────────────────────┴─► evaluate_e5_retrieval.py ─► reports/e5_finetune_eval/
```

| Script | Role |
|---|---|
| `backend/e5_ft_common.py` | torch-free: data selection, grouping, folds, batch masking, ranking, metrics, index checks |
| `backend/train_e5_from_reviewed_data.py` | fine-tunes `intfloat/multilingual-e5-base` -> `models/finetuned_e5/` |
| `backend/build_dense_index.py` | embeds all chunks (same on-disk format as before; adds `index_config.json`) |
| `backend/evaluate_e5_retrieval.py` | pretrained vs fine-tuned retrieval evaluation |
| `tests/test_e5_finetune_pipeline.py` | unit tests |

## Data and filtering

Source: `data/eval/questions_for_review.csv` (93 rows: 63 answerable, 30 unanswerable).
A row is used only if **all** hold: `answerable=true`, `reviewer_ok=Y`, and **every** gold chunk id exists in `data/chunks.jsonl`.
Rows marked `N` and unreviewed rows are never used; nothing is auto-accepted; gold ids and answers are never edited.

| | rows |
|---|---|
| answerable | 63 (57 `Y`, 6 `N`) |
| unanswerable, unreviewed | 30 |
| `Y` but **excluded**: gold chunk id missing from corpus | **6** (facts A15, A16 -> `ord_rulesandreg_ug_btechf_p29_c4`) |
| **eligible** | **51 questions = 17 facts x 3 languages (EN/HI/Hinglish)** |

> **Needs your decision.** The earlier structural audit said all gold ids exist; they do not. The corpus has
> `ord_rulesandreg_ug_btechf_p29_c1..c3` but no `c4`. A15/A16 are excluded rather than silently trimmed. To recover them,
> correct the gold ids in the CSV (and re-review if the passage changed), then re-run everything.

**Provenance:** `reviewer_ok` values are *assistant-assisted screening* of AI-drafted questions, **not** independent human verification.

Eligible facts share gold passages (17 distinct gold chunks in total), so the independent unit is a **group** = connected
component of fact <-> gold chunk. There are **8 groups** (sizes 4, 4, 2, 2, 2, 1, 1, 1 facts). Training yields 66 (question, gold-chunk) pairs.

## Training method

* Base model `intfloat/multilingual-e5-base`; inputs `query: <question>` / `passage: <chunk>` (same prefixes as `app/retrieval/dense.py`), cosine on L2-normalised embeddings.
* Objective: InfoNCE query -> passage, temperature 0.02. Negatives = other positives in the batch + 8 random corpus chunks that are gold for no eligible question (forward-only, to save memory).
* **Gold-aware masking:** any passage that is gold for the query is removed from its softmax, so other languages of the same fact, sibling facts and multi-gold questions never become false negatives. The old script (`.orig`) used plain `MultipleNegativesRankingLoss` over shuffled pairs, which does exactly that.
* Plain PyTorch loop (AdamW, linear warm-up/decay, grad-clip 1.0) instead of `SentenceTransformer.fit()` -> **no `datasets` / `accelerate` needed**, no new dependency.
* Defaults: 3 epochs, batch 8, lr 1e-5, weight-decay 0.01, warm-up 10 %, max length 512, seed 13, word-embedding matrix frozen (saves ~3 GB optimizer state), **CPU** (MPS only if `--device mps`). Hyper-parameters are fixed a priori, not tuned.
* Outputs in `models/finetuned_e5/`: model, `training_config.json`, `training_log.jsonl` (per-step loss), `training_metadata.json` (written only after a successful run and a save/reload check).

## Evaluation method

The eligible set is far too small for a fixed train/test split, and a random split would leak (languages/siblings share passages). So:

1. **pretrained** - no training, scored on all 51 questions (persisted index, alignment-checked).
2. **finetuned_cv_heldout** - grouped 4-fold CV: per fold a fresh base model is trained on the training groups only and scored on held-out groups; a leakage check (no shared fact, question or gold chunk) runs before every fold. This is the **valid** comparison against pretrained (paired, same questions).
3. **finetuned_final_resubstitution** - the saved `models/finetuned_e5` scored on the same questions it was trained on. **Not held-out**; only verifies the saved model/index.

Metrics: Hit@k, Recall@k (fraction of the full gold set in top-k; correct for multi-gold), MRR over the full 434-chunk ranking, nDCG@10, with k in {1,3,5,10}; group-level cluster-bootstrap 95 % intervals; per-language and per-group tables.
Outputs in `reports/e5_finetune_eval/`: `per_question_results.{jsonl,csv}`, `metrics.json`, `comparison.md`, `eval_config.json`, `fold_plan.json`, `eval_run.log`.

## Commands (macOS, from anywhere; paths resolve against the project root)

```bash
python -m pip install -r backend/requirements.txt          # sentence-transformers (+torch); nothing else new
python backend/train_e5_from_reviewed_data.py --dry-run     # check data selection first (no torch needed)
python backend/train_e5_from_reviewed_data.py               # train on CPU -> models/finetuned_e5
python backend/build_dense_index.py --model models/finetuned_e5 --out models/finetuned_e5_index --batch-size 8
python backend/evaluate_e5_retrieval.py                     # pretrained + 4-fold CV + final   (several x training time)
# or everything: bash backend/train_finetuned_e5.sh
```
Useful flags: `--skip-cv` (fast: pretrained + final only), `--folds 3`, `--resume` (reuse finished CV folds), `--batch-size 4` / `--max-seq-length 256` if memory is tight, `--overwrite` to retrain into a non-empty `models/finetuned_e5`.
The pretrained index in `phase4_outputs/dense_index/` is never modified. The first run downloads `intfloat/multilingual-e5-base` (~1.1 GB).

Run the tests: `python -m pytest tests/test_e5_finetune_pipeline.py -v` (the training-loop test needs torch).

## Limitations

* 51 questions / 17 facts / **8 independent groups**; the 3 language variants of a fact are not independent. Intervals will be very wide; small differences will not be statistically established.
* Questions are AI-drafted; labels are assistant-assisted, not independent human annotation. Gold lists may be incomplete (understates recall for all systems).
* CV fold models see fewer groups than the final model, so CV is a conservative proxy for it; the final model's own generalisation is not directly measurable with this data.
* Fine-tuning on ~66 pairs can overfit or change nothing; whether it helps is exactly what the evaluation determines.
* `data/eval/questions.jsonl` (the app's evaluation set, `verified: false`, dev/test split) is not used by this pipeline.
