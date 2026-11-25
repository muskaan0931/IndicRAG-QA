#!/usr/bin/env bash
# Full fine-tuning pipeline: train -> build separate fine-tuned index -> evaluate (pretrained vs fine-tuned).
# Extra arguments are passed to the training script, e.g.:  bash backend/train_finetuned_e5.sh --overwrite
set -euo pipefail
cd "$(dirname "$0")/.."
python backend/train_e5_from_reviewed_data.py "$@"
python backend/build_dense_index.py --model models/finetuned_e5 --out models/finetuned_e5_index --batch-size 8
python backend/evaluate_e5_retrieval.py
