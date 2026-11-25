#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r backend/requirements.txt
python backend/build_dense_index.py --model intfloat/multilingual-e5-base --out phase4_outputs/dense_index --batch-size 8
