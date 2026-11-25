"""Preflight check for IndicRAG-QA. Detects missing corpus/indexes and research-label caveats."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
checks={
 'chunks_file': ROOT/'data'/'chunks.jsonl',
 'pages_file': ROOT/'data'/'pages.jsonl',
 'pretrained_index': ROOT/'phase4_outputs'/'dense_index'/'dense_embeddings.npy',
 'fine_tuned_model': ROOT/'models'/'finetuned_e5',
}
print('IndicRAG-QA research preflight')
for name,path in checks.items():
 ok=path.exists()
 print(f'{name}: {"AVAILABLE" if ok else "MISSING"} ({path.relative_to(ROOT)})')
rows=[json.loads(x) for x in (ROOT/'data'/'eval'/'questions_expanded_draft.jsonl').read_text().splitlines() if x.strip()]
print(f'evaluation_questions: {len(rows)}')
print(f'human_verified: {sum(bool(x.get("verified")) for x in rows)} / {len(rows)}')
print('NOTICE: AI-drafted expanded questions are not human-reviewed gold labels; do not report benchmark scores as final.')
if not checks['chunks_file'].exists():
 print('BLOCKER: the corpus chunk file is not included in this release. Restore data/chunks.jsonl from the source corpus before running research evaluations.')
