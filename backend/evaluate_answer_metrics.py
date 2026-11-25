#!/usr/bin/env python3
"""Compute answer-level EM/F1/contains and citation validity from prediction JSONL.
Input rows require prediction, gold_answer, retrieved_passages (list), and optional cited_passages.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.evalkit import exact_match, token_f1, contains_gold
from app.generation import verify_answer
rows=[]
for line in open(sys.argv[1],encoding='utf8'):
 if line.strip(): rows.append(json.loads(line))
metrics=[]
for r in rows:
 passages=r.get('retrieved_passages',[]); pred=r.get('prediction',''); gold=r.get('gold_answer','')
 cited,warnings=verify_answer(pred,passages)
 metrics.append({'em':exact_match(pred,gold),'f1':token_f1(pred,gold),'contains':contains_gold(pred,gold),'citation_valid':float(bool(cited) and not any('not retrieved' in w for w in warnings)),'citation_present':float(bool(cited))})
summary={k:round(sum(x[k] for x in metrics)/len(metrics),4) for k in metrics[0]} if metrics else {}
report={'n':len(metrics),'metrics':summary,'rows':metrics,'provenance':'Computed from provided prediction/gold JSONL; ensure references are human-reviewed.'}
out=Path(sys.argv[2] if len(sys.argv)>2 else 'reports/answer_metrics.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));print('Saved',out)
