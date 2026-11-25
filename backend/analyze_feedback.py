#!/usr/bin/env python3
"""Summarize feedback JSONL into counts and a CSV for manual error analysis."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; f=ROOT/'data/feedback.jsonl'; out=ROOT/'reports/feedback_error_analysis.csv'
rows=[]
if f.exists():
 for line in f.read_text(encoding='utf8').splitlines():
  if line.strip(): rows.append(json.loads(line))
counts={'up':sum(r.get('rating')=='up' for r in rows),'down':sum(r.get('rating')=='down' for r in rows),'total':len(rows)}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf8',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['timestamp','rating','method','query','answer','cited_chunk_ids','note']);w.writeheader();w.writerows(rows)
print(counts);print('Export:',out)
