"""Policy-version conflict hints (heuristic, not legal/policy adjudication)."""
from datetime import datetime

def _year(s):
    if not s:return None
    import re
    m=re.search(r'(?:19|20)\d{2}',str(s));return int(m.group(0)) if m else None

def annotate_conflicts(results):
    # annotate records when multiple passages/documents contain explicit policy/version years
    years=[_year(r.get('version') or r.get('document_name') or '') or _year(r.get('text') or '') for r in results]
    valid=[y for y in years if y]
    newest=max(valid) if valid else None
    out=[]
    for r,y in zip(results,years):
        x=dict(r)
        if y and newest and len(set(valid))>1:
            x['version_conflict_hint']=True
            x['version_precedence_hint']='Potentially newer source in retrieved set' if y==newest else f'Potentially superseded by source dated {newest}'
        else:x['version_conflict_hint']=False
        out.append(x)
    return out
