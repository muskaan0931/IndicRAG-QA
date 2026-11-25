"""Optional multilingual lexical expansion for English-document retrieval.

For research transparency this uses a small editable dictionary, not a machine translation model.
Dense E5 remains the intended cross-lingual baseline. Expanded query is only used for BM25/hybrid lexical leg.
"""
import re

LEXICON = {
 "scholarship": ["छात्रवृत्ति", "scholarship"], "eligibility": ["पात्रता", "eligible"],
 "minimum attendance": ["न्यूनतम उपस्थिति", "attendance"], "attendance": ["उपस्थिति", "minimum attendance"],
 "exam": ["परीक्षा", "examination"], "examination": ["परीक्षा", "exam"], "refund": ["वापसी", "refund"],
 "fee": ["शुल्क", "fees"], "fees": ["शुल्क", "fee"], "credit": ["क्रेडिट", "credits"],
 "credits": ["क्रेडिट", "credit"], "semester": ["सेमेस्टर", "term"], "marks": ["अंक", "marks"],
 "cgpa": ["सीजीपीए", "grade point"], "grading": ["ग्रेड", "grade"],
 "leave": ["छुट्टी", "absence"], "absence": ["अनुपस्थिति", "leave"],
 "btech": ["बीटेक", "b.tech"], "student": ["छात्र", "student"], "students": ["छात्र", "student"],
}

def expand_query(query: str) -> str:
    q=(query or "").strip(); low=q.lower(); extras=[]
    for key, vals in LEXICON.items():
        if key in low or key in q or any(v in q for v in vals):
            extras.extend(v for v in vals if v.lower() not in low)
    # Avoid over-expanded strings: add at most 6 unique tokens/phrases.
    extras=list(dict.fromkeys(extras))[:6]
    return q + (" " + " ".join(extras) if extras else "")
