"""Lightweight, transparent query-language heuristic.

This is NOT a trained language-identification model. It separates four cases that matter
for this project using script and a small closed list of romanised-Hindi function words:

    English | Hindi | Hinglish (romanised Hindi + English) | Code-mixed (Devanagari + English)
"""
from __future__ import annotations

import re

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")

# Romanised-Hindi function/question words that are not ordinary English words.
# Deliberately excludes ambiguous tokens such as "to", "me", "do", "the".
_HINGLISH = {
    "kya", "kaise", "kaisa", "kitna", "kitni", "kitne", "kab", "kaun", "kahan", "kyun", "kyon",
    "hai", "hain", "tha", "thi", "hoga", "hogi", "hona", "honi", "hota", "hoti", "chahiye",
    "ke", "ka", "ki", "liye", "mein", "nahi", "nahin", "agar", "toh", "aur", "ya", "par",
    "karna", "karne", "kare", "kar", "karo", "milta", "milti", "milega", "wala", "wali",
    "kaunsa", "kaunsi", "batao", "bataye", "sakta", "sakti", "sakte", "jaye", "jata", "jati",
    "uska", "unka", "iska", "yeh", "ye", "woh", "se", "tak", "bhi", "sirf", "lekin",
}


def detect_language(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"label": "Unknown", "method": "heuristic", "evidence": "empty input"}

    dev_chars = len(_DEVANAGARI.findall(text))
    latin_words = _LATIN_WORD.findall(text)
    letters = dev_chars + sum(len(w) for w in latin_words)
    dev_ratio = dev_chars / letters if letters else 0.0
    hinglish_hits = [w.lower() for w in latin_words if w.lower() in _HINGLISH]

    if dev_chars and latin_words and dev_ratio < 0.85:
        label, why = "Code-mixed", f"Devanagari ({dev_ratio:.0%} of letters) mixed with Latin words"
    elif dev_chars:
        label, why = "Hindi", f"Devanagari script ({dev_ratio:.0%} of letters)"
    elif len(hinglish_hits) >= 2 or (hinglish_hits and len(latin_words) <= 6):
        label = "Hinglish"
        why = "romanised Hindi function words: " + ", ".join(sorted(set(hinglish_hits)))
    else:
        label, why = "English", "Latin script, no romanised-Hindi function words found"

    return {"label": label, "method": "heuristic", "evidence": why}
