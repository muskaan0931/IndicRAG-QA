"""Evidence-grounded answer generation with citations and abstention.

Two generators share one contract: they receive numbered passages and return an answer that cites them as
[1], [2], ... or the fixed abstention message. Nothing is returned without citations.

  * ExtractiveGenerator : offline, no model download. Picks the best-matching sentence from the passages.
                          It cannot paraphrase or combine facts, so it is conservative and limited.
  * LLMGenerator        : Anthropic Messages API (needs ANTHROPIC_API_KEY). Instructed to answer ONLY from the
                          passages, cite them, and reply INSUFFICIENT when the evidence does not contain the answer.

Every answer is post-checked by `verify_answer` (citations exist, numbers appear in the cited passages).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from app.retrieval.bm25 import tokenize

ABSTAIN_MESSAGE = "Insufficient information available in the provided documents."
CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-5-5"


@dataclass
class GeneratedAnswer:
    answer: str                      # final text shown to the user (abstain message when abstained)
    abstained: bool
    generator: str
    cited: list[int] = field(default_factory=list)       # 1-based passage numbers actually cited
    warnings: list[str] = field(default_factory=list)    # verification problems
    raw: str | None = None


# ---------------------------------------------------------------- verification
def verify_answer(answer: str, passages: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    """Return (valid citation numbers, warnings). Flags citations to passages that do not exist, answers with no
    citation at all, and numbers that appear in no cited passage (a cheap guard against invented figures)."""
    warnings: list[str] = []
    cited = sorted({int(n) for n in CITE_RE.findall(answer)})
    bad = [n for n in cited if not 1 <= n <= len(passages)]
    if bad:
        warnings.append(f"Answer cites passage(s) {bad} that were not retrieved.")
    cited = [n for n in cited if 1 <= n <= len(passages)]
    if not cited:
        warnings.append("Answer contains no citation.")
        return cited, warnings
    support = " ".join(passages[n - 1]["text"] for n in cited)
    support_nums = {x.replace(",", "") for x in NUM_RE.findall(support)}
    body = CITE_RE.sub("", answer)
    unsupported = [x for x in NUM_RE.findall(body) if x.replace(",", "") not in support_nums]
    if unsupported:
        warnings.append(f"Number(s) {sorted(set(unsupported))} do not appear in the cited passages.")
    return cited, warnings


# ---------------------------------------------------------------- extractive
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


_LIST_MARK = re.compile(r"^(?:\d{1,3}\s+)?(?:\((?:[ivxl]+|[a-z]|\d{1,2})\)\s*)+")
_SECTION_HEAD = re.compile(
    r"^(?:\d{1,3}\s+)?\d+(?:\.\d+)+\.?\s+"                                   # "16.2 " (optionally after a page number)
    r"(?:(?:[A-Z][\w/&\-]*|of|and|for|the|in|to|on|at)\s+){1,9}?"                # Title Case heading words
    r"(?=(?:It|A|An|The|All|Any|Students?|Every|Each|If|In|No|There|For|Candidates?|Those|Such|This)\b)")


def clean_sentence(sentence: str) -> str:
    """Remove page numbers, list markers and glued-on section headings so the answer reads as a normal sentence."""
    s = re.sub(r"\s+", " ", sentence or "").strip()
    for _ in range(2):
        s = _SECTION_HEAD.sub("", s, count=1)
        s = _LIST_MARK.sub("", s, count=1)
    s = s.strip(" -–—:;,")
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s


_ABBREV = re.compile(r"\b(i\.e|e\.g|etc|vs|No|Nos|Dr|Prof|Mr|Mrs|Ms|Sr|Jr|approx|viz|Rs|Ref)\.", re.I)
_DOT = "\u2024"  # one-dot leader, a stand-in so abbreviations are not treated as sentence ends


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0)[:-1] + _DOT, text or "")
    parts = [re.sub(r"\s+", " ", x).replace(_DOT, ".").strip() for x in _SENT_SPLIT.split(protected)]
    return [x for x in parts if len(x) >= min_chars and re.search(r"[A-Za-z]{3}", x)]


_STOP = frozenset("""a an the of in on at to for from by with and or but if is are was were be been being am do does did
has have had will shall can could should would may might must not no what which who whom whose when where why how
this that these those it its as than then there their they them he she his her we our you your i me my about into
over under per any all each every some such also only both either neither tell give please much many need needs needed""".split())
_NEEDS_NUMBER = re.compile(r"\b(minimum|maximum|how (?:many|much|long)|percentage|percent|cgpa|gpa|fee|fees|duration|limit|"
                           r"marks|credits?|days|years?|months?|weightage|cut.?off|deadline|attendance)\b", re.I)


def _stem(w: str) -> str:
    """Cheap prefix stemmer so attendance/attend, examination/examinations and required/require match."""
    return w if len(w) <= 6 else w[:6]


def is_noisy(sentence: str) -> bool:
    """OCR table soup: many pipes or a low share of letters/spaces/punctuation that normal prose uses."""
    if sentence.count("|") >= 2:
        return True
    visible = [c for c in sentence if not c.isspace()]
    return bool(visible) and sum(c.isalpha() or c in ".,;:()'’-%/" or c.isdigit() for c in visible) / len(visible) < 0.9


def content_terms(text: str) -> set[str]:
    return {_stem(w) for w in tokenize(text) if w not in _STOP and len(w) > 1}


class ExtractiveGenerator:
    """Selects the sentence with the highest IDF-weighted overlap with the question, from the retrieved passages.

    A sentence is accepted only if it matches at least `min_overlap` distinct question terms; otherwise the
    generator abstains. When a `sentence_scorer` (query, sentences -> scores) is supplied, for instance a
    multilingual encoder, it replaces lexical overlap, which is what makes Hindi questions workable."""
    name = "extractive"

    def __init__(self, min_overlap: int = 2, sentence_scorer: Callable[[str, list[str]], list[float]] | None = None,
                 min_score: float | None = None):
        self.min_overlap = min_overlap
        self.sentence_scorer = sentence_scorer
        self.min_score = min_score

    def generate(self, question: str, passages: list[dict[str, Any]]) -> GeneratedAnswer:
        sents: list[tuple[int, str]] = [(i, s) for i, p in enumerate(passages, start=1) for s in split_sentences(p["text"])]
        clean = [x for x in sents if not is_noisy(x[1])]
        sents = clean or sents                      # drop OCR table soup unless nothing else is left
        if not sents:
            return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
        texts = [s for _, s in sents]
        if self.sentence_scorer is not None:
            scores = self.sentence_scorer(question, texts)
            best = max(range(len(sents)), key=lambda i: scores[i])
            if self.min_score is not None and scores[best] < self.min_score:
                return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
        else:
            q_terms = content_terms(question)
            if not q_terms:
                return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
            term_sets = [content_terms(t) for t in texts]
            df: dict[str, int] = {}
            for ts in term_sets:
                for w in ts:
                    df[w] = df.get(w, 0) + 1
            n = len(texts)
            wants_number = bool(_NEEDS_NUMBER.search(question))
            substantive_q = q_terms - {"minimu", "maximum", "requir", "total", "least", "limit", "much", "many", "need"}

            def score(i: int) -> tuple[float, int]:
                hit = q_terms & term_sets[i]
                substantive_hit = substantive_q & term_sets[i]
                # Weight substantive domain terms (e.g. attendance, scholarship) higher than modifiers (minimum, requirement)
                base = sum((2.5 if w in substantive_q else 1.0) / (1 + df.get(w, 0) / n) for w in hit)
                if substantive_q and not substantive_hit:
                    base *= 0.2  # penalize sentences that match only modifiers without the core query subject
                if wants_number:                                # "minimum X" questions want the sentence with the figure
                    if re.search(r"\d\s?%|\d+(?:\.\d+)?\s?(?:CGPA|credits?|days|marks|years?|months?)", texts[i], re.I):
                        base *= 1.6
                    elif re.search(r"\b(?!(?:19|20)\d\d\b)\d+(?:\.\d+)?\b", texts[i]):
                        base *= 1.25
                base /= 1 + max(0, len(texts[i]) - 160) / 400   # run-on OCR text rarely makes a clean answer
                return base, len(hit)

            scored = [score(i) for i in range(len(sents))]
            best = max(range(len(sents)), key=lambda i: (scored[i][0], -sents[i][0], -i))
            if scored[best][1] < self.min_overlap:
                return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
            if substantive_q and not (substantive_q & term_sets[best]):
                return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
            if len(q_terms) <= 3 and scored[best][1] / len(q_terms) < 0.5:
                return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
        pnum, sentence = sents[best]
        sentence = clean_sentence(sentence)
        # A very short sentence is usually a fragment: add the sentence that follows it in the same passage.
        if len(sentence) < 70 and best + 1 < len(sents) and sents[best + 1][0] == pnum:
            sentence = f"{sentence} {clean_sentence(sents[best + 1][1])}"
        answer = f"{sentence} [{pnum}]"
        cited, warnings = verify_answer(answer, passages)
        return GeneratedAnswer(answer, False, self.name, cited, warnings)


# ---------------------------------------------------------------- LLM (Anthropic Messages API)
SYSTEM_RAG = (
    "You answer questions about Bennett University documents using ONLY the numbered passages provided.\n"
    "Rules:\n"
    "1. Use only facts stated in the passages. Never use outside knowledge.\n"
    "2. Cite every claim with the passage number in square brackets, e.g. [2]. Cite only passages you used.\n"
    "3. Be brief: one or two sentences. Copy numbers, dates and names exactly as written.\n"
    "4. If the passages do not contain the answer, reply with exactly: INSUFFICIENT\n"
    "5. Some passages come from noisy OCR; do not guess missing characters.\n"
    "{language_rule}"
)
SYSTEM_DIRECT = (
    "You answer questions about Bennett University (Greater Noida, India) from your own knowledge only. "
    "Be brief. If you do not know or are not sure, reply with exactly: INSUFFICIENT\n{language_rule}"
)


def _language_rule(answer_language: str | None, numbered: bool = True) -> str:
    text = ("Always answer in English, whatever language the question uses."
            if answer_language == "en" else "Answer in the same language and script as the question.")
    return f"6. {text}\n" if numbered else text + "\n"


class LLMGenerator:
    name = "llm"

    def __init__(self, api_key: str | None = None, model: str | None = None, answer_language: str | None = None,
                 post: Callable[[dict[str, Any]], dict[str, Any]] | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.answer_language = answer_language
        self.timeout = timeout
        self._post = post or self._http_post

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        req = urllib.request.Request(
            ANTHROPIC_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach the Anthropic API: {exc.reason}") from exc

    def _complete(self, system: str, user: str) -> str:
        data = self._post({"model": self.model, "max_tokens": 300, "temperature": 0, "system": system,
                           "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()

    def generate(self, question: str, passages: list[dict[str, Any]]) -> GeneratedAnswer:
        if not passages:
            return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name)
        ctx = "\n\n".join(f"[{i}] ({p['document_name']}, p. {p['page']})\n{p['text']}" for i, p in enumerate(passages, 1))
        raw = self._complete(SYSTEM_RAG.format(language_rule=_language_rule(self.answer_language)),
                             f"Passages:\n{ctx}\n\nQuestion: {question}")
        if raw.strip().upper().startswith("INSUFFICIENT"):
            return GeneratedAnswer(ABSTAIN_MESSAGE, True, self.name, raw=raw)
        cited, warnings = verify_answer(raw, passages)
        return GeneratedAnswer(raw, False, self.name, cited, warnings, raw=raw)

    def direct(self, question: str) -> GeneratedAnswer:
        """No retrieval: the model answers from memory. Used for the direct-LLM vs RAG experiment."""
        raw = self._complete(SYSTEM_DIRECT.format(language_rule=_language_rule(self.answer_language, numbered=False)), question)
        if raw.strip().upper().startswith("INSUFFICIENT"):
            return GeneratedAnswer(ABSTAIN_MESSAGE, True, "llm-direct", raw=raw)
        return GeneratedAnswer(raw, False, "llm-direct", [], ["No retrieval: answer is not grounded in any document."], raw=raw)