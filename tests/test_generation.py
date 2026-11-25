import conftest  # noqa: F401
from app.generation import (ABSTAIN_MESSAGE, ExtractiveGenerator, LLMGenerator, split_sentences, verify_answer)
from app.retrieval.bm25 import BM25Index, load_chunks
from app import config

PASSAGES = [
    {"document_name": "Exam-Manual.pdf", "page": 28, "text": "A student must have a minimum attendance of 75% in a course during a semester. "
                                                              "A student with less than 75% attendance will be awarded an X grade."},
    {"document_name": "Scholarship.pdf", "page": 7, "text": "Students should maintain a minimum 6.5 CGPA without any backlogs."},
]


def stub(text):
    return lambda payload: {"content": [{"type": "text", "text": text}]}


def test_verify_answer_flags_missing_bad_and_invented():
    assert verify_answer("75% is required [1]", PASSAGES) == ([1], [])
    cited, w = verify_answer("It is 80% [1]", PASSAGES)
    assert cited == [1] and "80" in w[0]
    cited, w = verify_answer("It is 75% [7]", PASSAGES)
    assert cited == [] and any("not retrieved" in x for x in w) and any("no citation" in x for x in w)
    assert verify_answer("75%", PASSAGES)[1] == ["Answer contains no citation."]


def test_extractive_answers_with_citation_and_abstains():
    g = ExtractiveGenerator()
    a = g.generate("What is the minimum attendance required in a course?", PASSAGES)
    assert not a.abstained and a.cited == [1] and "75%" in a.answer and a.answer.endswith("[1]") and not a.warnings
    b = g.generate("Who won the cricket world cup?", PASSAGES)
    assert b.abstained and b.answer == ABSTAIN_MESSAGE
    assert g.generate("", PASSAGES).abstained


def test_extractive_uses_sentence_scorer_when_given():
    sents = split_sentences(PASSAGES[1]["text"] + " " + PASSAGES[0]["text"])
    g = ExtractiveGenerator(sentence_scorer=lambda q, ss: [1.0 if "CGPA" in s else 0.0 for s in ss], min_score=0.5)
    assert "CGPA" in g.generate("anything", PASSAGES).answer and sents
    assert ExtractiveGenerator(sentence_scorer=lambda q, ss: [0.0] * len(ss), min_score=0.5).generate("x", PASSAGES).abstained


def test_llm_generator_paths():
    ok = LLMGenerator(api_key="k", post=stub("The minimum attendance is 75% [1].")).generate("q", PASSAGES)
    assert not ok.abstained and ok.cited == [1] and not ok.warnings
    ab = LLMGenerator(api_key="k", post=stub("INSUFFICIENT")).generate("q", PASSAGES)
    assert ab.abstained and ab.answer == ABSTAIN_MESSAGE
    bad = LLMGenerator(api_key="k", post=stub("It is 90% [1].")).generate("q", PASSAGES)
    assert bad.warnings and not bad.abstained
    assert LLMGenerator(api_key="k", post=stub("x")).generate("q", []).abstained


def test_llm_prompt_contains_rules_and_passages():
    seen = {}

    def post(payload):
        seen.update(payload)
        return {"content": [{"type": "text", "text": "ok [1]"}]}
    LLMGenerator(api_key="k", post=post, answer_language="en").generate("What?", PASSAGES)
    assert "ONLY" in seen["system"] and "INSUFFICIENT" in seen["system"] and "English" in seen["system"]
    assert "[1] (Exam-Manual.pdf, p. 28)" in seen["messages"][0]["content"] and seen["temperature"] == 0


def test_direct_answers_are_marked_ungrounded():
    d = LLMGenerator(api_key="k", post=stub("About 75%.")).direct("q")
    assert not d.abstained and d.cited == [] and d.warnings
    assert LLMGenerator(api_key="k", post=stub("INSUFFICIENT")).direct("q").abstained


def test_extractive_on_corpus_attendance_question():
    chunks = load_chunks(config.CHUNKS_PATH)
    q = "What is the minimum attendance required to appear in the end semester examination?"
    ev = BM25Index(chunks).search(q, 5)
    a = ExtractiveGenerator().generate(q, ev)
    assert not a.abstained and "75%" in a.answer and a.cited
