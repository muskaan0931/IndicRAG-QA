"""Automatic consistency checks for data/eval/questions.jsonl. These checks CANNOT replace human review:
they prove that gold passages exist and contain the gold answer, and that 'unanswerable' topics are absent from
the corpus, but not that a question is well-posed or that the gold list is complete.

Usage (from repo root):  python backend/verify_dataset.py
Exit code is non-zero if any hard check fails. Needs data/chunks.jsonl."""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import config  # noqa: E402
from app.corpus import read_jsonl  # noqa: E402
from app.evalkit import LANGUAGES, load_dataset, normalize  # noqa: E402
from app.language import detect_language  # noqa: E402


def main() -> int:
    if not config.CHUNKS_PATH.exists():
        print(f"ERROR corpus file missing: {config.CHUNKS_PATH}")
        return 2
    chunks = {c["chunk_id"]: c for c in read_jsonl(config.CHUNKS_PATH)}
    norm_text = {cid: f" {normalize(c['text'])} " for cid, c in chunks.items()}
    rows = load_dataset()
    errors, warnings, info = [], [], set()

    ids = [r["qid"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate qid values")

    facts = defaultdict(list)
    for r in rows:
        facts[r["fact_id"]].append(r)
    for fid, rs in facts.items():
        if {r["language"] for r in rs} != set(LANGUAGES):
            errors.append(f"{fid}: does not have exactly the three languages")
        if len({r["split"] for r in rs}) != 1:
            errors.append(f"{fid}: language variants are split across dev/test (leakage)")

    for r in rows:
        q = r["qid"]
        label = detect_language(r["question"])["label"]
        ok = {"English": {"English"}, "Hindi": {"Hindi", "Code-mixed"}, "Hinglish": {"Hinglish"}}[r["language"]]
        if label not in ok:
            warnings.append(f"{q}: labelled {r['language']} but heuristic says {label}")
        if r["answerable"]:
            missing = [c for c in r["gold_chunk_ids"] if c not in chunks]
            if missing or not r["gold_chunk_ids"]:
                errors.append(f"{q}: gold chunk(s) missing from corpus: {missing}")
                continue
            ans = normalize(r["answer_text"])
            if not any(f" {ans} " in norm_text[c] for c in r["gold_chunk_ids"]):
                errors.append(f"{q}: answer '{r['answer_text']}' not found in any gold chunk")
            if len(ans) >= 8:  # long answers are specific enough that other hits may be missing gold labels
                extra = [c for c, t in norm_text.items() if f" {ans} " in t and c not in r["gold_chunk_ids"]]
                if 0 < len(extra) <= 5:   # a handful of hits: likely missing gold labels, check them by hand
                    warnings.append(f"{q}: answer text also appears in non-gold chunks {extra} (gold list may be incomplete)")
                elif len(extra) > 5:
                    info.add(f"{r['fact_id']}: answer string '{r['answer_text']}' is generic (appears in {len(extra)} "
                             "other chunks); gold is judged by the question, so recall may be understated")
        else:
            for term in r["absent_terms"]:  # each term is a case-insensitive regex
                hits = [cid for cid, c in chunks.items() if re.search(term, c["text"], re.I)]
                if hits:
                    errors.append(f"{q}: 'unanswerable' but term '{term}' occurs in {hits[:3]}")
            if not r["absent_terms"]:
                errors.append(f"{q}: unanswerable question without absent_terms")

    n_ans = sum(r["answerable"] for r in rows)
    print(f"{len(rows)} questions · {n_ans} answerable · {len(rows) - n_ans} unanswerable · {len(facts)} facts")
    print("human-verified:", sum(bool(r.get("verified")) for r in rows), f"of {len(rows)}")
    for i in sorted(info):
        print("INFO ", i)
    for w in sorted(set(warnings)):
        print("WARN ", w)
    for e in sorted(set(errors)):
        print("ERROR", e)
    print("RESULT:", "FAILED" if errors else "all automatic checks passed (human review still required)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
