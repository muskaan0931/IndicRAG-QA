import collections
import json
import subprocess
import sys
from pathlib import Path

import conftest
from app import config
from app.evalkit import LANGUAGES, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_structure():
    rows = load_dataset()
    assert len(rows) == len({r["qid"] for r in rows}) == 93
    facts = collections.defaultdict(list)
    for r in rows:
        facts[r["fact_id"]].append(r)
    for fid, rs in facts.items():
        assert {r["language"] for r in rs} == set(LANGUAGES), fid
        assert len({r["split"] for r in rs}) == 1, f"{fid} leaks across splits"
    assert {r["split"] for r in rows} == {"dev", "test"}
    for r in rows:
        assert r["verified"] is False and r["source"] == "AI-drafted"           # never claim human verification
        assert bool(r["gold_chunk_ids"]) == r["answerable"] and bool(r["absent_terms"]) == (not r["answerable"])
    assert sum(not r["answerable"] for r in rows) == 30


def test_dev_and_test_both_have_both_classes():
    for split in ("dev", "test"):
        rs = load_dataset(split=split)
        assert {r["answerable"] for r in rs} == {True, False}


def test_builder_reproduces_checked_in_dataset(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bq", ROOT / "data" / "eval" / "build_questions.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.OUT = tmp_path
    mod.main()
    assert (tmp_path / "questions.jsonl").read_text(encoding="utf-8") == (ROOT / "data" / "eval" / "questions.jsonl").read_text(encoding="utf-8")


def test_verify_dataset_runs_on_available_corpus():
    out = subprocess.run([sys.executable, str(ROOT / "backend" / "verify_dataset.py")], capture_output=True, text=True,
                         env={**__import__("os").environ, "INDICRAG_DATA_DIR": str(config.DATA_DIR)})
    # With the real corpus this must pass; with the fixture, gold chunks for ids that the fixture lacks would fail,
    # so only the script mechanics are asserted there.
    assert "questions" in out.stdout
    if not conftest.USING_FIXTURE:
        assert out.returncode == 0, out.stdout
