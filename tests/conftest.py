"""Test environment. If the real corpus (data/chunks.jsonl) is not present, tests run against a SMALL SYNTHETIC
fixture (tests/fixtures/mini_chunks.jsonl) so the plumbing can be verified. Fixture text is condensed from the real
documents; it is NOT a substitute for evaluating on the full corpus."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

USING_FIXTURE = not (ROOT / "data" / "chunks.jsonl").exists() and "INDICRAG_DATA_DIR" not in os.environ
if USING_FIXTURE:
    tmp = Path(tempfile.mkdtemp(prefix="indicrag_test_"))
    (tmp / "data" / "eval").mkdir(parents=True)
    (tmp / "reports").mkdir()
    shutil.copy(ROOT / "tests" / "fixtures" / "mini_chunks.jsonl", tmp / "data" / "chunks.jsonl")
    shutil.copy(ROOT / "data" / "eval" / "questions.jsonl", tmp / "data" / "eval" / "questions.jsonl")
    os.environ["INDICRAG_DATA_DIR"] = str(tmp / "data")
    os.environ["INDICRAG_REPORTS_DIR"] = str(tmp / "reports")
