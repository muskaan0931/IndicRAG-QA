"""Helpers shared by the evaluate_*.py scripts: a cached retriever over the live engine and report writers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import config  # noqa: E402
from app.main import ALL_METHODS, MethodUnavailable, engine  # noqa: E402


def make_retrieve():
    cache: dict = {}

    def retrieve(question: str, method: str, k: int):
        key = (question, method, k)
        if key not in cache:
            cache[key] = engine.retrieve(question, method, k)
        return cache[key]
    return retrieve


def method_status(methods):
    """Split methods into usable ones and {method: reason} for those that cannot run here."""
    engine.ensure_core()
    ok, skipped = [], {}
    for m in methods:
        try:
            engine.retrieve("test", m, 1)
            ok.append(m)
        except MethodUnavailable as exc:
            skipped[m] = str(exc)
    return ok, skipped


def write_json(name: str, data) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = config.REPORTS_DIR / name
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
