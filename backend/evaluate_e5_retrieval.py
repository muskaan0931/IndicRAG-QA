#!/usr/bin/env python3
"""Retrieval evaluation: pretrained multilingual-E5 vs fine-tuned multilingual-E5 on the reviewed questions.

EVALUATION DESIGN (why it looks like this)
  * Eligible questions: answerable=true, reviewer_ok=Y, all gold ids in the corpus (same filter as training).
  * The reviewed set is tiny (a few independent groups), so there is no honest fixed train/test split. Instead:
      1. pretrained          - no training, so every eligible question is a valid test question.
      2. finetuned_cv        - grouped K-fold cross-validation. For each fold a FRESH base model is fine-tuned on the
                               training groups only and scored on the held-out groups. Groups = connected components
                               of (fact_id <-> gold chunk), so no fact, language variant or gold passage ever appears
                               on both sides. The pooled held-out results are a VALID generalisation estimate and
                               are what pretrained is compared against (paired, same questions).
      3. finetuned_final     - the saved models/finetuned_e5 model scored on all eligible questions. It was trained
                               on these very questions, so this is NOT a held-out result; it is reported only as a
                               sanity check of the saved model/index and is labelled as such.
  * Same corpus (all chunks in data/chunks.jsonl), same E5 prefixes, cosine over L2-normalised embeddings.
  * Metrics: Hit@k, Recall@k (fraction of the full gold set in the top k), MRR over the full ranking, nDCG@10.
  * Labels are assistant-assisted screening, not independent human verification.

Usage:
    python backend/evaluate_e5_retrieval.py --dry-run      # plan + leakage check, no torch
    python backend/evaluate_e5_retrieval.py                # full run (pretrained + CV + final)
    python backend/evaluate_e5_retrieval.py --skip-cv      # pretrained + final only (fast)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_ft_common import (  # noqa: E402
    BASE_MODEL, DEFAULT_EVAL_OUT, DEFAULT_FT_INDEX, DEFAULT_FT_MODEL, DEFAULT_PRETRAINED_INDEX, LABEL_PROVENANCE,
    METRICS, PASSAGE_PREFIX, QUERY_PREFIX, Fold, ReviewedQuestion, add_training_args, assign_groups, build_pairs,
    by_language, load_chunks, load_reviewed_questions, make_folds, negative_pool, paired_difference, resolve_path,
    score_questions, sha256_file, summarize, train_cfg, verify_index,
)

log = logging.getLogger("e5eval")


# ------------------------------------------------------------------------------------------ encoders
class STEncoder:
    """Wraps a SentenceTransformer with the E5 prefix convention used everywhere in the project."""

    def __init__(self, model, batch_size: int = 16):
        self.model, self.batch_size = model, batch_size

    def _enc(self, texts: list[str], prefix: str, progress: bool = False) -> np.ndarray:
        return np.asarray(self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                            convert_to_numpy=True, normalize_embeddings=True,
                                            show_progress_bar=progress), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, QUERY_PREFIX)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, PASSAGE_PREFIX, progress=True)


class FoldSystem:
    """What a fold produces: a query encoder, the corpus embeddings under that fold's model, and a cleanup hook."""

    def __init__(self, encode_queries: Callable[[list[str]], np.ndarray], passage_emb: np.ndarray,
                 close: Callable[[], None] | None = None, history: list[dict] | None = None):
        self.encode_queries, self.passage_emb, self._close, self.history = encode_queries, passage_emb, close, history or []

    def close(self) -> None:
        if self._close:
            self._close()


def make_real_fit_fn(args, cfg: dict, chunks: dict, all_questions: list[ReviewedQuestion]) -> Callable[[list, int], FoldSystem]:
    """Fold trainer: fresh base model -> fine-tune on train groups only -> embed the whole corpus."""
    from sentence_transformers import SentenceTransformer
    from train_e5_from_reviewed_data import fine_tune

    chunk_ids = list(chunks)
    pool = negative_pool(chunks, all_questions)   # excludes gold of train AND held-out -> clean

    def fit(train_qs: list[ReviewedQuestion], fold_idx: int) -> FoldSystem:
        model = SentenceTransformer(cfg["base_model"], device=cfg["device"])
        history = fine_tune(model, build_pairs(train_qs), chunks, pool, cfg)
        enc = STEncoder(model, args.encode_batch_size)
        pe = enc.encode_passages([chunks[c]["text"] for c in chunk_ids])

        def close() -> None:
            nonlocal model
            del model
            gc.collect()
        return FoldSystem(enc.encode_queries, pe, close, history)
    return fit


# ------------------------------------------------------------------------------------------ CV driver
def run_cv(questions: list[ReviewedQuestion], chunk_ids: list[str], folds: list[Fold],
           fit_fn: Callable[[list, int], FoldSystem], group_of: dict[str, str], out_dir: Path,
           signature: str, resume: bool) -> list[dict[str, Any]]:
    fold_of = {q.qid: f.index for f in folds for q in f.test}
    rows: list[dict[str, Any]] = []
    for f in folds:
        cache = out_dir / f"cv_fold_{f.index}.json"
        if resume and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("signature") == signature:
                log.info("fold %d: reusing cached results (%s)", f.index, cache.name)
                rows += data["rows"]
                continue
            log.warning("fold %d: cache signature mismatch, recomputing", f.index)
        log.info("fold %d/%d: train %d questions, held-out %d questions, groups %s",
                 f.index + 1, len(folds), len(f.train), len(f.test), list(f.test_groups))
        system = fit_fn(list(f.train), f.index)
        try:
            qe = system.encode_queries([q.question for q in f.test])
            fold_rows = score_questions("finetuned_cv_heldout", f.test, qe, system.passage_emb, chunk_ids, group_of, fold_of)
        finally:
            system.close()
        losses = [h["loss"] for h in system.history]
        for r in fold_rows:
            r["fold_train_loss_first_last"] = [losses[0], losses[-1]] if losses else None
        cache.write_text(json.dumps({"signature": signature, "rows": fold_rows}, ensure_ascii=False), encoding="utf-8")
        rows += fold_rows
    return rows


# ------------------------------------------------------------------------------------------ outputs
def write_rows(out: Path, rows: list[dict[str, Any]]) -> None:
    with open(out / "per_question_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["system", "qid", "fact_id", "group", "language", "split_original", "fold", "question", "gold_chunk_ids",
            "gold_best_rank", "top10_chunk_ids", "top10_scores", *METRICS]
    with open(out / "per_question_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gold_chunk_ids": ";".join(r["gold_chunk_ids"]), "top10_chunk_ids": ";".join(r["top10_chunk_ids"]),
                        "top10_scores": ";".join(map(str, r["top10_scores"]))})


def fmt(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['ci95_lo']:.3f}, {m['ci95_hi']:.3f}]"


def markdown_report(meta: dict, systems: dict, comparison: dict | None) -> str:
    L = ["# E5 retrieval evaluation (generated by backend/evaluate_e5_retrieval.py)", "",
         f"Generated {meta['generated_utc']}. All numbers below come from the run recorded in `eval_config.json`.", "",
         f"**Label provenance:** {LABEL_PROVENANCE}", "",
         f"**Questions:** {meta['n_questions']} ({meta['n_facts']} facts, {meta['n_groups']} independent groups). "
         f"Excluded although marked Y: {len(meta['excluded_after_Y'])} rows (gold chunk id missing from corpus).", ""]
    shown = ["hit@1", "hit@3", "hit@5", "hit@10", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
    L += ["## Valid comparison (held-out)", ""]
    valid = [s for s in ("pretrained", "finetuned_cv_heldout") if s in systems]
    if len(valid) == 2:
        L += ["| metric | pretrained | fine-tuned (grouped-CV, held-out) | paired diff (ft - pre) |", "|---|---|---|---|"]
        for m in shown:
            d = comparison["metrics"][m]
            L.append(f"| {m} | {fmt(systems['pretrained']['metrics'][m])} | {fmt(systems['finetuned_cv_heldout']['metrics'][m])} | "
                     f"{d['mean']:+.3f} [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}] |")
        L += ["", "Brackets: 95% group-level cluster bootstrap. With this few groups the intervals are very wide; "
              "treat differences smaller than the interval as not established.", "",
              "### Per-group MRR (pretrained -> fine-tuned held-out)", "", "| group | n | pretrained | fine-tuned |", "|---|---|---|---|"]
        for g, v in comparison["per_group_mrr"].items():
            L.append(f"| {g} | {v['n_questions']} | {v['a_mrr']:.3f} | {v['b_mrr']:.3f} |")
        L.append("")
    elif "pretrained" in systems:
        L += ["Grouped-CV fine-tuned results not available in this run (`--skip-cv`). Pretrained only:", "",
              "| metric | pretrained |", "|---|---|"] + [f"| {m} | {fmt(systems['pretrained']['metrics'][m])} |" for m in shown] + [""]
    if "finetuned_final_resubstitution" in systems:
        L += ["## Final saved model - NOT a held-out result", "",
              "`models/finetuned_e5` was trained on these same questions, so these numbers are optimistic by construction. "
              "They only confirm the saved model and its index load and behave sensibly. Do not cite them as generalisation.", "",
              "| metric | final model (trained on these questions) |", "|---|---|"]
        L += [f"| {m} | {fmt(systems['finetuned_final_resubstitution']['metrics'][m])} |" for m in shown] + [""]
    L += ["## By language", ""]
    for s in ("pretrained", "finetuned_cv_heldout"):
        if s in systems:
            L += [f"### {s}", "", "| language | n | hit@1 | hit@5 | mrr |", "|---|---|---|---|---|"]
            for lang, v in systems[s]["by_language"].items():
                L.append(f"| {lang} | {v['n_questions']} | {v['metrics']['hit@1']['mean']:.3f} | {v['metrics']['hit@5']['mean']:.3f} | {v['metrics']['mrr']['mean']:.3f} |")
            L.append("")
    L += ["## Limitations", "",
          "- Tiny, AI-drafted question set; labels are assistant-assisted, not independent human annotation.",
          "- The 3 language variants of each fact and facts sharing a gold passage are NOT independent; groups are the unit of analysis.",
          "- Hyper-parameters were fixed a priori (not tuned on held-out data). Fold models are trained on a subset of the groups, "
          "so they are a conservative stand-in for the final model, whose own generalisation is not directly measured.",
          "- Gold lists may be incomplete, which understates recall for every system equally.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(ap)
    ap.add_argument("--folds", type=int, default=4, help="grouped K-fold (capped at the number of groups)")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse cached cv_fold_*.json with a matching signature")
    ap.add_argument("--pretrained-index", default=str(DEFAULT_PRETRAINED_INDEX))
    ap.add_argument("--finetuned-model", default=str(DEFAULT_FT_MODEL))
    ap.add_argument("--finetuned-index", default=str(DEFAULT_FT_INDEX))
    ap.add_argument("--out", default=str(DEFAULT_EVAL_OUT))
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "eval_run.log", mode="a", encoding="utf-8")])

    chunks = load_chunks(args.chunks)
    chunk_ids = list(chunks)
    questions, report = load_reviewed_questions(args.review_csv, chunks)
    group_of = assign_groups(questions)
    folds = make_folds(questions, args.folds)
    cfg = train_cfg(args)
    n_groups = len(set(group_of.values()))
    log.info("eligible: %d questions, %d facts, %d groups; excluded-after-Y: %d", len(questions),
             report["eligible_facts"], n_groups, len(report["excluded_after_Y"]))
    for f in folds:
        log.info("fold %d: held-out groups %s (%d q) | train %d q | leakage check passed", f.index, list(f.test_groups), len(f.test), len(f.train))
    (out / "fold_plan.json").write_text(json.dumps([{"fold": f.index, "test_groups": list(f.test_groups),
        "test_qids": [q.qid for q in f.test], "train_qids": [q.qid for q in f.train]} for f in folds], indent=2), encoding="utf-8")
    if args.dry_run:
        log.info("dry run: nothing evaluated.")
        return

    from sentence_transformers import SentenceTransformer
    systems_rows: dict[str, list[dict]] = {}

    # 1. pretrained: persisted index (verified aligned) + pretrained query encoder
    pre_emb, pre_info = verify_index(args.pretrained_index, chunk_ids, expected_model=args.base_model)
    log.info("pretrained index OK: %s", pre_info)
    base = SentenceTransformer(args.base_model, device=args.device)
    qe = STEncoder(base, args.encode_batch_size).encode_queries([q.question for q in questions])
    systems_rows["pretrained"] = score_questions("pretrained", questions, qe, pre_emb, chunk_ids, group_of)
    del base
    gc.collect()

    # 2. grouped-CV fine-tuned (valid held-out estimate)
    if not args.skip_cv:
        signature = hashlib.sha256(json.dumps([cfg, [[q.qid for q in f.test] for f in folds], sha256_file(resolve_path(args.chunks)),
                                               sha256_file(resolve_path(args.review_csv))], sort_keys=True).encode()).hexdigest()[:16]
        fit = make_real_fit_fn(args, cfg, chunks, questions)
        systems_rows["finetuned_cv_heldout"] = run_cv(questions, chunk_ids, folds, fit, group_of, out, signature, args.resume)

    # 3. final saved model (trained on all eligible questions -> NOT held-out)
    ft_model = resolve_path(args.finetuned_model)
    if not args.skip_final and ft_model.exists() and (ft_model / "training_metadata.json").exists():
        ft_emb, ft_info = verify_index(args.finetuned_index, chunk_ids)
        log.info("fine-tuned index OK: %s", ft_info)
        # the index must have been built from THIS model's weights (guards against a stale index after retraining)
        icfg_p = resolve_path(args.finetuned_index) / "index_config.json"
        trained = json.loads((ft_model / "training_metadata.json").read_text(encoding="utf-8")).get("weights_sha256", {})
        if icfg_p.exists() and trained and json.loads(icfg_p.read_text(encoding="utf-8")).get("model_weights_sha256") != trained:
            raise SystemExit(f"{args.finetuned_index} was built from different weights than {ft_model}. Rebuild the index.")
        m = SentenceTransformer(str(ft_model), device=args.device)
        qe = STEncoder(m, args.encode_batch_size).encode_queries([q.question for q in questions])
        systems_rows["finetuned_final_resubstitution"] = score_questions("finetuned_final_resubstitution", questions, qe, ft_emb, chunk_ids, group_of)
    elif not args.skip_final:
        log.warning("final model/index not found at %s - skipped (train it first).", ft_model)

    # aggregate + write
    systems = {s: {**summarize(r), "by_language": by_language(r),
                   "valid_generalisation_estimate": s != "finetuned_final_resubstitution"} for s, r in systems_rows.items()}
    comparison = None
    if "finetuned_cv_heldout" in systems_rows:
        comparison = paired_difference(systems_rows["pretrained"], systems_rows["finetuned_cv_heldout"])
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "n_questions": len(questions),
            "n_facts": report["eligible_facts"], "n_groups": n_groups, "excluded_after_Y": report["excluded_after_Y"]}
    all_rows = [r for rows in systems_rows.values() for r in rows]
    write_rows(out, all_rows)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "systems": systems, "paired_pretrained_vs_finetuned_cv": comparison},
                                                  indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "eval_config.json").write_text(json.dumps({
        "args": vars(args), "train_cfg": cfg, "selection_report": report, "pretrained_index": pre_info,
        "chunks_sha256": sha256_file(resolve_path(args.chunks)), "review_csv_sha256": sha256_file(resolve_path(args.review_csv)),
        "label_provenance": LABEL_PROVENANCE}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "comparison.md").write_text(markdown_report(meta, systems, comparison), encoding="utf-8")
    log.info("wrote results to %s", out)
    print((out / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
