#!/usr/bin/env python3
"""Fine-tune multilingual E5 on the reviewed IndicRAG-QA question -> gold-passage pairs.

Training data (strict): rows of data/eval/questions_for_review.csv with answerable=true AND reviewer_ok=Y AND every
gold chunk id present in data/chunks.jsonl. Rows marked N, unreviewed rows and rows with unknown gold ids are NEVER
used. The Y/N labels are ASSISTANT-ASSISTED screening, not independent human verification.

Method: contrastive InfoNCE (query -> passage) with E5 prefixes ("query: " / "passage: "), cosine similarity,
temperature 0.02. Negatives are the other positives in the batch plus a few random unlabeled corpus chunks. Any
passage that is gold for the query is masked out, so sibling questions / other languages of the same fact (which
share passages) are never used as false negatives. A small explicit PyTorch loop is used instead of
SentenceTransformer.fit(), so the `datasets` and `accelerate` packages are NOT needed.

Nothing here reports metrics. Training metadata is written only after training actually completes; retrieval
quality is measured by backend/evaluate_e5_retrieval.py.

Usage (from anywhere; paths are resolved against the project root):
    python backend/train_e5_from_reviewed_data.py --dry-run          # data selection only, no torch needed
    python backend/train_e5_from_reviewed_data.py                    # train on CPU -> models/finetuned_e5
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_ft_common import (  # noqa: E402
    DEFAULT_FT_MODEL, LABEL_PROVENANCE, PASSAGE_PREFIX, QUERY_PREFIX, ROOT, add_training_args, assign_groups,
    batch_layout, build_pairs, load_chunks, load_reviewed_questions, negative_pool, resolve_path, sha256_file,
    train_cfg,
)

MIN_PAIRS = 20


# ------------------------------------------------------------------------------------------ core training
def _embed(model, texts: list[str], device: str, grad: bool):
    """SentenceTransformer forward pass with gradients, returning L2-normalised sentence embeddings."""
    import torch
    import torch.nn.functional as F

    feats = model.tokenize(texts)
    feats = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}
    with torch.set_grad_enabled(grad):
        out = model(feats)["sentence_embedding"]
    return F.normalize(out, p=2, dim=1)


def fine_tune(model, pairs, chunks: dict[str, dict], neg_pool: list[str], cfg: dict[str, Any],
              log: Callable[[dict], None] | None = None) -> list[dict[str, Any]]:
    """Train `model` (a SentenceTransformer) in place. Returns the per-step history."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = cfg["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available; use --device cpu.")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    model.max_seq_length = cfg["max_seq_length"]
    model.to(device)
    if cfg["freeze_embeddings"]:
        try:
            model[0].auto_model.embeddings.word_embeddings.weight.requires_grad_(False)
        except AttributeError:
            print("WARNING: could not locate the word-embedding matrix; training all parameters.", file=sys.stderr)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable, n_total = sum(p.numel() for p in params), sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_trainable:,} / {n_total:,}")

    steps_per_epoch = math.ceil(len(pairs) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(total_steps * cfg["warmup_ratio"]))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    history: list[dict[str, Any]] = []
    step, t0 = 0, time.monotonic()
    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for b in range(0, len(order), cfg["batch_size"]):
            batch = [pairs[i] for i in order[b:b + cfg["batch_size"]]]
            extras = rng.sample(neg_pool, min(cfg["extra_negatives"], len(neg_pool))) if neg_pool else []
            cols, target, mask = batch_layout(batch, extras)
            n_pos_cols = len({p.positive_id for p in batch})
            if len(cols) - 1 < 1:   # nothing to contrast against
                continue
            q = _embed(model, [QUERY_PREFIX + p.question for p in batch], device, grad=True)
            pos = _embed(model, [PASSAGE_PREFIX + chunks[c]["text"] for c in cols[:n_pos_cols]], device, grad=True)
            if len(cols) > n_pos_cols:   # extra negatives: forward only (no activation memory); still shape the query
                neg = _embed(model, [PASSAGE_PREFIX + chunks[c]["text"] for c in cols[n_pos_cols:]], device, grad=False)
                P = torch.cat([pos, neg.detach()], dim=0)
            else:
                P = pos
            logits = (q @ P.T) / cfg["temperature"]
            logits = logits.masked_fill(torch.tensor(mask, device=device), -1e9)
            loss = F.cross_entropy(logits, torch.tensor(target, device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": round(float(loss.item()), 6), "lr": sched.get_last_lr()[0],
                   "batch_pairs": len(batch), "columns": len(cols), "masked": int(sum(map(sum, mask))),
                   "elapsed_s": round(time.monotonic() - t0, 1)}
            history.append(rec)
            if log:
                log(rec)
            print(f"epoch {epoch}/{cfg['epochs']} step {step}/{total_steps} loss {rec['loss']:.4f} ({rec['elapsed_s']}s)")
    model.eval()
    return history


# ------------------------------------------------------------------------------------------ CLI
def _versions() -> dict[str, str]:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "sentence_transformers", "transformers", "numpy"):
        try:
            v[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            v[name] = "not installed"
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(ap)
    ap.add_argument("--output-dir", default=str(DEFAULT_FT_MODEL))
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output directory")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--dry-run", action="store_true", help="select/validate data and print the plan; no torch, no training")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    questions, report = load_reviewed_questions(args.review_csv, chunks)
    pairs = build_pairs(questions)
    pool = negative_pool(chunks, questions)
    groups = assign_groups(questions)
    cfg = train_cfg(args)
    out = resolve_path(args.output_dir)

    print("== data selection ==")
    print(json.dumps({k: v for k, v in report.items() if k != "label_provenance"}, indent=2, ensure_ascii=False))
    print(f"training pairs: {len(pairs)} from {len(questions)} questions / {report['eligible_facts']} facts / "
          f"{len(set(groups.values()))} independent groups; extra-negative pool: {len(pool)} chunks")
    print(LABEL_PROVENANCE)
    if report["excluded_after_Y"]:
        print("WARNING: rows marked Y but EXCLUDED (not repaired, not used):")
        for e in report["excluded_after_Y"]:
            print("   ", e)
    if len(pairs) < args.min_pairs:
        raise SystemExit(f"Need >= {args.min_pairs} reviewed training pairs; found {len(pairs)}.")
    if args.dry_run:
        print("dry run only - nothing trained.")
        return

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty. Re-run with --overwrite to write into it.")
    try:
        from sentence_transformers import SentenceTransformer
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install sentence-transformers (pulls in torch): pip install -r backend/requirements.txt") from exc

    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    log_path = out / "training_log.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(rec: dict) -> None:
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    model = SentenceTransformer(args.base_model, device=args.device)
    try:
        history = fine_tune(model, pairs, chunks, pool, cfg, log)
    finally:
        log_f.close()
    if not history:
        raise SystemExit("No optimisation steps were executed; refusing to save a model.")
    model.save(str(out))

    # Integrity check: the saved folder must reload and reproduce the in-memory model's embeddings.
    import numpy as np
    probes = [QUERY_PREFIX + questions[0].question, PASSAGE_PREFIX + chunks[questions[0].gold[0]]["text"]]
    a = model.encode(probes, normalize_embeddings=True, convert_to_numpy=True)
    reloaded = SentenceTransformer(str(out), device=args.device)
    b = reloaded.encode(probes, normalize_embeddings=True, convert_to_numpy=True)
    min_cos = float(np.min(np.sum(a * b, axis=1)))
    if min_cos < 0.999:
        raise SystemExit(f"Saved model does not reproduce in-memory embeddings (min cosine {min_cos:.5f}).")

    finished = datetime.now(timezone.utc)
    weights = {p.name: sha256_file(p) for p in sorted(out.glob("*.safetensors")) + sorted(out.glob("pytorch_model*.bin"))}
    (out / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    meta = {
        "status": "completed",
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "duration_s": round(time.monotonic() - t0, 1),
        "base_model": args.base_model, "config": cfg, "versions": _versions(),
        "objective": "InfoNCE query->passage, cosine/temperature, gold-aware masking, E5 query:/passage: prefixes",
        "data": {
            "selection_report": report, "n_questions": len(questions), "n_facts": report["eligible_facts"],
            "n_groups": len(set(groups.values())), "n_pairs": len(pairs), "extra_negative_pool": len(pool),
            "qids_used": [q.qid for q in questions],
            "chunks_sha256": sha256_file(resolve_path(args.chunks)), "review_csv_sha256": sha256_file(resolve_path(args.review_csv)),
        },
        "label_provenance": LABEL_PROVENANCE,
        "training": {"steps": len(history), "first_step_loss": history[0]["loss"], "last_step_loss": history[-1]["loss"],
                     "note": "training loss only; it says nothing about retrieval quality"},
        "reload_check_min_cosine": round(min_cos, 6),
        "weights_sha256": weights,
        "evaluation": "NOT performed here. Run backend/evaluate_e5_retrieval.py",
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fine-tuned model to {out} ({len(pairs)} pairs, {len(history)} steps, reload check cos>={min_cos:.5f})")


if __name__ == "__main__":
    main()
