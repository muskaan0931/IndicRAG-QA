"""Build and persist a dense passage index for a pretrained or fine-tuned E5 model.

On-disk format (unchanged, read by app/retrieval/dense.py): dense_embeddings.npy, dense_chunks.jsonl, dense_model.txt.
New, additive: index_config.json (model identification + build configuration). Existing indexes are never touched
unless you point --out at them.

Relative paths are resolved against the project root, so the working directory does not matter.

Examples:
  # pretrained multilingual E5 -> phase4_outputs/dense_index   (already built; do not rebuild unless needed)
  python backend/build_dense_index.py --model intfloat/multilingual-e5-base --out phase4_outputs/dense_index

  # fine-tuned model -> models/finetuned_e5_index  (separate directory; the pretrained index is preserved)
  python backend/build_dense_index.py --model models/finetuned_e5 --out models/finetuned_e5_index --batch-size 8

Device defaults to CPU. MPS is used only if you pass --device mps.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.retrieval.bm25 import load_chunks  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402


def _abs(p: str) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--model", default="intfloat/multilingual-e5-base",
                   help="Hugging Face model id, or a local SentenceTransformer folder (relative to the project root)")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                   help="default cpu; 'auto' lets sentence-transformers choose (may select MPS)")
    a = p.parse_args()

    chunks_path, out = _abs(a.chunks), _abs(a.out)
    local_model = _abs(a.model)
    model_ref = str(local_model) if local_model.exists() else a.model   # local folder wins over a hub id
    if local_model.exists() and not (local_model / "modules.json").exists():
        raise SystemExit(f"{local_model} exists but is not a SentenceTransformer folder (no modules.json). "
                         "Train first: python backend/train_e5_from_reviewed_data.py")

    chunks = load_chunks(str(chunks_path))
    r = DenseRetriever(model_ref, device=None if a.device == "auto" else a.device)
    emb = r.build(chunks, batch_size=a.batch_size)
    r.model_name = a.model          # record the portable name/path as given, not a machine-specific absolute path
    r.save(out)

    # verify what was written: reload against the corpus (count, ids, order) and check normalisation
    import numpy as np
    r.load(out, expected_chunk_ids=[c["chunk_id"] for c in chunks])   # re-reads the files just written
    norms = np.linalg.norm(r.embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit("Built embeddings are not L2-normalised.")

    weights = {}
    if local_model.exists():
        for f in sorted(local_model.glob("*.safetensors")) + sorted(local_model.glob("pytorch_model*.bin")):
            weights[f.name] = _sha256(f)
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except Exception:  # noqa: BLE001
        st_version = None
    cfg = {
        "built_utc": datetime.now(timezone.utc).isoformat(), "model": a.model, "model_is_local_folder": local_model.exists(),
        "model_weights_sha256": weights, "n_chunks": len(chunks), "embedding_dim": int(emb.shape[1]),
        "dtype": "float32", "l2_normalised": True, "passage_prefix": "passage: ", "query_prefix": "query: ",
        "batch_size": a.batch_size, "device": a.device, "chunks_path": str(chunks_path.relative_to(ROOT)) if chunks_path.is_relative_to(ROOT) else str(chunks_path),
        "chunks_sha256": _sha256(chunks_path), "sentence_transformers": st_version,
    }
    (out / "index_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks with {a.model}; embedding shape {emb.shape}; saved to {out} (verified)")


if __name__ == "__main__":
    main()
