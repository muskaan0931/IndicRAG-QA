# Corpus recovery status

The six supplied Bennett University PDFs are bundled in `data/source_pdfs/`.

- `data/pages.jsonl` and `data/chunks.jsonl` have been rebuilt from PDF extraction for the original five-domain corpus, and the 2025 SCSET brochure has been appended using native PDF text extraction.
- Several supplied policy PDFs are scanned/image-based. Current generated corpus files therefore contain blank text on some pages. The recovery script `backend/restore_corpus_from_pdfs.py` has an OCR fallback (Tesseract, English) for low-text pages. Run it locally to regenerate page and chunk files with OCR; inspect extraction quality before use.
- Dense embeddings/index are not bundled because `sentence-transformers` is not installed in this build environment and model downloads are not verified.
- Fine-tuned E5 weights are not bundled. Training requires human-reviewed relevance labels and a completed training run.
- Question pool remains draft AI-generated and not human-verified. Scores are not reported as experimental results.
