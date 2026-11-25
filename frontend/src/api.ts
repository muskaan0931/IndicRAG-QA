// Typed API layer. Every page reads real backend responses through these functions.
export const API_BASE: string = (import.meta as any).env?.VITE_API_URL ?? 'http://127.0.0.1:8000';

export type MethodKey = 'bm25' | 'pretrained' | 'hybrid';

export interface Passage {
  rank: number; score: number | null; chunk_id: string; doc_id: string; document_name: string;
  category?: string | null; version?: string | null; page: number | null; text: string;
  language?: string | null; extraction_method?: string | null; ocr_confidence?: number | null;
  source_warnings?: string[]; bm25_rank?: number | null; dense_rank?: number | null;
  hybrid_fallback?: boolean; hybrid_fallback_reason?: string; reranker_score?: number; version_conflict_hint?: boolean; version_precedence_hint?: string;
}
export interface Citation { citation_id: string; chunk_id: string; doc_id: string; document_name: string; category?: string | null; version?: string | null; page: number | null; label: string }
export interface Lang { label: string; method: string; evidence: string }
export interface AskResponse {
  query: string; detected_language: Lang; retrieval_method: MethodKey; method_label: string; score_type: 'bm25' | 'cosine' | 'rrf';
  top_k: number; retrieval_status: 'ok' | 'no_results'; answer_mode: 'extractive' | 'llm' | 'abstained' | 'evidence_only' | string;
  generator?: string | null; answer: string | null; abstained?: boolean; cited_passages?: number[]; answer_warnings?: string[];
  generation_error?: string | null;
  answerability: { status: 'answerable' | 'unanswerable' | 'not_calibrated' | string; reason: string; signal?: number | null; threshold?: number | null; signal_type?: string };
  citations: Citation[]; evidence: Passage[]; latency_ms: number;
}
export type GeneratorKey = 'auto' | 'extractive' | 'llm' | 'none';
export interface MethodStatus { available: boolean; reason: string | null; loaded: boolean; label: string; last_error: string | null }
export interface DocumentInfo {
  doc_id: string; document_name: string; category: string | null; version: string | null; language: string | null;
  pages: number; chunks: number; extraction_methods: string[]; ocr_confidence_mean: number | null;
  ocr_confidence_min: number | null; low_confidence_pages: number[]; note: string | null;
}
export interface CorpusOverview {
  documents: number; pages: number; chunks: number; ocr_pages: number; native_pages: number;
  ocr_confidence_mean: number | null; low_confidence_pages: number; chunk_chars_median: number; chunk_chars_max: number; categories: string[];
}
export interface OverviewResponse {
  corpus: CorpusOverview; methods: Record<MethodKey, MethodStatus>;
  artifacts: { pretrained_index_present: boolean; finetuned_model_present: boolean; bm25_pilot_present: boolean; mlqa_results_present: boolean; answer_generation: boolean; answerability_calibrated?: boolean; eval_dataset_present?: boolean };
}
export interface CompareBlock { status: 'ok' | 'no_results' | 'unavailable' | 'error'; results: Passage[]; score_type: string; message: string | null; label: string; latency_ms: number }
export interface CompareResponse {
  query: string; top_k: number; detected_language: Lang; methods: Record<string, CompareBlock>;
  overlap: { chunk_id: string; document_name: string; page: number | null; ranks: Record<string, number> }[];
}
export interface ChunkSearch { total: number; offset: number; limit: number; chunks: Passage[] }
export interface EvalSummary { evaluated_queries: number; excluded_queries: number;[k: string]: number }
export interface MlqaRow {
  split: string; context_language: string; question_language: string; setting: string; method: string; model: string | null;
  corpus_passages: number; questions: number; 'recall@1': number; 'recall@3': number; 'recall@5': number; 'recall@10': number;
  'mrr@10': number; 'ndcg@10': number; seconds: number; source_file: string;
}
export interface EvaluationResponse {
  bm25_pilot: {
    summary: EvalSummary | null; per_query: Record<string, string>[]; label_provenance: string;
    judgements: { query_id: string; query: string; relevant_chunk_ids: string[]; relevant_ranks: Record<string, number | null>; judged: boolean }[];
  };
  per_method: Record<string, { label: string; summary: EvalSummary; source_file: string }>;
  mlqa: MlqaRow[];
  qa_eval?: Record<string, any>;
  modules: { id: number; title: string; status: string; note: string }[];
  metric_notes: Record<string, string>;
}
export interface ErrorCase {
  case_id: string; query: string; language: string; expected: string; retrieved: string; failure_type: string;
  method: string; observed: string; source: string;
}

export class ApiError extends Error {
  constructor(message: string, public status?: number, public offline = false) { super(message); }
}

async function request<T>(path: string, init?: RequestInit, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, signal });
  } catch (e: any) {
    if (e?.name === 'AbortError') throw e;
    throw new ApiError(`Cannot reach the API at ${API_BASE}. Start the backend (cd backend && uvicorn app.main:app --reload) and try again.`, undefined, true);
  }
  let body: any = null;
  try { body = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const d = body?.detail;
    const msg = typeof d === 'string' ? d : Array.isArray(d) ? d.map((x: any) => x.msg).join('; ') : `Request failed (HTTP ${res.status})`;
    throw new ApiError(msg, res.status);
  }
  return body as T;
}

const post = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const api = {
  health: (s?: AbortSignal) => request<any>('/health', undefined, s),
  overview: (s?: AbortSignal) => request<OverviewResponse>('/overview', undefined, s),
  ask: (query: string, method: MethodKey, top_k: number, s?: AbortSignal, generator: GeneratorKey = 'auto') =>
    request<AskResponse>('/ask', post({ query, method, top_k, generator }), s),
  compare: (query: string, top_k: number, s?: AbortSignal) => request<CompareResponse>('/compare', post({ query, top_k }), s),
  documents: (s?: AbortSignal) => request<{ documents: DocumentInfo[]; overview: CorpusOverview }>('/documents', undefined, s),
  documentDetail: (id: string, page?: number, s?: AbortSignal) =>
    request<{ document: DocumentInfo } & ChunkSearch>(`/documents/${encodeURIComponent(id)}${page ? `?page=${page}` : ''}`, undefined, s),
  chunks: (q: string, doc_id: string, offset: number, s?: AbortSignal) =>
    request<ChunkSearch>(`/chunks?${new URLSearchParams({ q, ...(doc_id ? { doc_id } : {}), limit: '10', offset: String(offset) })}`, undefined, s),
  evaluation: (s?: AbortSignal) => request<EvaluationResponse>('/evaluation', undefined, s),
  errors: (s?: AbortSignal) => request<{ cases: ErrorCase[]; required_minimum: number; documented: number }>('/errors', undefined, s),
  feedback: (payload: { query: string; answer: string; rating: "up" | "down"; method: string; cited_chunk_ids: string[]; note?: string }) => request<{ saved: boolean; message: string }>('/feedback', post(payload)),
};

// Frontend preview heuristic mirroring backend/app/language.py (the backend result is authoritative).
export function detectPreview(q: string): string {
  const dev = (q.match(/[\u0900-\u097F]/g) || []).length;
  const latin = q.match(/[A-Za-z]+/g) || [];
  if (dev && latin.length) return 'Code-mixed';
  if (dev) return 'Hindi';
  const hits = latin.filter(w => HINGLISH.has(w.toLowerCase()));
  if (hits.length >= 2 || (hits.length && latin.length <= 6)) return 'Hinglish';
  return 'English';
}
const HINGLISH = new Set('kya kaise kaisa kitna kitni kitne kab kaun kahan kyun kyon hai hain tha thi hoga hogi hona honi hota hoti chahiye ke ka ki liye mein nahi nahin agar toh aur ya par karna karne kare kar karo milta milti milega wala wali kaunsa kaunsi batao bataye sakta sakti sakte jaye jata jati uska unka iska yeh ye woh se tak bhi sirf lekin'.split(' '));