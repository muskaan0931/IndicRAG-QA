import { Fragment, useEffect, useState, type ReactNode } from 'react';
import { Activity, AlertTriangle, ArrowLeft, ArrowRight, Check, ChevronDown, ChevronRight, ClipboardList, Copy, FileSearch, FileText, Quote, Sparkles, Globe2, Layers3, Library, Loader2, Microscope, RefreshCw, Search, ShieldCheck } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, type AskResponse, type CompareResponse, type DocumentInfo, type ErrorCase, type EvaluationResponse, type MethodKey, type MlqaRow, type OverviewResponse, type Passage } from './api';

export const METHOD_OPTIONS: { label: string; key: MethodKey }[] = [
  { label: 'Hybrid Retrieval (RRF)', key: 'hybrid' },
  { label: 'Pretrained Dense (E5)', key: 'pretrained' },
  { label: 'BM25 Lexical', key: 'bm25' },
];

export function PageHead({ eyebrow, title, desc, action }: { eyebrow: string; title: string; desc: string; action?: ReactNode }) {
  return <div className="page-head"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{desc}</p></div>{action}</div>;
}

/* ---------- shared helpers ---------- */
function useLoad<T>(fn: (s: AbortSignal) => Promise<T>, deps: unknown[] = []) {
  const [state, set] = useState<{ data: T | null; error: string; loading: boolean }>({ data: null, error: '', loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const c = new AbortController();
    set(s => ({ ...s, loading: true, error: '' }));
    fn(c.signal).then(d => set({ data: d, error: '', loading: false }))
      .catch((e: any) => { if (e?.name !== 'AbortError') set({ data: null, error: e?.message || 'Request failed', loading: false }); });
    return () => c.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { ...state, reload: () => setTick(t => t + 1) };
}

function Loading({ text }: { text: string }) {
  return <div className="state-box" role="status"><Loader2 size={22} className="spin" /><b>{text}</b></div>;
}
function ErrorBox({ title, message, onRetry, children }: { title: string; message: string; onRetry?: () => void; children?: ReactNode }) {
  return <div className="state-box error" role="alert"><AlertTriangle size={22} /><b>{title}</b><span>{message}</span>
    <div className="state-actions">{onRetry && <button className="btn secondary" onClick={onRetry}><RefreshCw size={14} /> Try again</button>}{children}</div></div>;
}
const fmt = (v: unknown, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
const SCORE_LABEL: Record<string, string> = { bm25: 'BM25 score', cosine: 'Cosine similarity', rrf: 'RRF score' };

/* ---------- passage card ---------- */
/** Wraps the sentences of `mark` that appear verbatim (ignoring case and whitespace) in `text` in <mark>. */
function Highlighted({ text, mark }: { text: string; mark?: string }) {
  if (!mark) return <>{text}</>;
  const sentences = mark.replace(/\[\d+\]/g, '').split(/(?<=[.!?])\s+/).map(x => x.trim().replace(/[.!?]+$/, '')).filter(x => x.length > 12);
  if (!sentences.length) return <>{text}</>;
  const esc = (x: string) => x.split(/\s+/).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+');
  let re: RegExp;
  try { re = new RegExp(`(${sentences.map(esc).join('|')})`, 'i'); } catch { return <>{text}</>; }
  return <>{text.split(re).map((part, i) => (i % 2 ? <mark className="evidence-mark" key={i}>{part}</mark> : <Fragment key={i}>{part}</Fragment>))}</>;
}

export function PassageCard({ p, scoreType, gold, mark }: { p: Passage; scoreType: string; gold?: boolean; mark?: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const long = p.text.length > 420;
  const cite = `${p.document_name}, p. ${p.page} (chunk ${p.chunk_id}${p.version ? `, version ${p.version}` : ''})`;
  const copy = async () => { try { await navigator.clipboard.writeText(cite); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard unavailable */ } };
  return <article className={`passage ${gold ? 'gold' : ''}`}>
    <div className="passage-head">
      <span className="rank-badge">{p.rank}</span>
      <div className="passage-title"><b>{p.document_name}</b><span>Page {p.page ?? '—'}</span></div>
      {gold && <span className="chip good" title="Marked relevant in the AI-assisted pilot judgements"><Check size={11} /> judged relevant</span>}
      <button className="icon-btn" onClick={copy} aria-label="Copy citation" title="Copy citation">{copied ? <Check size={15} /> : <Copy size={15} />}</button>
    </div>
    {p.version_precedence_hint && <small className="answer-warn">{p.version_precedence_hint} — heuristic, verify the source text.</small>}
    <div className="chips">
      <span className="chip">{SCORE_LABEL[scoreType] ?? 'Score'} {fmt(p.score, scoreType === 'rrf' ? 5 : 3)}</span>
      {p.bm25_rank != null && <span className="chip">BM25 rank {p.bm25_rank}</span>}
      {p.dense_rank != null && <span className="chip">Dense rank {p.dense_rank}</span>}
      {p.reranker_score != null && <span className="chip">Reranker {fmt(p.reranker_score, 3)}</span>}
      {p.version_conflict_hint && <span className="chip warn" title="Heuristic only: verify effective date and supersession text">Policy-version conflict hint</span>}
      <span className="chip">Doc ID {p.doc_id}</span>
      <span className="chip">Chunk {p.chunk_id}</span>
      {p.version && <span className="chip">Version {p.version}</span>}
      {p.category && <span className="chip">{p.category.replace(/_/g, ' ')}</span>}
      {p.extraction_method && <span className="chip">{p.extraction_method === 'ocr' ? `OCR${p.ocr_confidence != null ? ` ${p.ocr_confidence.toFixed(0)}%` : ''}` : 'native text'}</span>}
    </div>
    {(p.source_warnings || []).map((w, i) => <div className="chip warn block" key={i}><AlertTriangle size={11} /> {w}</div>)}
    <p className={`passage-text ${long && !open && !mark ? 'clamped' : ''}`}><Highlighted text={p.text} mark={mark} /></p>
    {long && !mark && <button className="link-btn" onClick={() => setOpen(!open)}>{open ? 'Show less' : 'Show full passage'}</button>}
  </article>;
}

/* ---------- Ask: grounded answer card ---------- */
const ANSWERABILITY_LABEL: Record<string, string> = { answerable: 'Answerable', unanswerable: 'Unanswerable', not_calibrated: 'Not calibrated', insufficient_evidence: 'Insufficient evidence' };

function CitedText({ text, max }: { text: string; max: number }) {
  return <>{text.split(/(\[\d+\])/).map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    return m && +m[1] >= 1 && +m[1] <= max ? <a key={i} className="cite-link" href={`#passage-${m[1]}`}>{part}</a> : <Fragment key={i}>{part}</Fragment>;
  })}</>;
}

const GENERATOR_LABEL: Record<string, string> = { extractive: 'Extractive · works offline', llm: 'LLM · grounded in passages' };

export function AnswerCard({ d }: { d: AskResponse }) {
  const [copied, setCopied] = useState(false);
  const gate = d.answerability;
  const mode = d.abstained ? 'abstained' : d.answer ? d.answer_mode : 'evidence_only';
  const warnings = d.answer_warnings ?? [];
  const cited = (d.cited_passages ?? []).filter(n => n >= 1 && n <= d.evidence.length);
  const grounded = mode !== 'abstained' && !!d.answer && cited.length > 0;
  const verified = grounded && warnings.length === 0;
  const plain = (d.answer ?? '').replace(/\s*\[\d+\]/g, '').trim();
  const copy = async () => { try { await navigator.clipboard.writeText(`${plain}\n\nSource: ${cited.map(n => `${d.evidence[n - 1].document_name}, p. ${d.evidence[n - 1].page}`).join('; ')}`); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard unavailable */ } };
  const badge = mode === 'abstained' ? { cls: 'abstain', icon: <AlertTriangle size={13} />, text: 'Not enough evidence' }
    : verified ? { cls: 'ok', icon: <ShieldCheck size={13} />, text: 'Grounded & verified answer' }
    : mode === 'evidence_only' ? { cls: 'neutral', icon: <FileSearch size={13} />, text: 'Evidence only' }
    : { cls: 'warn', icon: <AlertTriangle size={13} />, text: 'Answer needs checking' };

  return <div className="answer-card" data-testid="answer-card" data-mode={mode}>
    <div className="answer-top">
      <span className={`answer-badge ${badge.cls}`}>{badge.icon} {badge.text}</span>
      <div className="answer-meta">
        {d.generator && mode !== 'abstained' && <span className="chip"><Sparkles size={10} /> {GENERATOR_LABEL[d.generator] ?? d.generator}</span>}
        <span className="chip">{d.detected_language.label} question</span>
        {gate.status === 'answerable' && <span className="chip good">Answerability gate: pass</span>}
      </div>
    </div>

    <small className="answer-label">ANSWER</small>
    {d.answer
      ? <p className="answer-text" lang={mode === 'llm' && d.detected_language.label === 'Hindi' ? 'hi' : undefined}>
          {mode === 'abstained' ? d.answer : <CitedText text={d.answer} max={d.evidence.length} />}</p>
      : <p className="answer-text muted">No answer was generated for this request.</p>}

    {mode === 'abstained' && <div className="answer-help">
      <b>What this means</b>
      <span>{gate.status === 'unanswerable' && gate.reason ? gate.reason : 'The retrieved passages do not contain a clear answer, so the system declines instead of guessing.'}</span>
      <span>Try rephrasing with specific terms (for example a policy or course name), or switch the retrieval method.</span>
    </div>}
    {d.generation_error && <p className="answer-error">Answer generation failed ({d.generation_error}). Retrieved evidence is shown below.</p>}
    {mode === 'evidence_only' && !d.generation_error && <p className="answer-note">Answer generation is switched off for this request; only retrieved evidence is shown.</p>}

    {grounded && <div className="answer-sources">
      <small className="answer-label"><FileText size={11} /> SOURCES</small>
      {cited.map(n => { const e = d.evidence[n - 1]; return <a key={n} className="source-card" href={`#passage-${n}`}>
        <span className="source-n">{n}</span>
        <span className="source-body"><b>{e.document_name}</b><i>Page {e.page ?? '—'}{e.version ? ` · version ${e.version}` : ''}{e.category ? ` · ${e.category.replace(/_/g, ' ')}` : ''}</i></span>
        <ArrowRight size={14} /></a>; })}
    </div>}

    {grounded && <ul className="verify-list" aria-label="Answer verification">
      <li className={warnings.some(w => /citation|retrieved/i.test(w)) ? 'bad' : 'good'}><Check size={12} /> Citations point to retrieved passages</li>
      <li className={warnings.some(w => /number/i.test(w)) ? 'bad' : 'good'}><Check size={12} /> Numbers and figures appear in the cited text</li>
      {d.answer_mode === 'extractive' && <li className="good"><Quote size={12} /> Wording is copied from the source, nothing is invented</li>}
    </ul>}
    {warnings.map(w => <p className="answer-warn" key={w}>⚠ {w}</p>)}

    <div className="answer-foot">
      <span><ShieldCheck size={14} /> Checked automatically after generation. Please confirm against the highlighted source passage below.</span>
      {d.answer && mode !== 'abstained' && <button className="btn secondary small" onClick={copy}>{copied ? <Check size={13} /> : <Copy size={13} />} {copied ? 'Copied' : 'Copy answer'}</button>}
    </div>
  </div>;
}

/* ---------- Ask: dedicated results view ---------- */
export type AskState = { status: 'idle' | 'loading' | 'done' | 'error'; data?: AskResponse; error?: string; offline?: boolean; httpStatus?: number };
export interface AskRequestSnapshot { query: string; method: MethodKey; topK: number }

export function AskResults({ state, request, onBack, onRetry, onSwitch, onCompare }: {
  state: AskState; request: AskRequestSnapshot; onBack: () => void; onRetry: () => void; onSwitch: (m: MethodKey) => void; onCompare: () => void;
}) {
  const [feedbackStatus, setFeedbackStatus] = useState<string>('');
  const methodLabel = METHOD_OPTIONS.find(m => m.key === request.method)?.label ?? request.method;
  const d = state.data;
  const status = state.status === 'loading' ? 'Retrieving…' : state.status === 'error' ? 'Failed' : d?.retrieval_status === 'ok' ? `${d.evidence.length} passage${d.evidence.length === 1 ? '' : 's'} retrieved` : 'No matching passages';
  const hindiLike = d && ['Hindi', 'Code-mixed', 'Hinglish'].includes(d.detected_language.label);
  return <section className="results-view" aria-live="polite">
    <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Edit question</button>
    <div className="result-heading"><div><div className="eyebrow">02 / RETRIEVED EVIDENCE</div><h2>Query results</h2></div>
      <span className="demo-pill">{state.status === 'loading' ? 'RETRIEVING EVIDENCE' : state.status === 'error' ? 'RETRIEVAL FAILED' : 'LIVE BACKEND'}</span></div>

    <div className="panel query-card"><small>ORIGINAL QUERY</small><p lang={d?.detected_language.label === 'Hindi' ? 'hi' : undefined}>{request.query}</p></div>

    <div className="analysis-grid two-rows">
      <div><small>DETECTED LANGUAGE</small><b>{d ? d.detected_language.label : '—'}</b>{d && <i title={d.detected_language.evidence}>heuristic</i>}</div>
      <div><small>RETRIEVAL METHOD</small><b>{d?.method_label ?? methodLabel}</b></div>
      <div><small>TOP-K</small><b>{request.topK} passages</b></div>
      <div><small>RETRIEVAL STATUS</small><b>{status}</b></div>
      <div><small>ANSWERABILITY</small><b title={d?.answerability.reason}>{d ? ANSWERABILITY_LABEL[d.answerability.status] ?? d.answerability.status : '—'}</b></div>
      <div><small>RESPONSE TIME</small><b>{d ? `${d.latency_ms} ms` : '—'}</b></div>
    </div>

    {state.status === 'loading' && <><Loading text="Retrieving evidence from the document corpus…" />
      {[0, 1, 2].map(i => <div className="passage skeleton" key={i} aria-hidden="true" />)}</>}

    {state.status === 'error' && <ErrorBox title={state.offline ? 'Backend not reachable' : 'Retrieval failed'} message={state.error || 'Unknown error'} onRetry={onRetry}>
      {request.method !== 'bm25' && state.httpStatus === 503 && <button className="btn primary" onClick={() => onSwitch('bm25')}>Run with BM25 instead</button>}
      <button className="btn secondary" onClick={onBack}>Back to question</button></ErrorBox>}

    {state.status === 'done' && d && <>
      {d.evidence.some(e => e.hybrid_fallback) && <div className="notice fallback" role="status"><AlertTriangle size={17} /><span><b>Hybrid ran in BM25-only mode</b> · the dense (E5) half is not available on this server, so these results come from keyword matching alone. {d.evidence[0]?.hybrid_fallback_reason ? `Reason: ${d.evidence[0].hybrid_fallback_reason}` : ''}</span></div>}
      <AnswerCard d={d} />

      {d.retrieval_status === 'no_results' ? <div className="state-box"><FileSearch size={24} /><b>No passages matched this query</b>
        <span>{request.method === 'bm25' && hindiLike
          ? 'BM25 lexical expansion covers only selected Hindi/Hinglish terms. Dense multilingual retrieval is the intended broader cross-lingual method.'
          : 'None of the indexed passages shares terms with this query. Try rephrasing, or use a different retrieval method.'}</span>
        <div className="state-actions">{request.method === 'bm25' && <button className="btn primary" onClick={() => onSwitch('hybrid')}>Try hybrid retrieval</button>}
          <button className="btn secondary" onClick={onBack}>Edit question</button></div></div>
        : <>
          <div className="panel citations"><div className="eyebrow">CITATIONS</div>
            <ol>{d.citations.map(c => <li key={c.chunk_id}><b>{c.citation_id}</b> {c.label}<span> · {c.doc_id} · {c.chunk_id}{c.version ? ` · version ${c.version}` : ''}{c.category ? ` · ${c.category.replace(/_/g, ' ')}` : ''}</span></li>)}</ol></div>
          <div className="eyebrow list-label">RETRIEVED PASSAGES · RANKED</div>
          {d.evidence.map((p, i) => <div id={`passage-${i + 1}`} key={p.chunk_id}><PassageCard p={p} scoreType={d.score_type} mark={!d.abstained && d.answer_mode === 'extractive' && (d.cited_passages ?? []).includes(i + 1) ? d.answer ?? undefined : undefined} /></div>)}
          <div className="notice"><Microscope size={17} /><span><b>Reading scores</b> · {SCORE_LABEL[d.score_type]} is only meaningful for ranking within this list. Scores from different methods are not comparable and are not confidence estimates.</span></div>
        </>}
      {d.answer && <div className="panel feedback-panel"><b>Was this answer helpful?</b><div className="feedback-actions"><button className="btn secondary" onClick={async()=>{try{await api.feedback({query:request.query,answer:d.answer||"",rating:"up",method:request.method,cited_chunk_ids:d.evidence.filter((_,i)=>(d.cited_passages||[]).includes(i+1)).map(x=>x.chunk_id)});setFeedbackStatus("Thanks — feedback logged locally.")}catch(e:any){setFeedbackStatus(e?.message||"Could not save feedback")}}}>👍 Helpful</button><button className="btn secondary" onClick={async()=>{try{await api.feedback({query:request.query,answer:d.answer||"",rating:"down",method:request.method,cited_chunk_ids:d.evidence.filter((_,i)=>(d.cited_passages||[]).includes(i+1)).map(x=>x.chunk_id)});setFeedbackStatus("Thanks — feedback logged locally.")}catch(e:any){setFeedbackStatus(e?.message||"Could not save feedback")}}}>👎 Not helpful</button></div>{feedbackStatus&&<small>{feedbackStatus}</small>}</div>}
      <div className="results-actions"><button className="btn secondary" onClick={onCompare}><Layers3 size={14} /> Compare this query across methods</button>
        <button className="btn secondary" onClick={onBack}><ArrowLeft size={14} /> Ask another question</button></div>
    </>}
  </section>;
}

/* ---------- Overview (live stats) ---------- */
export function OverviewStats({ info, error, offline }: { info: OverviewResponse | null; error: string; offline: boolean }) {
  const c = info?.corpus;
  const m = info?.methods;
  const avail = m ? (Object.keys(m) as MethodKey[]).filter(k => m[k].available).length : null;
  const cards: [string, string, string][] = [
    [c ? String(c.documents) : '—', 'Documents', c ? `${c.pages} pages · ${c.chunks} passages` : error || 'Loading…'],
    [c?.ocr_confidence_mean != null ? `${c.ocr_confidence_mean}%` : '—', 'Mean OCR confidence', c ? `${c.ocr_pages} OCR pages · ${c.low_confidence_pages} below 70%` : '—'],
    [avail != null ? `${avail}/4` : '—', 'Retrieval methods ready', m ? (Object.keys(m) as MethodKey[]).map(k => `${m[k].label.split(' (')[0]}: ${m[k].available ? 'ready' : 'unavailable'}`).join(' · ') : '—'],
    [info?.artifacts.answerability_calibrated ? 'On' : 'Off', 'Answerability gate', info?.artifacts.answerability_calibrated ? 'Threshold calibrated on dev' : 'Not calibrated yet · run evaluate_answerability.py'],
  ];
  return <>
    <div className="stat-grid">{cards.map(([n, t, d], i) => <div className="stat-card" key={t}><div className="stat-top"><span>{['CORPUS', 'EXTRACTION', 'RETRIEVAL', 'GENERATION'][i]}</span><span className="stat-icon">{[<Library />, <FileSearch />, <Layers3 />, <ShieldCheck />][i]}</span></div><strong>{n}</strong><b>{t}</b><small>{d}</small></div>)}</div>
    {offline && <div className="notice warnbox"><AlertTriangle size={17} /><span><b>Backend offline</b> · {error}</span></div>}
  </>;
}

/* ---------- Document Library ---------- */
export function Documents({ search, setSearch, filter, setFilter }: { search: string; setSearch: (s: string) => void; filter: string; setFilter: (s: string) => void }) {
  const { data, error, loading, reload } = useLoad(s => api.documents(s));
  const [openId, setOpenId] = useState<string | null>(null);
  const docs = (data?.documents ?? []).filter(d =>
    (filter === 'All categories' || d.category === filter) && (!search.trim() || d.document_name.toLowerCase().includes(search.trim().toLowerCase())));
  const o = data?.overview;
  return <>
    <PageHead eyebrow="KNOWLEDGE BASE" title="Document Library." desc="The Bennett University source documents currently indexed, with extraction quality and version caveats. The corpus is built offline; this page is read-only." />
    <div className="stat-grid compact">{[[o ? String(o.documents) : '—', 'Indexed documents', 'From data/chunks.jsonl'], [o ? String(o.chunks) : '—', 'Total passages', o ? `median ${o.chunk_chars_median} characters` : ''], [o ? String(o.pages) : '—', 'Pages', o ? `${o.ocr_pages} OCR · ${o.native_pages} native text` : '']].map(([n, t, d]) =>
      <div className="stat-card" key={t}><strong>{n}</strong><b>{t}</b><small>{d}</small></div>)}</div>
    <div className="panel">
      <div className="doc-toolbar"><div className="search-field"><Search size={16} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search document library" aria-label="Search documents" /></div>
        <select value={filter} onChange={e => setFilter(e.target.value)} aria-label="Filter by category"><option>All categories</option>{(o?.categories ?? []).map(x => <option key={x} value={x}>{x}</option>)}</select></div>
      {loading && <Loading text="Loading documents…" />}
      {error && <ErrorBox title="Could not load documents" message={error} onRetry={reload} />}
      {data && <div className="table-wrap"><table><thead><tr><th>DOCUMENT</th><th>CATEGORY</th><th>VERSION</th><th>PAGES</th><th>PASSAGES</th><th>EXTRACTION</th></tr></thead>
        <tbody>{docs.length === 0 && <tr><td colSpan={6}><div className="empty-state"><Library size={28} /><b>No documents match</b><span>Change the search text or category filter.</span></div></td></tr>}
          {docs.map(d => <DocRow key={d.doc_id} d={d} open={openId === d.doc_id} toggle={() => setOpenId(openId === d.doc_id ? null : d.doc_id)} />)}</tbody></table></div>}
    </div></>;
}
function DocRow({ d, open, toggle }: { d: DocumentInfo; open: boolean; toggle: () => void }) {
  return <>
    <tr className="clickable" onClick={toggle} tabIndex={0} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } }} aria-expanded={open}>
      <td><span className="row-toggle">{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span><b>{d.document_name}</b><br /><small className="muted">{d.doc_id}</small></td>
      <td>{d.category?.replace(/_/g, ' ')}</td><td>{d.version}</td><td>{d.pages}</td><td>{d.chunks}</td>
      <td>{d.extraction_methods.join(', ')}{d.ocr_confidence_mean != null && <><br /><small className="muted">mean {d.ocr_confidence_mean}% · min {d.ocr_confidence_min}%</small></>}</td></tr>
    {open && <tr><td colSpan={6}><DocDetail d={d} /></td></tr>}
  </>;
}
function DocDetail({ d }: { d: DocumentInfo }) {
  const [page, setPage] = useState<number | undefined>(undefined);
  const { data, error, loading, reload } = useLoad(s => api.documentDetail(d.doc_id, page, s), [d.doc_id, page]);
  return <div className="doc-detail">
    {d.note && <div className="chip warn block"><AlertTriangle size={11} /> {d.note}</div>}
    {d.low_confidence_pages.length > 0 && <div className="chip warn block"><AlertTriangle size={11} /> Low-OCR-confidence pages (below 70%): {d.low_confidence_pages.join(', ')}</div>}
    <label className="inline-field">Page <select value={page ?? ''} onChange={e => setPage(e.target.value ? Number(e.target.value) : undefined)}><option value="">All pages</option>{Array.from({ length: d.pages }, (_, i) => i + 1).map(n => <option key={n} value={n}>{n}</option>)}</select></label>
    {loading && <Loading text="Loading passages…" />}
    {error && <ErrorBox title="Could not load passages" message={error} onRetry={reload} />}
    {data && <>{data.chunks.length === 0 ? <div className="empty-state"><b>No passages on this page</b></div> :
      <div className="detail-list">{data.chunks.slice(0, 30).map(c => <PassageCard key={c.chunk_id} p={{ ...c, rank: c.page ?? 0, score: null }} scoreType="" />)}</div>}
      {data.total > 30 && <small className="muted">Showing 30 of {data.total} passages. Pick a page to narrow the list.</small>}</>}
  </div>;
}

/* ---------- compare columns (Explorer + Comparison) ---------- */
function MethodColumns({ data, gold }: { data: CompareResponse; gold?: Set<string> }) {
  return <div className="method-cols">{Object.entries(data.methods).map(([key, b]) => <div className="panel method-col" key={key}>
    <div className="panel-head"><div><span className="eyebrow">{b.score_type.toUpperCase()}</span><h3>{b.label}</h3></div><span className={`not-evaluated ${b.status === 'ok' ? '' : 'off'}`}>{b.status === 'ok' ? `${b.results.length} passages · ${b.latency_ms} ms` : b.status === 'no_results' ? 'no matches' : b.status}</span></div>
    {b.status === 'unavailable' && <div className="state-box slim"><AlertTriangle size={18} /><span>{b.message}</span></div>}
    {b.status === 'error' && <div className="state-box slim error"><AlertTriangle size={18} /><span>{b.message}</span></div>}
    {b.status === 'no_results' && <div className="state-box slim"><FileSearch size={18} /><span>{b.message}</span></div>}
    {b.results.map(p => <PassageCard key={p.chunk_id} p={p} scoreType={b.score_type} gold={gold?.has(p.chunk_id)} />)}
  </div>)}</div>;
}
function Overlap({ data }: { data: CompareResponse }) {
  const keys = Object.keys(data.methods).filter(k => data.methods[k].results.length);
  if (!keys.length) return null;
  return <div className="panel"><div className="panel-head"><div><span className="eyebrow">AGREEMENT</span><h3>Passages and the rank each method gave them</h3></div></div>
    <div className="table-wrap"><table><thead><tr><th>PASSAGE</th>{keys.map(k => <th key={k}>{data.methods[k].label.split(' (')[0].toUpperCase()}</th>)}</tr></thead>
      <tbody>{data.overlap.map(o => <tr key={o.chunk_id}><td>{o.document_name} · p.{o.page}<br /><small className="muted">{o.chunk_id}</small></td>{keys.map(k => <td key={k}>{o.ranks[k] ?? '—'}</td>)}</tr>)}</tbody></table></div></div>;
}

/* ---------- Research Explorer ---------- */
export function Explorer() {
  const [q, setQ] = useState('');
  const [run, setRun] = useState<{ q: string; n: number } | null>(null);
  const cmp = useLoad(s => run ? api.compare(run.q, 5, s) : Promise.resolve(null as CompareResponse | null), [run?.q, run?.n]);
  const [kw, setKw] = useState(''); const [kwDoc, setKwDoc] = useState(''); const [offset, setOffset] = useState(0);
  const docs = useLoad(s => api.documents(s));
  const chunks = useLoad(s => api.chunks(kw, kwDoc, offset, s), [kw, kwDoc, offset]);
  const [kwInput, setKwInput] = useState('');
  return <>
    <PageHead eyebrow="RETRIEVAL INSPECTION" title="Research Explorer." desc="Run one query through every available retrieval method, and search the corpus text directly." />
    <form className="panel explorer-search" onSubmit={e => { e.preventDefault(); if (q.trim()) setRun(r => ({ q: q.trim(), n: (r?.n ?? 0) + 1 })); }}>
      <Search size={18} /><input value={q} onChange={e => setQ(e.target.value)} placeholder="Enter a research query to inspect retrieval behaviour..." aria-label="Research query" />
      <button className="btn primary" type="submit" disabled={!q.trim()}>Run exploration</button></form>
    {!run && <div className="panel"><div className="empty-state"><FileSearch size={28} /><b>Passages will appear here</b><span>Run a query to see BM25, pretrained multilingual E5, and hybrid rankings side by side. Methods whose index is missing are reported as unavailable instead of being faked.</span></div></div>}
    {run && cmp.loading && <Loading text="Running all retrieval methods…" />}
    {run && cmp.error && <ErrorBox title="Exploration failed" message={cmp.error} onRetry={cmp.reload} />}
    {run && cmp.data && <><div className="notice"><Globe2 size={17} /><span>Detected language: <b>{cmp.data.detected_language.label}</b> (heuristic) · top 5 per method</span></div><MethodColumns data={cmp.data} /><Overlap data={cmp.data} /></>}

    <div className="panel"><div className="panel-head"><div><span className="eyebrow">CORPUS TEXT SEARCH</span><h3>Find exact text in the passages</h3></div><span className="not-evaluated">{chunks.data ? `${chunks.data.total} matching passages` : '…'}</span></div>
      <form className="doc-toolbar" onSubmit={e => { e.preventDefault(); setOffset(0); setKw(kwInput.trim()); }}>
        <div className="search-field"><Search size={16} /><input value={kwInput} onChange={e => setKwInput(e.target.value)} placeholder="Substring, e.g. 75% or CGPA" aria-label="Corpus text search" /></div>
        <select value={kwDoc} onChange={e => { setOffset(0); setKwDoc(e.target.value); }} aria-label="Document filter"><option value="">All documents</option>{(docs.data?.documents ?? []).map(d => <option key={d.doc_id} value={d.doc_id}>{d.document_name}</option>)}</select>
        <button className="btn secondary" type="submit">Search</button></form>
      {chunks.loading && <Loading text="Searching…" />}
      {chunks.error && <ErrorBox title="Search failed" message={chunks.error} onRetry={chunks.reload} />}
      {chunks.data && (chunks.data.chunks.length === 0 ? <div className="empty-state"><b>No passages contain that text</b><span>Text search is exact (case-insensitive); OCR errors can hide matches.</span></div> :
        <div className="detail-list">{chunks.data.chunks.map(c => <PassageCard key={c.chunk_id} p={{ ...c, rank: c.page ?? 0, score: null }} scoreType="" />)}</div>)}
      {chunks.data && chunks.data.total > 10 && <div className="pager"><button className="btn secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 10))}>Previous</button>
        <small className="muted">{offset + 1}–{Math.min(offset + 10, chunks.data.total)} of {chunks.data.total}</small>
        <button className="btn secondary" disabled={offset + 10 >= chunks.data.total} onClick={() => setOffset(offset + 10)}>Next</button></div>}
    </div></>;
}

/* ---------- Retrieval Comparison ---------- */
const METRIC_METHODS: [string, string][] = [['bm25', 'BM25 Lexical'], ['dense', 'Pretrained Multilingual E5'], ['hybrid', 'Hybrid (BM25 + E5 RRF)']];
export function Comparison({ presetQuery, clearPreset }: { presetQuery: string; clearPreset: () => void }) {
  const ev = useLoad(s => api.evaluation(s));
  const [q, setQ] = useState(presetQuery);
  const [run, setRun] = useState<{ q: string; n: number } | null>(presetQuery ? { q: presetQuery, n: 1 } : null);
  useEffect(() => { if (presetQuery) clearPreset(); /* consume once */ // eslint-disable-next-line
  }, []);
  const cmp = useLoad(s => run ? api.compare(run.q, 5, s) : Promise.resolve(null as CompareResponse | null), [run?.q, run?.n]);
  const pilot = ev.data?.bm25_pilot;
  const goldFor = (query: string) => new Set(pilot?.judgements.find(j => j.query === query)?.relevant_chunk_ids ?? []);
  return <>
    <PageHead eyebrow="RETRIEVAL EVALUATION" title="Compare retrieval methods." desc="Measured retrieval metrics for each method, and a live side-by-side of the ranked passages." />
    {ev.loading && <Loading text="Loading evaluation artifacts…" />}
    {ev.error && <ErrorBox title="Could not load metrics" message={ev.error} onRetry={ev.reload} />}
    {ev.data && <div className="method-metrics">{METRIC_METHODS.map(([key, label], i) => {
      const s = ev.data!.per_method[key]?.summary;
      return <div className="panel metric-card" key={key}><span className="eyebrow">APPROACH 0{i + 1}</span><h3>{label}</h3>
        {s ? <><strong>{fmt(s['mean_recall@5'], 2)}</strong><small>Recall@5 · {s.evaluated_queries} judged queries</small><div className="metric-line" />
          <p>Precision@5 <b>{fmt(s['mean_precision@5'], 2)}</b></p><p>nDCG@5 <b>{fmt(s['mean_ndcg@5'], 2)}</b></p><p>MRR <b>{fmt(s.mean_reciprocal_rank, 2)}</b></p></>
          : <><strong>—</strong><small>Not run: no <code>{key}_retrieval_runs.csv</code> in reports/</small><div className="metric-line" /><p>Run the Phase 4 notebook or the retrieval script, then reload.</p></>}</div>;
    })}</div>}
    {pilot && <div className="notice warnbox"><AlertTriangle size={17} /><span><b>Read with care</b> · {pilot.label_provenance} Only {pilot.summary?.evaluated_queries ?? 0} of {pilot.judgements.length} queries have a relevant passage, so Recall@K is near-saturated and these figures cannot rank the methods.</span></div>}
    <div className="panel"><div className="panel-head"><div><span className="eyebrow">QUERY-LEVEL INSPECTION</span><h3>Ranked passage comparison</h3></div></div>
      <form className="doc-toolbar" onSubmit={e => { e.preventDefault(); if (q.trim()) setRun(r => ({ q: q.trim(), n: (r?.n ?? 0) + 1 })); }}>
        <select value="" onChange={e => { if (e.target.value) { setQ(e.target.value); setRun(r => ({ q: e.target.value, n: (r?.n ?? 0) + 1 })); } }} aria-label="Pilot queries"><option value="">Pilot queries…</option>{(pilot?.judgements ?? []).map(j => <option key={j.query_id} value={j.query}>{j.query_id} · {j.query.slice(0, 48)}</option>)}</select>
        <div className="search-field"><Search size={16} /><input value={q} onChange={e => setQ(e.target.value)} placeholder="Or type any query" aria-label="Comparison query" /></div>
        <button className="btn primary" type="submit" disabled={!q.trim()}>Compare</button></form>
      {!run && <div className="empty-state"><Activity size={28} /><b>Choose or type a query</b><span>Passages marked “judged relevant” come from the AI-assisted pilot labels and exist only for the pilot queries.</span></div>}
    </div>
    {run && cmp.loading && <Loading text="Running all retrieval methods…" />}
    {run && cmp.error && <ErrorBox title="Comparison failed" message={cmp.error} onRetry={cmp.reload} />}
    {run && cmp.data && <><MethodColumns data={cmp.data} gold={goldFor(cmp.data.query)} /><Overlap data={cmp.data} /></>}
  </>;
}

/* ---------- Evaluation Lab ---------- */
const STATUS_LABEL: Record<string, string> = { measured: 'Measured', partial: 'Partly measured', partial_bm25_only: 'BM25 only', in_progress: 'In progress', not_evaluated: 'Not evaluated', not_implemented: 'Not implemented', ready_not_run: 'Ready, not run yet' };
export function Evaluation() {
  const { data, error, loading, reload } = useLoad(s => api.evaluation(s));
  const [split, setSplit] = useState('test'); const [setting, setSetting] = useState('all');
  const rows: MlqaRow[] = (data?.mlqa ?? []).filter(r => r.split === split && (setting === 'all' || r.setting === setting));
  const chart = rows.map(r => ({ pair: `ctx ${r.context_language} / q ${r.question_language}`, 'Recall@10': r['recall@10'], 'MRR@10': r['mrr@10'] }));
  return <>
    <PageHead eyebrow="EXPERIMENT WORKBENCH" title="Evaluation Lab." desc="Genuine results from the artifacts in reports/. Everything here measures retrieval only." />
    {loading && <Loading text="Loading evaluation artifacts…" />}
    {error && <ErrorBox title="Could not load evaluation" message={error} onRetry={reload} />}
    {data && <>
      <div className="module-grid">{data.modules.map(m => <div className="panel module-card" key={m.id}><div className="module-top"><span>MODULE 0{m.id}</span><span className={`not-evaluated ${m.status === 'measured' ? '' : 'off'}`}>{STATUS_LABEL[m.status] ?? m.status}</span></div><h3>{m.title}</h3><div className="module-bottom"><span>{m.note}</span></div></div>)}</div>

      <div className="panel"><div className="panel-head"><div><span className="eyebrow">BENNETT CORPUS · PILOT</span><h3>Retrieval on judged queries</h3></div><span className="demo-pill">AI-ASSISTED LABELS</span></div>
        <p className="panel-note">{data.bm25_pilot.label_provenance} Metrics are averaged over {data.bm25_pilot.summary?.evaluated_queries ?? 0} queries; {data.bm25_pilot.summary?.excluded_queries ?? 0} query has no relevant passage and is excluded.</p>
        <div className="table-wrap"><table><thead><tr><th>METHOD</th><th>QUERIES</th><th>RECALL@5</th><th>RECALL@10</th><th>PRECISION@5</th><th>PRECISION@10</th><th>NDCG@5</th><th>NDCG@10</th><th>MRR</th></tr></thead>
          <tbody>{Object.entries(data.per_method).map(([k, m]) => <tr key={k}><td><b>{m.label}</b><br /><small className="muted">{m.source_file}</small></td><td>{m.summary.evaluated_queries}</td>{['mean_recall@5', 'mean_recall@10', 'mean_precision@5', 'mean_precision@10', 'mean_ndcg@5', 'mean_ndcg@10', 'mean_reciprocal_rank'].map(f => <td key={f}>{fmt(m.summary[f])}</td>)}</tr>)}
            {['bm25', 'dense', 'hybrid'].filter(k => !data.per_method[k]).map(k => <tr key={k} className="muted-row"><td>{METRIC_METHODS.find(x => x[0] === k)![1]}</td><td colSpan={8}>Not run · no {k}_retrieval_runs.csv in reports/</td></tr>)}</tbody></table></div>
        <div className="eyebrow list-label">PER QUERY · BM25</div>
        <div className="table-wrap"><table><thead><tr><th>ID</th><th>QUERY</th><th>RELEVANT PASSAGES</th><th>FIRST RELEVANT RANK</th><th>RR</th></tr></thead>
          <tbody>{data.bm25_pilot.judgements.map(j => { const pq = data.bm25_pilot.per_query.find(r => r.query_id === j.query_id); return <tr key={j.query_id}><td>{j.query_id}</td><td>{j.query}</td><td>{j.relevant_chunk_ids.length || 'none identified'}</td><td>{pq?.first_relevant_rank || '—'}</td><td>{pq ? Number(pq.reciprocal_rank).toFixed(2) : 'excluded'}</td></tr>; })}</tbody></table></div></div>

      <div className="panel"><div className="panel-head"><div><span className="eyebrow">EXTERNAL BENCHMARK · MLQA</span><h3>Passage retrieval, Hindi and English</h3></div>
        <div className="filter-inline"><select value={split} onChange={e => setSplit(e.target.value)} aria-label="Split"><option value="test">test</option><option value="dev">dev</option></select>
          <select value={setting} onChange={e => setSetting(e.target.value)} aria-label="Setting"><option value="all">All settings</option><option value="monolingual">Monolingual</option><option value="cross-lingual">Cross-lingual</option></select></div></div>
        <p className="panel-note">Corpus = all unique context paragraphs of each MLQA file; gold = the paragraph the question was written for. MLQA is Wikipedia text, so these numbers describe the method, not the Bennett corpus. Only BM25 has been run; dense and hybrid need a model download (see README).</p>
        {rows.length === 0 ? <div className="empty-state"><Activity size={26} /><b>No MLQA results for this selection</b><span>Run backend/evaluate_mlqa.py to produce them.</span></div> : <>
          <div className="chart-box"><ResponsiveContainer width="100%" height={240}><BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="#e8e5dd" /><XAxis dataKey="pair" tick={{ fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 10 }} /><Tooltip formatter={(v: any) => Number(v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 11 }} /><Bar dataKey="Recall@10" fill="#477c78" /><Bar dataKey="MRR@10" fill="#c58a45" /></BarChart></ResponsiveContainer></div>
          <div className="table-wrap"><table><thead><tr><th>CONTEXT / QUESTION</th><th>SETTING</th><th>METHOD</th><th>PASSAGES</th><th>QUESTIONS</th><th>R@1</th><th>R@5</th><th>R@10</th><th>MRR@10</th><th>NDCG@10</th></tr></thead>
            <tbody>{rows.map((r, i) => <tr key={i}><td>{r.context_language} / {r.question_language}</td><td>{r.setting}</td><td>{r.method}{r.model ? ` · ${r.model}` : ''}</td><td>{r.corpus_passages}</td><td>{r.questions}</td><td>{fmt(r['recall@1'])}</td><td>{fmt(r['recall@5'])}</td><td>{fmt(r['recall@10'])}</td><td>{fmt(r['mrr@10'])}</td><td>{fmt(r['ndcg@10'])}</td></tr>)}</tbody></table></div></>}
      </div>

      {data.qa_eval?.retrieval_by_language && (
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">MODULES 1–3 · MULTILINGUAL RETRIEVAL</span>
              <h3>Cross-Lingual Evaluation across English, Hindi, and Hinglish</h3>
            </div>
            <span className="demo-pill">TEST SPLIT BENCHMARK</span>
          </div>
          <p className="panel-note">
            Empirical evaluation over 39 answerable test questions across English (monolingual), Hindi (Devanagari cross-lingual), and Hinglish (code-mixed). Hybrid RRF combines BM25 keyword matching with multilingual E5 dense embeddings.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>METHOD</th>
                  <th>ENGLISH (HIT@5)</th>
                  <th>HINDI (HIT@5)</th>
                  <th>HINGLISH (HIT@5)</th>
                  <th>CROSS-LINGUAL (HIT@5)</th>
                  <th>ALL (HIT@5)</th>
                  <th>MRR</th>
                  <th>NDCG@10</th>
                </tr>
              </thead>
              <tbody>
                {['bm25', 'pretrained', 'hybrid'].map((m) => {
                  const res = data.qa_eval?.retrieval_by_language?.results?.[m];
                  if (!res) return null;
                  const label = m === 'bm25' ? 'BM25' : m === 'pretrained' ? 'Pretrained E5' : 'Hybrid (RRF)';
                  return (
                    <tr key={m}>
                      <td><b>{label}</b></td>
                      <td>{fmt(res.English?.['hit@5']?.mean)}</td>
                      <td>{fmt(res.Hindi?.['hit@5']?.mean)}</td>
                      <td>{fmt(res.Hinglish?.['hit@5']?.mean)}</td>
                      <td>{fmt(res['Cross-lingual (Hindi+Hinglish)']?.['hit@5']?.mean)}</td>
                      <td><b>{fmt(res.All?.['hit@5']?.mean)}</b></td>
                      <td>{fmt(res.All?.mrr?.mean)}</td>
                      <td>{fmt(res.All?.['ndcg@10']?.mean)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {data.qa_eval?.answerability_eval && (
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">MODULES 4 &amp; 5 · GATE &amp; GENERATION</span>
              <h3>Answerability Detection and RAG Abstention Gate</h3>
            </div>
            <span className="demo-pill">CALIBRATED ON DEV · TEST SCORES</span>
          </div>
          <p className="panel-note">
            Answerability gate classifies questions as answerable vs unanswerable based on retrieval signal thresholds fitted on DEV split and evaluated on held-out TEST split (57 test instances).
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>METHOD</th>
                  <th>SIGNAL TYPE</th>
                  <th>CALIBRATED THRESHOLD</th>
                  <th>TEST ACCURACY</th>
                  <th>PRECISION</th>
                  <th>RECALL</th>
                  <th>F1 SCORE</th>
                  <th>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {['bm25', 'pretrained', 'hybrid'].map((m) => {
                  const ev = data.qa_eval?.answerability_eval?.methods?.[m];
                  if (!ev) return null;
                  const label = m === 'bm25' ? 'BM25' : m === 'pretrained' ? 'Pretrained E5' : 'Hybrid (RRF)';
                  const t = ev.test?.overall;
                  return (
                    <tr key={m}>
                      <td><b>{label}</b></td>
                      <td><code>{m === 'bm25' ? 'bm25_top1' : 'dense_cosine_top1'}</code></td>
                      <td>{ev.threshold?.toFixed(4)}</td>
                      <td>{fmt(t?.accuracy)}</td>
                      <td>{fmt(t?.precision)}</td>
                      <td>{fmt(t?.recall)}</td>
                      <td><b>{fmt(t?.f1)}</b></td>
                      <td>{ev.test?.auc ? fmt(ev.test.auc) : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="panel metric-list"><div className="panel-head"><div><span className="eyebrow">HOW TO READ THESE METRICS</span><h3>What each measure does and does not tell you</h3></div></div>
        <dl className="metric-notes">{Object.entries(data.metric_notes).map(([k, v]) => <div key={k}><dt>{k === 'mrr' ? 'MRR' : k === 'scope' ? 'Scope' : k.replace('@k', '@K').replace(/^./, c => c.toUpperCase())}</dt><dd>{v}</dd></div>)}</dl>
        <div className="metric-chips">{['Exact Match', 'Token-level F1', 'Semantic similarity', 'Answer correctness', 'Answerability accuracy', 'Confusion matrix'].map(x => <span key={x}>{x}<i>Not measured · no answer generation</i></span>)}</div></div>
    </>}
  </>;
}

/* ---------- Error Analysis ---------- */
export function Errors() {
  const { data, error, loading, reload } = useLoad(s => api.errors(s));
  const cats = ['Code-mixed query', 'Cross-lingual vocabulary mismatch', 'OCR noise / spelling variation', 'Semantically similar passages / ranking', 'Unsupported / possibly unanswerable question'];
  const count = (c: string) => (data?.cases ?? []).filter((x: ErrorCase) => x.failure_type === c).length;
  return <>
    <PageHead eyebrow="QUALITATIVE EVALUATION" title="Error Analysis." desc="Difficult cases derived from the real artifacts: reviewed judgements, live BM25 runs on the probe queries, and OCR confidence." />
    {loading && <Loading text="Loading cases…" />}
    {error && <ErrorBox title="Could not load cases" message={error} onRetry={reload} />}
    {data && <>
      <div className="error-cats">{cats.map((c, i) => <div className="error-cat" key={c}><span>{count(c)}</span><b>{c}</b></div>)}</div>
      <div className="panel"><div className="panel-head"><div><span className="eyebrow">CASE REGISTER</span><h3>Difficult query analysis</h3></div>
        <span className={`not-evaluated ${data.documented >= data.required_minimum ? '' : 'off'}`}>{data.documented} of {data.required_minimum} required cases</span></div>
        <p className="panel-note">Only cases backed by an artifact are listed. Cases for numbers, dates and named entities still need to be run and written up.</p>
        <div className="table-wrap"><table><thead><tr>{['QUERY', 'LANGUAGE', 'EXPECTED PASSAGE', 'RETRIEVED PASSAGE', 'FAILURE TYPE', 'METHOD', 'OBSERVED ISSUE', 'SOURCE'].map(x => <th key={x}>{x}</th>)}</tr></thead>
          <tbody>{data.cases.length === 0 && <tr><td colSpan={8}><div className="empty-state"><ClipboardList size={28} /><b>No cases yet</b></div></td></tr>}
            {data.cases.map(c => <tr key={c.case_id}><td lang={c.language === 'Hindi' ? 'hi' : undefined}>{c.query}</td><td>{c.language}</td><td>{c.expected}</td><td>{c.retrieved}</td><td>{c.failure_type}</td><td>{c.method}</td><td>{c.observed}</td><td><small className="muted">{c.source}</small></td></tr>)}</tbody></table></div></div></>}
  </>;
}

/* ---------- About ---------- */
export function About({ info }: { info: OverviewResponse | null }) {
  const c = info?.corpus;
  const row = (label: string, done: boolean | null, note: string) => <li><span className={`dot ${done === null ? '' : done ? 'ok' : 'todo'}`} /><div><b>{label}</b><small>{note}</small></div></li>;
  return <>
    <PageHead eyebrow="PROJECT PROFILE" title="Research with accountability." desc="What the project set out to do, what is implemented today, and the limitations that apply to every result." />
    <div className="about-hero"><div className="eyebrow light">CSET 346 · NATURAL LANGUAGE PROCESSING</div><h2>IndicRAG-QA</h2><p>Evidence-Grounded Cross-Lingual Question Answering for Indic and Code-Mixed Languages.</p><div className="about-tags"><span>Bennett University</span><span>Project PJ-09</span><span>Team ARTHAI</span></div></div>
    <div className="about-grid"><div className="panel"><span className="eyebrow">RESEARCH QUESTION</span><h3>Does multilingual retrieval improve grounded QA for Indic and code-mixed queries?</h3>
      <p>The project compares lexical retrieval with multilingual dense and hybrid retrieval over English university documents, queried in English, Hindi and Hinglish, and aims to keep answers grounded in retrieved evidence.</p>
      <div className="about-rule" /><span className="eyebrow">STATUS OF THE PLANNED PIPELINE</span>
      <ul className="status-list">
        {row('Document corpus', true, c ? `${c.documents} Bennett University documents · ${c.pages} pages · ${c.chunks} page-aware passages` : 'Loading…')}
        {row('BM25 lexical baseline', true, 'Implemented and evaluated across English and multilingual queries')}
        {row('Multilingual dense retrieval (multilingual-e5-base)', info ? info.methods.pretrained.available : null, info?.methods.pretrained.available ? 'Index found; ready' : 'Needs the embedding index and sentence-transformers on the server')}
        {row('Hybrid retrieval (RRF)', info ? info.methods.hybrid.available : null, 'Reciprocal Rank Fusion of BM25 and pretrained multilingual E5')}
        {row('Answer generation / RAG', true, 'Implemented: short answer with [n] citations (extractive offline, or LLM with an API key), post-checked')}
        {row('Answerability detection', info ? !!info.artifacts.answerability_calibrated : null, info?.artifacts.answerability_calibrated ? 'Threshold fitted on dev split; evaluated on test split' : 'Implemented; needs a calibration run (backend/evaluate_answerability.py) on the full corpus')}
        {row('Human-verified relevance labels', false, 'Labels so far are AI-assisted screening; human-verification workflow in progress')}
      </ul></div>
      <div className="panel team-panel"><span className="eyebrow">PROJECT TEAM</span><h3>ARTHAI</h3>{[['Muskan Singh', 'S24CSEU2095'], ['Arisha Ali', 'S24CSEU2060'], ['Tulsi Sanskriti Tomar', 'S24CSEU0084'], ['Shermin Reaz', 'S24CSEU1730']].map(([n, id]) => <div className="team-row" key={id}><div className="team-avatar">{n[0]}</div><div><b>{n}</b><small>{id}</small></div></div>)}<div className="about-rule" /><small>INSTRUCTOR</small><b>Dr. Ankur Kumar Singhal</b><div className="about-rule" /><small>SCHOOL</small><b>School of Computer Science Engineering and Technology</b></div></div>
    <div className="panel"><span className="eyebrow">LIMITATIONS</span>
      <ul className="limit-list">
        <li>The corpus has six documents; the Student Handbook (2019-20) and Examination Manual are OCR-derived and noisy on some pages. The 2016-17 ordinances may be superseded by newer policy.</li>
        <li>Current relevance labels reflect assistant-assisted screening; human-reviewed evaluation benchmarks are being expanded.</li>
        <li>Fine-tuned E5 is planned as future research work on an independently validated query-passage corpus; the production system deploys BM25, pretrained multilingual E5, and Hybrid RRF.</li>
        <li>Language detection is a script and word-list heuristic, not a trained model.</li>
        <li>The MLQA benchmark uses Wikipedia text and measures retrieval only, not answer accuracy.</li>
      </ul></div>
  </>;
}
