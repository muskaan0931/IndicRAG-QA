import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowDownRight, ArrowRight, BookOpen, Brain, Check, ChevronDown, ChevronRight, CircleHelp, ClipboardList, Database, FileSearch, FileText, FlaskConical, Globe2, GraduationCap, Layers3, Library, Menu, MessageSquareText, Microscope, Search, ShieldCheck, SlidersHorizontal, Sparkles, Upload, X } from 'lucide-react';
import { api, detectPreview, type MethodKey, type OverviewResponse } from './api';
import { About, AskResults, Comparison, Documents, Errors, Evaluation, Explorer, METHOD_OPTIONS, OverviewStats, PageHead, type AskRequestSnapshot, type AskState } from './views';

type Page = 'Overview' | 'Ask a Question' | 'Research Explorer' | 'Document Library' | 'Retrieval Comparison' | 'Evaluation Lab' | 'Error Analysis' | 'About the Project';
const pages: { name: Page; icon: any; group: string }[] = [
    { name: 'Overview', icon: Activity, group: 'WORKSPACE' }, { name: 'Ask a Question', icon: MessageSquareText, group: 'WORKSPACE' }, { name: 'Research Explorer', icon: FileSearch, group: 'RESEARCH' }, { name: 'Document Library', icon: Library, group: 'RESEARCH' }, { name: 'Retrieval Comparison', icon: Layers3, group: 'RESEARCH' }, { name: 'Evaluation Lab', icon: FlaskConical, group: 'RESEARCH' }, { name: 'Error Analysis', icon: ClipboardList, group: 'RESEARCH' }, { name: 'About the Project', icon: CircleHelp, group: 'PROJECT' }
];
const examples = ['What is the minimum attendance requirement?', 'न्यूनतम उपस्थिति कितनी होनी चाहिए?', 'Scholarship ke liye minimum eligibility kya hai?', 'What happens if a student misses an examination?', 'Is there a refund policy for withdrawn admissions?'];
function App() {
    const [page, setPage] = useState<Page>('Overview'); const [mobile, setMobile] = useState(false); const [query, setQuery] = useState(''); const [method, setMethod] = useState('Hybrid Retrieval'); const [topK, setTopK] = useState('5'); const [docSearch, setDocSearch] = useState(''); const [filter, setFilter] = useState('All categories');
    const [askView, setAskView] = useState<'form' | 'results'>('form'); const [askState, setAskState] = useState<AskState>({ status: 'idle' }); const [request, setRequest] = useState<AskRequestSnapshot>({ query: '', method: 'hybrid', topK: 5 }); const [compareQuery, setCompareQuery] = useState('');
    const [info, setInfo] = useState<OverviewResponse | null>(null); const [infoError, setInfoError] = useState(''); const [offline, setOffline] = useState(false);
    const reqId = useRef(0); const abortRef = useRef<AbortController | null>(null);
    useEffect(() => { const c = new AbortController(); api.overview(c.signal).then(d => { setInfo(d); setOffline(false); setInfoError(''); }).catch((e: any) => { if (e?.name === 'AbortError') return; setInfoError(e?.message || 'Backend unavailable'); setOffline(!!e?.offline); }); return () => c.abort(); }, [page]);
    useEffect(() => { window.scrollTo({ top: 0 }); }, [askView, page]);
    const nav = (p: Page) => { if (p === 'Ask a Question' && page === 'Ask a Question') setAskView('form'); setPage(p); setMobile(false); };
    const keyOf = (label: string): MethodKey => METHOD_OPTIONS.find(m => m.label === label)?.key || 'hybrid';
    const runAsk = async (q: string, key: MethodKey, k: number) => {
        abortRef.current?.abort(); const c = new AbortController(); abortRef.current = c; const id = ++reqId.current;
        setRequest({ query: q, method: key, topK: k }); setAskView('results'); setPage('Ask a Question'); setAskState({ status: 'loading' });
        try { const data = await api.ask(q, key, k, c.signal); if (id !== reqId.current) return; setAskState({ status: 'done', data }); }
        catch (e: any) { if (e?.name === 'AbortError' || id !== reqId.current) return; setAskState({ status: 'error', error: e?.message || 'Unable to reach the backend', offline: !!e?.offline, httpStatus: e?.status }); }
    };
    const ask = (q: string = query) => { const text = (typeof q === 'string' ? q : query).trim(); if (!text) return; setQuery(text); runAsk(text, keyOf(method), Number(topK)); };
    const switchMethod = (key: MethodKey) => { const label = METHOD_OPTIONS.find(m => m.key === key)!.label; setMethod(label); runAsk(request.query, key, request.topK); };
    const goCompare = () => { setCompareQuery(request.query); setPage('Retrieval Comparison'); };
    return <div className="shell">
        <aside className={`sidebar ${mobile ? 'open' : ''}`}>
            <div className="brand"><div className="brand-mark">IR</div><div><div className="brand-name">IndicRAG-QA</div><div className="brand-sub">CROSS-LINGUAL RESEARCH</div></div><button className="mobile-close" onClick={() => setMobile(false)}><X size={18} /></button></div>
            <div className="side-rule" />
            {['WORKSPACE', 'RESEARCH', 'PROJECT'].map(g => <div className="nav-group" key={g}><div className="nav-label">{g}</div>{pages.filter(p => p.group === g).map(item => <button key={item.name} onClick={() => nav(item.name)} className={`nav-item ${page === item.name ? 'active' : ''}`}><item.icon size={17} /><span>{item.name}</span>{page === item.name && <span className="active-dot" />}</button>)}</div>)}
            <div className="side-bottom"><div className="university-icon"><GraduationCap size={18} /></div><div><b>Bennett University</b><span>School of Computer Science</span><span>CSET 346 · NLP · PJ-09</span></div></div>
        </aside>
        {mobile && <div className="scrim" onClick={() => setMobile(false)} />}
        <main className="main">
            <header className="topbar"><button className="mobile-menu" onClick={() => setMobile(true)}><Menu size={20} /></button><div className="crumb">Research workspace <ChevronRight size={14} /> <b>{page}</b></div><div className="top-right"><span className="status"><i /> {offline ? 'API offline' : info ? `API connected · ${info.corpus.chunks} passages` : 'Connecting…'}</span><div className="avatar">A</div></div></header>
            <div className="content">
                {page === 'Overview' && <Overview nav={nav} ask={ask} info={info} infoError={infoError} offline={offline} />}
                {page === 'Ask a Question' && <Ask query={query} setQuery={setQuery} method={method} setMethod={setMethod} topK={topK} setTopK={setTopK} ask={ask} askView={askView} askState={askState} request={request} backToForm={() => setAskView('form')} retry={() => runAsk(request.query, request.method, request.topK)} switchMethod={switchMethod} goCompare={goCompare} />}
                {page === 'Research Explorer' && <Explorer />}
                {page === 'Document Library' && <Documents search={docSearch} setSearch={setDocSearch} filter={filter} setFilter={setFilter} />}
                {page === 'Retrieval Comparison' && <Comparison presetQuery={compareQuery} clearPreset={() => setCompareQuery('')} />}
                {page === 'Evaluation Lab' && <Evaluation />}
                {page === 'Error Analysis' && <Errors />}
                {page === 'About the Project' && <About info={info} />}
            </div>
            <footer className="footer"><span>INDIRAG-QA · RESEARCH PROTOTYPE</span><span>Evidence first. Answers with accountability.</span></footer>
        </main>
    </div>
}
function Overview({ nav, ask, info, infoError, offline }: any) {
    return <><section className="hero">
        <div className="hero-copy"><div className="eyebrow light">BENNETT UNIVERSITY <span /> SCHOOL OF COMPUTER SCIENCE</div><h1>Answers grounded<br />in <em>evidence.</em><br />Across languages.</h1><p>An experimental cross-lingual question-answering system that connects English academic documents with questions asked in English, Hindi and Hinglish.</p><div className="hero-actions"><button className="btn ivory" onClick={() => nav('Research Explorer')}>Explore the research <ArrowRight size={16} /></button><button className="btn ghost" onClick={() => nav('Ask a Question')}>Ask a question</button></div></div>
        <div className="hero-art"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="hero-core"><BookOpen size={34} /><span>Evidence<br />Engine</span></div><div className="float-card fc-top"><span className="mini-tag">EXAMPLE QUERY</span><b>कितनी उपस्थिति?</b><small>Hindi · example</small></div><div className="float-card fc-bottom"><span className="mini-tag teal">RETRIEVAL</span><b>{info ? `${info.corpus.chunks} passages indexed` : 'Connecting…'}</b><small>Bennett University documents</small></div><div className="orbit-dot" /></div>
    </section>
        <section className="language-strip"><div className="strip-label">ONE KNOWLEDGE BASE<br /><b>THREE WAYS TO ASK</b></div><div className="language"><small>ENGLISH</small><span>“What is the minimum attendance requirement?”</span></div><div className="language hindi"><small>हिन्दी · HINDI</small><span>“न्यूनतम उपस्थिति कितनी होनी चाहिए?”</span></div><div className="language"><small>HINGLISH</small><span>“Minimum attendance kitni honi chahiye?”</span></div></section>
        <div className="section-heading"><div><div className="eyebrow">THE RESEARCH PIPELINE</div><h2>From documents to defensible answers.</h2></div><span className="muted-note">Proposed system workflow</span></div>
        <div className="pipeline">{['Document collection', 'Passage segmentation', 'Lexical & dense indexing', 'Cross-lingual retrieval', 'Evidence gate', 'RAG generation', 'Answer verification'].map((s, i) => <div className="pipe-step" key={s} style={i >= 4 ? { opacity: .5 } : undefined} title={i >= 4 ? 'Planned, not implemented yet' : undefined}><div className={`pipe-icon ${i === 4 ? 'highlight' : ''}`}>{[<Library />, <FileText />, <Database />, <Globe2 />, <ShieldCheck />, <Brain />, <Check />][i]}</div><span>{s}{i >= 4 && ' (planned)'}</span>{i < 6 && <ArrowRight className="pipe-arrow" size={14} />}</div>)}</div>
        <OverviewStats info={info} error={infoError} offline={offline} />
        <section className="objective"><div><div className="eyebrow light">RESEARCH INTENT</div><h2>Five questions.<br /><em>One accountable system.</em></h2><p>Investigating how retrieval strategies and multilingual representations affect the reliability of question answering over English academic documents.</p></div><div className="objective-list">{['How does multilingual dense retrieval compare with lexical retrieval?', 'How reliably can the system handle Hindi and code-mixed queries?', 'Does hybrid retrieval improve evidence relevance?', 'When does RAG outperform direct LLM question answering?', 'Can answerability detection reduce unsupported responses?'].map((x, i) => <div key={x}><span>0{i + 1}</span>{x}</div>)}</div></section>
        <div className="notice"><Sparkles size={17} /><span><b>Research integrity note</b> · Retrieval numbers in the Evaluation Lab come from a 4-query pilot with AI-assisted labels and from the external MLQA benchmark. They are diagnostic, not final results. The system returns cited evidence; it does not generate answers.</span></div>
    </>
}
function Ask({ query, setQuery, method, setMethod, topK, setTopK, ask, askView, askState, request, backToForm, retry, switchMethod, goCompare }: any) {
    if (askView === 'results') return <AskResults state={askState} request={request} onBack={backToForm} onRetry={retry} onSwitch={switchMethod} onCompare={goCompare} />;
    return <><PageHead eyebrow="CROSS-LINGUAL QUESTION ANSWERING" title="Ask across languages." desc="Ask in English, Hindi or Hinglish. The system retrieves the passages that support your question and cites them." />
        <div className="ask-layout"><div className="ask-main"><div className="panel ask-panel"><div className="panel-head"><div><span className="eyebrow">01 / YOUR QUESTION</span><h3>What would you like to know?</h3></div><span className="language-pill"><Globe2 size={13} /> EN · HI · HINGLISH</span></div><textarea value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && query.trim()) ask(); }} aria-label="Your question" placeholder="Scholarship ke liye minimum eligibility kya hai?" /><div className="detected-row"><span className="detect-dot" /> Language detection <b>{query.trim() ? detectPreview(query) : 'Waiting for input'}</b><span className="auto-label">Heuristic preview · backend confirms</span></div><div className="control-row"><label>RETRIEVAL METHOD<select value={method} onChange={e => setMethod(e.target.value)}>{METHOD_OPTIONS.map(m => <option key={m.key}>{m.label}</option>)}</select></label><label>TOP-K PASSAGES<select value={topK} onChange={e => setTopK(e.target.value)}><option>3</option><option>5</option><option>10</option></select></label><button type="button" className="btn primary" onClick={() => ask()} disabled={!query.trim()}>Analyse question <ArrowRight size={16} /></button></div></div>
            <div className="examples"><div className="eyebrow">TRY A RESEARCH QUERY</div><div className="example-list">{examples.map((e, i) => <button key={e} onClick={() => { setQuery(e) }}><span>{['EN', 'हि', 'HI+EN', 'EN', 'EN'][i]}</span>{e}<ArrowDownRight size={14} /></button>)}</div></div>
        </div><aside className="ask-aside"><div className="aside-card"><div className="aside-icon"><ShieldCheck size={20} /></div><h3>Evidence before answers.</h3><p>IndicRAG-QA returns the source passages that bear on your question, with document, page and chunk citations. It does not write an answer yet.</p><div className="aside-rule" /><div className="aside-point"><Check /> Retrieved passages</div><div className="aside-point"><Check /> Traceable citations</div><div className="aside-point" style={{ opacity: .55 }}><Check /> Answerability decision (planned)</div></div><div className="aside-note"><b>About retrieval methods</b><p>BM25 matches exact English words, so Hindi questions find nothing with it. Dense and hybrid retrieval need the embedding index on the server and report clearly when it is missing.</p></div></aside></div></>
}
export default App;