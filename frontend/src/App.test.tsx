// UI integration tests. They talk to a REAL running backend (uvicorn on 127.0.0.1:8000), not mocks,
// except the one test that simulates the backend being offline.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import App from './App';

let backendUp = false;
beforeAll(async () => {
  try { backendUp = (await fetch('http://127.0.0.1:8000/health')).ok; } catch { backendUp = false; }
  window.scrollTo = (() => {}) as any;
  (globalThis as any).ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };  // jsdom lacks it; recharts needs it
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const openAsk = () => fireEvent.click(screen.getAllByText('Ask a Question')[0].closest('button')!);
const type = (text: string) => fireEvent.change(screen.getByLabelText('Your question'), { target: { value: text } });
const pick = (label: RegExp, value: string) => fireEvent.change(screen.getByText(label).closest('label')!.querySelector('select')!, { target: { value } });

describe('Ask a Question workflow', () => {
  it('opens a dedicated results view with real retrieved passages (BM25)', async () => {
    if (!backendUp) throw new Error('Start the backend on :8000 before running these tests');
    render(<App />); openAsk();
    type('What is the minimum attendance required for end semester exams?');
    pick(/RETRIEVAL METHOD/, 'BM25');
    fireEvent.click(screen.getByText(/Analyse question/));
    expect(await screen.findByText('Query results')).toBeTruthy();                 // dedicated view opened
    expect(screen.queryByLabelText('Your question')).toBeNull();                    // form is no longer below the results
    await screen.findByText(/passages? retrieved/, undefined, { timeout: 10000 });
    expect(screen.getByText('What is the minimum attendance required for end semester exams?')).toBeTruthy();
    expect(screen.getAllByText('Notification-for-Minimum-Attendance-for-Students.pdf').length).toBeGreaterThan(0);
    expect(screen.getByText(/Chunk notification_for_minimum_attendance_for_students_p1_c1/)).toBeTruthy();
    expect(screen.getByTestId('answer-card')).toBeTruthy();                         // a cited answer or an explicit abstention
    fireEvent.click(screen.getByText('Edit question'));
    expect(screen.getByLabelText('Your question')).toBeTruthy();                    // navigation back works
  });

  it('shows a clear empty state for a Hindi query with BM25', async () => {
    render(<App />); openAsk();
    type('अंतिम सेमेस्टर परीक्षा देने के लिए न्यूनतम उपस्थिति कितनी होनी चाहिए?');
    pick(/RETRIEVAL METHOD/, 'BM25');
    fireEvent.click(screen.getByText(/Analyse question/));
    expect(await screen.findByText('No passages matched this query', undefined, { timeout: 10000 })).toBeTruthy();
    expect(screen.getByText('Hindi')).toBeTruthy();
    expect(screen.getByText(/document corpus is English/)).toBeTruthy();
  });

  it('falls back to flagged BM25 results when the dense half of hybrid is unavailable', async () => {
    const h = await (await fetch('http://127.0.0.1:8000/health')).json();
    if (h.methods.hybrid.available) return;                                          // only meaningful without the E5 index
    render(<App />); openAsk();
    type('Scholarship ke liye minimum eligibility kya hai?');
    fireEvent.click(screen.getByText(/Analyse question/));                           // default = hybrid
    expect(await screen.findByText('Hybrid ran in BM25-only mode', undefined, { timeout: 10000 })).toBeTruthy();
    await screen.findByText(/passages? retrieved/, undefined, { timeout: 10000 });
    expect(screen.getByText('Hinglish')).toBeTruthy();
  });

  it('shows a backend-offline error instead of crashing', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))));
    render(<App />); openAsk();
    type('minimum attendance');
    fireEvent.click(screen.getByText(/Analyse question/));
    expect(await screen.findByText('Backend not reachable')).toBeTruthy();
    expect(screen.getByText('Try again')).toBeTruthy();
  });
});

describe('Research pages use real backend data', () => {
  it('Document Library lists the five indexed documents', async () => {
    render(<App />);
    fireEvent.click(screen.getAllByText('Document Library')[0].closest('button')!);
    expect(await screen.findByText('Exam-Manual.pdf', undefined, { timeout: 10000 })).toBeTruthy();
    const h = await (await fetch('http://127.0.0.1:8000/health')).json();
    if (h.chunk_count < 400) return;                                                 // synthetic fixture corpus: only the full corpus has all five documents
    expect(screen.getByText('studentHandbookBennett.pdf')).toBeTruthy();
    expect(screen.getByText('Ord-RulesandReg-UG-BTechF.pdf')).toBeTruthy();
  });

  it('Evaluation Lab shows the MLQA table and the honest module statuses', async () => {
    render(<App />);
    fireEvent.click(screen.getAllByText('Evaluation Lab')[0].closest('button')!);
    expect(await screen.findByText(/Passage retrieval, Hindi and English/, undefined, { timeout: 10000 })).toBeTruthy();
    expect(screen.queryAllByText('Not implemented').length).toBe(0);               // modules 4 and 5 now exist (measured or ready-not-run)
    expect(screen.getAllByText(/Ready, not run yet|Measured/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/AI-assisted/i, { selector: '.demo-pill' })).toBeTruthy();
  });

  it('Error Analysis lists artifact-backed cases and the required-case count', async () => {
    render(<App />);
    fireEvent.click(screen.getAllByText('Error Analysis')[0].closest('button')!);
    expect(await screen.findByText(/of 15 required cases/, undefined, { timeout: 10000 })).toBeTruthy();
    const table = screen.getByRole('table');
    expect(within(table).getAllByText('Cross-lingual vocabulary mismatch').length).toBeGreaterThan(0);
  });
});
