# IndicRAG-QA: Live Demonstration & Viva Script

This script provides a step-by-step, reproducible demonstration of **IndicRAG-QA** for viva examiners, peer reviewers, and evaluation sessions.

---

## 1. Environment & Services Startup

### Terminal 1 — Backend (FastAPI API)
```bash
# Navigate to project root
cd ~/Downloads/IndicRAG/IndicRAG_QA_FINAL

# Activate virtual environment
source .venv/bin/activate

# Launch backend server
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Verify: Open `http://127.0.0.1:8000/docs` to inspect interactive Swagger documentation.*

### Terminal 2 — Frontend (React + Vite)
```bash
# Navigate to frontend
cd ~/Downloads/IndicRAG/IndicRAG_QA_FINAL/frontend

# Launch frontend development server
npm run dev
```
*Verify: Open `http://localhost:5173` in your browser.*

---

## 2. Core Demonstration Walkthrough

### Test Case 1: Monolingual English Grounded QA with Citations
1. Open the **Ask Question** tab in the UI (`http://localhost:5173`).
2. Enter the query:
   ```text
   What is the minimum attendance requirement to appear in the end semester examination?
   ```
3. Select Method: **Hybrid (BM25 + Pretrained E5, RRF)**.
4. Click **Ask**.
5. **Expected Output**:
   - **Answer**: *"A student must have a minimum attendance of 75% in a course during a semester, in lectures, tutorials or practical’s taken together (as applicable). [1]"*
   - **Answerability Gate**: `Status: answerable` (signal above calibrated threshold).
   - **Citations**: `[1] Exam-Manual.pdf, p. 28`.
   - **Evidence Cards**: Ranked evidence passages from `Exam-Manual.pdf` and `Notification-for-Minimum-Attendance-for-Students.pdf` showing RRF fusion score, page numbers, and highlighted text.
   - **Feedback**: Click the 👍 or 👎 button to verify local feedback logging in `data/feedback.jsonl`.

---

### Test Case 2: Cross-Lingual Hindi Query (Addressing Vocabulary Mismatch)
1. In the **Ask Question** tab, enter:
   ```text
   अंतिम सेमेस्टर परीक्षा में बैठने के लिए न्यूनतम उपस्थिति कितनी आवश्यक है?
   ```
2. First, switch method to **BM25** and click **Ask**:
   - **Observed**: BM25 pure keyword tokenizer produces 0 tokens on Devanagari script, returning no relevant passages.
3. Switch method to **Hybrid (BM25 + Pretrained E5, RRF)** and click **Ask**:
   - **Observed**: Multilingual E5 embeds the Devanagari query into the shared cross-lingual semantic space, accurately retrieving the English attendance policy chunk!
   - **Answer**: Generated with citation `[1]`.

---

### Test Case 3: Code-Mixed Hinglish Query
1. In the **Ask Question** tab, enter:
   ```text
   Sports scholarship continue karne ke liye minimum CGPA kitna chahiye?
   ```
2. Select Method: **Hybrid (BM25 + Pretrained E5, RRF)**.
3. Click **Ask**.
4. **Expected Output**:
   - Detected Language: `Hinglish`.
   - **Answer**: Retrieves `scholarship_policy_2024_25_final_p7_c2` specifying minimum 6.5 CGPA with valid citation.

---

### Test Case 4: Out-of-Domain Unanswerable Query (Abstention & Hallucination Guard)
1. In the **Ask Question** tab, enter:
   ```text
   What is the MBBS admission fee and cut off at Bennett University?
   ```
2. Click **Ask**.
3. **Expected Output**:
   - The query asks about MBBS (Bennett University offers B.Tech, Law, Management, Media, but no medical MBBS degree).
   - **Answerability Gate**: Identifies insufficient signal or low relevance.
   - **Answer**: Exactly `"Insufficient information available in the provided documents."`
   - **Abstention**: `True` (prevents hallucinating figures or guessing).

---

### Test Case 5: 4-Way Retriever Comparison (Compare Tab)
1. Navigate to the **Compare Retrievers** tab.
2. Enter:
   ```text
   What are the scholarship renewal conditions?
   ```
3. Set Top K = `5`. Click **Compare Methods**.
4. **Inspect Side-by-Side**:
   - **BM25**: Pure lexical TF-IDF ranking.
   - **Pretrained Multilingual E5**: Cosine similarity in 768-d multilingual semantic space.
   - **Hybrid RRF**: Reciprocal Rank Fusion combining lexical precision with dense semantic recall.
   - **Overlap Matrix**: Displays shared passages across methods and rank divergence.

---

### Test Case 6: Experiment Workbench & Live Evaluation Lab
1. Navigate to the **Evaluation Lab** tab.
2. **Module Cards**: All 6 research modules are displayed with real empirical status:
   - Module 01: TF-IDF / BM25 vs multilingual dense retrieval (`Measured`).
   - Module 02: Monolingual vs cross-lingual vs code-mixed queries (`Measured`).
   - Module 03: Lexical vs dense vs hybrid retrieval (`Measured`).
   - Module 04: Direct LLM QA vs retrieval-augmented QA (`Measured`).
   - Module 05: Hallucination and answerability testing (`Measured`).
   - Module 06: Qualitative analysis of 15+ difficult cases (`Measured`).
3. **Cross-Lingual Benchmark Table**:
   - BM25 All: Hit@5 = `0.59`, MRR = `0.51`
   - Pretrained E5 All: Hit@5 = `0.74`, MRR = `0.65`
   - **Hybrid RRF All: Hit@5 = 0.82, MRR = 0.69, nDCG@10 = 0.71**
   - Cross-lingual Hindi Hit@5: BM25 `0.23` vs Hybrid `0.62` (+39 point gain).
4. **Answerability Benchmark Table**:
   - Pretrained E5 & Hybrid: Test Accuracy = `84.2%`, Precision = `91.7%`, F1 = `0.88`, ROC-AUC = `0.9444`.

---

### Test Case 7: Error Analysis & Qualitative Case Register
1. Navigate to the **Error Analysis** tab.
2. Inspect the **16 documented failure cases** categorized by:
   - Cross-lingual lexical collapse (Devanagari queries vs English index)
   - Code-mixed partial loanword matching
   - Lexical keyword collisions (e.g. grading chunk outscoring attendance regulation in BM25)
   - Out-of-corpus unanswerable queries requiring abstention
   - OCR noisy / low-confidence page extractions

---

## 3. Automated Test Suite Execution
To prove system reliability to evaluators, execute the full test suite in terminal:
```bash
source .venv/bin/activate
pytest -v
```
**Expected**: All 53 unit and integration tests pass in ~8 seconds.
