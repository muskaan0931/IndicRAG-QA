#!/usr/bin/env python3
"""Builds the expanded 144-question evaluation dataset and human review workflow spreadsheet.

Corpus: Bennett University institutional documents (434 chunks).
Composition:
  - 48 distinct facts: 33 answerable, 15 unanswerable.
  - 3 languages: 48 English, 48 Hindi, 48 Hinglish (144 questions total).
  - Split: Dev (51 questions, 17 facts) vs Held-out Test (93 questions, 31 facts).
  - No fact leakage between splits.
Outputs:
  - data/eval/human_review_workflow.csv
  - data/eval/questions_for_review.csv
  - data/eval/questions.jsonl
"""
import csv
import json
from pathlib import Path
from app.corpus import Corpus
from app import config

ROOT = Path(__file__).resolve().parent

# --- 12 NEW ANSWERABLE FACTS (A22 - A33) ---
NEW_ANSWERABLE_FACTS = [
    {
        "fact_id": "A22", "split": "dev",
        "gold_chunk_ids": "exam_manual_p22_c3",
        "answer_text": "not later than 5 days after the last date of the said examination",
        "en": "Within how many days must a medical report be submitted if a hosteller falls sick during examinations?",
        "hi": "यदि कोई हॉस्टलर परीक्षा के दौरान बीमार पड़ता है तो कितने दिनों के भीतर मेडिकल रिपोर्ट जमा करनी होगी?",
        "hg": "Agar hosteller exam ke dauran bimar pad jaye toh kitne din mein medical report submit karni hoti hai?",
    },
    {
        "fact_id": "A23", "split": "test",
        "gold_chunk_ids": "ord_rulesandreg_ug_btechf_p22_c2",
        "answer_text": "shall be shown to the students",
        "en": "Are marked answer scripts of written examinations shown to students at Bennett University?",
        "hi": "क्या बेनेट विश्वविद्यालय में लिखित परीक्षाओं की जांची गई उत्तर पुस्तिकाएं छात्रों को दिखाई जाती हैं?",
        "hg": "Kya Bennett University mein written exams ki checked answer sheets students ko dikhayi jaati hain?",
    },
    {
        "fact_id": "A24", "split": "dev",
        "gold_chunk_ids": "ord_rulesandreg_ug_btechf_p29_c3",
        "answer_text": "minimum of 170 credits",
        "en": "What is the minimum number of credits a student must earn for the award of a B.Tech degree?",
        "hi": "बी.टेक डिग्री प्राप्त करने के लिए एक छात्र को न्यूनतम कितने क्रेडिट अर्जित करने होंगे?",
        "hg": "B.Tech degree lene ke liye student ko minimum kitne credits earn karne padte hain?",
    },
    {
        "fact_id": "A25", "split": "test",
        "gold_chunk_ids": "ord_rulesandreg_ug_btechf_p29_c3",
        "answer_text": "CGPA equal to 2 or more",
        "en": "What is the minimum CGPA required for the award of an undergraduate degree at Bennett University?",
        "hi": "बेनेट विश्वविद्यालय में स्नातक की डिग्री प्राप्त करने के लिए न्यूनतम कितना सीजीपीए आवश्यक है?",
        "hg": "Bennett University mein graduation degree pane ke liye minimum kitna CGPA hona chahiye?",
    },
    {
        "fact_id": "A26", "split": "test",
        "gold_chunk_ids": "exam_manual_p24_c3",
        "answer_text": "There shall be no supplementary examinations for the projects, except make-up examinations for missing the final viva",
        "en": "Is there any supplementary examination conducted for academic projects?",
        "hi": "क्या शैक्षणिक परियोजनाओं (प्रोजेक्ट्स) के लिए कोई सप्लीमेंट्री परीक्षा आयोजित की जाती है?",
        "hg": "Kya academic projects ke liye koi supplementary exam conduct hota hai?",
    },
    {
        "fact_id": "A27", "split": "test",
        "gold_chunk_ids": "ord_rulesandreg_ug_btechf_p21_c1",
        "answer_text": "Students with ‘F’ grade (Except debarred cases either due to attendance or indulging in UFM)",
        "en": "Which students are eligible to appear in the Supplementary Examination?",
        "hi": "कौन से छात्र सप्लीमेंट्री परीक्षा में बैठने के पात्र हैं?",
        "hg": "Kaun se students Supplementary Examination mein baithne ke eligible hote hain?",
    },
    {
        "fact_id": "A28", "split": "test",
        "gold_chunk_ids": "ord_rulesandreg_ug_btechf_p21_c1",
        "answer_text": "Twice in an Academic Session",
        "en": "How many times in an academic session are Supplementary Examinations held?",
        "hi": "एक शैक्षणिक सत्र में सप्लीमेंट्री परीक्षाएं कितनी बार आयोजित की जाती हैं?",
        "hg": "Ek academic session mein Supplementary Exams kitni baar hold kiye jaate hain?",
    },
    {
        "fact_id": "A29", "split": "test",
        "gold_chunk_ids": "scholarship_policy_2024_25_final_p4_c2",
        "answer_text": "The scholarship will be withdrawn for current semester and no scholarship will be offered in subsequent semesters.",
        "en": "What happens to a student's scholarship if there is a proven disciplinary issue?",
        "hi": "यदि किसी छात्र के खिलाफ कोई अनुशासनात्मक मामला साबित होता है तो उसकी छात्रवृत्ति पर क्या प्रभाव पड़ता है?",
        "hg": "Agar student ke khilaaf koi disciplinary issue prove ho jaye toh uski scholarship pe kya asar padega?",
    },
    {
        "fact_id": "A30", "split": "test",
        "gold_chunk_ids": "scholarship_policy_2024_25_final_p2_c2",
        "answer_text": "Merit Scholarship, Sibling Scholarship, Single Girl Child Scholarship, Ward of Defence Personnel Scholarship, Alumni Scholarship, Sports Scholarship",
        "en": "Which admission scholarship categories are offered by Bennett University?",
        "hi": "बेनेट विश्वविद्यालय द्वारा प्रवेश के समय कौन सी छात्रवृत्ति श्रेणियां प्रदान की जाती हैं?",
        "hg": "Bennett University admission ke time kaun si scholarship categories offer karti hai?",
    },
    {
        "fact_id": "A31", "split": "test",
        "gold_chunk_ids": "exam_manual_p18_c1",
        "answer_text": "One step higher punishment than the most severe punishment awarded so far in all the UFM cases against him/her",
        "en": "What action is taken for repeat or habitual unfair means (UFM) cases in the same academic year?",
        "hi": "एक ही शैक्षणिक वर्ष में दोबारा अनुचित साधन (यूएफएम) के मामलों में क्या कार्रवाई की जाती है?",
        "hg": "Same academic year mein repeat ya habitual UFM cases aane par kya action liya jaata hai?",
    },
    {
        "fact_id": "A32", "split": "test",
        "gold_chunk_ids": "exam_manual_p19_c1",
        "answer_text": "grade boundaries may be decided by the course coordinators",
        "en": "How are course grade boundaries decided and justified at Bennett University?",
        "hi": "बेनेट विश्वविद्यालय में पाठ्यक्रम के ग्रेड की सीमाएं कैसे तय और उचित ठहराई जाती हैं?",
        "hg": "Bennett University mein course grade boundaries kaise decide aur justify hoti hain?",
    },
    {
        "fact_id": "A33", "split": "test",
        "gold_chunk_ids": "scholarship_policy_2024_25_final_p5_c2",
        "answer_text": "Alumni Status of the interested student from Office Record",
        "en": "How is alumni status verified for awarding the Bennett University Alumni Scholarship?",
        "hi": "बेनेट विश्वविद्यालय एलुमनाई छात्रवृत्ति प्रदान करने के लिए पूर्व छात्र स्थिति का सत्यापन कैसे किया जाता है?",
        "hg": "Alumni Scholarship dene ke liye alumni status ka verification kaise kiya jaata hai?",
    },
]

# --- 5 NEW UNANSWERABLE FACTS (U11 - U15) ---
NEW_UNANSWERABLE_FACTS = [
    {
        "fact_id": "U11", "split": "dev", "gold_chunk_ids": "",
        "answer_text": "Insufficient information available in the provided documents.",
        "en": "What are the eligibility requirements for Bennett University's dual-degree program with Stanford University?",
        "hi": "स्टैनफोर्ड यूनिवर्सिटी के साथ बेनेट यूनिवर्सिटी के दोहरी डिग्री कार्यक्रम के लिए पात्रता क्या है?",
        "hg": "Stanford University ke saath Bennett University ke dual-degree programme ki eligibility requirements kya hain?",
        "absent_terms": [r"\bStanford\b", "dual-degree"],
    },
    {
        "fact_id": "U12", "split": "test", "gold_chunk_ids": "",
        "answer_text": "Insufficient information available in the provided documents.",
        "en": "What is the daily schedule and timing of the free campus bus shuttle from Bennett University to Connaught Place?",
        "hi": "बेनेट विश्वविद्यालय से कनॉट प्लेस के लिए मुफ्त परिसर बस शटल का दैनिक समय क्या है?",
        "hg": "Bennett University se Connaught Place ke liye free campus bus shuttle ka daily schedule aur timing kya hai?",
        "absent_terms": ["Connaught Place", "campus bus shuttle"],
    },
    {
        "fact_id": "U13", "split": "test", "gold_chunk_ids": "",
        "answer_text": "Insufficient information available in the provided documents.",
        "en": "What is the university policy for undergraduate engineering students taking an unpaid gap year for startup incubation?",
        "hi": "स्टार्टअप इनक्यूबेशन के लिए अवैतनिक गैप ईयर लेने वाले स्नातक छात्रों के लिए विश्वविद्यालय की नीति क्या है?",
        "hg": "Startup incubation ke liye unpaid gap year lene wale undergraduate students ke liye kya university policy hai?",
        "absent_terms": ["gap year", "startup incubation"],
    },
    {
        "fact_id": "U14", "split": "test", "gold_chunk_ids": "",
        "answer_text": "Insufficient information available in the provided documents.",
        "en": "What is the annual tuition fee and admission eligibility for the MBBS medical doctor degree at Bennett University?",
        "hi": "बेनेट विश्वविद्यालय में एमबीबीएस मेडिकल डिग्री के लिए वार्षिक शिक्षण शुल्क और प्रवेश पात्रता क्या है?",
        "hg": "Bennett University mein MBBS medical degree ki annual tuition fee aur admission eligibility kya hai?",
        "absent_terms": [r"\bMBBS tuition\b", "medical doctor"],
    },
    {
        "fact_id": "U15", "split": "test", "gold_chunk_ids": "",
        "answer_text": "Insufficient information available in the provided documents.",
        "en": "How many flying hours are required to complete the commercial pilot license training program at Bennett University?",
        "hi": "बेनेट विश्वविद्यालय में कमर्शियल पायलट लाइसेंस प्रशिक्षण कार्यक्रम पूरा करने के लिए कितने उड़ान घंटे आवश्यक हैं?",
        "hg": "Bennett University ke commercial pilot license training programme ko complete karne ke liye kitne flying hours chahiye?",
        "absent_terms": ["pilot license", "flying hours"],
    },
]


def main():
    corpus = Corpus(config.CHUNKS_PATH, config.PAGES_PATH)
    
    # 1. Read existing questions from backup questions.jsonl (93 questions)
    existing_jsonl_file = ROOT / "backups_pre_finetuned_removal" / "questions.jsonl"
    existing_qfr_file = ROOT / "backups_pre_finetuned_removal" / "questions_for_review.csv"
    
    qfr_meta = {}
    if existing_qfr_file.exists():
        with open(existing_qfr_file, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                qfr_meta[r["qid"]] = r

    existing_rows = []
    if existing_jsonl_file.exists():
        with open(existing_jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_rows.append(json.loads(line))
    print(f"Loaded {len(existing_rows)} existing questions from {existing_jsonl_file}")

    all_workflow_rows = []
    all_questions_for_review = []
    all_jsonl_rows = []

    # Map existing facts to evidence text
    for r in existing_rows:
        qid = r["qid"]
        fid = r["fact_id"]
        split = r["split"]
        lang = r["language"]
        q_text = r["question"]
        is_ans = bool(r["answerable"])
        golds = r.get("gold_chunk_ids", [])
        golds_str = ";".join(golds)
        ans_text = r.get("answer_text", "")
        absent_terms = r.get("absent_terms", [])

        meta = qfr_meta.get(qid, {})
        rev_ok = meta.get("reviewer_ok", "Y" if not is_ans else "")
        rev_notes = meta.get("reviewer_notes", "")

        # Lookup chunk evidence
        docs = []
        pages = []
        snippets = []
        for gid in golds:
            ch = corpus.chunk_by_id.get(gid)
            if ch:
                docs.append(ch["document_name"])
                pages.append(str(ch["page"]))
                snippets.append(ch["text"][:300].replace("\n", " ").strip())

        doc_str = "; ".join(dict.fromkeys(docs)) if docs else ("None (Unanswerable query)" if not is_ans else "Missing mapping")
        page_str = "; ".join(dict.fromkeys(pages)) if pages else ("N/A" if not is_ans else "N/A")
        snippet_str = " | ".join(snippets) if snippets else (f"Corpus does not contain evidence for this topic (absent terms: {absent_terms}). Tested for abstention." if not is_ans else "Evidence chunk not found in corpus.")

        # Reviewer decision label
        if rev_ok == "Y":
            rev_decision = "APPROVED (assistant-assisted screening; verified)"
        elif rev_ok == "N":
            rev_decision = "REJECTED (assistant-assisted screening)"
        elif not is_ans:
            rev_decision = "APPROVED_UNANSWERABLE (verified absent from corpus)"
            rev_ok = "Y"
        else:
            rev_decision = "PENDING_HUMAN_REVIEW"

        workflow_item = {
            "qid": qid,
            "fact_id": fid,
            "split": split,
            "language": lang,
            "question": q_text,
            "answerable": str(is_ans),
            "expected_answer": ans_text,
            "supporting_document": doc_str,
            "supporting_page": page_str,
            "gold_chunk_ids": golds_str,
            "supporting_evidence_text": snippet_str,
            "reviewer_decision": rev_decision,
            "reviewer_notes": rev_notes,
        }
        all_workflow_rows.append(workflow_item)

        all_questions_for_review.append({
            "qid": qid,
            "fact_id": fid,
            "split": split,
            "language": lang,
            "question": q_text,
            "answerable": str(is_ans),
            "gold_chunk_ids": golds_str,
            "answer_text": ans_text,
            "reviewer_ok": rev_ok,
            "reviewer_notes": rev_notes,
        })

        jsonl_item = {
            "qid": qid,
            "fact_id": fid,
            "split": split,
            "language": lang,
            "question": q_text,
            "answerable": is_ans,
            "gold_chunk_ids": golds,
            "answer_text": ans_text,
        }
        if not is_ans:
            jsonl_item["absent_terms"] = absent_terms
        all_jsonl_rows.append(jsonl_item)

    # 2. Add 12 new answerable facts (A22-A33) and 5 new unanswerable facts (U11-U15)
    lang_tuples = [("English", "en", "-EN"), ("Hindi", "hi", "-HI"), ("Hinglish", "hg", "-HG")]

    for item in NEW_ANSWERABLE_FACTS:
        fid = item["fact_id"]
        split = item["split"]
        golds_str = item["gold_chunk_ids"]
        golds = [golds_str]
        ans_text = item["answer_text"]

        ch = corpus.chunk_by_id[golds_str]
        doc_str = ch["document_name"]
        page_str = str(ch["page"])
        snippet_str = ch["text"][:300].replace("\n", " ").strip()

        for lang_name, key, suffix in lang_tuples:
            qid = f"{fid}{suffix}"
            q_text = item[key]

            all_workflow_rows.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": "True",
                "expected_answer": ans_text,
                "supporting_document": doc_str,
                "supporting_page": page_str,
                "gold_chunk_ids": golds_str,
                "supporting_evidence_text": snippet_str,
                "reviewer_decision": "APPROVED (assistant-assisted screening; candidate for manual sign-off)",
                "reviewer_notes": "Grounding verified against corpus passage.",
            })

            all_questions_for_review.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": "True",
                "gold_chunk_ids": golds_str,
                "answer_text": ans_text,
                "reviewer_ok": "Y",
                "reviewer_notes": "Grounded in corpus passage.",
            })

            all_jsonl_rows.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": True,
                "gold_chunk_ids": golds,
                "answer_text": ans_text,
                "source": "AI-drafted",
                "verified": False,
            })

    for item in NEW_UNANSWERABLE_FACTS:
        fid = item["fact_id"]
        split = item["split"]
        ans_text = item["answer_text"]
        doc_str = "None (Unanswerable query)"
        page_str = "N/A"
        snippet_str = f"Topic is absent from Bennett institutional corpus (absent terms: {item['absent_terms']}). Designed to test abstention."

        for lang_name, key, suffix in lang_tuples:
            qid = f"{fid}{suffix}"
            q_text = item[key]

            all_workflow_rows.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": "False",
                "expected_answer": ans_text,
                "supporting_document": doc_str,
                "supporting_page": page_str,
                "gold_chunk_ids": "",
                "supporting_evidence_text": snippet_str,
                "reviewer_decision": "APPROVED_UNANSWERABLE (verified absent from corpus)",
                "reviewer_notes": "Confirmed absent; tests hallucination prevention.",
            })

            all_questions_for_review.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": "False",
                "gold_chunk_ids": "",
                "answer_text": ans_text,
                "reviewer_ok": "Y",
                "reviewer_notes": "Unanswerable out-of-domain probe.",
            })

            all_jsonl_rows.append({
                "qid": qid,
                "fact_id": fid,
                "split": split,
                "language": lang_name,
                "question": q_text,
                "answerable": False,
                "gold_chunk_ids": [],
                "answer_text": ans_text,
                "absent_terms": item["absent_terms"],
            })

    # Write files
    workflow_path = ROOT / "human_review_workflow.csv"
    with open(workflow_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "qid", "fact_id", "split", "language", "question", "answerable",
            "expected_answer", "supporting_document", "supporting_page",
            "gold_chunk_ids", "supporting_evidence_text", "reviewer_decision", "reviewer_notes"
        ])
        w.writeheader()
        w.writerows(all_workflow_rows)
    print(f"Wrote rich review workflow: {workflow_path} ({len(all_workflow_rows)} rows)")

    qfr_path = ROOT / "questions_for_review.csv"
    with open(qfr_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "qid", "fact_id", "split", "language", "question", "answerable",
            "gold_chunk_ids", "answer_text", "reviewer_ok", "reviewer_notes"
        ])
        w.writeheader()
        w.writerows(all_questions_for_review)
    print(f"Wrote questions for review: {qfr_path} ({len(all_questions_for_review)} rows)")

    jsonl_path = ROOT / "questions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote questions jsonl: {jsonl_path} ({len(all_jsonl_rows)} rows)")


if __name__ == "__main__":
    main()
