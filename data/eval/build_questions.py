"""Regenerates data/eval/questions.jsonl and data/eval/questions_for_review.csv.

Every question is AI-drafted (source="AI-drafted", verified=False). Gold chunk ids were located by reading the corpus
text; backend/verify_dataset.py re-checks them mechanically. A HUMAN must still review each row (see the CSV).
Run from the repository root:  python data/eval/build_questions.py
"""
import collections
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
N = "notification_for_minimum_attendance_for_students_p1_c1"
EM, ORD, SC = "exam_manual_", "ord_rulesandreg_ug_btechf_", "scholarship_policy_2024_25_final_"
LANG_CODE = {"English": "EN", "Hindi": "HI", "Hinglish": "HG"}

# (fact_id, split, gold_chunk_ids, answer_text, English, Hindi, Hinglish)
A = [
("A01","dev",[N,EM+"p28_c1"],"75%","What is the minimum attendance required to appear in the end semester examination?","अंतिम सेमेस्टर परीक्षा में बैठने के लिए न्यूनतम उपस्थिति कितनी आवश्यक है?","End semester exam mein baithne ke liye minimum attendance kitni chahiye?"),
("A02","test",[EM+"p28_c1"],"X grade","Which grade is awarded to a student with less than 75% attendance in a course?","किसी कोर्स में 75% से कम उपस्थिति होने पर छात्र को कौन सा ग्रेड दिया जाता है?","Kisi course mein 75% se kam attendance ho toh kaunsa grade milta hai?"),
("A03","test",[N],"AY 2022-23","From which academic year does the 75% attendance rule for the end semester examination apply?","अंतिम सेमेस्टर परीक्षा के लिए 75% उपस्थिति का नियम किस शैक्षणिक वर्ष से लागू है?","End semester exam ke liye 75% attendance ka rule kis academic year se lagu hai?"),
("A04","dev",[N],"08t September, 2022","On what date was the minimum attendance notification issued?","न्यूनतम उपस्थिति की अधिसूचना किस तारीख को जारी की गई थी?","Minimum attendance ka notification kis date ko jaari hua tha?"),
("A05","test",[SC+"p7_c2"],"6.5 CGPA","What is the minimum CGPA required to continue a sports scholarship?","स्पोर्ट्स स्कॉलरशिप जारी रखने के लिए न्यूनतम CGPA कितना होना चाहिए?","Sports scholarship continue karne ke liye minimum CGPA kitna chahiye?"),
("A06","test",[SC+"p10_c1"],"8.5","What minimum CGPA is required to continue the merit scholarship for B.Tech AI from the second year?","दूसरे वर्ष से बी.टेक एआई की मेरिट स्कॉलरशिप जारी रखने के लिए न्यूनतम CGPA कितना है?","B.Tech AI ki merit scholarship second year se continue karne ke liye minimum CGPA kitna hai?"),
("A07","dev",[SC+"p10_c1"],"10% of Tuition Fees","What percentage of tuition fees is the Academic Scholarship?","एकेडमिक स्कॉलरशिप ट्यूशन फीस का कितना प्रतिशत होती है?","Academic scholarship tuition fees ka kitna percent hoti hai?"),
("A08","test",[SC+"p10_c1"],"not eligible","Is a student who already has an admission scholarship eligible for the Academic Scholarship?","क्या पहले से एडमिशन स्कॉलरशिप पाने वाला छात्र एकेडमिक स्कॉलरशिप के लिए पात्र है?","Kya jis student ke paas pehle se admission scholarship hai woh academic scholarship ke liye eligible hai?"),
("A09","test",[EM+"p23_c2"],"5 days","Within how many days of the last date of the examination must a make-up examination request be submitted?","परीक्षा की अंतिम तिथि के कितने दिनों के भीतर मेक-अप परीक्षा का अनुरोध जमा करना होता है?","Make-up exam ka request exam ki last date ke kitne din ke andar dena hota hai?"),
("A10","dev",[EM+"p23_c2"],"Vice Chancellor","Who gives the final approval for a make-up examination?","मेक-अप परीक्षा को अंतिम मंज़ूरी कौन देता है?","Make-up exam ko final approval kaun deta hai?"),
("A11","test",[EM+"p23_c3",ORD+"p19_c1"],"no further chance","What happens if a student misses the make-up examination?","यदि कोई छात्र मेक-अप परीक्षा भी छोड़ देता है तो क्या होता है?","Agar student make-up exam bhi miss kar de toh kya hota hai?"),
("A12","test",[EM+"p23_c2",EM+"p23_c3"],"between Mid Term and End Semester Examination","When is the mid-term make-up examination held?","मिड टर्म की मेक-अप परीक्षा कब आयोजित की जाती है?","Mid term ka make-up exam kab hota hai?"),
("A13","dev",[ORD+"p16_c3"],"one credit","How many credits is one lecture hour per week per semester worth?","प्रति सप्ताह एक लेक्चर घंटे के लिए कितने क्रेडिट दिए जाते हैं?","Ek lecture hour per week ke liye kitne credit milte hain?"),
("A14","test",[ORD+"p16_c3"],"half credit","How many credits is one laboratory hour per week assigned?","प्रति सप्ताह एक प्रयोगशाला घंटे के लिए कितने क्रेडिट मिलते हैं?","Ek lab hour per week ko kitne credit milte hain?"),
("A15","test",[ORD+"p10_c1"],"four years","What is the minimum duration of the B.Tech programme?","बी.टेक कार्यक्रम की न्यूनतम अवधि कितनी है?","B.Tech programme ki minimum duration kitni hoti hai?"),
("A16","dev",[ORD+"p10_c1"],"six years","Up to how many years can the B.Tech duration be extended?","बी.टेक की अवधि अधिकतम कितने वर्षों तक बढ़ाई जा सकती है?","B.Tech ki duration maximum kitne saal tak badhayi ja sakti hai?"),
("A17","test",[EM+"p13_c4",EM+"p13_c3"],"calculators","Which electronic device is allowed inside the examination hall?","परीक्षा कक्ष के अंदर कौन सा इलेक्ट्रॉनिक उपकरण ले जाने की अनुमति है?","Exam hall ke andar kaunsa electronic device le ja sakte hain?"),
("A18","dev",[ORD+"p25_c2",ORD+"p25_c3"],"above 75%","What attendance is required for the NP audit grade?","ऑडिट कोर्स में NP ग्रेड के लिए कितनी उपस्थिति आवश्यक है?","Audit course mein NP grade ke liye kitni attendance chahiye?"),
("A19","test",[ORD+"p25_c2"],"first two weeks of a semester","During which period can students apply for an audit course?","ऑडिट कोर्स के लिए आवेदन कब किया जा सकता है?","Audit course ke liye apply kab kar sakte hain?"),
("A20","dev",[SC+"p7_c3"],"every year","How often must a student apply to continue the sports scholarship?","स्पोर्ट्स स्कॉलरशिप जारी रखने के लिए छात्र को कितनी बार आवेदन करना होता है?","Sports scholarship continue karne ke liye student ko kitni baar apply karna padta hai?"),
("A21","test",[EM+"p5_c1"],"Vice-Chancellor","Who appoints the Controller of Examination?","परीक्षा नियंत्रक की नियुक्ति कौन करता है?","Controller of Examination ko kaun appoint karta hai?"),
]
# (fact_id, split, absent-term regexes, English, Hindi, Hinglish)
U = [
("U01","dev",[r"\bMBBS\b",r"medical college",r"NEET cut.?off"],"What is the MBBS course fee and admission process at Bennett University?","बेनेट यूनिवर्सिटी में एमबीबीएस कोर्स की फीस और प्रवेश प्रक्रिया क्या है?","Bennett University mein MBBS course ki fee aur admission process kya hai?"),
("U02","test",[r"gold medal"],"Who is eligible for the gold medal at convocation?","दीक्षांत समारोह में गोल्ड मेडल के लिए कौन पात्र है?","Convocation mein gold medal ke liye kaun eligible hai?"),
("U03","test",[r"\bNIRF\b",r"\branking"],"What is the NIRF ranking of the university?","विश्वविद्यालय की NIRF रैंकिंग क्या है?","University ki NIRF ranking kya hai?"),
("U04","dev",[r"\bduplicate"],"What is the fee for a duplicate degree certificate?","डुप्लिकेट डिग्री सर्टिफिकेट की फीस कितनी है?","Duplicate degree certificate ki fee kitni hai?"),
("U05","test",[r"yoga (?:class|timing|schedule)"],"What are the timings of the daily yoga classes?","दैनिक योग कक्षाओं का समय क्या है?","Daily yoga classes ka timing kya hai?"),
("U06","test",[r"pilot license",r"flight training",r"\baviation\b"],"Does the university offer a flight training programme for commercial pilot license?","क्या विश्वविद्यालय कमर्शियल पायलट लाइसेंस के लिए उड़ान प्रशिक्षण कार्यक्रम प्रदान करता है?","Kya university commercial pilot license ke liye flight training program provide karti hai?"),
("U07","dev",[r"alumni meet",r"alumni reunion"],"When is the annual alumni meet held each year?","वार्षिक पूर्व छात्र सम्मेलन (एलुमनाई मीट) प्रत्येक वर्ष कब आयोजित होता है?","Annual alumni meet har saal kab conduct hoti hai?"),
("U08","test",[r"\bcr[eè]che"],"Is there a creche facility for the children of staff?","क्या कर्मचारियों के बच्चों के लिए क्रेच की सुविधा है?","Kya staff ke bachchon ke liye creche ki suvidha hai?"),
("U09","test",[r"\btaxi",r"\bcab\b"],"How can I book a taxi from the campus?","कैंपस से टैक्सी कैसे बुक करें?","Campus se taxi kaise book karein?"),
("U10","dev",[r"\bnationalit",r"international student"],"Is there a seat quota for international students based on nationality?","क्या राष्ट्रीयता के आधार पर अंतरराष्ट्रीय छात्रों के लिए सीट कोटा है?","Kya nationality ke basis par international students ke liye seat quota hai?"),
]


def main():
    rows = []
    for fid, split, gold, ans, *qs in A:
        for lang, q in zip(LANG_CODE, qs):
            rows.append(dict(qid=f"{fid}-{LANG_CODE[lang]}", fact_id=fid, split=split, language=lang, question=q,
                             answerable=True, gold_chunk_ids=gold, answer_text=ans, absent_terms=[],
                             source="AI-drafted", verified=False))
    for fid, split, absent, *qs in U:
        for lang, q in zip(LANG_CODE, qs):
            rows.append(dict(qid=f"{fid}-{LANG_CODE[lang]}", fact_id=fid, split=split, language=lang, question=q,
                             answerable=False, gold_chunk_ids=[], answer_text="", absent_terms=absent,
                             source="AI-drafted", verified=False))
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "questions_for_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "fact_id", "split", "language", "question", "answerable", "gold_chunk_ids", "answer_text",
                    "reviewer_ok(Y/N)", "reviewer_notes"])
        for r in rows:
            w.writerow([r["qid"], r["fact_id"], r["split"], r["language"], r["question"], r["answerable"],
                        ";".join(r["gold_chunk_ids"]), r["answer_text"], "", ""])
    print(len(rows), collections.Counter((r["split"], r["answerable"]) for r in rows))


if __name__ == "__main__":
    main()
