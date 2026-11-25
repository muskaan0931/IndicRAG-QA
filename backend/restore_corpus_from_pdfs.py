#!/usr/bin/env python3
"""Extract local source PDFs into page-aware data/pages.jsonl and data/chunks.jsonl.
Place the five original PDFs in data/source_pdfs/. Extraction is reproducible but OCR quality
and chunk boundaries must be checked. Uses PyMuPDF (fitz).
"""
import json, re, subprocess, tempfile
from pathlib import Path
try:
 import fitz
except ImportError as e:
 raise SystemExit('Install PyMuPDF first: pip install pymupdf') from e
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data'/'source_pdfs'; OUT=ROOT/'data'
MAP={
 'Exam-Manual.pdf':('exam_manual','examination','unspecified'),
 'Notification-for-Minimum-Attendance-for-Students.pdf':('notification_for_minimum_attendance_for_students','attendance','2022'),
 'Ord-RulesandReg-UG-BTechF.pdf':('ord_rulesandreg_ug_btechf','academic_regulations','2016-17'),
 'Scholarship-Policy-2024-25_Final.pdf':('scholarship_policy_2024_25_final','scholarship','2024-25'),
 'studentHandbookBennett.pdf':('studenthandbookbennett','student_handbook','2019-20'),
 'Bennett-University-SCSET-Brochure-AY-2025.pdf':('bennett_university_scset_brochure_ay_2025','university_brochure','2025'),
}
def chunks(text, size=1100, overlap=160):
 text=re.sub(r'\s+',' ',text).strip()
 if not text:return []
 out=[]; start=0
 while start<len(text):
  end=min(len(text),start+size)
  if end<len(text):
   cut=text.rfind(' ',start,end)
   if cut>start+size//2:end=cut
  out.append(text[start:end].strip())
  if end>=len(text):break
  start=max(end-overlap,start+1)
 return out

def main():
 if not SRC.exists(): SRC.mkdir(parents=True); print('Created',SRC,'— add the five original PDFs, then rerun.'); return
 pages=[]; ch=[]; absent=[]
 for filename,(doc_id,cat,ver) in MAP.items():
  f=SRC/filename
  if not f.exists(): absent.append(filename); continue
  doc=fitz.open(f)
  for pi,p in enumerate(doc,1):
   txt=p.get_text('text') or ''
   method='native'; confidence=None
   if len(txt.strip()) < 30:
    pix=p.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False)
    with tempfile.NamedTemporaryFile(suffix='.png') as im:
     pix.save(im.name)
     try:
      txt=subprocess.run(['tesseract', im.name, 'stdout', '-l', 'eng', '--psm', '6'],capture_output=True,text=True,timeout=90).stdout
      method='ocr'; confidence=None
     except Exception as e:
      txt=''; method='ocr_failed'
   pages.append(dict(doc_id=doc_id,page=pi,text=txt,extraction_method=method,ocr_confidence=confidence))
   for ci,t in enumerate(chunks(txt),1):
    ch.append(dict(chunk_id=f'{doc_id}_p{pi}_c{ci}',doc_id=doc_id,document_name=filename,category=cat,version=ver,language='English',page=pi,text=t))
 for name,rows in [('pages.jsonl',pages),('chunks.jsonl',ch)]:
  with (OUT/name).open('w',encoding='utf8') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(f'Extracted {len(pages)} pages, {len(ch)} chunks.')
 if absent:print('Missing PDFs:',', '.join(absent))
 print('Check extracted text and chunk boundaries before research evaluation.')
if __name__=='__main__':main()
