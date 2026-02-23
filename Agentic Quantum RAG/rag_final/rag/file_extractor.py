import csv, json, os, re, tempfile, zipfile
from typing import List, Dict, Any

def _pdf(p):
    try:
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    except: return ""

def _docx(p):
    try:
        import docx
        return "\n".join(par.text for par in docx.Document(p).paragraphs)
    except: return ""

def _txt(p):
    try:
        with open(p,"r",encoding="utf-8",errors="ignore") as f: return f.read()
    except: return ""

def _csv(p):
    try:
        with open(p,newline="",encoding="utf-8",errors="ignore") as f:
            return "\n".join(" | ".join(row) for row in csv.reader(f))
    except: return ""

def _json(p):
    try:
        with open(p,"r",encoding="utf-8") as f: return json.dumps(json.load(f),indent=2)
    except: return ""

def _image(p):
    try:
        import cv2, pytesseract
        return pytesseract.image_to_string(cv2.imread(p))
    except: return ""

def _audio(p):
    try:
        import whisper
        return whisper.load_model("base").transcribe(p)["text"]
    except: return ""

def _zip(p):
    text = ""
    try:
        with zipfile.ZipFile(p,"r") as z:
            with tempfile.TemporaryDirectory() as tmp:
                z.extractall(tmp)
                for root,_,files in os.walk(tmp):
                    for name in files: text += extract_text(os.path.join(root,name))+"\n"
    except: pass
    return text

_EXT = {".pdf":_pdf,".docx":_docx,".txt":_txt,".md":_txt,
        ".csv":_csv,".json":_json,".jpg":_image,".jpeg":_image,
        ".png":_image,".mp3":_audio,".wav":_audio,".mp4":_audio,".zip":_zip}

def extract_text(path: str) -> str:
    fn = _EXT.get(os.path.splitext(path)[1].lower())
    return re.sub(r"\s+"," ",fn(path)).strip() if fn else ""

def load_files(paths: List[str]) -> List[Dict[str,Any]]:
    docs = []
    for p in paths:
        if not os.path.exists(p):
            print("File not found: "+p); continue
        text = extract_text(p)
        if text:
            docs.append({"text":text,"metadata":{"source":os.path.basename(p)}})
    return docs
