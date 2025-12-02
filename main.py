from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import subprocess
from datetime import datetime
from typing import Dict, Any

import fitz  # PyMuPDF
from pptx import Presentation
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai

# Ai model 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEM_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ----------------- CONFIG -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")
DOC_DIR = os.path.join(UPLOAD_DIR, "docs")
SESSIONS_PATH = os.path.join(BASE_DIR, "sessions.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

app = FastAPI(title="Helektron Study Assistant")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ----------------- SESSION STORAGE -----------------
def load_sessions() -> Dict[str, Any]:
    if not os.path.exists(SESSIONS_PATH):
        return {}
    with open(SESSIONS_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_sessions(sessions: Dict[str, Any]) -> None:
    with open(SESSIONS_PATH, "w") as f:
        json.dump(sessions, f, indent=2)

sessions: Dict[str, Any] = load_sessions()

def get_or_create_session(session_id: str | None) -> str:
    global sessions
    if session_id and session_id in sessions:
        return session_id
    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "files": [],
    }
    save_sessions(sessions)
    return new_id

def add_text_to_session(session_id: str, filename: str, file_type: str, text: str) -> None:
    global sessions
    if session_id not in sessions:
        session_id = get_or_create_session(None)
    sessions[session_id]["files"].append({
        "name": filename,
        "type": file_type,
        "text": text,
        "added_at": datetime.utcnow().isoformat()
    })
    save_sessions(sessions)

def get_session_text(session_id: str) -> str:
    session = sessions.get(session_id)
    if not session:
        return ""
    blocks = [f"--- {f['name']} ({f['type']}) ---\n{f['text']}" for f in session["files"] if f.get("text")]
    return "\n\n".join(blocks)

# ----------------- HELPERS: FILE TEXT EXTRACTION -----------------
def extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)

def extract_pptx(path: str) -> str:
    prs = Presentation(path)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)

def transcribe_audio(path: str) -> str:
    """Transcribe audio using whisper.cpp"""
    
    candidates = [
        os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "whisper-cli"),
        os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "main"),
        os.path.join(BASE_DIR, "whisper.cpp", "main"),
    ]
    
    whisper_bin = next((c for c in candidates if os.path.exists(c)), None)
    if not whisper_bin:
        raise RuntimeError(f"Whisper binary not found.")

    model_candidates = [
        os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-base.en.bin"),
        os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-base.bin"),
        os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-small.en.bin"),
        os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-tiny.en.bin"),
    ]
    
    model_path = next((m for m in model_candidates if os.path.exists(m)), None)
    if not model_path:
        raise RuntimeError("Whisper model not found.")

    wav_path = path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install with: brew install ffmpeg")

    out_prefix = path + "_out"
    try:
        result = subprocess.run(
            [whisper_bin, "-m", model_path, "-f", wav_path, "-otxt", "-of", out_prefix],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError:
        try:
            result = subprocess.run(
                [whisper_bin, "-m", model_path, "-f", wav_path, "--output-txt", "--output-file", out_prefix],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(f"Whisper transcription failed: {e2.stderr}")

    txt_path = out_prefix + ".txt"
    if not os.path.exists(txt_path):
        raise RuntimeError(f"Transcript file not created.")
    
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        transcript = f.read().strip()

    for p in [wav_path, txt_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

    return transcript if transcript else "[No speech detected]"

# ----------------- GPT PROMPTS -----------------
def get_summary_prompt(transcript: str) -> str:
    return f"""
Based on the following combined study materials (lectures, slides, PDFs, notes, transcripts),
create a **structured, detailed, and academically accurate study summary**.

Organize the output into the following clearly labeled sections:

- **Overview**: Briefly describe the overall topic and goals.
- **Key Concepts Introduced**: List and explain major ideas, theories, formulas, or processes.
- **Detailed Topic Breakdown**: Group related ideas and summarize explanations, reasoning steps, and relationships.
- **Important Definitions**: Provide concise definitions for technical terms or domain-specific vocabulary.
- **Examples or Applications**: Summarize any examples, demonstrations, or real-world applications.
- **Main Takeaways**: What a student should remember after studying this material.

Expectations:
- Use clear, concise bullet points.
- Prioritize conceptual correctness.
- Use simple academic language suitable for undergraduate study.
- If information is unclear or missing, label it as `TBD`.

---BEGIN MATERIAL---
{transcript}
---END MATERIAL---
"""

def get_keyterms_prompt(text: str) -> str:
    return f"""
From the combined study materials below, extract **10–20 key terms** with short definitions.

For each term, include:
- The term
- A 1–2 sentence definition
- (Optional) A quick note on why it matters in this context.

Format as a bulleted list.

---BEGIN MATERIAL---
{text}
---END MATERIAL---
"""

def get_questions_prompt(text: str) -> str:
    return f"""
Create **8–12 practice questions** based on the combined study materials below.

Include a mix of:
- Conceptual understanding questions
- Short-answer questions
- Application/problem-style questions (where possible)

Do NOT provide answers, only questions.
Group them into sections if appropriate (e.g., 'Conceptual', 'Application').

---BEGIN MATERIAL---
{text}
---END MATERIAL---
"""

def get_resources_prompt(text: str) -> str:
    return f"""
Based on the topics and concepts in the combined study materials below,
recommend **3–7 high-quality external resources** for further study.

Each resource should include:
- Title
- Type (e.g., textbook chapter, .edu article, video lecture)
- Source (e.g., university, well-known platform)
- 1–2 sentences on why it is relevant.

Prefer:
- .edu domains
- Reputable textbooks
- Well-known educational channels

You do NOT need to provide URLs, just clear, identifiable references.

---BEGIN MATERIAL---
{text}
---END MATERIAL---
"""

def call_gemini(prompt: str) -> str:
    try:
        response = GEM_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini API Error: {e}]"

# ----------------- ROUTES -----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Get latest transcript for a session
@app.get("/get_latest_transcript/{session_id}")
def get_latest_transcript(session_id: str):
    """Return the latest audio transcription for a session"""
    global sessions
    sessions = load_sessions()  # Reload to get fresh data
    
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"text": None, "error": "Session not found"}, status_code=404)
    
    audio_files = [f for f in session["files"] if f.get("type") == "audio-live"]
    if not audio_files:
        return JSONResponse({"text": None, "error": "No transcriptions found"})
    
    latest = audio_files[-1]
    return JSONResponse({
        "text": latest.get("text", ""),
        "name": latest.get("name", ""),
        "added_at": latest.get("added_at", "")
    })

# Upload any material
@app.post("/upload", response_class=HTMLResponse)
async def upload_material(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
):
    global sessions

    session_id = get_or_create_session(session_id)
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext in ["mp4", "m4a", "wav", "webm"]:
        save_dir = AUDIO_DIR
        kind = "audio"
    else:
        save_dir = DOC_DIR
        kind = "document"

    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.join(save_dir, f"{uuid.uuid4().hex}_{filename}")

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    try:
        if ext == "txt":
            text = extract_txt(saved_path)
        elif ext == "pdf":
            text = extract_pdf(saved_path)
        elif ext in ["pptx"]:
            text = extract_pptx(saved_path)
        elif ext in ["mp4", "m4a", "wav", "webm"]:
            text = transcribe_audio(saved_path)
        else:
            text = f"[Unsupported file type: {ext}]"
    except Exception as e:
        text = f"[Error processing file {filename}: {e}]"

    add_text_to_session(session_id, filename, kind, text)

    session = sessions[session_id]
    return templates.TemplateResponse(
        "fragments/upload_status.html",
        {
            "request": request,
            "session_id": session_id,
            "files": session["files"],
        },
    )

# Live transcription
@app.post("/upload_live_audio", response_class=HTMLResponse)
async def upload_live_audio(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str | None = Form(None),
):
    global sessions
    
    session_id = get_or_create_session(session_id)
    filename = audio.filename or "live_recording.webm"
    ext = filename.split(".")[-1].lower()

    os.makedirs(AUDIO_DIR, exist_ok=True)
    saved_path = os.path.join(AUDIO_DIR, f"live_{uuid.uuid4().hex}.{ext}")

    with open(saved_path, "wb") as f:
        f.write(await audio.read())

    try:
        text = transcribe_audio(saved_path)
    except Exception as e:
        text = f"[Error transcribing live audio: {e}]"

    add_text_to_session(session_id, filename, "audio-live", text)
    
    # Reload sessions to get fresh data
    sessions = load_sessions()
    session = sessions[session_id]

    return templates.TemplateResponse(
        "fragments/upload_status.html",
        {
            "request": request,
            "session_id": session_id,
            "files": session["files"],
        },
    )

# ---- Study tools ----

@app.get("/summary/{session_id}", response_class=HTMLResponse)
def generate_summary(request: Request, session_id: str):
    text = get_session_text(session_id)
    if not text.strip():
        content = "No material uploaded yet."
    else:
        prompt = get_summary_prompt(text)
        content = call_gemini(prompt)

    return templates.TemplateResponse(
        "fragments/summary.html",
        {"request": request, "content": content},
    )

@app.get("/keyterms/{session_id}", response_class=HTMLResponse)
def generate_keyterms(request: Request, session_id: str):
    text = get_session_text(session_id)
    if not text.strip():
        content = "No material uploaded yet."
    else:
        prompt = get_keyterms_prompt(text)
        content = call_gemini(prompt)

    return templates.TemplateResponse(
        "fragments/keyterms.html",
        {"request": request, "content": content},
    )

@app.get("/questions/{session_id}", response_class=HTMLResponse)
def generate_questions_view(request: Request, session_id: str):
    text = get_session_text(session_id)
    if not text.strip():
        content = "No material uploaded yet."
    else:
        prompt = get_questions_prompt(text)
        content = call_gemini(prompt)

    return templates.TemplateResponse(
        "fragments/questions.html",
        {"request": request, "content": content},
    )

@app.get("/resources/{session_id}", response_class=HTMLResponse)
def generate_resources_view(request: Request, session_id: str):
    text = get_session_text(session_id)
    if not text.strip():
        content = "No material uploaded yet."
    else:
        prompt = get_resources_prompt(text)
        content = call_gemini(prompt)

    return templates.TemplateResponse(
        "fragments/resources.html",
        {"request": request, "content": content},
    )