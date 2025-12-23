# Helektrik Study Assistant

A web application for uploading lecture materials (PDF, slides, audio, notes) and generating AI-powered study content including summaries, key terms, practice questions, and resource recommendations using RAG (Retrieval-Augmented Generation).

## Features

- **File Upload**: Support for `.txt`, `.pdf`, `.pptx`, `.mp4`, `.m4a`, `.wav`, `.webm`
- **Live Transcription**: Record audio directly in the browser and transcribe using Whisper.cpp
- **RAG-Powered Generation**: Retrieval-Augmented Generation for context-aware AI responses
- **Study Tools**: Generate summaries, key terms, practice questions, and external resources
- **HTMX Interface**: Responsive UI without JavaScript frameworks

## Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.10+**
2. **Ollama** (for LLM and embeddings)
3. **FFmpeg** (for audio processing)
4. **Whisper.cpp** (for audio transcription)

## Installation

### Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull Required Models

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the models:
ollama pull qwen2.5:7b          # LLM model
ollama pull nomic-embed-text    # Embedding model for RAG
```

### Step 3: Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Step 4: Install Whisper.cpp

```bash
# Clone whisper.cpp into your project directory
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# Build
make

# Download a model
bash ./models/download-ggml-model.sh base.en
```

### Step 5: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## Running the Application

### Step 1: Start Ollama

```bash
ollama serve
```

### Step 2: Start the FastAPI Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Open Your Browser

Open: **http://localhost:8000**

--- 

## Project Structure

```
helektron/
├── main.py                      # Main application (all backend logic)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── sessions.json                # Persistent session storage (auto-created)
├── .gitignore                   # Git ignore file
├── whisper.cpp/                 # Whisper.cpp for audio transcription
│   ├── build/bin/
│   └── models/
├── templates/
│   ├── base.html                # Base HTML template
│   ├── index.html               # Main page template
│   └── fragments/
│       ├── upload_status.html   # File list fragment (HTMX)
│       ├── summary.html         # Summary output fragment
│       ├── keyterms.html        # Key terms output fragment
│       ├── questions.html       # Questions output fragment
│       └── resources.html       # Resources output fragment
├── static/
│   ├── app.js                   # Frontend JavaScript
│   └── styles.css               # CSS styles
└── upload/                      # Upload directories (auto-created)
    ├── audio/                   # Audio files
    ├── docs/                    # Document files
    └── vector_stores/           # RAG vector stores
```

---

## Usage Guide

### Uploading Files

1. Click the **Upload** button and select a file (PDF, TXT, PPTX, or audio)
2. The file will be processed and text will be extracted
3. The content is automatically indexed in the RAG vector store

### Generating Study Materials

After uploading at least one file:

1. Click **Summary** to generate a summary
2. Click **Key Terms** to extract important key terms and definitions
3. Click **Practice Questions** to create study questions
4. Click **External Resources** to get recommended learning resources

### Live Recording

1. Click **🎙 Start Recording** to begin recording audio
2. Click **⏹ Stop Recording** when done
3. The audio will be transcribed using Whisper.cpp
4. View, copy, or download the transcript

### Deleting Files

- Click the 🗑️ button next to any file to remove it
- The RAG vector store is automatically rebuilt after deletion

---

## API Endpoints

### UI Endpoints (HTML/HTMX)

| Method | Endpoint                     | Description          |
|--------|------------------------------|----------------------|
| GET    | `/`                          | Main page            |
| POST   | `/upload`                    | Upload a file        |
| POST   | `/upload_live_audio`         | Upload recorded audio|
| GET    | `/summary/{session_id}`      | Generate summary     |
| GET    | `/keyterms/{session_id}`     | Generate key terms   |
| GET    | `/questions/{session_id}`    | Generate questions   |
| GET    | `/resources/{session_id}`    | Generate resources   |
| DELETE | `/session/{id}/file/{index}` | Delete a file (HTMX) |

### API Endpoints (JSON)

| Method | Endpoint                        | Description          |
|--------|---------------------------------|----------------------|
| GET    | `/api/session/{id}`             | Get session details  |
| PUT    | `/api/session/{id}`             | Update session name  |
| DELETE | `/api/session/{id}`             | Delete entire session|
| DELETE | `/api/session/{id}/file/{index}`| Delete a file        |
| GET    | `/api/transcript/{id}`          | Get latest transcript|
| GET    | `/api/rag/stats/{id}`           | Get RAG statistics   |
| GET    | `/health`                       | Health check         |

---

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Get session details (replace {session_id})
curl http://localhost:8000/api/session/{session_id}

# Get RAG stats
curl http://localhost:8000/api/rag/stats/{session_id}
```

## Knowledge Goals Demonstrated

This project demonstrates all 10 Knowledge Goals (KG1-KG10):

| KG  | Concept                     | Implementation                        |
|-----|-----------------------------|---------------------------------------|
| KG1 | Endpoint Definitions        | 14+ unique URL paths                  |
| KG2 | HTTP Methods & Status Codes | GET, POST, PUT, DELETE + 200, 400, 404|
| KG3 | Endpoint Validation         | File extension, UUID, size validation |
| KG4 | Dependency Injection        | FastAPI `Depends()` pattern           |
| KG5 | Data Model                  | Pydantic schemas                      |
| KG6 | CRUD Operations             | JSON file persistence                 |
| KG7 | API Endpoints               | RESTful JSON responses                |
| KG8 | UI Endpoints & HTMX         | Dynamic HTML fragments                |
| KG9 | User Interaction CRUD       | Upload, View, Delete UI               |
| KG10| Separation of Concerns      | 9 organized code sections             |

---

## Troubleshooting

### "Ollama connection refused"
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"
```bash
# Pull the required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### "Whisper binary not found"
Ensure whisper.cpp is built and located in the project directory:
```bash
cd whisper.cpp && make
```

### "FFmpeg not found"
Install FFmpeg for your operating system (see Installation section).

### "No speech detected"
- Make sure your microphone is working
- Speak clearly and at a reasonable volume
- Check that the audio file is not corrupted
---


## Technologies Used

- **FastAPI** - Modern Python web framework
- **HTMX** - Dynamic HTML without JavaScript complexity
- **Ollama** - Local LLM and embeddings
- **Whisper.cpp** - Fast speech-to-text transcription
- **Pydantic** - Data validation
- **PyMuPDF** - PDF text extraction
- **python-pptx** - PowerPoint text extraction


