# Veltrix PeopleOps Assistant

An AI-powered HR assistant for Veltrix Systems. Employees can ask HR policy questions and manage leave through a conversational chat interface.

---

## Features

- **Policy Q&A** — Answers questions from 30 HR documents (PDF, DOCX, HTML, Markdown) using RAG
- **Leave management** — Check balance, apply leave, cancel leave via natural language
- **Role-based access control** — HR docs hidden from employees, manager docs hidden from regular staff
- **Document precedence** — Regional policy overrides global; newer effective date wins
- **Two-step confirmation** — Leave actions require explicit user confirmation before execution
- **Multi-provider LLM** — Supports Anthropic, OpenAI, Groq (free), Google Gemini
- **Scanned PDF support** — Tesseract OCR extracts text from image-based PDFs

---

## Architecture

```
User message (X-Employee-Id header)
        ↓
FastAPI /chat endpoint
        ↓
Session guard (prevents session hijacking)
        ↓
Retriever — FAISS semantic search → filter by role/region/date → 3000 token budget
        ↓
LLM Agent — reads question + document chunks → calls tools as needed
        ↓
Tools: get_leave_balance | list_leave_requests | get_team_leave | apply_leave | cancel_leave
        ↓
Structured response (route, answer, citations, pending_confirmation)
```

**Stack:**
- `FastAPI` + `uvicorn` — REST API
- `sentence-transformers` + `FAISS` — semantic search
- `SQLite` — document chunks, employee data, leave records
- `pdfplumber` + `pytesseract` — document extraction + OCR
- `Groq / Anthropic / OpenAI / Google` — LLM provider (configurable)
- `Streamlit` — chat UI

---

## Quick Start (Local — No Docker)

### Prerequisites

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr -y   # for scanned PDF support
```

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key. The free option is Groq:

```env
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 2. Start the API server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Wait for `/health` to return `{"status": "ok"}` (ingestion runs at startup, ~20 seconds):

```bash
curl http://localhost:8000/health
```

### 3. Start the chat UI (new terminal)

```bash
streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## Quick Start (Docker) — Recommended

Just 3 commands:

```bash
# 1. Copy env file
cp .env.example .env
```

Edit `.env` and set your API key (free option — Groq):
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at **https://console.groq.com**

```bash
# 2. Start everything (API + UI together)
docker compose up

# 3. Open browser
http://localhost:8501
```

Docker handles everything automatically:
- Builds the API server (with Tesseract OCR)
- Builds the Streamlit UI
- Runs ingestion at startup (~20 seconds)
- UI waits for API to be ready before opening

To stop:
```bash
docker compose down
```

---

## API Usage

### Check health
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-Employee-Id: E1001" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "What is my annual leave balance?"}'
```

### Apply leave (two-step)
```bash
# Step 1 — propose
curl -X POST http://localhost:8000/chat \
  -H "X-Employee-Id: E1001" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "Apply annual leave from Nov 10 to Nov 12"}'

# Step 2 — confirm
curl -X POST http://localhost:8000/chat \
  -H "X-Employee-Id: E1001" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "yes"}'
```

---

## Test Employees

| Employee ID | Name | Region | Role |
|---|---|---|---|
| E1001 | Alice Smith | India | Employee |
| E1002 | Bob Jones | USA | Employee |
| E1003 | Clara Muller | Germany | Employee |
| E2001 | Dan Manager | India | Manager |
| E2002 | Eva Manager | Germany | Manager |
| E3001 | HR Hannah | India | HR |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` / `openai` / `groq` / `google` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name for chosen provider |
| `ANTHROPIC_API_KEY` | — | Required if LLM_PROVIDER=anthropic |
| `OPENAI_API_KEY` | — | Required if LLM_PROVIDER=openai |
| `GROQ_API_KEY` | — | Required if LLM_PROVIDER=groq |
| `GOOGLE_API_KEY` | — | Required if LLM_PROVIDER=google |
| `CORPUS_DIR` | `./data/corpus` | Folder with HR documents |
| `MANIFEST_PATH` | `./data/manifest.json` | Document manifest |
| `DB_PATH` | `./data/veltrix.db` | SQLite database |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model (runs locally) |
| `APP_TODAY` | *(system date)* | Override today's date `YYYY-MM-DD` |

---

## Run Tests

```bash
make test
# or
pytest tests/ -v
```

All 70 tests pass. Tests use in-memory SQLite and a mock LLM — **no API key required**.

---

## Project Structure

```
app/
  config.py              — Settings from .env
  main.py                — FastAPI app, startup ingestion
  models.py              — Pydantic data models
  ingestion/
    pipeline.py          — Document ingestion, chunking
    extractors.py        — PDF/DOCX/HTML/MD + Tesseract OCR
  retrieval/
    retriever.py         — FAISS search, access control, token budget
  orchestration/
    agent.py             — LLM agent, tool loop, confirmation flow
  tools/
    leave_tools.py       — Leave balance, apply, cancel tools
  llm/
    client.py            — Unified adapter (Anthropic/OpenAI/Groq/Google)
  db/
    database.py          — SQLite queries

ui/
  app.py                 — Streamlit chat interface

tests/
  conftest.py            — Fixtures (in-memory DB, mock LLM)
  test_access_control.py — 17 tests
  test_chat.py           — 10 tests
  test_leave_rules.py    — 15 tests
  test_precedence.py     — 16 tests
  test_working_days.py   — 12 tests

data/
  corpus/                — 30 HR documents
  manifest.json          — Document metadata (access_level, regions, dates)
  veltrix.db             — SQLite database

research/                — Learning scripts (not part of submission)
```

---

## Key Design Decisions

1. **Tesseract OCR** — Free local OCR for scanned PDFs. Falls back to Claude Vision if available.
2. **final_answer tool** — Forces structured JSON output from any LLM without regex parsing.
3. **3000-token corpus budget** — Controls LLM cost; enforced in retriever, logged per request.
4. **Access control at retriever** — Restricted docs never reach the LLM (not filtered after).
5. **Two-step write confirmation** — apply/cancel require explicit "yes" before DB write.
6. **Unified LLM adapter** — One agent codebase works with all 4 providers.
7. **Session hijack protection** — Same session_id with different employee_id → HTTP 403.
