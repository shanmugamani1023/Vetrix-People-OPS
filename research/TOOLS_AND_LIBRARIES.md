# Veltrix Project — Tools & Libraries Reference

---

## PART 1 — Python Libraries Used (Technical Stack)

### 1. Document Reading
| Library | Why |
|---|---|
| `pdfplumber` | Read text + tables from normal PDFs |
| `python-docx` | Read Word (.docx) files |
| `BeautifulSoup + lxml` | Read HTML files — strips tags, gets clean text |
| `pytesseract` | OCR — read text from scanned image PDFs |
| `pdf2image` | Convert PDF pages to images so tesseract can read them |

### 2. Chunking & Token Counting
| Library | Why |
|---|---|
| `re` (Python built-in) | Regex — detect headings to split text at right places |
| `tiktoken` | Count tokens accurately — enforce 400 token max per chunk |

### 3. Embeddings & Search
| Library | Why |
|---|---|
| `sentence-transformers` | Library to run embedding models locally |
| `all-MiniLM-L6-v2` | The actual AI model — converts text → 384 numbers. Free, fast, offline |
| `faiss-cpu` | Facebook's library — stores 384-number vectors in RAM, finds similar ones instantly |

### 4. Database
| Library | Why |
|---|---|
| `SQLite` (built-in) | Store chunks, employees, leave balances, leave requests. No server needed |

### 5. AI / LLM
| Library | Why |
|---|---|
| `groq` | Python client to call Groq API (free, fast) |
| `anthropic` | Python client for Claude API (paid, more powerful) |
| `openai` | Python client for OpenAI API |

### 6. API Server
| Library | Why |
|---|---|
| `FastAPI` | Build /health and /chat REST API endpoints |
| `uvicorn` | Runs the FastAPI server |
| `pydantic` | Validate request/response data types automatically |

### 7. UI
| Library | Why |
|---|---|
| `streamlit` | Chat UI in browser with pure Python. No HTML/CSS/JS needed |
| `requests` | Streamlit calls the FastAPI server using this |

### 8. Logging & Testing
| Library | Why |
|---|---|
| `structlog` | Structured logging as JSON key=value pairs |
| `pytest` | Run all 70 tests |

### 9. Configuration
| Library | Why |
|---|---|
| `pydantic-settings` | Read .env file into typed Python settings |

---

## PART 2 — Agentic Tools (What the AI Can Call)

These are NOT Python libraries. These are functions the AI agent calls
during a conversation to get live data or perform actions.

Think of it like: the AI has 6 "buttons" it can press to get information
or take action. Based on the user's question, it decides which button to press.

---

### Tool 1 — `get_leave_balance`
**What it does:** Check remaining leave days for an employee
**When AI uses it:** User asks "How many leave days do I have?"
**Security:** Employee can only check their OWN balance. Manager can check direct reports.

```
Input : employee_id, leave_type (annual/sick)
Output: { balance_days: 10.0, leave_type: "annual" }
```

---

### Tool 2 — `list_leave_requests`
**What it does:** Show list of past/current leave requests
**When AI uses it:** User asks "Show my leave history" or "What leave did I take?"
**Security:** Can only see own requests

```
Input : employee_id, status (pending/approved/cancelled)
Output: list of leave requests with dates and status
```

---

### Tool 3 — `get_team_leave`
**What it does:** Show all leave of direct reports in a date range
**When AI uses it:** Manager asks "Who is on leave next week?"
**Security:** ONLY available to managers and HR. Regular employees get refused.

```
Input : start_date, end_date
Output: list of team members on leave in that period
```

---

### Tool 4 — `apply_leave`
**What it does:** Submit a new leave request
**When AI uses it:** User says "Apply annual leave from Nov 3 to Nov 5"
**Security:** Two-step — AI proposes first, waits for user to say "yes", then executes

```
Step 1: validate dates, check balance, count working days
Step 2: show proposal → "3 working days, balance: 10→7. Confirm?"
Step 3: user says "yes" → save to DB, deduct balance
```

---

### Tool 5 — `cancel_leave`
**What it does:** Cancel an existing leave request
**When AI uses it:** User says "Cancel my leave request LR-001"
**Security:** Two-step confirmation. Can only cancel own requests. Must be future leave.

```
Step 1: check request exists and belongs to employee
Step 2: show proposal → "Cancel LR-001 (Nov 3-5). Confirm?"
Step 3: user says "yes" → status=cancelled, restore balance
```

---

### Tool 6 — `final_answer`
**What it does:** Forces the AI to return a structured response
**When AI uses it:** Always — as the LAST action in every conversation
**Why we need it:** Without this, AI returns free text. With this, we get structured JSON.

```
Output:
{
  "route": "rag / tool / hybrid / clarify / refuse",
  "answer": "Your response text",
  "citations": [{"doc_id": "...", "locator": "..."}],
  "pending_confirmation": true/false
}
```

---

## How AI decides which tool to call

```
User: "What is my leave balance?"
        ↓
AI thinks: this needs live data → call get_leave_balance
        ↓
get_leave_balance returns: 10 days annual, 5 days sick
        ↓
AI thinks: I have the answer → call final_answer
        ↓
Returns structured response to user

---

User: "What is the maternity leave policy?"
        ↓
AI thinks: this is a policy question → check document chunks
           no tool needed, answer from documents
        ↓
AI calls final_answer directly with citations
        ↓
Returns answer with citation to POL-LEAVE-GLOBAL-V2

---

User: "Apply leave Nov 3-5"
        ↓
AI thinks: write action needed → call apply_leave
        ↓
apply_leave validates: 3 working days, balance ok
        ↓
AI calls final_answer with pending_confirmation=true
        ↓
"3 days will be deducted. Confirm?"
        ↓
User says "yes" → execute → balance updated
```

---

## Difference: Libraries vs Agentic Tools

| | Python Libraries | Agentic Tools |
|---|---|---|
| What | Code packages installed via pip | Functions the AI decides to call |
| Who calls them | Our code calls them | The AI (LLM) calls them |
| Example | `faiss.search()` | `get_leave_balance(E1001)` |
| Purpose | Build the system | Give AI capabilities at runtime |
| Count | ~15 libraries | 6 tools |
