# Veltrix Project — Learning Q&A Log

Questions asked while learning how the project was built.

---

## Q1. What is the assignment? What are they expecting?
**Asked:** Session 2

The assignment is to build a Veltrix PeopleOps HR assistant chatbot for an AI/ML Engineer assessment. It requires:
- Ingestion pipeline (read 30 HR docs, chunk, store in DB)
- FAISS semantic search retriever with access control
- LLM agent with tool calling (leave management)
- FastAPI server (/health, /chat endpoints)
- Streamlit UI
- 70 passing tests

Key challenges: access control, document precedence, two-step leave confirmation, 3000-token budget, security.

---

## Q2. How did you build this step by step?
**Asked:** Session 2

8 steps:
1. Read documents (PDF/DOCX/HTML/MD extractors)
2. Make documents searchable with FAISS (semantic vector search)
3. Access control — filter docs by employee role before sending to AI
4. AI Agent — reads question + relevant docs, decides what to do
5. Tools — live DB queries for leave balance, apply, cancel
6. Two-step confirmation — propose → user says yes → execute
7. FastAPI server wrapping everything
8. Streamlit UI calling the API

---

## Q3. Everything is completed?
**Asked:** Session 2

Yes. 70/70 tests pass. Server running on port 8000. UI on port 8501. Using Groq (free) with qwen/qwen3.8-27b model.

---

## Q4. How many employees are there?
**Asked:** Session 2

6 employees: E1001 Alice (IN, employee), E1002 Bob (US, employee), E1003 Clara (DE, employee), E2001 Dan (IN, manager), E2002 Eva (DE, manager), E3001 Hannah (IN, hr).

---

## Q5. Does all data expose to all? Can employee see HR data?
**Asked:** Session 2

No. Access control enforced at 3 layers:
1. Retriever filters docs before LLM sees them (access_level check)
2. Citation filter strips any restricted doc IDs from response
3. Tools enforce employee-level permissions in code, not prompts
Tested: E1001 asking for compensation bands → refused correctly.

---

## Q6. Show me the chunk of one PDF document
**Asked:** Session 2

POL-LEAVE-GLOBAL-V2 (global_leave_policy_v2.pdf) splits into 3 chunks:
- Chunk 1 (482 tokens): Header, purpose, definitions, sections 1-5
- Chunk 2 (116 tokens): Comparison table V1 vs V2
- Chunk 3 (558 tokens): All actual rules (sections 6-12)

---

## Q7. We take one PDF, split into 3 chunks — each chunk will have 400 tokens?
**Asked:** Session 2

No. 400 tokens is the MAXIMUM target, not a fixed size. Actual sizes:
- Chunk 1: 482 tokens (slightly over — tolerated)
- Chunk 2: 116 tokens (small table, no need to fill to 400)
- Chunk 3: 558 tokens (big section, no paragraph split point available)
Splitting happens at headings first, then paragraphs if still too large. If no split point exists, kept as-is.

---

## Q8. How does splitting work? Which method of chunking is best? How many types of chunking exist?
**Asked:** Session 2

*(See full answer below)*

---
