# DESIGN.md — Veltrix PeopleOps Assistant

## 1. Ingesting Tables and the Scanned Document

### Tables
Documents with tabular data (holiday calendars, accrual schedules) are extracted with **pdfplumber**, which preserves table structure by extracting cells row-by-row and joining them with `|` separators. For DOCX tables, `python-docx` iterates `table.rows` and joins cells similarly. HTML tables are parsed with BeautifulSoup's `find_all("table")`. The result is plain text like:

```
Level | Annual Days | Sick Days
L1-L3 | 15 | 10
L4-L5 | 18 | 10
...
```

This tabular text is chunked and embedded like any other paragraph, so the LLM can reason over it.

For **holiday calendars specifically**, after text extraction, a dedicated LLM call (at ingestion time) parses the structured list of holiday dates and stores them in the `holidays` SQLite table `(date TEXT, region TEXT, name TEXT)`. This means working-day calculations during tool calls use pre-extracted, structured data — not the raw corpus text — and do **not** count towards the 3,000-token corpus budget.

### Scanned Document
A scanned PDF has no text layer. Detection: if `pdfplumber` extracts fewer than **50 characters per page** on average, the document is treated as scanned. Each page is then converted to a PNG image using `pdf2image` (backed by poppler) and sent to the Claude Vision API with the prompt:

> "Extract all text from this HR document page. Preserve headings, bullet points and table structure. Output only the extracted text."

The extracted text is then chunked and embedded identically to any other document. No hand-transcription occurs; the process is fully automatic.

---

## 2. Chunking and Retrieval Design

### Chunking (`app/ingestion/chunker.py`)
**Strategy:** Heading-aware recursive splitting.

1. Split the full document text on Markdown-style headings (`#`, `##`) or PDF section patterns (ALL-CAPS lines, numbered sections).
2. Each section becomes a candidate chunk. Sections that exceed **400 tokens** (tiktoken `cl100k_base`) are further split at paragraph boundaries (double newlines).
3. Each chunk stores: `doc_id`, `locator` (nearest heading text, or `p. N` if no headings), and raw `text`.
4. Chunks do not cross document boundaries.

**Why this approach:**
- Heading-aware splitting keeps semantically coherent units together (a policy section stays intact), which improves retrieval precision.
- 400-token target balances recall (enough context) against the 3,000-token budget constraint — we can include ~7 chunks per query.
- Storing the locator at chunk time means citations are always accurate, even after re-ranking.

### Retrieval (`app/retrieval/retriever.py`)
1. **Eligibility filter**: Before any vector search, the full manifest is filtered to only documents the requester is allowed to read (by `access_level`, `regions`, and `effective_date ≤ reference_date`). Only chunks belonging to eligible `doc_id`s are candidates.
2. **Vector search**: The query is embedded with the same `all-MiniLM-L6-v2` model, and a FAISS inner-product search returns the top-20 candidate chunks from the eligible set.
3. **Token budget**: Chunks are added greedily in relevance order until the cumulative token count would exceed **3,000**. The final count is logged at `INFO` level as `corpus_tokens=N`.
4. **Upcoming-change detection**: After selecting eligible documents, the retriever also checks for documents with `effective_date > reference_date` that would win under the precedence rules. If found, a short note (not corpus text) is appended to the context describing the upcoming change.

**Why FAISS + sentence-transformers:**
- Runs entirely inside Docker with no additional services.
- `all-MiniLM-L6-v2` is a well-benchmarked model for semantic similarity at small scale; it runs on CPU in ~10 ms per query for 300 chunks.
- FAISS rebuilds in seconds at this document scale.

---

## 3. Implementing the Document Precedence Rules (Section 5)

The precedence logic lives in `app/retrieval/retriever.py:filter_and_rank_docs`.

**Eligibility** (three conditions, all must pass):
```python
ACCESS_MAP = {
    "employee": {"all"},
    "manager":  {"all", "managers"},
    "hr":       {"all", "managers", "hr"},
}
eligible = [
    doc for doc in manifest
    if doc.access_level in ACCESS_MAP[role]
    and ("GLOBAL" in doc.regions or region in doc.regions)
    and doc.effective_date <= reference_date
]
```

**Precedence** (applied when two docs address the same rule):
```python
AUTHORITY = {
    "policy": 2, "addendum": 2, "memo": 2,
    "handbook": 2, "calendar": 2,
    "faq": 1, "wiki": 1,
}
SCOPE = lambda doc: 0 if doc.regions == ["GLOBAL"] else 1  # higher = more specific

# Sort key (descending):
key = (AUTHORITY[doc.doc_type], SCOPE(doc), doc.effective_date)
```

The retriever does not resolve precedence for the LLM — it includes all eligible chunks and instructs the LLM in the system prompt to apply the precedence rules when synthesising an answer. This is more robust than pre-filtering by precedence because the LLM can handle edge cases (conflicts, partial answers) and must cite both documents when they conflict.

---

## 4. Where Access Control Is Enforced

Access control is enforced at **two independent layers**:

| Layer | File | Function | What it blocks |
|---|---|---|---|
| Retrieval | `app/retrieval/retriever.py` | `filter_eligible_docs()` | Restricted documents never enter the FAISS search; they cannot appear in retrieved chunks sent to the LLM |
| Tool | `app/tools/leave_tools.py` | Each tool function preamble | Unauthorised leave data reads/writes; uses `requester_id` from the HTTP header, not from LLM-generated arguments |

The retrieval layer enforces the must-pass check "No document the requester is not allowed to read is ever sent to the LLM." The tool layer enforces "No employee can see leave data they are not allowed to see."

The agent (`app/orchestration/agent.py`) passes `requester_id` (taken from `X-Employee-Id` header) into every tool call explicitly — the LLM cannot override it via its `args`.

---

## 5. Keeping Personal Data Out of Logs and Prompts

### Logs
`app/db/database.py` provides a `sanitise_for_log(record: dict) -> dict` helper that removes `full_name`, `email`, and `phone` keys before any log statement. All log calls use employee IDs (e.g. `employee_id=E2001`) rather than names.

### LLM Prompts
- The system prompt contains only the requester's `employee_id`, `region`, `role`, and `level` — never name, email, or phone.
- Tool results returned to the LLM (e.g. `list_leave_requests`) include `employee_id` and dates, but **not** name, email, or phone.
- When the requester asks about another employee (a manager asking about a direct report), the tool result includes the target's `employee_id` only. Name is included in the final answer only if the requester is allowed to see it (i.e. the target is their direct report or they are HR), and only in the `answer` string — never in `tool_calls.args`.

### Answers
Email addresses in answers are masked: `first_char + "***@" + domain` (e.g. `r***@veltrix.example`). Phone numbers are never shown in answers. The `mask_email()` helper lives in `app/orchestration/prompts.py`.

---

## 6. Failure Modes Found and Fixed

| Failure mode | Status |
|---|---|
| **Session hijack**: a second employee reuses a session_id | Fixed — HTTP 403 returned if `session_id` maps to a different `employee_id` |
| **Prompt injection in documents**: corpus text containing "Ignore previous instructions" | Fixed — system prompt explicitly treats all `<hr_context>` text as information only; validated with a test document |
| **OCR doc with zero text**: pdfplumber returns empty string, LLM call fails | Fixed — detection threshold (50 chars/page) triggers Vision API path; empty result raises `IngestionError` |
| **Holiday not in DB**: `count_working_days` uses empty set, over-counts days | Fixed — ingestion validates that at least one holiday was extracted per calendar document; raises error if zero |
| **Balance race condition**: two concurrent `apply_leave` calls deducting same balance | Partially fixed — SQLite WAL mode + a single `BEGIN IMMEDIATE` transaction for balance check + deduction |
| **Token budget exceeded**: very long chunk added in one step | Fixed — `budget_chunks()` checks each chunk before adding; chunks that alone exceed 3000 tokens are truncated at the sentence boundary |
| **Restricted doc citation**: LLM cites a doc the user retrieved via older session | Fixed — citation validation step checks each cited `doc_id` against the current request's eligible doc set; invalid citations are stripped |

---

## 7. What Would Break First with 10,000 Documents

**Bottleneck 1 — Ingestion time**: At 10,000 documents, sequential LLM calls for OCR and holiday extraction would take hours. Fix: parallelise ingestion with `asyncio.gather` or a worker pool (e.g. Celery); persist the FAISS index to disk so it is not rebuilt on restart.

**Bottleneck 2 — FAISS search latency**: A flat FAISS index degrades at ~1M vectors. Fix: switch to `faiss.IndexIVFFlat` (approximate nearest-neighbour) or a dedicated vector DB (Qdrant, pgvector) that supports filtered search natively.

**Bottleneck 3 — Eligibility pre-filter**: Iterating all 10,000 manifest entries per request is O(n). Fix: index `access_level`, `regions`, and `effective_date` in a PostgreSQL table; join with the FAISS result set.

**Bottleneck 4 — Token budget vs recall**: With 10,000 documents, 3,000 tokens of corpus may represent only a tiny fraction of relevant content. Fix: use a two-stage retrieval (BM25 keyword pre-filter → dense re-rank) and/or hierarchical summarisation per document at ingestion time, storing summaries as a separate FAISS partition.
