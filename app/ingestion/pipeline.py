"""
Ingestion pipeline: load manifest, extract text, chunk, extract holidays.
Builds chunks for the retriever to embed and index.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
import tiktoken

from app.config import Settings, get_settings
from app.db.database import get_conn, init_schema, insert_holidays
from app.ingestion.extractors import extract_text
from app.llm.client import build_llm_client
from app.models import Chunk, ManifestDoc

log = structlog.get_logger()

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 400


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ── Chunking ──────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(#{1,6}\s+\S.*|"          # ## Markdown heading
    r"\d+(?:\.\d+)*\s+[A-Z]\S.*|"  # 1.2 Numbered heading
    r"[A-Z][A-Z &/-]{3,}:?\s*)$",  # ALL-CAPS heading
    re.MULTILINE,
)


def _split_by_paragraphs(doc_id: str, text: str, locator: str) -> list[Chunk]:
    """Group paragraphs into chunks ≤ MAX_CHUNK_TOKENS."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0

    for para in paras:
        pt = count_tokens(para)
        if buf and buf_tokens + pt > MAX_CHUNK_TOKENS:
            body = "\n\n".join(buf)
            chunks.append(Chunk(doc_id=doc_id, locator=locator, text=body, token_count=count_tokens(body)))
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += pt

    if buf:
        body = "\n\n".join(buf)
        chunks.append(Chunk(doc_id=doc_id, locator=locator, text=body, token_count=count_tokens(body)))

    return chunks or [Chunk(doc_id=doc_id, locator=locator, text=text[:2000], token_count=count_tokens(text[:2000]))]


def chunk_text(doc_id: str, text: str) -> list[Chunk]:
    """Heading-aware recursive chunking."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return _split_by_paragraphs(doc_id, text, "p. 1")

    chunks: list[Chunk] = []

    # Text before first heading
    pre = text[: matches[0].start()].strip()
    if pre:
        chunks.extend(_split_by_paragraphs(doc_id, pre, "p. 1"))

    for i, m in enumerate(matches):
        heading = m.group().strip()[:80]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if not section:
            continue
        if count_tokens(section) <= MAX_CHUNK_TOKENS:
            chunks.append(Chunk(doc_id=doc_id, locator=heading, text=section, token_count=count_tokens(section)))
        else:
            chunks.extend(_split_by_paragraphs(doc_id, section, heading))

    return chunks or [Chunk(doc_id=doc_id, locator="p. 1", text=text[:2000], token_count=count_tokens(text[:2000]))]


# ── Holiday extraction ────────────────────────────────────────────────────────

def _extract_holidays_llm(text: str, region: str, llm_client: Any) -> list[dict]:
    """Ask Claude Haiku to extract holiday dates from calendar text."""
    prompt = (
        f"Extract all public holiday dates from the HR calendar document below for region '{region}'.\n"
        "Return a JSON array only, with objects: {\"date\": \"YYYY-MM-DD\", \"region\": \"2-letter code\", \"name\": \"holiday name\"}.\n"
        "Only include 2025 and 2026 dates. Use the region code provided if not specified in the document.\n"
        "Output only valid JSON, nothing else.\n\n"
        f"{text[:4000]}"
    )
    resp = llm_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        rows = json.loads(m.group())
        return [
            {"date": r["date"], "region": r["region"], "name": r["name"]}
            for r in rows
            if isinstance(r, dict) and all(k in r for k in ("date", "region", "name"))
        ]
    except (json.JSONDecodeError, KeyError):
        log.warning("holiday_json_parse_failed", region=region)
        return []


# ── Main ingestion ────────────────────────────────────────────────────────────

def run_ingestion(settings: Settings | None = None) -> list[Chunk]:
    """
    Full ingestion pipeline:
    1. Init DB schema
    2. For each document: extract text, chunk, store chunks
    3. For calendar docs: extract holidays and store
    Returns list of all chunks (for FAISS building).
    """
    if settings is None:
        settings = get_settings()

    db_path = settings.db_path
    corpus_dir = Path(settings.corpus_dir)
    manifest_path = Path(settings.manifest_path)

    init_schema(db_path)

    with open(manifest_path, encoding="utf-8") as f:
        docs = [ManifestDoc(**d) for d in json.load(f)]

    # Build LLM client for OCR + holiday extraction
    llm_client = build_llm_client(settings)

    # Clear old chunks
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM doc_chunks")
        conn.commit()

    all_chunks: list[Chunk] = []

    for doc in docs:
        file_path = corpus_dir / doc.file
        if not file_path.exists():
            log.warning("corpus_file_missing", doc_id=doc.doc_id, file=doc.file)
            continue

        log.info("ingesting_doc", doc_id=doc.doc_id, file=doc.file)

        try:
            text = extract_text(str(file_path), llm_client)
        except Exception as exc:
            log.error("extract_failed", doc_id=doc.doc_id, error=str(exc))
            continue

        if not text.strip():
            log.warning("empty_doc", doc_id=doc.doc_id)
            continue

        # Holiday extraction for calendar docs
        if doc.doc_type == "calendar" and llm_client is not None:
            regions = [r for r in doc.regions if r != "GLOBAL"] or ["IN", "US", "DE"]
            for reg in regions:
                try:
                    holidays = _extract_holidays_llm(text, reg, llm_client)
                    if holidays:
                        insert_holidays(db_path, holidays)
                        log.info("holidays_stored", doc_id=doc.doc_id, region=reg, count=len(holidays))
                    else:
                        log.warning("no_holidays_found", doc_id=doc.doc_id, region=reg)
                except Exception as exc:
                    log.error("holiday_extraction_error", doc_id=doc.doc_id, error=str(exc))

        # Chunk
        chunks = chunk_text(doc.doc_id, text)
        log.info("chunks_created", doc_id=doc.doc_id, count=len(chunks))

        # Store in DB
        with get_conn(db_path) as conn:
            with conn:
                for ch in chunks:
                    conn.execute(
                        "INSERT INTO doc_chunks (doc_id, locator, text) VALUES (?, ?, ?)",
                        (ch.doc_id, ch.locator, ch.text),
                    )

        all_chunks.extend(chunks)

    log.info("ingestion_complete", total_chunks=len(all_chunks))
    return all_chunks


