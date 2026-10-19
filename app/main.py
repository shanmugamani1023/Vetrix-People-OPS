"""
FastAPI application: /health and /chat endpoints.
Ingestion runs at startup; /health returns 503 until complete.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.database import get_employee, get_or_create_session
from app.ingestion.pipeline import run_ingestion
from app.llm.client import build_llm_client
from app.models import ChatRequest, ChatResponse
from app.orchestration.agent import Agent
from app.retrieval.retriever import Retriever

log = structlog.get_logger()

# ── Global app state ──────────────────────────────────────────────────────────

_health_ready = False
_agent: Agent | None = None
_settings = None


def _run_ingestion_background():
    global _health_ready, _agent, _settings
    try:
        settings = get_settings()
        _settings = settings

        import json
        from pathlib import Path
        from app.models import ManifestDoc

        with open(settings.manifest_path, encoding="utf-8") as f:
            manifest_docs = [ManifestDoc(**d) for d in json.load(f)]

        # Ingest documents
        chunks = run_ingestion(settings)

        # Build FAISS index
        retriever = Retriever(settings.embed_model)
        if chunks:
            # Load chunk IDs from DB (they were assigned during ingestion)
            from app.db.database import get_conn
            with get_conn(settings.db_path) as conn:
                rows = conn.execute("SELECT chunk_id, doc_id, locator, text FROM doc_chunks").fetchall()

            from app.models import Chunk
            from app.ingestion.pipeline import count_tokens
            db_chunks = []
            chunk_ids = []
            for row in rows:
                ch = Chunk(
                    doc_id=row["doc_id"],
                    locator=row["locator"],
                    text=row["text"],
                    token_count=count_tokens(row["text"]),
                )
                db_chunks.append(ch)
                chunk_ids.append(row["chunk_id"])

            retriever.build_index(db_chunks, chunk_ids)
        else:
            retriever._ready = True

        # Build LLM client (supports anthropic, openai, groq, google)
        llm_client = build_llm_client(settings)

        _agent = Agent(llm_client, retriever, manifest_docs, settings)
        _health_ready = True
        log.info("service_ready")

    except Exception as exc:
        log.error("ingestion_failed", error=str(exc))
        # Keep _health_ready=False so /health returns 503


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_run_ingestion_background, daemon=True)
    thread.start()
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)


@app.get("/health")
async def health():
    if _health_ready:
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "starting"})


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    x_employee_id: str = Header(..., alias="X-Employee-Id"),
):
    if not _health_ready or _agent is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    settings = _settings or get_settings()
    db_path = settings.db_path

    # Validate employee
    employee = get_employee(db_path, x_employee_id)
    if employee is None:
        raise HTTPException(status_code=401, detail="Employee not found")

    # Session ownership check
    try:
        session, _ = get_or_create_session(body.session_id, x_employee_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session belongs to another employee")

    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        response = _agent.handle_message(
            session_id=body.session_id,
            employee=employee,
            message=body.message.strip(),
            session=session,
            db_path=db_path,
        )
        return response
    except Exception as exc:
        log.error("chat_error", employee_id=x_employee_id, error=str(exc))
        return ChatResponse(
            request_id=str(uuid.uuid4()),
            answer="An internal error occurred. Please try again.",
            route="refuse",
            citations=[],
            tool_calls=[],
            pending_confirmation=False,
        )


@app.exception_handler(400)
async def bad_request_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
