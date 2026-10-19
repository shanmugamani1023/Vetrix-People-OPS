"""Integration tests for POST /chat (spec §9). Uses mock LLM — no API key needed."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ── App setup ─────────────────────────────────────────────────────────────────

@pytest.fixture
def app_client(db_path, sample_manifest, mock_llm_client, tmp_path):
    """TestClient with ingestion bypassed and agent pre-wired."""
    import app.main as main_module

    # Write minimal manifest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([d.model_dump() for d in sample_manifest]))

    from app.retrieval.retriever import Retriever
    retriever = Retriever("all-MiniLM-L6-v2")
    retriever._ready = True  # No actual chunks needed for mock tests

    from app.orchestration.agent import Agent
    from app.config import Settings
    settings = Settings(
        db_path=db_path,
        manifest_path=str(manifest_path),
        corpus_dir=str(tmp_path),
        app_today="2026-10-13",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        anthropic_api_key="test-key",
    )

    agent = Agent(mock_llm_client, retriever, sample_manifest, settings)

    main_module._health_ready = True
    main_module._agent = agent
    main_module._settings = settings

    with patch("app.main.get_settings", return_value=settings):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client

    # Cleanup
    main_module._health_ready = False
    main_module._agent = None


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_ready(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Auth / identity ───────────────────────────────────────────────────────────

def test_missing_employee_id_returns_401(app_client):
    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "hello"},
    )
    assert resp.status_code == 422  # FastAPI rejects missing required header


def test_unknown_employee_returns_401(app_client):
    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "hello"},
        headers={"X-Employee-Id": "E9999"},
    )
    assert resp.status_code == 401


def test_session_hijack_returns_403(app_client):
    sid = str(uuid.uuid4())
    # First message establishes session for E1001
    app_client.post(
        "/chat",
        json={"session_id": sid, "message": "hi"},
        headers={"X-Employee-Id": "E1001"},
    )
    # Second message with different employee
    resp = app_client.post(
        "/chat",
        json={"session_id": sid, "message": "hi"},
        headers={"X-Employee-Id": "E1002"},
    )
    assert resp.status_code == 403


def test_valid_request_returns_200(app_client):
    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "What is the leave policy?"},
        headers={"X-Employee-Id": "E1001"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "route" in data
    assert data["route"] in ("rag", "tool", "hybrid", "clarify", "no_answer", "refuse")


# ── Response structure ────────────────────────────────────────────────────────

def test_response_has_required_fields(app_client):
    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "hello"},
        headers={"X-Employee-Id": "E1001"},
    )
    data = resp.json()
    assert all(k in data for k in ("request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"))


def test_citations_list(app_client, mock_llm_client):
    from app.llm.client import LLMResponse, ToolUse
    mock_llm_client.chat.return_value = LLMResponse(
        tool_uses=[ToolUse(id="tu_c1", name="final_answer", args={
            "route": "rag",
            "answer": "You have 15 annual days.",
            "citations": [{"doc_id": "POL-LEAVE-GLOBAL-V2", "locator": "Section 3"}],
            "pending_confirmation": False,
        })],
        text="", stop_reason="tool_use", raw=None,
    )

    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "how many leave days do I have?"},
        headers={"X-Employee-Id": "E1001"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "rag"
    for c in data["citations"]:
        assert c["doc_id"] == "POL-LEAVE-GLOBAL-V2"


def test_restricted_citation_filtered(app_client, mock_llm_client):
    """Citations to docs the employee can't read must be filtered out."""
    from app.llm.client import LLMResponse, ToolUse
    mock_llm_client.chat.return_value = LLMResponse(
        tool_uses=[ToolUse(id="tu_r1", name="final_answer", args={
            "route": "rag",
            "answer": "Some answer.",
            "citations": [
                {"doc_id": "HR-ONLY-DOC", "locator": "p. 1"},
                {"doc_id": "POL-LEAVE-GLOBAL-V1", "locator": "Section 1"},
            ],
            "pending_confirmation": False,
        })],
        text="", stop_reason="tool_use", raw=None,
    )

    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "what is the policy?"},
        headers={"X-Employee-Id": "E1001"},
    )
    data = resp.json()
    citation_ids = {c["doc_id"] for c in data["citations"]}
    assert "HR-ONLY-DOC" not in citation_ids


# ── Pending confirmation flow ─────────────────────────────────────────────────

def test_pending_confirmation_flag(app_client, mock_llm_client):
    """apply_leave proposal sets pending_confirmation=True."""
    from app.llm.client import LLMResponse, ToolUse

    call_count = [0]
    def side_effect(**_):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: Claude proposes apply_leave
            return LLMResponse(
                tool_uses=[ToolUse(id="tu_al1", name="apply_leave",
                                   args={"leave_type": "annual", "start_date": "2026-11-03", "end_date": "2026-11-05"})],
                text="", stop_reason="tool_use", raw=None,
            )
        # Second call: Claude confirms proposal
        return LLMResponse(
            tool_uses=[ToolUse(id="tu_fa1", name="final_answer", args={
                "route": "tool",
                "answer": "I propose 3 days annual leave Nov 3-5. Confirm?",
                "citations": [],
                "pending_confirmation": True,
            })],
            text="", stop_reason="tool_use", raw=None,
        )

    mock_llm_client.chat.side_effect = side_effect

    sid = str(uuid.uuid4())
    resp = app_client.post(
        "/chat",
        json={"session_id": sid, "message": "Apply annual leave from Nov 3 to Nov 5"},
        headers={"X-Employee-Id": "E1001"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pending_confirmation"] is True


def test_empty_message_returns_400(app_client):
    resp = app_client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": ""},
        headers={"X-Employee-Id": "E1001"},
    )
    assert resp.status_code == 400
