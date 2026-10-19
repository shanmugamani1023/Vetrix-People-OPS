"""
Shared fixtures for all tests.
Uses in-memory SQLite — no real API key required.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from app.models import Employee, ManifestDoc


# ── In-memory DB helpers ──────────────────────────────────────────────────────

def _make_db(path: str) -> None:
    """Create the full schema (existing + new tables) and seed test data."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            full_name   TEXT NOT NULL,
            email       TEXT NOT NULL,
            phone       TEXT NOT NULL,
            region      TEXT NOT NULL,
            join_date   TEXT NOT NULL,
            level       TEXT NOT NULL,
            role        TEXT NOT NULL,
            manager_id  TEXT REFERENCES employees(employee_id)
        );
        CREATE TABLE IF NOT EXISTS leave_balances (
            employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type   TEXT NOT NULL,
            balance_days REAL NOT NULL,
            PRIMARY KEY (employee_id, leave_type)
        );
        CREATE TABLE IF NOT EXISTS leave_requests (
            request_id   TEXT PRIMARY KEY,
            employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type   TEXT NOT NULL,
            start_date   TEXT NOT NULL,
            end_date     TEXT NOT NULL,
            working_days INTEGER NOT NULL,
            status       TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS holidays (
            date   TEXT NOT NULL,
            region TEXT NOT NULL,
            name   TEXT NOT NULL,
            PRIMARY KEY (date, region)
        );
        CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id   TEXT NOT NULL,
            locator  TEXT NOT NULL,
            text     TEXT NOT NULL
        );
    """)
    # Seed employees — managers/HR first so FK references resolve
    conn.executemany(
        "INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("E2001", "Dan Manager","d@veltrix.example", "+91-8888888888", "IN", "2019-01-01", "L5", "manager",  None),
            ("E2002", "Eva Manager","e@veltrix.example", "+49-30-9876543", "DE", "2018-06-01", "L6", "manager",  None),
            ("E3001", "HR Hannah",  "h@veltrix.example", "+91-7777777777", "IN", "2020-09-01", "L5", "hr",       None),
            ("E1001", "Alice Smith","a@veltrix.example", "+91-9999999999", "IN", "2022-01-01", "L3", "employee", "E2001"),
            ("E1002", "Bob Jones",  "b@veltrix.example", "+1-555-0001",   "US", "2021-06-01", "L4", "employee", "E2001"),
            ("E1003", "Clara Muller","c@veltrix.example","+49-30-1234567","DE", "2020-03-15", "L3", "employee", "E2002"),
        ],
    )
    # Seed leave balances
    conn.executemany(
        "INSERT INTO leave_balances VALUES (?,?,?)",
        [
            ("E1001", "annual", 15.0), ("E1001", "sick", 10.0),
            ("E1002", "annual", 8.0),  ("E1002", "sick", 5.0),
            ("E1003", "annual", 20.0), ("E1003", "sick", 8.0),
            ("E2001", "annual", 12.0), ("E2001", "sick", 10.0),
            ("E2002", "annual", 10.0), ("E2002", "sick", 10.0),
            ("E3001", "annual", 15.0), ("E3001", "sick", 10.0),
        ],
    )
    # Seed some holidays (IN region)
    conn.executemany(
        "INSERT INTO holidays VALUES (?,?,?)",
        [
            ("2026-01-26", "IN", "Republic Day"),
            ("2026-08-15", "IN", "Independence Day"),
            ("2026-10-02", "IN", "Gandhi Jayanti"),
            ("2026-12-25", "IN", "Christmas"),
            ("2026-07-04", "US", "Independence Day"),
            ("2026-11-26", "US", "Thanksgiving"),
            ("2026-12-25", "US", "Christmas"),
            ("2026-10-03", "DE", "German Unity Day"),
            ("2026-12-25", "DE", "Christmas"),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / "test.db")
    _make_db(path)
    return path


# ── Employee fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def employee_in() -> Employee:
    return Employee(employee_id="E1001", full_name="Alice Smith", email="a@veltrix.example",
                    phone="+91-9999", region="IN", join_date="2022-01-01", level="L3",
                    role="employee", manager_id="E2001")


@pytest.fixture
def employee_us() -> Employee:
    return Employee(employee_id="E1002", full_name="Bob Jones", email="b@veltrix.example",
                    phone="+1-555", region="US", join_date="2021-06-01", level="L4",
                    role="employee", manager_id="E2001")


@pytest.fixture
def manager_in() -> Employee:
    return Employee(employee_id="E2001", full_name="Dan Manager", email="d@veltrix.example",
                    phone="+91-8888", region="IN", join_date="2019-01-01", level="L5",
                    role="manager", manager_id=None)


@pytest.fixture
def hr_employee() -> Employee:
    return Employee(employee_id="E3001", full_name="HR Hannah", email="h@veltrix.example",
                    phone="+91-7777", region="IN", join_date="2020-09-01", level="L5",
                    role="hr", manager_id=None)


# ── Manifest docs fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_manifest() -> list[ManifestDoc]:
    return [
        ManifestDoc(doc_id="POL-LEAVE-GLOBAL-V1", file="leave_v1.pdf", title="Global Leave Policy v1",
                    doc_type="policy", regions=["GLOBAL"], effective_date="2025-01-01", access_level="all"),
        ManifestDoc(doc_id="POL-LEAVE-GLOBAL-V2", file="leave_v2.pdf", title="Global Leave Policy v2",
                    doc_type="policy", regions=["GLOBAL"], effective_date="2026-04-01", access_level="all"),
        ManifestDoc(doc_id="ADD-LEAVE-IN", file="leave_in.pdf", title="India Leave Addendum",
                    doc_type="addendum", regions=["IN"], effective_date="2025-06-01", access_level="all"),
        ManifestDoc(doc_id="FAQ-LEAVE", file="leave_faq.html", title="Leave FAQ",
                    doc_type="faq", regions=["GLOBAL"], effective_date="2026-05-01", access_level="all"),
        ManifestDoc(doc_id="MGR-HANDBOOK", file="manager_handbook.pdf", title="Manager Handbook",
                    doc_type="handbook", regions=["GLOBAL"], effective_date="2025-01-01", access_level="managers"),
        ManifestDoc(doc_id="HR-ONLY-DOC", file="hr_doc.pdf", title="HR Compensation Bands",
                    doc_type="policy", regions=["GLOBAL"], effective_date="2025-01-01", access_level="hr"),
        ManifestDoc(doc_id="POL-LEAVE-GLOBAL-V3", file="leave_v3.pdf", title="Global Leave Policy v3",
                    doc_type="policy", regions=["GLOBAL"], effective_date="2027-01-01", access_level="all"),
    ]


# ── Mock LLM client ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """
    Mock that mimics the unified LLMClient adapter interface:
    - client.chat(...) → LLMResponse
    - client.make_assistant_message(raw) → dict
    - client.make_tool_result_message(id, content) → dict
    """
    from app.llm.client import LLMResponse, ToolUse

    def _make_llm_response(route="rag", answer="Test answer", citations=None, tool_uses=None):
        if tool_uses is None:
            tool_uses = [ToolUse(
                id="tu_001",
                name="final_answer",
                args={
                    "route": route,
                    "answer": answer,
                    "citations": citations or [],
                    "pending_confirmation": False,
                },
            )]
        return LLMResponse(tool_uses=tool_uses, text="", stop_reason="tool_use", raw=None)

    client = MagicMock()
    client.chat.return_value = _make_llm_response()
    client.make_assistant_message.return_value = {"role": "assistant", "content": "Test answer"}
    client.make_tool_result_message.return_value = {"role": "user", "content": "tool result"}
    client._make_llm_response = _make_llm_response
    return client
