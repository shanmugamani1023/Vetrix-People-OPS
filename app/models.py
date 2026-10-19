from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel


# ── Manifest ────────────────────────────────────────────────────────────────

class ManifestDoc(BaseModel):
    doc_id: str
    file: str
    title: str
    doc_type: Literal["policy", "addendum", "memo", "handbook", "calendar", "faq", "wiki"]
    regions: list[str]          # ["GLOBAL"] or ["IN", "US", …]
    effective_date: str         # YYYY-MM-DD
    access_level: Literal["all", "managers", "hr"]


# ── Employee / DB rows ───────────────────────────────────────────────────────

class Employee(BaseModel):
    employee_id: str
    full_name: str
    email: str
    phone: str
    region: str
    join_date: str
    level: str
    role: Literal["employee", "manager", "hr"]
    manager_id: str | None = None


class LeaveBalance(BaseModel):
    employee_id: str
    leave_type: str
    balance_days: float


class LeaveRequest(BaseModel):
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: Literal["pending", "approved", "cancelled"]
    created_at: str


# ── API contract ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class Citation(BaseModel):
    doc_id: str
    locator: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    status: Literal["pending_confirmation", "success", "error"]


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    pending_confirmation: bool = False


# ── Retrieval ────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    doc_id: str
    locator: str        # section heading or "p. N"
    text: str
    token_count: int = 0


# ── Session state ────────────────────────────────────────────────────────────

class PendingAction(BaseModel):
    tool_name: str                  # "apply_leave" | "cancel_leave"
    args: dict[str, Any]
    summary: str                    # human-readable summary shown to user
    working_days: int = 0           # for apply_leave
    validated: bool = True


class SessionState(BaseModel):
    employee_id: str
    history: list[dict] = []        # [{role, content}] for LLM
    pending: PendingAction | None = None
