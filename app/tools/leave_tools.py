"""
Leave tools with permission checks enforced in tool code (not in prompts).
Tool functions return dicts; the agent wraps them in ToolCall objects.
"""

from __future__ import annotations

import uuid
from datetime import date

import structlog

from app.db.database import (
    apply_leave_db,
    cancel_leave_db,
    count_working_days,
    get_direct_reports,
    get_employee,
    get_leave_balances,
    get_leave_request_by_id,
    get_leave_requests,
    get_team_leave_requests,
    has_overlapping_request,
)
from app.models import Employee, LeaveRequest

log = structlog.get_logger()

_FIXED_REFUSAL = "I can't share information about that employee."


# ── Permission helpers ────────────────────────────────────────────────────────

def _can_view_employee(requester: Employee, target_id: str, db_path: str) -> bool:
    """Return True if requester may view target's leave data."""
    if requester.employee_id == target_id:
        return True
    if requester.role == "hr":
        return True
    if requester.role == "manager":
        reports = get_direct_reports(db_path, requester.employee_id)
        return any(r.employee_id == target_id for r in reports)
    return False


def _format_request(req: LeaveRequest) -> dict:
    return {
        "request_id": req.request_id,
        "leave_type": req.leave_type,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "working_days": req.working_days,
        "status": req.status,
        "created_at": req.created_at,
    }


# ── Tool: get_leave_balance ───────────────────────────────────────────────────

def tool_get_leave_balance(
    employee_id: str,
    leave_type: str | None,
    requester: Employee,
    db_path: str,
) -> dict:
    if not _can_view_employee(requester, employee_id, db_path):
        log.info("leave_balance_denied", requester_id=requester.employee_id, target_id=employee_id)
        return {"status": "error", "error": _FIXED_REFUSAL}

    # Verify target exists
    target = get_employee(db_path, employee_id)
    if target is None:
        return {"status": "error", "error": _FIXED_REFUSAL}

    balances = get_leave_balances(db_path, employee_id, leave_type)
    return {
        "status": "success",
        "employee_id": employee_id,
        "balances": [{"leave_type": b.leave_type, "balance_days": b.balance_days} for b in balances],
    }


# ── Tool: list_leave_requests ─────────────────────────────────────────────────

def tool_list_leave_requests(
    employee_id: str,
    status: str | None,
    requester: Employee,
    db_path: str,
) -> dict:
    if not _can_view_employee(requester, employee_id, db_path):
        log.info("leave_list_denied", requester_id=requester.employee_id, target_id=employee_id)
        return {"status": "error", "error": _FIXED_REFUSAL}

    target = get_employee(db_path, employee_id)
    if target is None:
        return {"status": "error", "error": _FIXED_REFUSAL}

    requests = get_leave_requests(db_path, employee_id, status)
    return {
        "status": "success",
        "employee_id": employee_id,
        "requests": [_format_request(r) for r in requests],
    }


# ── Tool: get_team_leave ──────────────────────────────────────────────────────

def tool_get_team_leave(
    start_date: str,
    end_date: str,
    requester: Employee,
    db_path: str,
) -> dict:
    if requester.role != "manager":
        return {"status": "error", "error": "Only managers can view team leave."}

    requests = get_team_leave_requests(db_path, requester.employee_id, start_date, end_date)
    return {
        "status": "success",
        "start_date": start_date,
        "end_date": end_date,
        "requests": [_format_request(r) for r in requests],
    }


# ── Tool: validate apply_leave (proposal, no DB write) ───────────────────────

def validate_apply_leave(
    leave_type: str,
    start_date: str,
    end_date: str,
    employee: Employee,
    db_path: str,
    today: date,
) -> dict:
    """
    Validate an apply_leave request against all spec §6.2 rules.
    Returns {"status": "ok", "working_days": N, "summary": "..."} on success,
    or {"status": "error", "error": "reason"} on failure.
    Does NOT write to the database.
    """
    if leave_type not in ("annual", "sick"):
        return {"status": "error", "error": f"Invalid leave type '{leave_type}'. Must be 'annual' or 'sick'."}

    try:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
    except ValueError:
        return {"status": "error", "error": "Invalid date format. Use YYYY-MM-DD."}

    if s.year != 2026 or e.year != 2026:
        return {"status": "error", "error": "Leave dates must be within the 2026 calendar year."}

    if s < today:
        return {"status": "error", "error": f"start_date {start_date} is in the past. Must be on or after today ({today})."}

    if e < s:
        return {"status": "error", "error": "end_date must be on or after start_date."}

    working_days = count_working_days(s, e, employee.region, db_path)
    if working_days == 0:
        return {"status": "error", "error": "The date range contains no working days (all weekends or holidays)."}

    balances = get_leave_balances(db_path, employee.employee_id, leave_type)
    if not balances:
        return {"status": "error", "error": f"No {leave_type} leave balance found."}
    balance = balances[0].balance_days

    if working_days > balance:
        return {
            "status": "error",
            "error": f"Insufficient balance. Requested {working_days} working days but only {balance} {leave_type} days available.",
        }

    if has_overlapping_request(db_path, employee.employee_id, start_date, end_date):
        return {"status": "error", "error": "This date range overlaps with an existing pending or approved leave request."}

    summary = (
        f"{working_days} working day(s) of {leave_type} leave "
        f"from {start_date} to {end_date} "
        f"(current balance: {balance} days)"
    )
    return {"status": "ok", "working_days": working_days, "summary": summary, "balance": balance}


def execute_apply_leave(
    leave_type: str,
    start_date: str,
    end_date: str,
    working_days: int,
    employee: Employee,
    db_path: str,
    today: date,
) -> dict:
    """Execute a previously validated apply_leave. Returns result dict."""
    # Re-validate to guard against race conditions
    check = validate_apply_leave(leave_type, start_date, end_date, employee, db_path, today)
    if check["status"] != "ok":
        return {"status": "error", "error": check["error"]}

    request_id = str(uuid.uuid4())
    apply_leave_db(db_path, request_id, employee.employee_id, leave_type, start_date, end_date, working_days)

    balances = get_leave_balances(db_path, employee.employee_id, leave_type)
    new_balance = balances[0].balance_days if balances else 0

    return {
        "status": "success",
        "request_id": request_id,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "working_days": working_days,
        "new_balance": new_balance,
    }


# ── Tool: validate cancel_leave (proposal, no DB write) ──────────────────────

def validate_cancel_leave(
    request_id: str,
    employee: Employee,
    db_path: str,
    today: date,
) -> dict:
    """
    Validate a cancel_leave request against spec §6.3 rules.
    Returns {"status": "ok", "request": ..., "summary": "..."} or {"status": "error", ...}.
    Does NOT write to the database.
    """
    req = get_leave_request_by_id(db_path, request_id)
    if req is None or req.employee_id != employee.employee_id:
        return {"status": "error", "error": "Leave request not found or does not belong to you."}

    if req.status == "cancelled":
        return {"status": "error", "error": "This request is already cancelled."}

    if req.status not in ("pending", "approved"):
        return {"status": "error", "error": f"Cannot cancel a request with status '{req.status}'."}

    start = date.fromisoformat(req.start_date)
    if start <= today:
        return {"status": "error", "error": "Cannot cancel a leave request that has already started or passed."}

    summary = (
        f"Cancel {req.leave_type} leave from {req.start_date} to {req.end_date} "
        f"({req.working_days} working days, status: {req.status}). "
        f"This will restore {req.working_days} day(s) to your balance."
    )
    return {"status": "ok", "request": _format_request(req), "summary": summary}


def execute_cancel_leave(
    request_id: str,
    employee: Employee,
    db_path: str,
    today: date,
) -> dict:
    """Execute a previously validated cancel_leave."""
    check = validate_cancel_leave(request_id, employee, db_path, today)
    if check["status"] != "ok":
        return {"status": "error", "error": check["error"]}

    req_data = check["request"]
    try:
        days_restored = cancel_leave_db(db_path, request_id, employee.employee_id, today)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    balances = get_leave_balances(db_path, employee.employee_id, req_data["leave_type"])
    new_balance = balances[0].balance_days if balances else 0

    return {
        "status": "success",
        "request_id": request_id,
        "days_restored": days_restored,
        "leave_type": req_data["leave_type"],
        "new_balance": new_balance,
    }
