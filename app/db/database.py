"""
Database layer for Veltrix PeopleOps Assistant.

Responsibilities:
- Read employees, leave balances, leave requests from veltrix.db
- Manage the holidays table (created at ingestion)
- Manage in-memory session store
- Personal data sanitisation for logs
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from typing import Generator

import structlog

from app.models import (
    Employee, LeaveBalance, LeaveRequest, SessionState
)

log = structlog.get_logger()

# ── In-memory session store (thread-safe) ────────────────────────────────────
_sessions: dict[str, SessionState] = {}
_session_lock = threading.Lock()


# ── DB connection helper ─────────────────────────────────────────────────────

@contextmanager
def get_conn(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


# ── Schema setup (new tables only — never touch existing ones) ────────────────

def init_schema(db_path: str) -> None:
    with get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS holidays (
                date   TEXT NOT NULL,
                region TEXT NOT NULL,
                name   TEXT NOT NULL,
                PRIMARY KEY (date, region)
            );

            CREATE TABLE IF NOT EXISTS doc_chunks (
                chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    TEXT NOT NULL,
                locator   TEXT NOT NULL,
                text      TEXT NOT NULL
            );
        """)
        conn.commit()
    log.info("db_schema_ready", db_path=db_path)


# ── Personal data helpers ────────────────────────────────────────────────────

def mask_email(email: str) -> str:
    """r***@veltrix.example"""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    return f"{local[0]}***@{domain}"


def sanitise_for_log(record: dict) -> dict:
    """Remove PII fields before logging."""
    return {k: v for k, v in record.items()
            if k not in ("full_name", "email", "phone")}


# ── Employee queries ─────────────────────────────────────────────────────────

def get_employee(db_path: str, employee_id: str) -> Employee | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
    if row is None:
        return None
    return Employee(**dict(row))


def get_direct_reports(db_path: str, manager_id: str) -> list[Employee]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM employees WHERE manager_id = ?", (manager_id,)
        ).fetchall()
    return [Employee(**dict(r)) for r in rows]


# ── Leave balance queries ────────────────────────────────────────────────────

def get_leave_balances(db_path: str, employee_id: str, leave_type: str | None = None) -> list[LeaveBalance]:
    with get_conn(db_path) as conn:
        if leave_type:
            rows = conn.execute(
                "SELECT * FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
                (employee_id, leave_type)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM leave_balances WHERE employee_id = ?",
                (employee_id,)
            ).fetchall()
    return [LeaveBalance(**dict(r)) for r in rows]


# ── Leave request queries ────────────────────────────────────────────────────

def get_leave_requests(
    db_path: str,
    employee_id: str,
    status: str | None = None
) -> list[LeaveRequest]:
    with get_conn(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM leave_requests WHERE employee_id = ? AND status = ? ORDER BY start_date",
                (employee_id, status)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM leave_requests WHERE employee_id = ? ORDER BY start_date",
                (employee_id,)
            ).fetchall()
    return [LeaveRequest(**dict(r)) for r in rows]


def get_team_leave_requests(
    db_path: str,
    manager_id: str,
    start_date: str,
    end_date: str
) -> list[LeaveRequest]:
    """Pending + approved leave for direct reports overlapping date range."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT lr.* FROM leave_requests lr
            JOIN employees e ON e.employee_id = lr.employee_id
            WHERE e.manager_id = ?
              AND lr.status IN ('pending', 'approved')
              AND lr.start_date <= ?
              AND lr.end_date   >= ?
            ORDER BY lr.start_date
            """,
            (manager_id, end_date, start_date)
        ).fetchall()
    return [LeaveRequest(**dict(r)) for r in rows]


def has_overlapping_request(
    db_path: str,
    employee_id: str,
    start_date: str,
    end_date: str
) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM leave_requests
            WHERE employee_id = ?
              AND status IN ('pending', 'approved')
              AND start_date <= ?
              AND end_date   >= ?
            LIMIT 1
            """,
            (employee_id, end_date, start_date)
        ).fetchone()
    return row is not None


# ── Apply / cancel leave (atomic transactions) ────────────────────────────────

def apply_leave_db(
    db_path: str,
    request_id: str,
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    working_days: int
) -> None:
    from datetime import timezone
    created_at = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        with conn:  # BEGIN … COMMIT / ROLLBACK
            conn.execute(
                """
                INSERT INTO leave_requests
                  (request_id, employee_id, leave_type, start_date, end_date,
                   working_days, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (request_id, employee_id, leave_type,
                 start_date, end_date, working_days, created_at)
            )
            conn.execute(
                """
                UPDATE leave_balances
                SET balance_days = balance_days - ?
                WHERE employee_id = ? AND leave_type = ?
                """,
                (working_days, employee_id, leave_type)
            )
    log.info("leave_applied", employee_id=employee_id, request_id=request_id,
             leave_type=leave_type, working_days=working_days)


def cancel_leave_db(db_path: str, request_id: str, employee_id: str, today: date | None = None) -> int:
    """Cancel request and restore balance. Returns working_days restored."""
    effective_today = (today or date.today()).isoformat()
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM leave_requests WHERE request_id = ? AND employee_id = ?",
            (request_id, employee_id)
        ).fetchone()
        if row is None:
            raise ValueError("Request not found")
        req = LeaveRequest(**dict(row))
        if req.status == "cancelled":
            raise ValueError("Request is already cancelled")
        if req.start_date <= effective_today:
            raise ValueError("Cannot cancel a request that has already started or passed")

        with conn:
            conn.execute(
                "UPDATE leave_requests SET status = 'cancelled' WHERE request_id = ?",
                (request_id,)
            )
            conn.execute(
                """
                UPDATE leave_balances
                SET balance_days = balance_days + ?
                WHERE employee_id = ? AND leave_type = ?
                """,
                (req.working_days, employee_id, req.leave_type)
            )
    log.info("leave_cancelled", employee_id=employee_id, request_id=request_id,
             working_days_restored=req.working_days)
    return req.working_days


def get_leave_request_by_id(db_path: str, request_id: str) -> LeaveRequest | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
    if row is None:
        return None
    return LeaveRequest(**dict(row))


# ── Holidays ─────────────────────────────────────────────────────────────────

def insert_holidays(db_path: str, holidays: list[dict]) -> None:
    """holidays: [{date: 'YYYY-MM-DD', region: 'IN', name: '...'}]"""
    with get_conn(db_path) as conn:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO holidays (date, region, name) VALUES (?, ?, ?)",
                [(h["date"], h["region"], h["name"]) for h in holidays]
            )
    log.info("holidays_inserted", count=len(holidays))


def get_holidays(db_path: str, region: str) -> set[date]:
    """Return set of holiday dates for a region (includes GLOBAL if stored)."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT date FROM holidays WHERE region = ?", (region,)
        ).fetchall()
    return {date.fromisoformat(r["date"]) for r in rows}


# ── Working day calculator ───────────────────────────────────────────────────

def count_working_days(
    start: date,
    end: date,
    region: str,
    db_path: str
) -> int:
    holidays = get_holidays(db_path, region)
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            count += 1
        from datetime import timedelta
        current += timedelta(days=1)
    return count


# ── Session store ────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str, employee_id: str) -> tuple[SessionState, bool]:
    """
    Returns (session, created).
    Raises PermissionError if session belongs to a different employee.
    """
    with _session_lock:
        if session_id in _sessions:
            session = _sessions[session_id]
            if session.employee_id != employee_id:
                raise PermissionError("Session belongs to another employee")
            return session, False
        session = SessionState(employee_id=employee_id)
        _sessions[session_id] = session
        return session, True


def get_session(session_id: str) -> SessionState | None:
    with _session_lock:
        return _sessions.get(session_id)


def update_session(session_id: str, session: SessionState) -> None:
    with _session_lock:
        _sessions[session_id] = session


def clear_session_pending(session_id: str) -> None:
    with _session_lock:
        if session_id in _sessions:
            _sessions[session_id].pending = None
