"""Tests for apply_leave and cancel_leave rules (spec §6.2, §6.3)."""

import uuid
from datetime import date
import pytest
from app.tools.leave_tools import validate_apply_leave, validate_cancel_leave, execute_apply_leave
from app.db.database import apply_leave_db, get_leave_balances


TODAY = date(2026, 10, 13)


# ── apply_leave validation ────────────────────────────────────────────────────

def test_apply_leave_ok(db_path, employee_in):
    result = validate_apply_leave("annual", "2026-10-20", "2026-10-22", employee_in, db_path, TODAY)
    assert result["status"] == "ok"
    assert result["working_days"] == 3


def test_apply_leave_invalid_type(db_path, employee_in):
    result = validate_apply_leave("vacation", "2026-10-20", "2026-10-22", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "Invalid leave type" in result["error"]


def test_apply_leave_past_date(db_path, employee_in):
    result = validate_apply_leave("annual", "2026-10-01", "2026-10-05", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "past" in result["error"].lower() or "after today" in result["error"].lower()


def test_apply_leave_end_before_start(db_path, employee_in):
    result = validate_apply_leave("annual", "2026-10-22", "2026-10-20", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "end_date" in result["error"]


def test_apply_leave_year_constraint(db_path, employee_in):
    result = validate_apply_leave("annual", "2027-01-10", "2027-01-12", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "2026" in result["error"]


def test_apply_leave_zero_working_days(db_path, employee_in):
    # Weekend only
    result = validate_apply_leave("annual", "2026-10-17", "2026-10-18", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "no working days" in result["error"].lower()


def test_apply_leave_insufficient_balance(db_path, employee_in):
    # E1001 has 15 annual days; request 20 days
    result = validate_apply_leave("annual", "2026-11-02", "2026-11-27", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "Insufficient balance" in result["error"] or "balance" in result["error"].lower()


def test_apply_leave_overlap_rejected(db_path, employee_in):
    # First apply succeeds
    rid = str(uuid.uuid4())
    apply_leave_db(db_path, rid, "E1001", "annual", "2026-10-20", "2026-10-22", 3)

    # Overlapping request should fail
    result = validate_apply_leave("annual", "2026-10-21", "2026-10-23", employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "overlap" in result["error"].lower()


def test_apply_leave_holiday_excluded_from_working_days(db_path, employee_in):
    # 2026-10-02 Gandhi Jayanti (Friday) for IN
    # Oct 1 (Thu) to Oct 5 (Mon): Oct 2 is holiday → 2 working days
    result = validate_apply_leave("annual", "2026-10-01", "2026-10-05", employee_in, db_path, TODAY)
    # start_date is in the past for TODAY=Oct 13, so this will fail validation for different reason
    # Let's use a future range that includes a holiday
    # 2026-12-23 (Wed) to 2026-12-28 (Mon): Dec 25 holiday → 4 working days (23, 24, 28, and 26 Sat/27 Sun skipped)
    result = validate_apply_leave("annual", "2026-12-23", "2026-12-28", employee_in, db_path, TODAY)
    assert result["status"] == "ok"
    assert result["working_days"] == 3  # 23(Wed), 24(Thu), 25=holiday, 26=Sat, 27=Sun, 28(Mon)


def test_execute_apply_leave_deducts_balance(db_path, employee_in):
    balances_before = get_leave_balances(db_path, "E1001", "annual")
    before = balances_before[0].balance_days

    result = execute_apply_leave("annual", "2026-10-20", "2026-10-22", 3, employee_in, db_path, TODAY)
    assert result["status"] == "success"
    assert result["working_days"] == 3

    balances_after = get_leave_balances(db_path, "E1001", "annual")
    after = balances_after[0].balance_days
    assert after == before - 3


# ── cancel_leave validation ───────────────────────────────────────────────────

def test_cancel_pending_request_ok(db_path, employee_in):
    rid = str(uuid.uuid4())
    apply_leave_db(db_path, rid, "E1001", "annual", "2026-11-10", "2026-11-12", 3)

    result = validate_cancel_leave(rid, employee_in, db_path, TODAY)
    assert result["status"] == "ok"
    assert "summary" in result


def test_cancel_nonexistent_request(db_path, employee_in):
    result = validate_cancel_leave("nonexistent-id", employee_in, db_path, TODAY)
    assert result["status"] == "error"


def test_cancel_other_employee_request(db_path, employee_in, employee_us):
    rid = str(uuid.uuid4())
    apply_leave_db(db_path, rid, "E1002", "annual", "2026-11-10", "2026-11-12", 3)

    # E1001 cannot cancel E1002's request
    result = validate_cancel_leave(rid, employee_in, db_path, TODAY)
    assert result["status"] == "error"


def test_cancel_already_started_request(db_path, employee_in):
    rid = str(uuid.uuid4())
    # Request that started before today
    apply_leave_db(db_path, rid, "E1001", "annual", "2026-10-01", "2026-10-05", 3)

    result = validate_cancel_leave(rid, employee_in, db_path, TODAY)
    assert result["status"] == "error"
    assert "started" in result["error"].lower() or "passed" in result["error"].lower()


def test_cancel_restores_balance(db_path, employee_in):
    rid = str(uuid.uuid4())
    apply_leave_db(db_path, rid, "E1001", "annual", "2026-11-10", "2026-11-12", 3)

    before = get_leave_balances(db_path, "E1001", "annual")[0].balance_days

    from app.tools.leave_tools import execute_cancel_leave
    result = execute_cancel_leave(rid, employee_in, db_path, TODAY)
    assert result["status"] == "success"

    after = get_leave_balances(db_path, "E1001", "annual")[0].balance_days
    assert after == before + 3
