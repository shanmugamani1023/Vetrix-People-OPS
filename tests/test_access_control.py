"""Tests for access control: documents and leave data (spec §7)."""

from app.tools.leave_tools import (
    tool_get_leave_balance,
    tool_list_leave_requests,
    tool_get_team_leave,
)
from app.retrieval.retriever import filter_eligible_docs
from datetime import date


TODAY = date(2026, 10, 13)


# ── Document access ───────────────────────────────────────────────────────────

def test_employee_cannot_see_managers_docs(sample_manifest, employee_in):
    eligible = filter_eligible_docs(sample_manifest, employee_in, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "MGR-HANDBOOK" not in ids


def test_manager_can_see_managers_docs(sample_manifest, manager_in):
    eligible = filter_eligible_docs(sample_manifest, manager_in, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "MGR-HANDBOOK" in ids


def test_hr_can_see_hr_docs(sample_manifest, hr_employee):
    eligible = filter_eligible_docs(sample_manifest, hr_employee, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "HR-ONLY-DOC" in ids


def test_manager_cannot_see_hr_docs(sample_manifest, manager_in):
    eligible = filter_eligible_docs(sample_manifest, manager_in, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "HR-ONLY-DOC" not in ids


def test_employee_cannot_see_hr_docs(sample_manifest, employee_in):
    eligible = filter_eligible_docs(sample_manifest, employee_in, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "HR-ONLY-DOC" not in ids


# ── Leave data access ─────────────────────────────────────────────────────────

def test_employee_can_view_own_balance(db_path, employee_in):
    result = tool_get_leave_balance("E1001", None, employee_in, db_path)
    assert result["status"] == "success"
    assert result["employee_id"] == "E1001"
    assert len(result["balances"]) == 2


def test_employee_cannot_view_other_balance(db_path, employee_in):
    # E1001 cannot view E1002's balance
    result = tool_get_leave_balance("E1002", None, employee_in, db_path)
    assert result["status"] == "error"
    assert result["error"] == "I can't share information about that employee."


def test_manager_can_view_direct_report_balance(db_path, manager_in):
    # E2001 is E1001's manager
    result = tool_get_leave_balance("E1001", None, manager_in, db_path)
    assert result["status"] == "success"


def test_manager_cannot_view_non_report_balance(db_path, manager_in):
    # E1003 is under E2002, not E2001
    result = tool_get_leave_balance("E1003", None, manager_in, db_path)
    assert result["status"] == "error"


def test_hr_can_view_any_employee_balance(db_path, hr_employee):
    for emp_id in ["E1001", "E1002", "E1003", "E2001"]:
        result = tool_get_leave_balance(emp_id, None, hr_employee, db_path)
        assert result["status"] == "success", f"HR should see {emp_id}"


def test_employee_can_view_own_requests(db_path, employee_in):
    result = tool_list_leave_requests("E1001", None, employee_in, db_path)
    assert result["status"] == "success"


def test_employee_cannot_view_other_requests(db_path, employee_in):
    result = tool_list_leave_requests("E1002", None, employee_in, db_path)
    assert result["status"] == "error"


def test_manager_can_view_report_requests(db_path, manager_in):
    result = tool_list_leave_requests("E1001", None, manager_in, db_path)
    assert result["status"] == "success"


def test_get_team_leave_requires_manager(db_path, employee_in):
    result = tool_get_team_leave("2026-10-01", "2026-10-31", employee_in, db_path)
    assert result["status"] == "error"
    assert "manager" in result["error"].lower()


def test_get_team_leave_works_for_manager(db_path, manager_in):
    result = tool_get_team_leave("2026-10-01", "2026-10-31", manager_in, db_path)
    assert result["status"] == "success"


def test_nonexistent_employee_returns_fixed_refusal(db_path, employee_in):
    result = tool_get_leave_balance("E9999", None, employee_in, db_path)
    assert result["status"] == "error"
    # Must give same fixed message whether employee exists or not
    assert result["error"] == "I can't share information about that employee."


def test_nonexistent_employee_hr_returns_fixed_refusal(db_path, hr_employee):
    # HR gets same fixed message for non-existent employees
    result = tool_get_leave_balance("E9999", None, hr_employee, db_path)
    assert result["status"] == "error"
    assert result["error"] == "I can't share information about that employee."
