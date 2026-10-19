"""Tests for document eligibility and precedence rules (spec §5)."""

from datetime import date
from app.models import Employee, ManifestDoc
from app.retrieval.retriever import filter_eligible_docs, detect_upcoming_changes, precedence_key


TODAY = date(2026, 9, 15)


def _emp(role="employee", region="IN") -> Employee:
    return Employee(employee_id="E9999", full_name="Test", email="t@x.com",
                    phone="", region=region, join_date="2020-01-01", level="L3", role=role)


# ── Eligibility ───────────────────────────────────────────────────────────────

def test_eligible_global_policy(sample_manifest):
    emp = _emp(role="employee", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "POL-LEAVE-GLOBAL-V1" in ids
    assert "POL-LEAVE-GLOBAL-V2" in ids


def test_eligible_excludes_future(sample_manifest):
    emp = _emp(role="employee", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "POL-LEAVE-GLOBAL-V3" not in ids  # effective 2027


def test_eligible_manager_sees_mgr_docs(sample_manifest):
    emp = _emp(role="manager", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "MGR-HANDBOOK" in ids


def test_eligible_employee_not_see_mgr_docs(sample_manifest):
    emp = _emp(role="employee", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "MGR-HANDBOOK" not in ids


def test_eligible_hr_sees_all(sample_manifest):
    emp = _emp(role="hr", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "MGR-HANDBOOK" in ids
    assert "HR-ONLY-DOC" in ids


def test_eligible_employee_not_see_hr_docs(sample_manifest):
    emp = _emp(role="employee", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "HR-ONLY-DOC" not in ids


def test_eligible_region_filter(sample_manifest):
    emp = _emp(role="employee", region="US")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    # IN addendum not for US
    assert "ADD-LEAVE-IN" not in ids
    # GLOBAL docs still eligible
    assert "POL-LEAVE-GLOBAL-V1" in ids


def test_eligible_in_employee_sees_in_addendum(sample_manifest):
    emp = _emp(role="employee", region="IN")
    eligible = filter_eligible_docs(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in eligible}
    assert "ADD-LEAVE-IN" in ids


def test_eligible_reference_date_past(sample_manifest):
    # Using 2025-08-01 as reference date
    emp = _emp(role="employee", region="IN")
    past_date = date(2025, 8, 1)
    eligible = filter_eligible_docs(sample_manifest, emp, past_date)
    ids = {d.doc_id for d in eligible}
    assert "POL-LEAVE-GLOBAL-V1" in ids         # effective 2025-01-01 ✓
    assert "POL-LEAVE-GLOBAL-V2" not in ids     # effective 2026-04-01, not yet
    assert "ADD-LEAVE-IN" in ids                # effective 2025-06-01, before ref date


def test_eligible_reference_date_includes_june(sample_manifest):
    emp = _emp(role="employee", region="IN")
    ref = date(2025, 7, 1)
    eligible = filter_eligible_docs(sample_manifest, emp, ref)
    ids = {d.doc_id for d in eligible}
    assert "ADD-LEAVE-IN" in ids  # effective 2025-06-01


# ── Precedence ────────────────────────────────────────────────────────────────

def test_precedence_policy_over_faq():
    policy = ManifestDoc(doc_id="P", file="p.pdf", title="Policy", doc_type="policy",
                         regions=["GLOBAL"], effective_date="2026-01-01", access_level="all")
    faq = ManifestDoc(doc_id="F", file="f.html", title="FAQ", doc_type="faq",
                      regions=["GLOBAL"], effective_date="2026-06-01", access_level="all")
    # policy wins on authority even if faq is more recent
    assert precedence_key(policy) > precedence_key(faq)


def test_precedence_regional_over_global():
    global_pol = ManifestDoc(doc_id="G", file="g.pdf", title="Global", doc_type="policy",
                              regions=["GLOBAL"], effective_date="2026-04-01", access_level="all")
    regional = ManifestDoc(doc_id="R", file="r.pdf", title="Regional", doc_type="addendum",
                           regions=["IN"], effective_date="2025-06-01", access_level="all")
    # regional wins on scope even if older
    assert precedence_key(regional) > precedence_key(global_pol)


def test_precedence_recency_breaks_tie():
    v1 = ManifestDoc(doc_id="V1", file="v1.pdf", title="V1", doc_type="policy",
                     regions=["GLOBAL"], effective_date="2025-01-01", access_level="all")
    v2 = ManifestDoc(doc_id="V2", file="v2.pdf", title="V2", doc_type="policy",
                     regions=["GLOBAL"], effective_date="2026-04-01", access_level="all")
    assert precedence_key(v2) > precedence_key(v1)


# ── Upcoming changes ──────────────────────────────────────────────────────────

def test_upcoming_detects_future_docs(sample_manifest):
    emp = _emp(role="employee", region="IN")
    upcoming = detect_upcoming_changes(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in upcoming}
    assert "POL-LEAVE-GLOBAL-V3" in ids


def test_upcoming_excludes_current_docs(sample_manifest):
    emp = _emp(role="employee", region="IN")
    upcoming = detect_upcoming_changes(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in upcoming}
    assert "POL-LEAVE-GLOBAL-V1" not in ids
    assert "POL-LEAVE-GLOBAL-V2" not in ids


def test_upcoming_respects_access_level(sample_manifest):
    emp = _emp(role="employee", region="IN")
    upcoming = detect_upcoming_changes(sample_manifest, emp, TODAY)
    ids = {d.doc_id for d in upcoming}
    # HR-only future doc wouldn't appear for employee
    assert "HR-ONLY-DOC" not in ids
