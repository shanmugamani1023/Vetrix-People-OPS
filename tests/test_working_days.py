"""Tests for working-day calculation (spec §6.2)."""

from datetime import date
import pytest
from app.db.database import count_working_days, insert_holidays


def test_simple_week(db_path):
    # Mon 2026-10-12 to Fri 2026-10-16 = 5 working days
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 16), "IN", db_path) == 5


def test_weekend_excluded(db_path):
    # Sat 2026-10-17 to Sun 2026-10-18 = 0 working days
    assert count_working_days(date(2026, 10, 17), date(2026, 10, 18), "IN", db_path) == 0


def test_week_spanning_weekend(db_path):
    # Mon 2026-10-12 to Mon 2026-10-19 = 6 working days (12-16 + 19)
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 19), "IN", db_path) == 6


def test_holiday_excluded(db_path):
    # 2026-08-15 is Independence Day in IN (seeded in conftest)
    # Fri 2026-08-14 to Mon 2026-08-17: 14(Fri) + 17(Mon) = 2 days, holiday 15 is Sat anyway
    # Let's use Gandhi Jayanti 2026-10-02 (Friday)
    # Thu 2026-10-01 to Mon 2026-10-05: 01(Thu), 02(Fri=holiday), 05(Mon) = 2 working days
    assert count_working_days(date(2026, 10, 1), date(2026, 10, 5), "IN", db_path) == 2


def test_holiday_not_applied_to_other_region(db_path):
    # 2026-10-02 Gandhi Jayanti is only IN, not US
    # Thu 2026-10-01 to Mon 2026-10-05 for US: 01, 02, 05 = 3 working days
    assert count_working_days(date(2026, 10, 1), date(2026, 10, 5), "US", db_path) == 3


def test_single_day_holiday(db_path):
    # 2026-01-26 Republic Day (Monday) = 0 working days for IN
    assert count_working_days(date(2026, 1, 26), date(2026, 1, 26), "IN", db_path) == 0


def test_single_day_working(db_path):
    # 2026-10-12 Monday, not a holiday
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 12), "IN", db_path) == 1


def test_same_start_end_weekend(db_path):
    assert count_working_days(date(2026, 10, 17), date(2026, 10, 17), "IN", db_path) == 0


def test_long_range(db_path):
    # Full month Oct 2026 for US
    # Oct 2026: 22 weekdays. 2026-10-03 is German Unity Day not US holiday.
    # US holidays seeded: Jul 4, Nov 26, Dec 25 — none in Oct
    working = count_working_days(date(2026, 10, 1), date(2026, 10, 31), "US", db_path)
    assert working == 22


def test_us_thanksgiving_excluded(db_path):
    # 2026-11-26 is Thanksgiving (Thursday) for US
    # Nov 25 (Wed) to Nov 27 (Fri): 25, 27 = 2 days (26 is holiday)
    assert count_working_days(date(2026, 11, 25), date(2026, 11, 27), "US", db_path) == 2


def test_cross_month_range(db_path):
    # Oct 29 (Thu) to Nov 2 (Mon) IN: 29(Thu), 30(Fri), 2(Mon) = 3 days (no holidays)
    assert count_working_days(date(2026, 10, 29), date(2026, 11, 2), "IN", db_path) == 3


def test_insert_and_count_custom_holiday(db_path):
    insert_holidays(db_path, [{"date": "2026-10-14", "region": "IN", "name": "Test Holiday"}])
    # Wed 2026-10-14 is now a holiday
    # Mon Oct 12 to Fri Oct 16: 12, 13, 15, 16 = 4 (not 5)
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 16), "IN", db_path) == 4
