"""Tests for the event calendars and event studies.

The expiry tests matter most. A settlement date derived from a calendar rule is
wrong whenever the exchange was shut, and an event study aligned on wrong dates
produces a confident answer to a question nobody asked.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import events as ev

IDX = pd.bdate_range("2024-01-02", periods=600)


def test_third_wednesday_is_the_rule():
    assert ev.third_wednesday(2026, 11) == pd.Timestamp("2026-11-18")
    assert ev.third_wednesday(2026, 2) == pd.Timestamp("2026-02-18")


def test_expiry_is_derived_from_data_not_from_the_rule():
    """February 2026 settled on the 23rd because Lunar New Year shut the 18th.

    The rule gives the 18th. Deriving from the last session each contract
    actually traded gives the 23rd, which is the date that happened.
    """
    from twbt import cache

    try:
        fut = cache.load("futures_daily", fixture=True)
    except FileNotFoundError:
        pytest.skip("futures fixture not present")

    cal = ev.expiry_dates(fut)
    if "202602" not in cal.observed.index:
        pytest.skip("Feb 2026 not in the fixture")
    assert cal.observed["202602"] == pd.Timestamp("2026-02-23")
    assert cal.observed["202602"] != ev.third_wednesday(2026, 2)


def test_unexpired_contracts_are_estimated_not_observed():
    """A contract whose last session is the end of the sample has not expired.

    Treating it as settling on the final day of the data would put a fake event
    on the last session of every backtest.
    """
    from twbt import cache

    try:
        fut = cache.load("futures_daily", fixture=True)
    except FileNotFoundError:
        pytest.skip("futures fixture not present")

    cal = ev.expiry_dates(fut)
    end = fut["date"].max()
    assert (cal.observed < end).all(), "an observed expiry sits on the last session"
    assert len(cal.estimated) > 0, "front contracts should be unresolved"
    assert set(cal.observed.index).isdisjoint(cal.estimated.index)


def test_spread_and_weekly_contracts_are_excluded():
    df = pd.DataFrame({
        "date": [IDX[0]] * 4,
        "product": ["TX"] * 4,
        "session": ["一般"] * 4,
        "expiry": ["202401", "202401/202402", "202401W1", "nonsense"],
    })
    cal = ev.expiry_dates(df)
    assert set(cal.observed.index) | set(cal.estimated.index) <= {"202401"}


def test_days_to_expiry_counts_sessions_not_calendar_days():
    exp = pd.DatetimeIndex([IDX[10], IDX[30]])
    d = ev.days_to_expiry(IDX[:40], exp)
    assert d.iloc[10] == 0 and d.iloc[30] == 0
    assert d.iloc[9] == 1
    assert d.iloc[0] == 10                 # ten sessions, whatever the dates
    assert (d.dropna() >= 0).all()


def test_event_window_drops_events_too_close_to_the_edges():
    """Padding a half-window would silently reweight the tails of the average."""
    r = pd.Series(np.arange(len(IDX), dtype=float), index=IDX)
    w = ev.event_window(r, [IDX[2], IDX[300], IDX[-2]], before=10, after=10)
    assert len(w) == 1                      # only the middle one survives
    assert list(w.columns) == list(range(-10, 11))


def test_caar_is_cumulative_and_reports_n():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 50, len(IDX)), index=IDX)
    out = ev.caar(r, IDX[50:400:20], 5, 5)
    assert out["n"] > 10
    assert len(out["caar"]) == 11
    np.testing.assert_allclose(out["caar"].to_numpy(),
                               out["aar"].cumsum().to_numpy())


def test_caar_on_noise_has_no_significant_t():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 50, len(IDX)), index=IDX)
    out = ev.caar(r, IDX[40:500:11], 10, 10)
    assert abs(out["t"]).max() < 4.0, "noise should not produce a strong t"


def test_missing_event_file_is_empty_not_an_error():
    assert len(ev.load_events("definitely_not_here")) == 0


def test_event_file_round_trips(tmp_path):
    (tmp_path / "x.csv").write_text("date,label\n2024-03-01,A\n2024-06-03,B\n")
    out = ev.load_events("x", directory=tmp_path)
    assert len(out) == 2
    assert list(out["label"]) == ["A", "B"]
    assert out["date"].iloc[0] == pd.Timestamp("2024-03-01")


def test_event_file_without_a_date_column_says_so(tmp_path):
    (tmp_path / "bad.csv").write_text("when,label\n2024-03-01,A\n")
    with pytest.raises(ValueError, match="date"):
        ev.load_events("bad", directory=tmp_path)
