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


# --------------------------------------------------------------------------- #
# MOPS conference filings
# --------------------------------------------------------------------------- #

def test_results_calls_are_separated_from_broker_conferences():
    """The distinction the whole earnings feature rests on.

    A company files both under 法人說明會. One announces figures; the other
    explicitly restates figures already disclosed. TSMC filed 55 over ROC
    112-115 and 16 were results calls, so mixing them would average 16
    announcements with 37 non-events.
    """
    from twbt.sources import mops

    results = ("(1)公布本公司2024年第1季財務報告及2024年第2季業績展望。"
               "(2)參加方式：請參見https://investor.tsmc.com/")
    invited = ("本公司受邀參加摩根大通證券所舉辦之『J.P. Morgan Taiwan CEO-CFO "
               "Conference』，會中就本公司1/18法說會已公開之財務數字、經營績效等"
               "相關資訊進行說明。")
    venue = "台北君悅酒店三樓 凱悅廳一區 (台北市信義區松壽路2號)"

    assert mops.classify(results) == "earnings"
    assert mops.classify(invited) == "conference"
    assert mops.classify(venue) == "other"


def test_a_results_call_hosted_by_a_broker_is_still_a_results_call():
    """Both patterns can match; the figures are new whoever is in the room."""
    from twbt.sources import mops
    both = "本公司受邀參加某券商會議，會中公布本公司2024年第1季財務報告。"
    assert mops.classify(both) == "earnings"


def test_roc_dates_and_ranges_parse():
    from twbt.sources import mops
    assert mops._roc_to_date("113/01/18") == pd.Timestamp("2024-01-18")
    # a multi-day conference takes its first day: by the second it is public
    assert mops._roc_to_date("113/03/05 至 113/03/07") == pd.Timestamp("2024-03-05")
    assert mops._roc_to_date("not a date") is None


def test_to_frame_keeps_only_results_calls_by_default():
    from twbt.sources import mops
    items = [
        mops.Conference(pd.Timestamp("2024-01-18"), "earnings", "公布…財務報告"),
        mops.Conference(pd.Timestamp("2024-02-19"), "conference", "受邀參加…"),
        mops.Conference(pd.Timestamp("2024-03-05"), "other", "台北君悅酒店"),
    ]
    assert len(mops.to_frame(items)) == 1
    assert len(mops.to_frame(items, earnings_only=False)) == 3


def test_the_shipped_tsmc_file_is_quarterly():
    """A sanity check on the fetched file: four calls a year, one per quarter."""
    tbl = ev.load_events("tsmc_earnings")
    if tbl.empty:
        pytest.skip("tsmc_earnings.csv not present")
    assert len(tbl) >= 8
    months = sorted(set(tbl["date"].dt.month))
    assert months == [1, 4, 7, 10], f"expected quarterly cadence, got {months}"
    per_year = tbl.groupby(tbl["date"].dt.year).size()
    assert (per_year[per_year.index < per_year.index.max()] == 4).all()
