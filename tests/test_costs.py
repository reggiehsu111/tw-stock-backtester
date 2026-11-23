"""Tests for the cost convention.

The first test is the one that matters: a round trip must cost ONE spread, not two.
That single factor flipped a real research conclusion, and it is invisible in a diff.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import costs


def test_round_trip_charges_exactly_one_spread():
    """The headline convention. Fees aside, cost must equal the full width once."""
    m = costs.CostModel("TXF", spread_pts=4.0, fees_pts_per_side=0.0)
    assert m.round_trip_pts() == pytest.approx(4.0)       # not 8.0

    # and the halving happens once no matter how the model was built
    m2 = costs.CostModel("TXF", spread_pts=4.0, fees_pts_per_side=0.0).at_session("mid", "mid")
    assert m2.round_trip_pts() == pytest.approx(4.0)


def test_fees_are_charged_per_side():
    m = costs.CostModel("TXF", spread_pts=0.0, fees_pts_per_side=0.5)
    assert m.round_trip_pts() == pytest.approx(1.0)       # two sides


def test_session_widening_applies_per_leg_not_to_the_total():
    """Entering at the open and exiting mid-session costs 1.5 and 1.0 half-spreads."""
    m = costs.CostModel("TXF", spread_pts=4.0, fees_pts_per_side=0.0)
    wide = m.at_session(entry="open", exit="mid")
    # 4.0 * 1.5 / 2  +  4.0 * 1.0 / 2  =  3.0 + 2.0
    assert wide.round_trip_pts() == pytest.approx(5.0)
    assert wide.round_trip_pts() > m.round_trip_pts()


def test_bps_cost_falls_as_the_index_rises():
    """A fixed point cost is a shrinking percentage; the model must reflect that."""
    m = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)
    assert m.round_trip_bps(15_000) == pytest.approx(2.0)
    assert m.round_trip_bps(27_000) == pytest.approx(3.0 / 27_000 * 1e4)
    assert m.round_trip_bps(27_000) < m.round_trip_bps(15_000)


def test_bps_cost_accepts_a_price_series_and_handles_bad_prices():
    m = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)
    px = pd.Series([15_000.0, 0.0, -1.0, 20_000.0])
    out = m.round_trip_bps(px)
    assert out.iloc[0] == pytest.approx(2.0)
    assert out.iloc[1:3].isna().all()          # no division by zero, no negative cost
    assert out.iloc[3] > 0


def test_smaller_contracts_cost_more_in_bps():
    """Smaller contracts must never look cheaper than the one that was measured.

    A spread in index points is the same bps for all three -- bps = points / price,
    and contract size does not enter. The ordering therefore comes from fees, which
    are a fixed NT$ amount over a smaller point value. If a default ever made the
    micro look cheapest, that would be an artefact of a guess flattering the
    contracts nobody measured.
    """
    px = 23_000
    big = costs.CostModel("TXF").round_trip_bps(px)
    mini = costs.CostModel("MXF").round_trip_bps(px)
    micro = costs.CostModel("TMF").round_trip_bps(px)
    assert big < mini < micro
    assert all(v > 0 for v in (big, mini, micro))


def test_spread_in_points_is_contract_independent_in_bps():
    """The same point width must give the same bps on every contract."""
    a = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)
    b = costs.CostModel("TMF", spread_pts=3.0, fees_pts_per_side=0.0)
    assert a.round_trip_bps(20_000) == pytest.approx(b.round_trip_bps(20_000))


def test_scaled_turns_only_the_spread_knob():
    m = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.5)
    s = m.scaled(2.0)
    # spread doubles (3 -> 6), fees do not (1.0 either way)
    assert m.round_trip_pts() == pytest.approx(4.0)
    assert s.round_trip_pts() == pytest.approx(7.0)


def test_unknown_contract_is_rejected_at_construction():
    with pytest.raises(ValueError):
        costs.CostModel("NOPE")


def test_unknown_session_is_rejected():
    with pytest.raises(ValueError):
        costs.CostModel("TXF").at_session(entry="lunchtime")


def test_sensitivity_is_monotone_and_finds_the_crossing():
    """A 3 bps gross edge must survive a small cost and die under a large one."""
    idx = pd.bdate_range("2022-01-01", periods=600)
    rng = np.random.default_rng(4)
    gross = pd.Series(rng.normal(3.0, 20.0, 600), index=idx)
    price = pd.Series(20_000.0, index=idx)

    tbl = costs.sensitivity(gross, price, costs.CostModel("TXF"))

    assert tbl["cost_bps"].is_monotonic_increasing
    assert tbl["net_mean_bps"].is_monotonic_decreasing

    # Factor 0 removes the SPREAD, not the fees -- so net at factor 0 is the gross
    # mean less the fee-only cost, and is strictly below gross. A model where
    # factor 0 returned the gross mean exactly would be silently dropping fees.
    zero = tbl.loc[tbl.spread_factor == 0.0].iloc[0]
    assert zero["cost_bps"] > 0
    assert zero["net_mean_bps"] == pytest.approx(gross.mean() - zero["cost_bps"], abs=1e-6)
    assert zero["net_mean_bps"] < gross.mean()

    # and the edge dies somewhere in the range: net mean must go negative
    assert tbl["net_mean_bps"].iloc[-1] < 0 < tbl["net_mean_bps"].iloc[0]


def test_sensitivity_scales_with_turnover():
    idx = pd.bdate_range("2022-01-01", periods=300)
    gross = pd.Series(5.0, index=idx)
    price = pd.Series(20_000.0, index=idx)
    m = costs.CostModel("TXF")

    once = costs.sensitivity(gross, price, m, turnover=pd.Series(1.0, index=idx))
    twice = costs.sensitivity(gross, price, m, turnover=pd.Series(2.0, index=idx))

    assert twice["cost_bps"].iloc[-1] == pytest.approx(2 * once["cost_bps"].iloc[-1])
    assert (twice["net_mean_bps"] < once["net_mean_bps"]).all()


def test_breakeven_is_the_gross_mean():
    s = pd.Series([1.0, 3.0, 5.0, np.nan])
    assert costs.breakeven_bps(s) == pytest.approx(3.0)


def test_describe_mentions_the_round_trip():
    text = costs.CostModel("MXF").describe()
    assert "round trip" in text and "MXF" in text
