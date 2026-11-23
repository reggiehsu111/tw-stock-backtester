"""Tests for the engine.

The two load-bearing ones are ``test_a_one_day_hold_pays_exactly_one_round_trip``
and ``test_publication_lag_is_applied_once``. Both were written after the
corresponding bug: masking the cost series to days with a non-zero position drops
the exit leg and charges half a round trip, and a lag applied in both the strategy
and the engine is applied twice while a lag applied in neither is free money.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from twbt import backtest, costs

IDX = pd.bdate_range("2020-01-01", periods=1000, name="date")


def _panel(ret=None, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if ret is None:
        ret = rng.normal(0, 80, len(IDX))
    return pd.DataFrame({
        "ret": pd.Series(ret, index=IDX),
        "close": pd.Series(20_000.0, index=IDX),
        "flow": pd.Series(rng.normal(0, 1, len(IDX)), index=IDX),
    })


@dataclass(frozen=True)
class FixedPositions(backtest.Strategy):
    """A strategy that replays a position series given to it, for exact arithmetic."""

    pattern: tuple = ()

    def signal(self, panel):
        v = np.resize(np.array(self.pattern, dtype=float), len(panel))
        return pd.Series(v, index=panel.index)


@dataclass(frozen=True)
class FlowSign(backtest.Strategy):
    threshold: float = 0.0

    def signal(self, panel):
        return np.sign(panel["flow"].where(panel["flow"].abs() > self.threshold, 0.0))

    def grid(self):
        return [{"threshold": t} for t in (0.0, 0.5, 1.0)]


# --------------------------------------------------------------------------- #
# cost accounting
# --------------------------------------------------------------------------- #

def test_a_one_day_hold_pays_exactly_one_round_trip():
    """The regression test for the dropped-exit-leg bug.

    Position 0 -> 1 -> 0 is one round trip. Not half, not two.
    """
    pos = (0.0, 1.0, 0.0, 0.0, 0.0)
    p = _panel(ret=np.zeros(len(IDX)))           # zero returns isolate the cost
    strat = FixedPositions(name="fixed", publication_lag=0, pattern=pos)
    cm = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)

    r = backtest.run(p, strat, cm, contracts=("TXF",))
    one_rt = cm.round_trip_bps(20_000.0)

    assert len(r.main.trades) == len(IDX) // len(pos)
    # every trade pays one round trip, whole
    assert r.main.trades["cost_bps"].round(10).eq(round(one_rt, 10)).all()
    # and the daily series totals the same thing, so nothing is double-charged
    assert r.main.daily_cost.sum() == pytest.approx(one_rt * len(r.main.trades))


def test_a_multi_day_hold_also_pays_one_round_trip():
    """Holding longer must not cost more: it is still one entry and one exit."""
    p = _panel(ret=np.zeros(len(IDX)))
    cm = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)
    one_rt = cm.round_trip_bps(20_000.0)

    short = backtest.run(p, FixedPositions(name="s", publication_lag=0,
                                          pattern=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
                         cm, contracts=("TXF",))
    long_ = backtest.run(p, FixedPositions(name="l", publication_lag=0,
                                           pattern=(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)),
                         cm, contracts=("TXF",))

    assert short.main.trades["cost_bps"].mean() == pytest.approx(one_rt)
    assert long_.main.trades["cost_bps"].mean() == pytest.approx(one_rt)
    assert long_.main.trades["days"].iloc[0] == 3
    # every trade pays the same, including one left open when the data ends
    assert long_.main.trades["cost_bps"].round(9).nunique() == 1


def test_a_sign_flip_costs_two_round_trips():
    """1 -> -1 closes a long and opens a short: two round trips' worth of legs."""
    p = _panel(ret=np.zeros(len(IDX)))
    cm = costs.CostModel("TXF", spread_pts=3.0, fees_pts_per_side=0.0)
    one_rt = cm.round_trip_bps(20_000.0)

    r = backtest.run(p, FixedPositions(name="flip", publication_lag=0,
                                       pattern=(1.0, -1.0)),
                     cm, contracts=("TXF",))
    # Each transition moves 2 contracts, which is 2 legs = 1 round trip per day.
    # Day 0 is the exception: there is no prior position, so entering is 1 leg.
    daily = r.main.daily_cost
    assert daily.iloc[0] == pytest.approx(one_rt / 2)
    assert daily.iloc[1:].eq(daily.iloc[1]).all()
    assert daily.iloc[1] == pytest.approx(one_rt)


def test_zero_position_days_contribute_nothing():
    p = _panel(seed=2)
    r = backtest.run(p, FixedPositions(name="z", publication_lag=0,
                                       pattern=(0.0, 0.0, 0.0)),
                     contracts=("TXF",))
    assert len(r.main.trades) == 0
    assert r.main.daily_net.abs().sum() == 0
    assert "never took a position" in " ".join(r.verdict())


# --------------------------------------------------------------------------- #
# publication lag
# --------------------------------------------------------------------------- #

def test_publication_lag_is_applied_once():
    """A lag of 1 must earn day t+1's return from day t's signal, exactly."""
    ret = np.zeros(len(IDX))
    ret[5] = 100.0                        # a single non-zero day
    p = _panel(ret=ret)

    # signal is on everywhere, so position is on everywhere after the lag
    always = FixedPositions(name="a", publication_lag=1, pattern=(1.0,))
    r = backtest.run(p, always, costs.CostModel("TXF", spread_pts=0.0,
                                                fees_pts_per_side=0.0),
                     contracts=("TXF",))
    assert r.main.daily_gross.iloc[5] == pytest.approx(100.0)
    assert r.main.daily_gross.iloc[0] == 0.0          # lag means day 0 is flat


def test_lag_shifts_the_signal_not_the_return():
    """A signal that fires on exactly one day must trade on exactly the next one."""
    p = _panel(ret=np.arange(len(IDX), dtype=float))
    pattern = tuple([1.0] + [0.0] * 9)
    strat = FixedPositions(name="pulse", publication_lag=1, pattern=pattern)
    r = backtest.run(p, strat, costs.CostModel("TXF", spread_pts=0.0,
                                               fees_pts_per_side=0.0),
                     contracts=("TXF",))
    held = r.main.position[r.main.position != 0].index
    # signal fires on index 0, 10, 20...; with lag 1 the position is on 1, 11, 21...
    assert list(p.index.get_indexer(held))[:3] == [1, 11, 21]
    assert r.main.trades["gross_bps"].iloc[0] == pytest.approx(1.0)


def test_zero_lag_on_published_data_is_refused():
    """The guard that stops a flow strategy from trading on data it cannot have."""

    @dataclass(frozen=True)
    class Cheat(backtest.Strategy):
        def signal(self, panel):
            return np.sign(panel["flow"])

    bad = Cheat(name="cheat", publication_lag=0, published_columns=("flow",))
    with pytest.raises(ValueError, match="publication_lag=0"):
        backtest.run(_panel(), bad, contracts=("TXF",))

    # the same strategy with a lag is fine
    ok = Cheat(name="ok", publication_lag=1, published_columns=("flow",))
    assert backtest.run(_panel(), ok, contracts=("TXF",)).main.n_trades > 0


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError):
        backtest.run(_panel(), FixedPositions(name="x", publication_lag=-1,
                                              pattern=(1.0,)),
                     contracts=("TXF",))


# --------------------------------------------------------------------------- #
# cross-instrument and verdict
# --------------------------------------------------------------------------- #

def test_all_contracts_are_evaluated_and_cost_more_on_smaller_ones():
    p = _panel(ret=np.full(len(IDX), 10.0))
    r = backtest.run(p, FixedPositions(name="long", publication_lag=0,
                                       pattern=(0.0, 1.0, 0.0)))
    assert set(r.contracts) == {"TXF", "MXF", "TMF"}
    cons = r.consistency()
    assert len(cons) == 3
    # same gross everywhere, so net must be ordered purely by cost
    assert (r.contracts["TMF"].net_perf.mean
            < r.contracts["MXF"].net_perf.mean)


def test_verdict_flags_sign_disagreement_across_siblings():
    """Construct the pathology deliberately and check the engine says so.

    A gross edge small enough that cost flips its sign on the expensive contracts is
    exactly the pattern that should be reported as a failure, not a success.
    """
    p = _panel(ret=np.full(len(IDX), 1.6))     # ~1.6 bps/day gross
    r = backtest.run(p, FixedPositions(name="thin", publication_lag=0,
                                       pattern=(0.0, 1.0, 0.0)))
    means = [r.contracts[c].net_perf.mean for c in ("TXF", "MXF", "TMF")]
    if len(set(np.sign(means))) > 1:
        assert any("cross-instrument" in v for v in r.verdict())
    else:                                       # if cost did not flip it, say so
        assert any("Consistent in sign" in v for v in r.verdict())


def test_verdict_reports_the_noise_bar_when_variants_were_searched():
    p = _panel(seed=5)
    r = backtest.run(p, FlowSign(name="flow", publication_lag=1,
                                 published_columns=("flow",)),
                     contracts=("TXF",), n_variants_tested=40)
    assert np.isfinite(r.noise_bar_95)
    assert r.noise_bar_95 > 2.5
    text = " ".join(r.verdict())
    assert "noise bar" in text or "multiple-testing" in text


def test_equity_curve_is_the_cumulative_daily_net():
    p = _panel(seed=3)
    r = backtest.run(p, FixedPositions(name="e", publication_lag=0,
                                       pattern=(0.0, 1.0, 0.0)),
                     contracts=("TXF",))
    eq = r.main.equity
    assert len(eq) == len(IDX)
    assert eq.iloc[-1] == pytest.approx(r.main.daily_net.sum())
    # the trade-level net must total the same as the daily net
    assert r.main.trades["net_bps"].sum() == pytest.approx(r.main.daily_net.sum())


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #

def test_walk_forward_only_uses_prior_data_to_pick_parameters():
    p = _panel(seed=1)
    wf = backtest.walk_forward(p, FlowSign(name="flow", publication_lag=1,
                                           published_columns=("flow",)),
                               contract="TXF", min_train_years=1)
    assert len(wf) > 0
    assert {"year", "params", "n", "net_mean_bps", "net_t"} <= set(wf.columns)
    # the first evaluated year must be late enough to have a training sample
    assert wf["year"].min() > p.index.year.min()


def test_walk_forward_handles_a_strategy_with_no_grid():
    p = _panel(seed=4)
    wf = backtest.walk_forward(p, FixedPositions(name="f", publication_lag=1,
                                                 pattern=(0.0, 1.0, 0.0)),
                               contract="TXF", min_train_years=1)
    assert len(wf) > 0
    assert (wf["params"] == "fixed").all()
