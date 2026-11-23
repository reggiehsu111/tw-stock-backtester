"""Tests for the equity cost model and the cross-sectional simulator.

``test_tradability_rules_must_not_read_the_return_they_gate`` is the one that
earns its place. The bug it pins made results look *worse*, not better, which is
the unusual direction and exactly why it survived a first reading: a backtest
that gets less attractive feels conservative rather than broken.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import ops, xcosts, xsim

IDX = pd.bdate_range("2023-01-02", periods=600)
COLS = [f"{i:04d}" for i in range(1101, 1201)]


def _px(seed=0, drift=0.0002, vol=0.02):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(drift, vol, (len(IDX), len(COLS))), axis=0)),
        index=IDX, columns=COLS)


def _reversal(px):
    return ops.LongShortBalance(ops.Rank(ops.Demean(-ops.InstSum(5, ops.Returns(px)))))


# --------------------------------------------------------------------------- #
# the look-ahead bug in the tradability rules
# --------------------------------------------------------------------------- #

def test_tradability_rules_must_not_read_the_return_they_gate():
    """The regression test for the +0.80 correlation bug.

    shortable_t = (px_t > px_{t-1}) is literally (ret_t > 0). Gating a book with
    it deletes the shorts that were about to win. At the correct lag the rule
    must be essentially uncorrelated with the return the position earns.
    """
    px = _px()
    ret = px.pct_change().fillna(0.0)

    good = xcosts.shortable(px, lag=1).astype(float).stack().corr(ret.stack())
    bad = xcosts.shortable(px, lag=0).astype(float).stack().corr(ret.stack())

    assert abs(good) < 0.05, f"lag-1 rule still reads the return: corr={good:+.3f}"
    assert bad > 0.5, "lag-0 should reproduce the bug, or this test proves nothing"


def test_tradability_does_not_manufacture_an_edge_on_random_data():
    """On a random walk a reversal signal has no edge; tradability must not add one.

    Before the fix this pair read -0.49 and -15.05.
    """
    px = _px()
    sig = _reversal(px)
    free = xsim.simulate(sig, px, enforce_tradability=False)
    real = xsim.simulate(sig, px, enforce_tradability=True)

    assert abs(free.gross_sharpe) < 1.5
    assert abs(real.gross_sharpe) < 1.5, (
        f"tradability moved gross Sharpe to {real.gross_sharpe:.2f} on noise")
    assert abs(real.gross_sharpe - free.gross_sharpe) < 1.0


def test_limit_locks_block_the_right_direction():
    px = pd.DataFrame(100.0, index=IDX[:10], columns=["a"])
    px.iloc[3] = 112.0                    # +12%: locked limit-up at t=3
    up, dn = xcosts.limit_locked(px, lag=0)
    assert bool(up.iloc[3, 0]) and not bool(dn.iloc[3, 0])
    px2 = px.copy(); px2.iloc[3] = 88.0   # -12%
    up2, dn2 = xcosts.limit_locked(px2, lag=0)
    assert bool(dn2.iloc[3, 0]) and not bool(up2.iloc[3, 0])


def test_blocked_trades_carry_the_prior_weight_rather_than_dropping_the_name():
    """Dropping would silently rebalance into whatever was tradable."""
    px = _px()
    w = ops.GrossScale(_reversal(px).fillna(0.0)).shift(1).fillna(0.0)
    held = xcosts.apply_tradability(w, px)
    # the book must not be re-normalised back to full gross
    assert held.abs().sum(axis=1).mean() <= 1.0 + 1e-9
    assert held.abs().sum(axis=1).mean() > 0.5


# --------------------------------------------------------------------------- #
# asymmetric cost
# --------------------------------------------------------------------------- #

def test_the_transaction_tax_falls_on_sells_only():
    c = xcosts.TWEquityCost(brokerage=0.0, half_spread=0.0)
    assert c.buy_rate == 0.0
    assert c.sell_rate == pytest.approx(xcosts.TAX_SELL)
    assert c.sell_rate > c.buy_rate


def test_a_round_trip_costs_about_42bps_before_spread():
    c = xcosts.TWEquityCost(brokerage=0.0006, half_spread=0.0)
    assert c.round_trip_bps == pytest.approx(42.0, abs=0.1)


def test_tax_dominates_the_cost_breakdown():
    b = xcosts.TWEquityCost().breakdown()
    tax = b[b["component"].str.contains("證交稅")]["share"].iloc[0]
    assert tax > 0.4, "the tax should be the largest single component"


def test_buying_and_selling_are_costed_separately_not_netted():
    """|dw| * one_rate would halve the real cost of a long round trip."""
    c = xcosts.TWEquityCost(brokerage=0.0, half_spread=0.0)
    dw = pd.DataFrame({"a": [1.0, -1.0]}, index=IDX[:2])
    cost = c.trade_cost(dw)
    assert cost.iloc[0] == pytest.approx(0.0)              # the buy is tax-free
    assert cost.iloc[1] == pytest.approx(xcosts.TAX_SELL)  # the sell is not


def test_borrow_accrues_only_on_the_short_side():
    c = xcosts.TWEquityCost(borrow_annual=0.0252)
    w = pd.DataFrame({"a": [0.5, 0.0], "b": [-0.5, 0.0]}, index=IDX[:2])
    cost = c.borrow_cost(w)
    assert cost.iloc[0] == pytest.approx(0.5 * 0.0252 / 252)
    assert cost.iloc[1] == 0.0


def test_scaling_costs_leaves_the_statutory_tax_alone():
    base = xcosts.TWEquityCost()
    doubled = base.scaled(2.0)
    assert doubled.brokerage == pytest.approx(base.brokerage * 2)
    assert doubled._tax == base._tax


def test_day_trade_halves_the_tax():
    assert (xcosts.TWEquityCost(day_trade=True).sell_rate
            < xcosts.TWEquityCost(day_trade=False).sell_rate)


# --------------------------------------------------------------------------- #
# the simulator
# --------------------------------------------------------------------------- #

def test_execution_lag_is_owned_by_the_engine_and_zero_is_flagged():
    px = _px()
    sig = _reversal(px)
    r = xsim.simulate(sig, px, exec_shift=0)
    assert any("SAME-CLOSE" in v for v in r.verdict())
    assert not any("SAME-CLOSE" in v for v in xsim.simulate(sig, px).verdict())
    with pytest.raises(ValueError):
        xsim.simulate(sig, px, exec_shift=-1)


def test_lagging_more_cannot_improve_a_signal_with_no_edge():
    px = _px()
    sig = _reversal(px)
    sharpes = [abs(xsim.simulate(sig, px, exec_shift=k).gross_sharpe)
               for k in (1, 2, 3)]
    assert max(sharpes) < 1.5


def test_costs_always_reduce_pnl():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    assert (r.cost >= 0).all()
    assert r.net.sum() < r.gross.sum()


def test_pnl_parts_add_up():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    total = r.long_pnl + r.short_pnl
    pd.testing.assert_series_equal(total, r.gross, check_names=False,
                                   check_exact=False, rtol=1e-9)


def test_book_is_market_neutral_when_asked():
    px = _px()
    r = xsim.simulate(_reversal(px), px, enforce_tradability=False)
    assert r.weights.sum(axis=1).abs().max() < 1e-9


def test_summary_and_by_year_are_populated():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    s = r.summary()
    for k in ("sharpe", "t_stat", "avg_turnover", "avg_names", "days"):
        assert k in s and np.isfinite(s[k])
    yr = r.by_year()
    assert len(yr) >= 2 and {"year", "sharpe", "net_ret"} <= set(yr.columns)


def test_verdict_names_the_failure_mode():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    text = " ".join(r.verdict())
    assert "FAILS" in text or "Net positive" in text


def test_avgpath_averages_years_and_drops_short_stubs():
    s = pd.Series(1.0, index=pd.bdate_range("2021-01-01", periods=600))
    path = xsim.AvgPath(s)
    assert len(path) > 100
    assert path.is_monotonic_increasing


def test_sharpe_handles_degenerate_input():
    assert np.isnan(xsim.Sharpe(pd.Series(dtype=float)))
    assert np.isnan(xsim.Sharpe(pd.Series([1.0, 1.0, 1.0])))


# --------------------------------------------------------------------------- #
# configurable costs and translated output
# --------------------------------------------------------------------------- #

def test_every_cost_component_is_adjustable():
    """The right rates depend on your broker; a fixed dropdown asserts otherwise."""
    base = xcosts.TWEquityCost()
    cheap = xcosts.TWEquityCost(brokerage=0.0, half_spread=0.0, tax_sell=0.0)
    assert cheap.round_trip_bps == 0.0
    assert base.round_trip_bps > 0

    tax_only = xcosts.TWEquityCost(brokerage=0.0, half_spread=0.0)
    assert tax_only.round_trip_bps == pytest.approx(xcosts.TAX_SELL * 1e4)
    assert xcosts.TWEquityCost(brokerage=0.0, half_spread=0.0,
                               day_trade=True).round_trip_bps == pytest.approx(
        xcosts.TAX_SELL_DAYTRADE * 1e4)


def test_borrow_rate_is_adjustable_and_only_hits_shorts():
    w = pd.DataFrame({"a": [0.5], "b": [-0.5]}, index=IDX[:1])
    lo = xcosts.TWEquityCost(borrow_annual=0.0).borrow_cost(w).iloc[0]
    hi = xcosts.TWEquityCost(borrow_annual=0.10).borrow_cost(w).iloc[0]
    assert lo == 0.0 and hi > 0


def test_verdict_renders_in_both_languages_without_leaking_keys():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    en, zh = r.verdict("en"), r.verdict("zh")
    assert len(en) == len(zh) > 0
    # not "contains a space": a Chinese sentence legitimately has none
    assert all(isinstance(v, str) and len(v) > 8 for v in en + zh)
    # a Chinese verdict must actually contain Chinese
    assert any(any("一" <= ch <= "鿿" for ch in v) for v in zh)
    # and must not be identical to the English
    assert en != zh


def test_diagnostics_are_populated_for_the_extra_panels():
    px = _px()
    r = xsim.simulate(_reversal(px), px)
    assert r.raw_gross is not None and len(r.raw_gross) == len(r.gross)
    assert r.long_count is not None and r.long_count.max() > 0
    assert set(r.blocked) >= {"blocked_buy", "blocked_sell", "untraded_gmv"}
    assert (r.blocked["untraded_gmv"] >= 0).all()
    # the locks cost something, and the engine can say how much
    assert len(r.limit_drag) == len(r.gross)


def test_lag_sweep_and_buckets_produce_usable_series():
    px = _px()
    sig = _reversal(px)
    lags = xsim.lag_sweep(sig, px, lags=(1, 2, 3))
    assert set(lags) == {1, 2, 3}
    assert all(len(v[0]) > 100 and np.isfinite(v[1]) for v in lags.values())

    bkt = xsim.bucket_pnl(sig, px.pct_change().fillna(0.0))
    assert list(bkt) == [f"Q{i}" for i in range(1, 6)]
    assert all(len(v) > 100 for v in bkt.values())
