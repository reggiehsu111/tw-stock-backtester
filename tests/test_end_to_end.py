"""End-to-end tests over the committed fixture.

These run the documented path — build the panel, run both strategies, render the
tearsheet — with no network. If the README's quick start stops working, this is
what fails.

Everything here uses `--fixture`, so it is also the test that the fixture itself
is complete and committed.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import backtest, cache, costs, panel, pipeline as pl, strategies, xcosts

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def p():
    try:
        return panel.build(fixture=True)
    except FileNotFoundError as exc:
        pytest.skip(f"fixture not present: {exc}")


# --------------------------------------------------------------------------- #
# the fixture and the panel
# --------------------------------------------------------------------------- #

def test_fixture_contains_all_three_datasets():
    for name in ("futures_daily", "three_majors", "large_trader"):
        df = cache.load(name, fixture=True)
        assert len(df) > 100, f"{name} fixture is suspiciously small"
        assert "date" in df.columns


def test_panel_has_the_columns_the_strategies_need(p):
    assert len(p) > 400
    for c in ("ret", "close", "open", "high", "low", "volume", "vol_20"):
        assert c in p.columns, f"missing {c}"
    # the institutional columns, under the names the live endpoint uses
    assert "foreign_oi_net_lots" in p.columns
    assert "dealer_oi_net_lots" in p.columns
    assert p["foreign_oi_net_lots"].notna().sum() > 300


def test_panel_is_one_row_per_session_and_sorted(p):
    assert p.index.is_monotonic_increasing
    assert not p.index.has_duplicates
    assert isinstance(p.index, pd.DatetimeIndex)


def test_panel_returns_are_sane(p):
    """A bad front-month pick shows up as giant returns on roll days."""
    r = p["ret"].dropna()
    assert r.abs().max() < 1500, "a >15% daily move suggests a contract roll artifact"
    assert 20 < r.std() < 300


def test_large_trader_columns_survived_the_numeric_filters(p):
    assert "top5_net_frac" in p.columns
    frac = p["top5_net_frac"].dropna()
    assert len(frac) > 300, "the numeric-code filters matched nothing"
    assert frac.abs().max() <= 1.5


# --------------------------------------------------------------------------- #
# the strategies, end to end
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["institutional_flow", "momentum"])
def test_strategy_runs_and_produces_a_verdict(p, name):
    strat = strategies.get(name)()
    res = backtest.run(p, strat, costs.CostModel("TXF"),
                       n_variants_tested=getattr(strat, "n_variants", 1))

    assert res.main.n_trades > 0, f"{name} never traded on the fixture"
    assert len(res.contracts) == 3
    assert len(res.verdict()) > 0
    assert np.isfinite(res.main.net_perf.mean)
    # net must be worse than gross: costs are not free
    assert res.main.net_perf.mean < res.main.gross_perf.mean


@pytest.mark.parametrize("name", ["institutional_flow", "momentum"])
def test_reported_sharpe_is_coherent_with_the_t_stat(p, name):
    """Guards the sqrt(252) artifact: Sharpe 4 beside t=1 is not possible."""
    strat = strategies.get(name)()
    perf = backtest.run(p, strat, contracts=("TXF",)).main.net_perf
    assert np.sign(perf.sharpe) == np.sign(perf.t)
    # with n trades, Sharpe and t are related by the observation count; a Sharpe
    # many times the t-stat means the annualisation is inventing observations
    assert abs(perf.sharpe) < abs(perf.t) * 3 + 1


def test_walk_forward_runs_and_reports_trade_counts(p):
    strat = strategies.get("institutional_flow")()
    wf = backtest.walk_forward(p, strat, costs.CostModel("TXF"), min_train_years=1)
    assert len(wf) >= 1
    assert "n" in wf.columns
    assert wf["n"].sum() >= 0


def test_publication_lag_materially_changes_the_result(p):
    """The lag is not cosmetic: removing it should flatter the strategy.

    This is the check that the guard in run() is protecting something real. If
    lag 0 and lag 1 gave the same answer, the guard would be theatre.
    """
    base = strategies.get("institutional_flow")()
    lagged = backtest.run(p, base, contracts=("TXF",))

    # bypass the guard deliberately to measure what it prevents
    cheating = base.with_params(publication_lag=0, published_columns=())
    peeking = backtest.run(p, cheating, contracts=("TXF",))

    assert peeking.main.n_trades > 0 and lagged.main.n_trades > 0
    assert peeking.main.net_perf.mean != lagged.main.net_perf.mean


# --------------------------------------------------------------------------- #
# the tearsheet
# --------------------------------------------------------------------------- #

def test_every_chart_renders_from_a_real_result(p):
    from twbt.web import charts

    res = backtest.run(p, strategies.get("momentum")(), costs.CostModel("TXF"))
    m = res.main

    svgs = {
        "equity": charts.equity({"gross": m.daily_gross.cumsum(),
                                 "net": m.daily_net.cumsum()}),
        "by_year": charts.by_year(m.daily_net),
        "decomp": charts.decomposition(m.daily_net, m.position),
        "cost": charts.cost_curve(m.sensitivity),
    }
    for name, svg in svgs.items():
        assert svg.startswith("<svg"), name
        assert "empty" not in svg, f"{name} rendered a placeholder on real data"
        assert svg.count("<path") >= 1, name


# --------------------------------------------------------------------------- #
# measured spreads
# --------------------------------------------------------------------------- #

def test_panel_derives_a_measured_spread_from_the_quotes(p):
    assert "spread_bps" in p.columns
    sp = p["spread_bps"].dropna()
    assert len(sp) > 400, "the bid/ask columns are not being read"
    assert 0.3 < sp.median() < 5.0
    assert (sp >= 0).all()


def test_measured_spread_is_wider_on_volatile_days(p):
    """The reason a constant is not good enough, as an assertion.

    If this ever fails, the argument for carrying measured spreads through the
    engine has gone away and the extra machinery should go with it.
    """
    from twbt import stats

    d = p[["spread_bps", "range_bps"]].dropna()
    assert len(d) > 300
    hi = d[d.range_bps >= d.range_bps.median()]["spread_bps"]
    lo = d[d.range_bps < d.range_bps.median()]["spread_bps"]
    diff, t = stats.welch(hi, lo)
    assert diff > 0, "spread should be wider on high-range days"
    assert t > 2.5, f"expected a clear effect, got t={t:.2f}"


def test_using_the_measured_spread_charges_more_than_the_constant(p):
    """And it charges more on the days these strategies actually trade."""
    strat = strategies.get("momentum")()
    with_measured = backtest.run(p, strat, contracts=("TXF",)).main
    without = backtest.run(p.drop(columns=["spread_bps"]), strat,
                           contracts=("TXF",)).main

    assert with_measured.avg_cost_bps > without.avg_cost_bps
    assert with_measured.net_perf.mean < without.net_perf.mean


# --------------------------------------------------------------------------- #
# the cross-sectional path, over the committed fixture
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def eq():
    from twbt import equity_panel
    try:
        mats = equity_panel.build(fixture=True)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"equity fixture not present: {exc}")
    return mats, equity_panel.liquid_universe(mats)


def test_equity_fixture_is_a_real_cross_section(eq):
    mats, univ = eq
    px = mats["close"]
    assert len(px) > 400, "need enough sessions for a 12-month lookback"
    assert px.shape[1] > 100, "a cross-sectional rank needs a cross-section"
    assert univ.sum(axis=1).mean() > 50


def test_universe_membership_rotates(eq):
    """A static universe would be a survivorship screen wearing a mask."""
    _, univ = eq
    ever = univ.any()
    always = univ.all()
    assert int(ever.sum()) > int(always.sum()), "nobody ever enters or leaves"


def test_universe_is_decided_by_trailing_data_only(eq):
    """Corrupt the future; past membership must not move."""
    from twbt import equity_panel
    mats, univ = eq
    cut = len(mats["close"]) // 2

    poisoned = {k: v.copy() for k, v in mats.items()}
    poisoned["turnover"].iloc[cut:] = 1.0        # flatten all future liquidity
    dirty = equity_panel.liquid_universe(poisoned)

    pd.testing.assert_frame_equal(univ.iloc[:cut], dirty.iloc[:cut])


def test_delisted_and_newly_listed_names_are_present_as_nan(eq):
    """Survivorship: a name must not be dropped for failing to span the sample."""
    mats, _ = eq
    px = mats["close"]
    spans_all = px.notna().all()
    assert not spans_all.all(), (
        "every name spans the whole sample, which means short-lived names were "
        "dropped -- that is a survivorship screen")


@pytest.mark.parametrize("name", sorted(pl.PRESETS))
def test_every_preset_runs_on_the_fixture(eq, name):
    mats, univ = eq
    src = pl.build_sources(px=mats["close"], vol=mats.get("volume"),
                           amount=mats.get("turnover"), high=mats.get("high"),
                           low=mats.get("low"))
    pipe = pl.Pipeline.from_json(pl.PRESETS[name])
    _, res = pl.run_pipeline(pipe, src, mats["close"], universe=univ,
                             cost=xcosts.TWEquityCost())
    assert res.coverage.mean() > 5, f"{name} held almost nothing"
    assert np.isfinite(res.sharpe)
    assert res.net.sum() < res.gross.sum(), "costs must bite"


def test_costs_are_what_kills_the_turnover_heavy_preset(eq):
    """The project's thesis, as an assertion over real data.

    Short-term reversal turns over a quarter of the book a day. Whatever its
    gross merit, net must be materially worse -- if this ever stops being true
    the cost model has broken.
    """
    mats, univ = eq
    src = pl.build_sources(px=mats["close"], amount=mats.get("turnover"))
    pipe = pl.Pipeline.from_json(pl.PRESETS["short-term reversal"])
    _, res = pl.run_pipeline(pipe, src, mats["close"], universe=univ,
                             cost=xcosts.TWEquityCost())
    assert res.avg_turnover > 0.10
    assert res.sharpe < res.gross_sharpe - 1.0
