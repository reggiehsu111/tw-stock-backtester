"""Tests for the chart layer and the server.

Charts are strings, so they are cheap to assert on. The tests check the things
that actually broke: axis labels that collapse to the same value, and a chart
asked to draw data it does not have.
"""

import numpy as np
import pandas as pd
import pytest

from twbt.web import charts

IDX = pd.bdate_range("2023-01-02", periods=750)


def _pnl(seed=0):
    return pd.Series(np.random.default_rng(seed).normal(1, 40, len(IDX)), index=IDX)


# --------------------------------------------------------------------------- #
# axis labelling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ticks,expected", [
    ([0, 2000, 4000, 6000], 0),
    ([-0.5, 0.0, 0.5, 1.0], 2),
    ([1.00, 1.01, 1.02, 1.03], 3),
])
def test_tick_precision_adapts_to_the_step(ticks, expected):
    assert charts._tick_decimals(ticks) == expected


def test_narrow_axis_labels_stay_distinguishable():
    """The bug: a t-stat axis from 1.01 to 1.03 rendered as "1, 1, 1, 0, 0, -0"."""
    sens = pd.DataFrame({
        "spread_factor": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        "net_t": [0.115, 0.111, 0.107, 0.103, 0.100, 0.095],
        "cost_bps": [0.3, 0.8, 1.3, 1.8, 2.3, 3.3],
        "net_mean_bps": [9.0, 8.5, 8.0, 7.5, 7.0, 6.0],
    })
    svg = charts.cost_curve(sens)
    labels = [t for t in svg.split("ytick") if t]
    # the rendered labels must not all be the same string
    import re
    ylabels = re.findall(r'class="tick ytick"[^>]*>([^<]+)<', svg)
    assert len(ylabels) >= 3
    assert len(set(ylabels)) == len(ylabels), f"duplicate axis labels: {ylabels}"
    assert "-0.000" not in ylabels and "-0" not in ylabels


def test_zero_tick_is_not_negative_zero():
    svg = charts.equity({"net": _pnl().cumsum()})
    import re
    assert "-0<" not in svg
    for lab in re.findall(r'class="tick ytick"[^>]*>([^<]+)<', svg):
        assert not lab.startswith("-0.") or float(lab.replace(",", "")) != 0.0


# --------------------------------------------------------------------------- #
# charts render, and say so when they cannot
# --------------------------------------------------------------------------- #

def test_equity_draws_both_series_with_a_legend():
    gross, net = _pnl().cumsum(), (_pnl() - 2).cumsum()
    svg = charts.equity({"gross": gross, "net": net})
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "gross" in svg and "net" in svg
    assert svg.count('class="line"') == 2


def test_by_year_labels_each_year_with_its_sharpe():
    svg = charts.by_year(_pnl())
    for yr in (2023, 2024, 2025):
        assert str(yr) in svg
    assert "SR" in svg
    assert "AvgPath" in svg


def test_decomposition_omits_a_side_that_was_never_traded():
    pnl = _pnl()
    long_only = pd.Series(1.0, index=IDX)
    svg = charts.decomposition(pnl, long_only)
    assert "long" in svg
    assert ">short" not in svg          # no short side to draw


def test_decomposition_shows_both_sides_when_both_traded()  :
    pnl = _pnl()
    pos = pd.Series(np.where(np.arange(len(IDX)) % 2, 1.0, -1.0), index=IDX)
    svg = charts.decomposition(pnl, pos)
    assert "long" in svg and "short" in svg and "total" in svg


@pytest.mark.parametrize("fn,arg", [
    (charts.equity, {}),
    (charts.by_year, pd.Series(dtype=float)),
    (charts.cost_curve, pd.DataFrame()),
])
def test_charts_say_why_they_are_empty_rather_than_crashing(fn, arg):
    svg = fn(arg)
    assert svg.startswith("<svg")
    assert "empty" in svg


def test_charts_escape_text():
    svg = charts.equity({"<script>x</script>": _pnl().cumsum()})
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #

def test_app_serves_index_and_unknown_strategy_404s(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from twbt.web import app as app_mod

    monkeypatch.setattr(app_mod, "RESULTS_DIR", tmp_path)
    client = TestClient(app_mod.create_app())

    r = client.get("/")
    assert r.status_code == 200
    assert "Nothing saved yet" in r.text

    r = client.get("/s/does_not_exist")
    assert r.status_code == 404
