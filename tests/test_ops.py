"""Tests for the matrix operators.

The parametrised look-ahead test is the important one: it runs over every
operator in the registry, so adding an operator without thinking about the
future-blindness rule fails the suite automatically rather than silently
widening the hole.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import ops

IDX = pd.bdate_range("2023-01-02", periods=400)
COLS = [f"{i:04d}" for i in range(1101, 1161)]


def _mx(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (len(IDX), len(COLS))),
                        index=IDX, columns=COLS)


def _px(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(IDX), len(COLS))), axis=0)),
        index=IDX, columns=COLS)


# --------------------------------------------------------------------------- #
# the rule: no operator may see the future
# --------------------------------------------------------------------------- #

SIGNAL_OPS = [n for n, o in ops.ALL_OPS.items() if not o.lookahead]


@pytest.mark.parametrize("name", sorted(SIGNAL_OPS))
def test_no_operator_sees_the_future(name):
    """Corrupt the back half of the matrix; the front half must not move.

    Parametrised over the whole registry on purpose. A new operator is covered
    the moment it is registered, which is the only way this stays true.
    """
    mx = _mx()
    cut = len(mx) // 2

    clean = ops.apply_op(name, mx)

    poisoned = mx.copy()
    rng = np.random.default_rng(99)
    poisoned.iloc[cut:] = rng.normal(5e6, 1e6, (len(mx) - cut, len(mx.columns)))
    dirty = ops.apply_op(name, poisoned)

    a, b = clean.iloc[:cut], dirty.iloc[:cut]
    assert a.isna().equals(b.isna()), f"{name}: NaN pattern changed in the past"
    np.testing.assert_allclose(
        a.fillna(0).to_numpy(), b.fillna(0).to_numpy(), rtol=0, atol=0,
        err_msg=f"{name} leaked the future into the past")


def test_lead_is_the_only_lookahead_operator_and_is_refused():
    assert ops.ALL_OPS["Lead"].lookahead
    assert sum(o.lookahead for o in ops.ALL_OPS.values()) == 1
    with pytest.raises(ValueError, match="future"):
        ops.apply_op("Lead", _mx(), {"n": 1})


def test_lead_still_works_when_called_directly_for_building_targets():
    """It is legitimate for a forward return, just not inside a signal."""
    px = _px()
    fwd = ops.Lead(5, ops.Returns(px))
    assert fwd.iloc[0].notna().any()
    assert fwd.iloc[-1].isna().all()


# --------------------------------------------------------------------------- #
# time series
# --------------------------------------------------------------------------- #

def test_window_comes_first_not_the_matrix():
    """Lag(n, mx), matching the research vocabulary rather than pandas."""
    mx = _mx()
    pd.testing.assert_frame_equal(ops.Lag(3, mx), mx.shift(3))


def test_instrank_excludes_the_current_observation():
    mx = _mx()
    mx.iloc[200, 0] = 1e9
    r = ops.InstRank(60, mx)
    assert r.iloc[200, 0] == 1.0
    assert r.dropna(how="all").stack().between(0, 1).all()


def test_instzscore_does_not_let_a_value_set_its_own_scale():
    """A self-inclusive z-score is pulled toward zero exactly when it matters."""
    mx = pd.DataFrame(np.zeros((300, 2)), index=IDX[:300], columns=["a", "b"])
    rng = np.random.default_rng(1)
    mx[:] = rng.normal(0, 1, mx.shape)
    mx.iloc[250, 0] = 50.0
    z = ops.InstZscore(60, mx)
    assert z.iloc[250, 0] > 10, "an extreme value should produce an extreme z"


def test_forwardfill_is_bounded():
    mx = _mx()
    mx.iloc[10:200, 0] = np.nan
    out = ops.ForwardFill(5, mx)
    # limit=5 fills rows 10..14 and stops; row 15 is beyond the limit
    assert out.iloc[14, 0] == pytest.approx(mx.iloc[9, 0])
    assert np.isnan(out.iloc[15, 0]), "fill must stop at the limit"
    assert np.isnan(out.iloc[100, 0]), "fill must not run forever"


# --------------------------------------------------------------------------- #
# cross section
# --------------------------------------------------------------------------- #

def test_demean_makes_each_date_sum_to_zero():
    out = ops.Demean(_mx())
    assert out.sum(axis=1).abs().max() < 1e-9


def test_rank_is_within_date_not_within_name():
    """The classic confusion: Rank is cross-sectional, InstRank is time series."""
    mx = _mx()
    r = ops.Rank(mx)
    # every date's ranks span [0, 1] regardless of that date's level
    assert r.min(axis=1).max() < 0.1
    assert r.max(axis=1).min() == pytest.approx(1.0)
    # shifting one whole date up must not change that date's ranks
    bumped = mx.copy()
    bumped.iloc[5] += 1000
    pd.testing.assert_series_equal(ops.Rank(bumped).iloc[5], r.iloc[5])


def test_winsorize_clips_per_date_and_rejects_bad_quantiles():
    mx = _mx()
    mx.iloc[10, 0] = 1e6
    out = ops.Winsorize(0.05, 0.95, mx)
    assert out.iloc[10, 0] < 1e6
    with pytest.raises(ValueError):
        ops.Winsorize(0.9, 0.1, mx)


def test_neutralize_removes_the_factor():
    rng = np.random.default_rng(3)
    factor = _mx(seed=5)
    sig = 2.0 * factor + pd.DataFrame(
        rng.normal(0, 0.1, factor.shape), index=factor.index, columns=factor.columns)
    resid = ops.Neutralize(sig, factor)
    # residual must be essentially uncorrelated with the factor, per date
    c = [resid.loc[d].corr(factor.loc[d]) for d in resid.index[-50:]]
    assert np.nanmax(np.abs(c)) < 1e-6


# --------------------------------------------------------------------------- #
# portfolio construction
# --------------------------------------------------------------------------- #

def test_longshortbalance_is_dollar_neutral():
    out = ops.LongShortBalance(_mx())
    assert out.sum(axis=1).abs().max() < 1e-9


def test_grossscale_hits_the_target_exposure():
    out = ops.GrossScale(ops.LongShortBalance(_mx()), gmv=2.0)
    g = out.abs().sum(axis=1)
    assert g[g > 0].sub(2.0).abs().max() < 1e-9


def test_grossscale_survives_an_all_zero_date():
    mx = ops.LongShortBalance(_mx())
    mx.iloc[7] = 0.0
    out = ops.GrossScale(mx)
    assert np.isfinite(out.to_numpy()).all()
    assert out.iloc[7].abs().sum() == 0.0


def test_topn_keeps_both_tails():
    mx = _mx()
    out = ops.TopN(5, mx)
    held = out.notna().sum(axis=1)
    assert held.max() <= 10
    assert held.iloc[-1] == 10
    with pytest.raises(ValueError):
        ops.TopN(0, mx)


# --------------------------------------------------------------------------- #
# registry and composition
# --------------------------------------------------------------------------- #

def test_catalogue_is_grouped_and_complete():
    cat = ops.catalogue()
    assert len(cat) == len(ops.ALL_OPS)
    assert {c["group"] for c in cat} == {"time series", "cross section",
                                         "elementwise", "portfolio"}
    for c in cat:
        assert c["doc"], f"{c['name']} has no description for the palette"


def test_apply_op_validates_parameters():
    with pytest.raises(KeyError):
        ops.apply_op("NotAnOp", _mx())
    with pytest.raises(ValueError, match="must be int"):
        ops.apply_op("Lag", _mx(), {"n": "banana"})


def test_apply_op_handles_both_argument_orders():
    """Some operators take the matrix first; the registry must know which."""
    mx = ops.LongShortBalance(_mx())
    assert ops.apply_op("Lag", mx, {"n": 2}).equals(ops.Lag(2, mx))
    assert ops.apply_op("GrossScale", mx).equals(ops.GrossScale(mx))
    assert ops.apply_op("TopN", mx, {"n": 3}).equals(ops.TopN(3, mx))


def test_a_full_pipeline_composes_to_a_neutral_scaled_book():
    px = _px()
    sig = ops.GrossScale(ops.LongShortBalance(
        ops.Rank(ops.Demean(ops.InstRank(20, ops.Returns(px))))))
    g = sig.abs().sum(axis=1)
    assert g[g > 0].sub(1.0).abs().max() < 1e-9
    assert sig.sum(axis=1).abs().max() < 1e-9
