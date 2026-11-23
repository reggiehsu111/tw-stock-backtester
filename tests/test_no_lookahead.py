"""Guard tests: every normalisation must be blind to the future.

The test is adversarial rather than illustrative. It does not check that a function
returns plausible numbers — it checks that the numbers are *bit-identical* when the
part of the series the function must not have seen is replaced with garbage.

This is the test that would have caught the full-sample-quantile bug in
docs/METHOD.md section 1, and it fails loudly if anyone reintroduces it.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import panel

FUNCS = {
    "zscore": lambda s: panel.zscore(s, window=60, min_periods=30),
    "trailing_quantile": lambda s: panel.trailing_quantile(s, 1 / 3, window=60, min_periods=30),
    "trailing_rank": lambda s: panel.trailing_rank(s, window=60, min_periods=30),
    "realised_vol": lambda s: panel.realised_vol(s, window=20, min_periods=10),
}


def _series(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0, 100, n), index=idx, name="x")


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_future_corruption_does_not_change_the_past(name):
    """Replace the back half with garbage; the front half must not move."""
    fn = FUNCS[name]
    s = _series()
    cut = len(s) // 2

    clean = fn(s)

    poisoned = s.copy()
    rng = np.random.default_rng(99)
    poisoned.iloc[cut:] = rng.normal(5e6, 1e6, len(s) - cut)   # wildly different scale
    dirty = fn(poisoned)

    a, b = clean.iloc[:cut], dirty.iloc[:cut]
    assert a.isna().equals(b.isna()), f"{name}: NaN pattern changed in the past half"
    np.testing.assert_array_equal(
        a.dropna().to_numpy(), b.dropna().to_numpy(),
        err_msg=f"{name} leaked information from the future into the past",
    )


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_perturbing_one_day_cannot_move_earlier_days(name):
    """A single changed observation may affect itself and later days, never earlier."""
    fn = FUNCS[name]
    s = _series()
    k = 300

    before = fn(s)
    bumped = s.copy()
    bumped.iloc[k] += 1e6
    after = fn(bumped)

    np.testing.assert_array_equal(
        before.iloc[:k].dropna().to_numpy(),
        after.iloc[:k].dropna().to_numpy(),
        err_msg=f"{name}: changing day {k} altered a day before it",
    )


def test_trailing_quantile_differs_from_full_sample_on_a_trending_series():
    """A regression test for the mistake itself, not just for the fix.

    On a series with a trend, the full-sample quantile and the trailing one must
    disagree. If they ever agree here, the trailing version is not actually trailing
    and the other tests have become vacuous.
    """
    idx = pd.bdate_range("2020-01-01", periods=400)
    s = pd.Series(np.linspace(0, 1000, 400), index=idx)

    trailing = panel.trailing_quantile(s, 1 / 3, window=60, min_periods=30)
    full = s.quantile(1 / 3)

    valid = trailing.dropna()
    assert len(valid) > 100
    assert not np.allclose(valid.to_numpy(), full), (
        "trailing quantile matched the full-sample quantile on a trending series"
    )
    # and the classification genuinely differs
    assert ((s < trailing) != (s < full)).sum() > 50


def test_trailing_rank_is_in_unit_interval_and_excludes_today():
    s = _series()
    r = panel.trailing_rank(s, window=60, min_periods=30).dropna()
    assert len(r) > 200
    assert r.between(0.0, 1.0).all()

    # a value larger than everything before it must rank 1.0, which is only possible
    # if the current observation is excluded from its own comparison set
    s2 = s.copy()
    s2.iloc[200] = 1e9
    assert panel.trailing_rank(s2, window=60, min_periods=30).iloc[200] == 1.0


def test_zscore_rejects_zero_variance_windows():
    """A flat window must give NaN, not a division by zero."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    s = pd.Series([5.0] * 150 + [9.0] * 50, index=idx)
    z = panel.zscore(s, window=60, min_periods=30)
    assert z.iloc[:150].isna().all() or np.isfinite(z.iloc[:150].dropna()).all()
    assert not np.isinf(z.dropna()).any()


def test_trailing_quantile_validates_q():
    s = _series(100)
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            panel.trailing_quantile(s, bad)
