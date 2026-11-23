"""Tests for the significance machinery.

The important ones feed the functions data with *no* edge and assert they say so.
A test that a statistic is large when the effect is real is easy to pass; a test
that it is small when the effect is absent is the one that catches a broken
correction.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import stats

H = 30


def _overlapping_noise(seed: int, n: int = 3000, h: int = H):
    """Trailing h-mean and strictly-forward h-mean of the same white noise.

    There is no relationship between them by construction: x_t uses r[t-h+1..t],
    y_t uses r[t+1..t+h], and the two sets never intersect. But consecutive
    observations of each share h-1 of their h terms, which is exactly the
    autocorrelation that inflates OLS.
    """
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0, 100, n))
    x = r.rolling(h).mean()
    y = r.rolling(h).mean().shift(-h)
    return y, x


def test_ols_is_fooled_by_overlap_and_the_corrections_are_not():
    """The headline claim of docs/METHOD.md section 2, as an assertion.

    Across seeds of pure noise, OLS produces |t| above 2 most of the time, while
    the disjoint estimator behaves like the null it is: |t| above 2 rarely.
    """
    ols, nw, dj = [], [], []
    for seed in range(20):
        y, x = _overlapping_noise(seed)
        res = stats.regress(y, x, overlap=H)
        ols.append(abs(res.t_ols))
        nw.append(abs(res.t_nw))
        dj.append(abs(res.t_disjoint))

    ols, nw, dj = np.array(ols), np.array(nw), np.array(dj)

    # OLS is badly miscalibrated: it should reject ~5% of the time, and does far more
    assert (ols > 2).mean() > 0.40, f"expected OLS to over-reject on noise, got {(ols > 2).mean():.2f}"
    # both corrections cut it down substantially
    assert nw.mean() < ols.mean() / 2
    assert dj.mean() < ols.mean() / 2
    # and the disjoint estimator is roughly calibrated
    assert (dj > 2).mean() < 0.25, f"disjoint over-rejects on noise: {(dj > 2).mean():.2f}"


def test_inflation_is_on_the_order_of_sqrt_h():
    """Not an equality — the ratio is noisy — but the right order of magnitude."""
    ratios = []
    for seed in range(20):
        y, x = _overlapping_noise(seed)
        res = stats.regress(y, x, overlap=H)
        if np.isfinite(res.inflation):
            ratios.append(res.inflation)
    assert 2.0 < np.median(ratios) < 25.0


def test_no_overlap_means_all_three_t_stats_agree():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=800))
    y = 0.3 * x + pd.Series(rng.normal(size=800))
    res = stats.regress(y, x, overlap=1)
    assert res.t_ols == res.t_nw == res.t_disjoint == res.honest_t
    assert res.n == res.n_disjoint


def test_a_real_relationship_survives_the_disjoint_test():
    rng = np.random.default_rng(11)
    y, x = _overlapping_noise(5)
    y = y + 0.8 * x                      # inject a genuine slope
    res = stats.regress(y, x, overlap=H)
    assert res.t_disjoint > 3.0
    assert res.beta > 0.4


def test_honest_t_picks_disjoint_when_overlap_matters():
    y, x = _overlapping_noise(1)
    res = stats.regress(y, x, overlap=H)
    assert res.honest_t == res.t_disjoint
    assert abs(res.honest_t) < abs(res.t_ols)


def test_regress_returns_nan_rather_than_guessing_on_short_input():
    res = stats.regress([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], overlap=1)
    assert np.isnan(res.t_ols) and np.isnan(res.beta)


def test_regress_drops_non_finite_pairs():
    x = pd.Series([1.0, 2.0, np.nan, 4.0] * 40)
    y = pd.Series([1.0, np.inf, 3.0, 4.0] * 40)
    res = stats.regress(y, x, overlap=1)
    assert res.n == 80                   # the nan column and the inf column each cost 40
    assert np.isfinite(res.t_ols)


def test_noise_bar_rises_with_the_number_of_tests():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 100, 1500))
    b1 = stats.noise_bar(r, 1, iters=400)
    b2 = stats.noise_bar(r, 2, iters=400)
    b10 = stats.noise_bar(r, 10, iters=400)
    b40 = stats.noise_bar(r, 40, iters=400)

    assert b1 < b10 < b40
    # a single test's 95th percentile should land near the normal 1.96
    assert 1.6 < b1 < 2.4
    # two buckets is one comparison, not two -- each is the other's complement
    assert b1 == pytest.approx(b2)
    # and forty tests should demand well above the naive threshold
    assert b40 > 2.8


def test_noise_bar_is_deterministic_for_a_given_seed():
    r = pd.Series(np.random.default_rng(0).normal(size=600))
    assert stats.noise_bar(r, 8, iters=200, seed=42) == stats.noise_bar(r, 8, iters=200, seed=42)


def test_welch_compares_against_the_complement_not_zero():
    """A series with a non-zero mean must still give t~0 against its own complement."""
    rng = np.random.default_rng(9)
    r = rng.normal(50, 100, 2000)        # mean is 50, clearly non-zero
    diff, t = stats.welch(r[:1000], r[1000:])
    assert abs(t) < 2.5                  # but the halves do not differ
    # whereas a genuinely shifted bucket does
    _, t2 = stats.welch(r[:1000] + 60, r[1000:])
    assert t2 > 5.0


def test_performance_reports_by_year_and_drawdown():
    idx = pd.bdate_range("2021-01-01", periods=500)
    s = pd.Series(np.r_[np.full(250, 2.0), np.full(250, -1.0)], index=idx)
    p = stats.performance(s)
    assert p.total == pytest.approx(250.0)
    assert p.max_drawdown == pytest.approx(-250.0)
    assert set(p.by_year) == {2021, 2022}
    assert p.hit_rate == pytest.approx(0.5)
    assert "by_year" in p.as_dict()


def test_sandwich_does_not_allocate_an_n_by_n_matrix():
    """Regression test for the 488 GiB bug.

    ``X.T @ diag(e**2) @ X`` is mathematically identical to what we compute and
    completely unusable at scale. This runs a size that would be ~2 GiB as a dense
    diagonal, so it fails by dying rather than by asserting.
    """
    n = 200_000
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=n))
    y = pd.Series(rng.normal(size=n))
    res = stats.regress(y, x, overlap=5)
    assert np.isfinite(res.t_ols) and np.isfinite(res.t_nw)
