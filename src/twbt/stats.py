"""Significance testing that does not flatter the result.

Three tools, each aimed at a specific way backtests overstate themselves.

``regress``   returns OLS, Newey-West and disjoint-sample t together, because an
              overlapping-window regression inflates OLS by roughly sqrt(horizon)
              and you cannot tell by how much without looking.
``noise_bar`` simulates the largest |t| you would see from N tests on data with no
              edge, so "best of 40 variants, t=2.8" can be compared with the ~3.5
              that noise alone produces.
``welch``     compares a bucket with its COMPLEMENT rather than with zero, which
              is the difference between "this subgroup is non-zero" (nearly always
              true somewhere) and "this subgroup differs from the others".

A note on the HC0/HAC sandwich below: it is written as ``(X * w[:, None]).T @ X``
rather than ``X.T @ diag(w) @ X``. The second form allocates an n-by-n matrix —
at n = 255,800 that is 488 GiB, which is how this was discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class RegressionResult:
    """One univariate regression, reported three ways."""

    beta: float
    t_ols: float
    t_nw: float               # Newey-West, Bartlett kernel, lag = overlap
    t_disjoint: float         # every overlap-th observation, no kernel assumption
    n: int
    n_disjoint: int
    r2: float
    overlap: int

    @property
    def honest_t(self) -> float:
        """The one to quote. Disjoint when overlap matters, OLS when it cannot."""
        return self.t_ols if self.overlap <= 1 else self.t_disjoint

    @property
    def inflation(self) -> float:
        """How much the overlap flattered OLS. ~sqrt(overlap) when it bites."""
        if not np.isfinite(self.t_disjoint) or self.t_disjoint == 0:
            return np.nan
        return abs(self.t_ols) / abs(self.t_disjoint)

    def __str__(self) -> str:
        return (f"beta={self.beta:+.4f}  t_ols={self.t_ols:+.2f}  "
                f"t_nw={self.t_nw:+.2f}  t_disjoint={self.t_disjoint:+.2f}  "
                f"(n={self.n:,}, overlap={self.overlap})")


def _sandwich_t(y: np.ndarray, x: np.ndarray, lag: int = 0) -> tuple[float, float, float]:
    """OLS slope with HC0 (lag=0) or Newey-West (lag>0) errors. Returns (beta, t, r2)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    # meat matrix; never materialise diag(e^2) -- see module docstring
    S = (X * (e ** 2)[:, None]).T @ X
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        Xi, Xj, ei, ej = X[L:], X[:-L], e[L:], e[:-L]
        G = (Xi * (ei * ej)[:, None]).T @ Xj
        S = S + w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))[1]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (e ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return float(beta[1]), float(beta[1] / se) if se > 0 else np.nan, float(r2)


def regress(y, x, overlap: int = 1, min_obs: int = 60) -> RegressionResult:
    """Regress y on x, reporting OLS, Newey-West and disjoint t.

    ``overlap`` is the number of observations that share data — for a k-period
    forward return sampled every period, it is k. Pass 1 for non-overlapping data
    and the three t-statistics coincide by construction.
    """
    y = np.asarray(pd.Series(y), dtype=float)
    x = np.asarray(pd.Series(x), dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < min_obs:
        return RegressionResult(np.nan, np.nan, np.nan, np.nan, len(y), 0, np.nan, overlap)

    beta, t_ols, r2 = _sandwich_t(y, x, lag=0)
    t_nw = _sandwich_t(y, x, lag=max(overlap - 1, 0))[1] if overlap > 1 else t_ols

    if overlap > 1:
        sl = slice(None, None, overlap)
        yd, xd = y[sl], x[sl]
        t_dj = (_sandwich_t(yd, xd, lag=0)[1] if len(yd) >= min_obs else np.nan)
        n_dj = len(yd)
    else:
        t_dj, n_dj = t_ols, len(y)

    return RegressionResult(beta, t_ols, t_nw, t_dj, len(y), n_dj, r2, overlap)


def noise_bar(returns, n_tests: int, iters: int = 3000, pct: float = 95.0,
              seed: int = 0) -> float:
    """The |t| that ``n_tests`` tests on edgeless data would produce by chance.

    Randomly relabels the SAME returns into n_tests groups and records the largest
    |t| of a group against its complement, repeatedly. The returned percentile is
    the bar a real finding has to clear.

    Using the actual return series rather than normal draws matters: financial
    returns are fat-tailed, which raises the bar above what a textbook
    distribution would suggest.

    Note on small n_tests: the labels are always drawn over at least two groups,
    because a bucket needs a complement to be compared against. This also makes
    n_tests=1 and n_tests=2 agree, which is correct rather than a quirk — splitting
    a sample into two buckets yields one comparison, not two, since each bucket is
    the other's complement and the two |t| values are identical. The returned bar
    for either is close to the textbook 1.96.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 40 or n_tests < 1:
        return np.nan
    n_groups = max(n_tests, 2)
    rng = np.random.default_rng(seed)
    maxes = np.empty(iters)
    for i in range(iters):
        lab = rng.integers(0, n_groups, len(r))
        best = 0.0
        for g in range(n_tests):
            m = lab == g
            a, b = r[m], r[~m]
            if len(a) < 10 or len(b) < 10:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            if se > 0:
                best = max(best, abs(a.mean() - b.mean()) / se)
        maxes[i] = best
    return float(np.percentile(maxes, pct))


def welch(a, b) -> tuple[float, float]:
    """Difference in means and its t, for a bucket against its complement."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 8 or len(b) < 8:
        return np.nan, np.nan
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(diff), float(diff / se) if se > 0 else np.nan


@dataclass
class Performance:
    """Summary of a daily return series, in the units it was given (bps or fraction)."""

    n: int
    mean: float
    t: float
    sharpe: float
    hit_rate: float
    total: float
    max_drawdown: float
    best: float
    worst: float
    by_year: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "by_year"}
        d["by_year"] = {str(k): v for k, v in self.by_year.items()}
        return d


def observed_frequency(s: pd.Series, default: int = TRADING_DAYS) -> float:
    """Observations per year, measured from the index rather than assumed.

    This exists because annualising with a hardcoded sqrt(252) is how a strategy
    that trades sixteen times in three years reports a Sharpe of 4.0 beside a
    t-statistic of 1.0. The two cannot both be true: sqrt(252) claims 252
    independent observations a year, and there were five.

    Falls back to ``default`` for a series with no usable dates or too short a
    span to measure.
    """
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 2:
        return float(default)
    span = (s.index.max() - s.index.min()).days
    if span < 60:
        return float(default)
    return len(s) / (span / 365.25)


def performance(pnl, periods_per_year: int | None = None) -> Performance:
    """Headline statistics for a P&L series indexed by date.

    ``periods_per_year`` is measured from the index by default. Pass a number only
    when you know the series' frequency better than its own timestamps do.
    """
    s = pd.Series(pnl).dropna()
    if len(s) < 2:
        return Performance(len(s), *([np.nan] * 8))
    ppy = observed_frequency(s) if periods_per_year is None else float(periods_per_year)
    sd = s.std(ddof=1)
    t = s.mean() / (sd / np.sqrt(len(s))) if sd > 0 else np.nan
    sharpe = s.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan
    eq = s.cumsum()
    dd = float((eq - eq.cummax()).min())
    by_year = {}
    if isinstance(s.index, pd.DatetimeIndex):
        by_year = {int(y): float(g.sum()) for y, g in s.groupby(s.index.year)}
    return Performance(
        n=len(s), mean=float(s.mean()), t=float(t), sharpe=float(sharpe),
        hit_rate=float((s > 0).mean()), total=float(s.sum()), max_drawdown=dd,
        best=float(s.max()), worst=float(s.min()), by_year=by_year,
    )
