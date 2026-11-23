"""Matrix operators for cross-sectional research.

Every operator here takes a **wide matrix** — rows are trading days, columns are
tickers — and returns one of the same shape. That uniformity is the whole point:
operators compose, so a signal is written as a pipeline rather than as a loop.

    sig = LongShortBalance(Rank(Demean(InstRank(20, Returns(px)))))

The names and argument orders deliberately match the ``f`` research framework, so
a signal written against one reads the same against the other. This is an
independent implementation over public data, not a copy of that package.

Note the argument order on the time-series operators: the window comes **first**,
``Lag(1, mx)`` rather than ``Lag(mx, 1)``. It looks backwards next to pandas and is
kept because it is what the research vocabulary uses, and a silent argument swap
between two libraries that both accept two arguments is an ugly bug to find.

THE RULE THAT MATTERS. No operator may see the future. Every rolling window ends
at the current row, ``Lag`` shifts forward in time only, and ``Lead`` — which does
look ahead — exists solely to build the *target* of a study, never a signal.
``tests/test_ops_no_lookahead.py`` corrupts the back half of a matrix and asserts
the front half is bit-identical, for every operator in ``ALL_OPS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# time series: along the date axis, within each column
# --------------------------------------------------------------------------- #


def Lag(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Value from n rows earlier. The workhorse for making a signal tradable."""
    return mx.shift(n)


def Lead(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Value from n rows later.

    This is the one operator that looks ahead, and it is here to build the thing
    a signal is *tested against* — a forward return — not to build the signal.
    Using it inside a signal is look-ahead by definition, and the builder refuses
    to run a pipeline containing it.
    """
    return mx.shift(-n)


# Half the window before a trailing statistic is reported. The exact fraction is
# arbitrary; what matters is that it is the SAME fraction everywhere, so that
# "window 20" means one thing across the whole operator set.
MIN_FRACTION = 2


def _minp(n: int, floor: int = 2) -> int:
    return max(floor, n // MIN_FRACTION)


def InstMean(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Trailing mean over n rows, ending at the current row."""
    return mx.rolling(n, min_periods=_minp(n)).mean()


def InstStdev(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Trailing standard deviation over n rows."""
    return mx.rolling(n, min_periods=_minp(n, 2)).std(ddof=1)


def InstSkew(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Trailing skewness over n rows."""
    return mx.rolling(n, min_periods=_minp(n, 3)).skew()


def InstCount(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Count of finite observations in the trailing n rows."""
    return mx.notna().rolling(n, min_periods=1).sum()


def InstSum(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Trailing sum over n rows."""
    return mx.rolling(n, min_periods=_minp(n, 2)).sum()


def InstRank(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Where the current value sits inside its own trailing window, in [0, 1].

    The kernel is n+1 wide and ranks its last element against the n before it, so
    day t is compared with days t-1 and earlier and never with itself. This is the
    time-series analogue of ``Rank``, and the two are easy to confuse: ``InstRank``
    asks "is this name unusual *for itself*", ``Rank`` asks "is this name unusual
    *among its peers today*".
    """
    def _pos(w):
        prior = w[:-1]
        ok = np.isfinite(prior)
        if ok.sum() < 2 or not np.isfinite(w[-1]):
            return np.nan
        return float((prior[ok] < w[-1]).mean())

    return mx.rolling(n + 1, min_periods=3).apply(_pos, raw=True)


def InstZscore(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """(x - trailing mean) / trailing sd, using rows strictly before the current.

    Shifts before the window so the current observation does not set the scale it
    is measured against — a z-score that includes itself is pulled toward zero
    exactly when the value is extreme, which is when you care.
    """
    base = mx.shift(1)
    mu = base.rolling(n, min_periods=_minp(n)).mean()
    sd = base.rolling(n, min_periods=_minp(n)).std(ddof=1)
    return (mx - mu) / sd.where(sd > 0)


def InstBeta(n: int, mx: pd.DataFrame, benchmark) -> pd.DataFrame:
    """Trailing beta of each column against a benchmark.

    cov(x, b) / var(b) over the trailing n rows, column-wise, so a name with a
    short history gets NaN rather than a beta borrowed from its neighbours.

    ``benchmark`` may be a Series or a matrix. A matrix is collapsed to its
    cross-sectional mean, i.e. the equal-weighted market, which is both the
    thing you usually want and the only form the expression editor can supply —
    there is no Series-valued source, so a Series-only signature made this
    operator impossible to type.
    """
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.mean(axis=1)
    b = pd.Series(benchmark).reindex(mx.index).astype(float)
    mp = max(10, n // 3)
    var = b.rolling(n, min_periods=mp).var(ddof=1)
    cov = mx.rolling(n, min_periods=mp).cov(b)
    return cov.div(var.where(var > 0), axis=0)


def ForwardFill(limit: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Carry values forward, at most ``limit`` rows.

    Bounded on purpose. An unbounded fill turns a name that stopped reporting into
    a permanent stale position, and the backtest happily trades it for years.
    """
    return mx.ffill(limit=limit)


def Returns(px: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close return per name, as a fraction."""
    return px.pct_change()


def LogReturns(px: pd.DataFrame) -> pd.DataFrame:
    """Log close-to-close return per name. Additive across time, unlike Returns.

    A non-positive ratio becomes NaN rather than -inf plus a RuntimeWarning.
    That happens on a delisting, on a bad tick, and whenever this operator is
    pointed at a series that is not a price -- all cases where a blank is the
    honest answer and an infinity silently poisons every later operator.
    """
    ratio = px / px.shift(1)
    # `> 0` alone is not enough: a zero previous price gives inf, and inf > 0
    # is True, so log() returns inf and the guard does nothing.
    return np.log(ratio.where((ratio > 0) & np.isfinite(ratio)))


# --------------------------------------------------------------------------- #
# cross section: across names, within each date
# --------------------------------------------------------------------------- #


def Rank(mx: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank within each date, scaled to [0, 1].

    Rank rather than z-score is the usual default for a cross-section because one
    name limit-up at +10% would otherwise dominate the day's standard deviation.
    """
    return mx.rank(axis=1, pct=True)


def Demean(mx: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean of each date.

    Applied to returns this is the cheapest possible market-neutralisation: what
    is left is the part of a name's move that the whole market did not explain.
    """
    return mx.sub(mx.mean(axis=1), axis=0)


def Zscore(mx: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score within each date."""
    sd = mx.std(axis=1, ddof=1)
    return mx.sub(mx.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0)


def DemeanByGroup(mx: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Subtract the mean of each name's group, per date.

    ``group`` is a matrix of labels with the same shape. Neutralising by sector
    rather than by the whole market is what separates "this chemical stock is
    cheap" from "chemicals are cheap", and those are different trades.
    """
    out = pd.DataFrame(np.nan, index=mx.index, columns=mx.columns)
    g = group.reindex_like(mx)
    for dt in mx.index:
        row, lbl = mx.loc[dt], g.loc[dt]
        out.loc[dt] = row - row.groupby(lbl).transform("mean")
    return out


def GroupRank(mx: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Rank within each name's group, per date, scaled to [0, 1]."""
    out = pd.DataFrame(np.nan, index=mx.index, columns=mx.columns)
    g = group.reindex_like(mx)
    for dt in mx.index:
        row, lbl = mx.loc[dt], g.loc[dt]
        out.loc[dt] = row.groupby(lbl).rank(pct=True)
    return out


def Winsorize(lo_q: float, hi_q: float, mx: pd.DataFrame) -> pd.DataFrame:
    """Clip each date's cross-section to the given quantiles.

    Quantiles are taken per date, so this never uses tomorrow's distribution.
    """
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(f"need 0 <= lo_q < hi_q <= 1; got {lo_q}, {hi_q}")
    lo = mx.quantile(lo_q, axis=1)
    hi = mx.quantile(hi_q, axis=1)
    return mx.clip(lower=lo, upper=hi, axis=0)


# --------------------------------------------------------------------------- #
# elementwise
# --------------------------------------------------------------------------- #


def Abs(mx):
    """Absolute value."""
    return mx.abs()


def Log(mx):
    """Natural log; non-positive values become NaN rather than -inf."""
    return np.log(mx.where(mx > 0))


def Sqrt(mx):
    """Square root; negative values become NaN."""
    return np.sqrt(mx.where(mx >= 0))


def Sign(mx):
    """-1, 0 or +1. Throws away magnitude, which is often the point."""
    return np.sign(mx)


def Pow(mx, p):
    """Raise to a power."""
    return mx.pow(p)


def Neg(mx):
    """Flip the sign. Turns a momentum signal into a reversal one."""
    return -mx


def Clip(lo: float, hi: float, mx: pd.DataFrame) -> pd.DataFrame:
    """Clamp to a fixed range. Unlike Winsorize the bounds are absolute, not quantiles."""
    return mx.clip(lower=lo, upper=hi)


def Div(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """Elementwise division that yields NaN rather than inf on a zero divisor."""
    d = denominator.replace(0, np.nan) if hasattr(denominator, "replace") else denominator
    return numerator / d


def Finite(mx: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN. The guard before anything that sums or averages."""
    return mx.replace([np.inf, -np.inf], np.nan)


def And(left, right):
    """1.0 where both matrices are truthy, else 0.0. For combining filters."""
    return (left.astype(bool) & right.astype(bool)).astype(float)


def Or(left, right):
    """1.0 where either matrix is truthy, else 0.0."""
    return (left.astype(bool) | right.astype(bool)).astype(float)


def Mask(mx: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    """Keep values where ``condition`` is truthy, NaN elsewhere."""
    return mx.where(condition.astype(bool))


# --------------------------------------------------------------------------- #
# portfolio construction
# --------------------------------------------------------------------------- #


def LongShortBalance(mx: pd.DataFrame) -> pd.DataFrame:
    """Scale each date to +0.5 long and -0.5 short, so gross is 1.0 and net is 0.

    This was previously implemented as ``mx.sub(mx.mean(axis=1), axis=0)`` --
    byte-identical to ``Demean``. Two registry entries did the same thing, and
    the name promised something neither of them delivered.

    Subtracting the mean does make the two sides equal in gross, because the
    positive and negative parts of a mean-zero row have the same magnitude. What
    it does not do is fix the LEVEL: gross varies row to row with the dispersion
    of the signal, so a day when the cross-section happened to be wide takes a
    bigger book than a day when it was narrow, and the P&L mixes the view with
    how large the bet was.

    Scaling each side separately fixes both: dollar-neutral and unit-gross.

    **It does not centre the input, and that matters.** Given an all-positive
    matrix -- which is what ``Rank`` produces -- there is nothing to put on the
    short side, and the result is a long-only book with gross 0.5 and net +0.5.
    Centre first: ``LongShortBalance(Demean(Rank(x)))``, not
    ``LongShortBalance(Rank(x))``. The builder warns when the second shape is
    assembled, because the failure is silent otherwise.
    """
    pos = mx.clip(lower=0)
    neg = mx.clip(upper=0)
    return (pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).mul(0.5).fillna(0.0)
            + neg.div(neg.sum(axis=1).replace(0, np.nan), axis=0).mul(-0.5).fillna(0.0))


def GrossScale(mx: pd.DataFrame, gmv: float = 1.0) -> pd.DataFrame:
    """Scale each date to a target gross exposure, so |weights| sums to ``gmv``.

    Without this, a signal's P&L is confounded by how large the book happened to
    be that day, and a quiet period looks like skill.
    """
    g = mx.abs().sum(axis=1).replace(0, np.nan)
    return mx.div(g, axis=0).mul(gmv).fillna(0.0)


def TopN(n: int, mx: pd.DataFrame) -> pd.DataFrame:
    """Keep only the n largest and n smallest values per date, NaN elsewhere.

    The honest way to express "I would only trade my best names", and it changes
    the result a great deal — concentration raises both the mean and the variance.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    r = mx.rank(axis=1, ascending=False, na_option="keep")
    rr = mx.rank(axis=1, ascending=True, na_option="keep")
    return mx.where((r <= n) | (rr <= n))


def Neutralize(mx: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Residual of a cross-sectional regression of mx on factor, per date.

    Use it to answer "is there anything left after the obvious explanation?" — the
    question that killed several signals in the research behind this project. A
    signal that vanishes under ``Neutralize(sig, size)`` was a size bet.
    """
    out = pd.DataFrame(np.nan, index=mx.index, columns=mx.columns)
    f = factor.reindex_like(mx)
    for dt in mx.index:
        y, x = mx.loc[dt], f.loc[dt]
        ok = y.notna() & x.notna()
        if ok.sum() < 10:
            continue
        yv, xv = y[ok].to_numpy(float), x[ok].to_numpy(float)
        X = np.column_stack([np.ones(len(xv)), xv])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        out.loc[dt, y[ok].index] = yv - X @ beta
    return out


# --------------------------------------------------------------------------- #
# registry, for the interactive builder
# --------------------------------------------------------------------------- #


class Op:
    """One operator as the builder sees it: name, params, and a description."""

    def __init__(self, fn, group, params=(), doc="", lookahead=False,
                 matrix_args=1, hint=""):
        self.fn = fn
        self.name = fn.__name__
        self.group = group
        self.params = params           # (param_name, type, default) before the matrix
        self.doc = doc or (fn.__doc__ or "").strip().split("\n")[0]
        # A plain-English line written for the palette tooltip. The docstring is
        # written for someone reading the source; this is written for someone
        # deciding whether to drag the block, which is a different sentence.
        self.hint = hint or self.doc
        self.lookahead = lookahead
        # How many MATRIX inputs this operator takes. Anything above one cannot
        # appear on the block canvas, which draws a linear chain, but is
        # perfectly usable in the expression editor.
        self.matrix_args = matrix_args

    @property
    def signature(self) -> str:
        """The call as you would type it, read off the real function."""
        from inspect import Parameter, signature
        ps = [p for p in signature(self.fn).parameters.values()
              if p.kind in (Parameter.POSITIONAL_ONLY,
                            Parameter.POSITIONAL_OR_KEYWORD)]
        parts = [p.name if p.default is Parameter.empty
                 else f"{p.name}={p.default!r}" for p in ps]
        return f"{self.name}({', '.join(parts)})"

    def as_dict(self) -> dict:
        return {"name": self.name, "group": self.group, "doc": self.doc,
                "hint": self.hint, "signature": self.signature,
                "lookahead": self.lookahead, "matrix_args": self.matrix_args,
                "params": [{"name": n, "type": t.__name__, "default": d}
                           for n, t, d in self.params]}


_TS = "time series"
_XS = "cross section"
_EW = "elementwise"
_PC = "portfolio"

ALL_OPS: dict[str, Op] = {o.name: o for o in [
    Op(Lag, _TS, [("n", int, 1)], hint='Push the signal n days later. Use it to make a signal tradable, or to test whether the edge survives delay.'),
    Op(Lead, _TS, [("n", int, 1)], lookahead=True, hint='Look n days AHEAD. Only for building the thing you are predicting — never inside a signal.'),
    Op(InstMean, _TS, [("n", int, 20)], hint="Average of each name's own last n days. Smooths a noisy input."),
    Op(InstStdev, _TS, [("n", int, 20)], hint="Each name's own volatility over the last n days. The usual denominator for risk-scaling."),
    Op(InstSkew, _TS, [("n", int, 60)], hint="Skewness of each name's last n days. Lottery-like names score high."),
    Op(InstCount, _TS, [("n", int, 20)], hint='How many of the last n days actually had data. Use it to drop names with thin history.'),
    Op(InstSum, _TS, [("n", int, 20)], hint="Total over each name's last n days. InstSum(5, Returns(px)) is last week's return."),
    Op(InstRank, _TS, [("n", int, 60)], hint="Where today sits inside this name's OWN last n days, 0 to 1. Asks 'unusual for this stock?'"),
    Op(InstZscore, _TS, [("n", int, 60)], hint='How many of its own standard deviations today is from its own mean. Comparable across names.'),
    Op(ForwardFill, _TS, [("limit", int, 5)], hint='Carry the last value forward, at most n days. Bounded so a dead name does not become a permanent position.'),
    Op(Returns, _TS, hint='Day-over-day percentage change. The usual starting point.'),
    Op(LogReturns, _TS, hint='Log change. Adds up cleanly over time, unlike plain returns.'),

    Op(Rank, _XS, hint='Rank every name against the OTHER NAMES on the same day, 0 to 1. The cross-sectional workhorse.'),
    Op(Demean, _XS, hint="Subtract the day's average across names. Removes the market move, leaving the relative one."),
    Op(Zscore, _XS, hint='Standardise across names within each day. Like Rank but keeps magnitude, so outliers still shout.'),
    Op(Winsorize, _XS, [("lo_q", float, 0.01), ("hi_q", float, 0.99)], hint="Clip each day's cross-section at the given quantiles. Tames outliers without deleting them."),

    Op(Abs, _EW, hint='Drop the sign, keep the size.'),
    Op(Log, _EW, hint='Natural log. Compresses a long-tailed input like turnover. Non-positive values become blank.'),
    Op(Sqrt, _EW, hint='Square root. A gentler compression than Log.'),
    Op(Sign, _EW, hint='Reduce to -1, 0 or +1. Throws away magnitude, which is often what you want.'),
    Op(Neg, _EW, hint='Drop the sign, keep the size.'),
    Op(Clip, _EW, [("lo", float, -3.0), ("hi", float, 3.0)], hint='Clamp to fixed numeric bounds. Unlike Winsorize these are absolute, not quantiles.'),
    Op(Finite, _EW, hint='Turn infinities into blanks. Put it after a division.'),

    Op(LongShortBalance, _PC, hint='Shift each day so longs and shorts are equal. The book stops being a bet on market direction.'),
    Op(GrossScale, _PC, hint='Scale each day so the position sizes add to a constant. P&L then reflects the view, not the book size.'),
    Op(TopN, _PC, [("n", int, 50)], hint='Keep only the n best and n worst names each day. Concentrates the book — higher mean, much higher variance.'),

    # Two-matrix operators. Expression editor only -- a linear chain has one
    # input, so these cannot be a block on the canvas.
    Op(Div, _EW, matrix_args=2, hint='Divide one matrix by another, blanking a zero denominator rather than returning infinity.'),
    Op(And, _EW, matrix_args=2, hint='1 where BOTH inputs are non-zero. For combining two filters.'),
    Op(Or, _EW, matrix_args=2, hint='1 where EITHER input is non-zero.'),
    Op(Mask, _EW, matrix_args=2, hint='Keep the first matrix only where the second is non-zero. Blank elsewhere.'),
    Op(Neutralize, _XS, matrix_args=2, hint="Regress the signal on a factor each day and keep the residual. Answers 'is anything left after the obvious explanation?'"),
    Op(DemeanByGroup, _XS, matrix_args=2, hint="Subtract the group average instead of the market average. Separates 'cheap stock' from 'cheap sector'."),
    Op(GroupRank, _XS, matrix_args=2, hint="Rank within each name's group rather than the whole market."),
    Op(InstBeta, _TS, [("n", int, 60)], matrix_args=2, hint="Each name's trailing beta to the market. Pass a matrix as the benchmark; its row-mean is the market."),
    Op(Pow, _EW, [("p", float, 2.0)], hint='Raise to a power. Above 1 emphasises extremes, below 1 flattens them.'),
]}

# Operators whose first argument is the matrix rather than a parameter.
#
# Nearly everything here takes its parameters first -- Lag(n, mx), TopN(n, mx),
# Winsorize(lo, hi, mx) -- matching the research vocabulary. These two do not.
# Getting the set wrong does not raise a TypeError, because both arguments are
# accepted positionally: TopN(mx, n) runs `if n < 1` against a DataFrame and
# dies several frames away with "truth value is ambiguous".
MATRIX_FIRST = {"GrossScale", "Pow"}


def apply_op(name: str, mx: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Apply a registered operator by name, with validated parameters.

    Raises on an unknown operator and on ``Lead``, which cannot appear in a signal
    without making the result meaningless.
    """
    if name not in ALL_OPS:
        raise KeyError(f"unknown operator {name!r}; have {sorted(ALL_OPS)}")
    op = ALL_OPS[name]
    if op.lookahead:
        raise ValueError(
            f"{name} looks into the future and cannot be part of a signal. "
            "It exists for building the target of a study, not the predictor.")

    params = params or {}
    args = []
    for pname, ptype, default in op.params:
        raw = params.get(pname, default)
        try:
            args.append(ptype(raw))
        except (TypeError, ValueError):
            raise ValueError(f"{name}: parameter {pname} must be "
                             f"{ptype.__name__}, got {raw!r}") from None

    if name in MATRIX_FIRST:
        return op.fn(mx, *args)
    return op.fn(*args, mx)


def catalogue(max_matrix_args: int | None = None) -> list[dict]:
    """Every operator, grouped, for the builder's palette.

    ``max_matrix_args=1`` restricts the result to operators that fit on the block
    canvas. The expression editor passes nothing and gets everything.
    """
    order = {_TS: 0, _XS: 1, _EW: 2, _PC: 3}
    items = [o.as_dict() for o in ALL_OPS.values()
             if max_matrix_args is None or o.matrix_args <= max_matrix_args]
    return sorted(items, key=lambda d: (order.get(d["group"], 9), d["name"]))
