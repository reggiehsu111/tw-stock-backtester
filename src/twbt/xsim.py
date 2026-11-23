"""Cross-sectional simulation: a signal matrix becomes a book, and a book a P&L.

The shape of the output mirrors the research workflow this project borrows its
vocabulary from: total / long / short P&L, a per-year view with each year's
Sharpe, and the average path across years.

Three things the engine does that a naive `(weights * returns).sum()` does not.

EXECUTION LAG. A signal computed from today's close cannot be traded at today's
close. ``exec_shift`` defaults to **1** — decide on t, hold over t+1 — and the
engine owns the shift so a signal author cannot apply it twice or not at all.
Setting it to 0 is permitted but prints as "SAME-CLOSE EXECUTION" in the result,
because it is not a backtest of anything tradable.

TRADABILITY. Taiwan has 漲停/跌停 locks and 平盤下不得放空. A blocked trade does
not happen, so yesterday's weight carries forward; the engine does this rather
than dropping the name, because dropping it silently rebalances into whatever
*was* tradable — a different and much better strategy than the one you tested.

ASYMMETRIC COST. The 0.30% 證交稅 falls on sells only, so the engine costs buys
and sells separately from the signed weight change, and accrues borrow on the
short side per day held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import i18n, xcosts
from .ops import GrossScale

TRADING_DAYS = 252


def Sharpe(x: pd.Series) -> float:
    """Annualised Sharpe of a daily return series."""
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.std(ddof=0) == 0:
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * s.mean() / s.std(ddof=0))


def PnlParts(w: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Total, long and short P&L series from weights and returns.

    The short leg is returned as its **contribution** — positive means the shorts
    made money — which is what you want when reading the two sides against each
    other.
    """
    tot = (w * r).sum(axis=1)
    lng = (w.clip(lower=0) * r).sum(axis=1)
    sht = (w.clip(upper=0) * r).sum(axis=1)
    return tot, lng, sht


def AvgPath(x: pd.Series, min_frac: float = 0.6) -> pd.Series:
    """Average cumulative path across calendar years.

    Each year is re-based to zero at its first session and the paths averaged by
    session-of-year. Years shorter than ``min_frac`` of the longest are dropped,
    so a two-week stub at the end of the sample does not drag the tail around.
    """
    s = pd.Series(x).dropna()
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    paths = {int(y): g.cumsum().reset_index(drop=True)
             for y, g in s.groupby(s.index.year)}
    if not paths:
        return pd.Series(dtype=float)
    longest = max(len(v) for v in paths.values())
    keep = {k: v for k, v in paths.items() if len(v) >= min_frac * longest}
    return pd.DataFrame(keep or paths).mean(axis=1, skipna=True)


@dataclass
class XSimResult:
    """Everything the tearsheet needs from one cross-sectional backtest."""

    gross: pd.Series                 # daily, before costs
    net: pd.Series                   # daily, after costs
    long_pnl: pd.Series
    short_pnl: pd.Series
    cost: pd.Series
    turnover: pd.Series              # one-way, as a fraction of gross
    coverage: pd.Series              # names held per date
    weights: pd.DataFrame
    exec_shift: int
    notes: list[str] = field(default_factory=list)

    # diagnostics. `raw_gross` is the P&L of the book you INTENDED, before the
    # tradability rules took it away, so the gap between it and `gross` is the
    # cost of 漲停/跌停 and 平盤下不得放空 -- a number usually left unmeasured.
    raw_gross: pd.Series | None = None
    long_count: pd.Series | None = None
    short_count: pd.Series | None = None
    blocked: dict = field(default_factory=dict)
    signal: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None

    @property
    def limit_drag(self) -> pd.Series:
        """Daily P&L lost to tradability rules."""
        if self.raw_gross is None:
            return pd.Series(0.0, index=self.gross.index)
        return self.raw_gross - self.gross

    # ---- headline statistics -------------------------------------------- #
    @property
    def sharpe(self) -> float: return Sharpe(self.net)

    @property
    def gross_sharpe(self) -> float: return Sharpe(self.gross)

    @property
    def t_stat(self) -> float:
        s = self.net.dropna()
        sd = s.std(ddof=1)
        return float(s.mean() / (sd / np.sqrt(len(s)))) if len(s) > 2 and sd > 0 else np.nan

    @property
    def ann_return(self) -> float: return float(self.net.mean() * TRADING_DAYS)

    @property
    def max_drawdown(self) -> float:
        eq = self.net.cumsum()
        return float((eq - eq.cummax()).min())

    @property
    def hit_rate(self) -> float: return float((self.net > 0).mean())

    @property
    def avg_turnover(self) -> float: return float(self.turnover.mean())

    @property
    def cost_share(self) -> float:
        """Fraction of the gross edge eaten by costs. Above 1.0 means it is gone."""
        g = self.gross.mean()
        return float(self.cost.mean() / g) if g > 0 else np.nan

    def by_year(self) -> pd.DataFrame:
        rows = []
        for y, g in self.net.groupby(self.net.index.year):
            gg = self.gross.loc[g.index]
            rows.append({"year": int(y), "days": len(g),
                         "gross_ret": float(gg.sum()), "net_ret": float(g.sum()),
                         "sharpe": Sharpe(g), "hit": float((g > 0).mean())})
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        return {
            "sharpe": self.sharpe, "gross_sharpe": self.gross_sharpe,
            "t_stat": self.t_stat, "ann_return": self.ann_return,
            "max_drawdown": self.max_drawdown, "hit_rate": self.hit_rate,
            "avg_turnover": self.avg_turnover, "cost_share": self.cost_share,
            "days": int(len(self.net)), "avg_names": float(self.coverage.mean()),
        }

    # Verdict lines that report a failure. Kept as a set of KEYS rather than
    # detected from the rendered text: a previous version matched on the prefix
    # "FAILS", which meant any rewording -- or translating the sentence at all --
    # silently stopped the line being highlighted as a failure.
    FAIL_KEYS = frozenset({"same_close", "no_edge", "cost_kills"})

    def verdict_items(self, lang: str = "en") -> list[dict]:
        """Findings as {key, text, fail}, written to be unflattering where true.

        Takes a language because these sentences are the most-read output on the
        page, and a tool that reports its own failures only in English is less
        useful to the people who most need to read them.
        """
        def item(key, **kw):
            return {"key": key,
                    "text": i18n.t(i18n.VERDICT, key, lang).format(**kw),
                    "fail": key in self.FAIL_KEYS}

        out = []
        for n in self.notes:
            if isinstance(n, tuple):
                out.append(item(n[0], **n[1]))
            else:                                   # a plain pre-rendered string
                out.append({"key": None, "text": str(n), "fail": False})

        if self.exec_shift == 0:
            out.append(item("same_close"))

        g, n = self.gross.mean(), self.net.mean()
        if g <= 0:
            out.append(item("no_edge"))
        elif n <= 0:
            out.append(item("cost_kills", share=self.cost.mean() / g,
                            turn=self.avg_turnover))
        else:
            out.append(item("net_positive", sharpe=self.sharpe, t=self.t_stat,
                            share=self.cost_share))

        yr = self.by_year()
        if len(yr) > 1:
            pos = int((yr["net_ret"] > 0).sum())
            out.append(item("years", pos=pos, n=len(yr)))
            if pos <= 1 and len(yr) >= 3:
                out.append(item("one_year"))

        lng, sht = self.long_pnl.sum(), self.short_pnl.sum()
        if lng != 0 or sht != 0:
            dom = ("side_long" if abs(lng) > abs(sht) * 3
                   else ("side_short" if abs(sht) > abs(lng) * 3 else None))
            if dom:
                out.append(item("one_side",
                                side=i18n.t(i18n.VERDICT, dom, lang),
                                lng=lng, sht=sht))
        return out

    def verdict(self, lang: str = "en") -> list[str]:
        """The same findings, as plain sentences."""
        return [i["text"] for i in self.verdict_items(lang)]


def simulate(sig: pd.DataFrame, px: pd.DataFrame,
             exec_shift: int = 1,
             cost: xcosts.TWEquityCost | None = None,
             gmv: float = 1.0,
             enforce_tradability: bool = True,
             market_neutral: bool = True) -> XSimResult:
    """Run a signal matrix as a book and return its P&L.

    ``sig`` and ``px`` are wide matrices on the same index and columns. The signal
    is scaled to a constant gross exposure so the P&L reflects the view rather
    than how large the book happened to be that day, lagged by ``exec_shift``,
    passed through the tradability rules, and then charged asymmetric costs.
    """
    px = px.reindex_like(sig)
    ret = px.pct_change()
    cost = cost or xcosts.TWEquityCost()
    notes: list[str] = []

    w = sig.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if market_neutral:
        # only over names with a live signal, so the neutralisation is not
        # dragged by a few thousand zeros
        live = sig.notna() & (sig != 0)
        n_live = live.sum(axis=1).replace(0, np.nan)
        w = w.sub((w.where(live).sum(axis=1) / n_live).fillna(0.0), axis=0).where(live, 0.0)
    w = GrossScale(w, gmv)

    if exec_shift < 0:
        raise ValueError(f"exec_shift must be >= 0; got {exec_shift}")
    held = w.shift(exec_shift).fillna(0.0)

    intended = held.copy()
    info: dict = {}
    if enforce_tradability:
        # lag >= 1: the rules must be assessed with prices known when the trade
        # was placed, never over the period the position earns its return
        held, info = xcosts.apply_tradability(
            held, px, lag=max(exec_shift, 1), return_info=True)
        moved = (held - intended).abs().sum().sum() / max(
            intended.abs().sum().sum(), 1e-9)
        if moved > 0.001:
            notes.append(("tradability", {"moved": moved}))

    r0 = ret.fillna(0.0)
    gross, lng, sht = PnlParts(held, r0)
    # the P&L of the book you INTENDED: the gap is what the locks cost you
    raw_gross = (intended * r0).sum(axis=1)
    dw = held.diff().fillna(held)
    trade_cost = cost.trade_cost(dw)
    borrow = cost.borrow_cost(held)
    total_cost = (trade_cost + borrow).reindex(gross.index).fillna(0.0)

    return XSimResult(
        gross=gross, net=gross - total_cost, long_pnl=lng, short_pnl=sht,
        cost=total_cost, turnover=dw.abs().sum(axis=1) / 2.0,
        coverage=(held != 0).sum(axis=1), weights=held,
        exec_shift=exec_shift, notes=notes,
        raw_gross=raw_gross,
        long_count=(held > 0).sum(axis=1), short_count=(held < 0).sum(axis=1),
        blocked=info, signal=sig, returns=r0)


# --------------------------------------------------------------------------- #
# the extra analyses a full tearsheet shows
# --------------------------------------------------------------------------- #

def lag_sweep(sig: pd.DataFrame, px: pd.DataFrame, lags=(1, 2, 3, 5, 10),
              **kw) -> dict:
    """Re-run the same signal at several execution lags.

    The decay shape is diagnostic. An edge that survives a few days of delay is
    a slow effect you could actually implement; one that is gone by lag 2 needed
    to be executed on the close it was computed from, which nobody can do. It
    also catches the opposite problem: a "signal" whose P&L is identical at
    every lag is usually tracking something static, like size.
    """
    out = {}
    for k in lags:
        try:
            r = simulate(sig, px, exec_shift=k, **kw)
        except Exception:                                # noqa: BLE001
            continue
        out[int(k)] = (r.net.cumsum(), Sharpe(r.net))
    return out


def bucket_pnl(sig: pd.DataFrame, ret: pd.DataFrame, n: int = 5,
               exec_shift: int = 1) -> dict:
    """Equal-weighted P&L of each quantile of the signal's own cross-section.

    Long/short construction assumes the relationship is monotone across the
    cross-section. Often it is not: the whole effect lives in one tail, and the
    middle buckets are noise. That is a different and much more fragile object
    than a monotone factor, and it is invisible in the headline Sharpe.
    """
    r = sig.rank(axis=1, pct=True)
    fwd = ret.shift(-exec_shift) if exec_shift else ret
    out = {}
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        m = (r > lo) & (r <= hi) if i else (r >= 0) & (r <= hi)
        w = m.astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
        out[f"Q{i + 1}"] = (w * fwd).sum(axis=1).fillna(0.0).cumsum()
    return out


def group_pnl(weights: pd.DataFrame, ret: pd.DataFrame,
              group: pd.DataFrame) -> dict:
    """P&L split by a categorical label per name, e.g. sector or listing board.

    Two paths. When the labels are constant down the date axis -- which is true
    of both sector and board, because those come from a snapshot -- the whole
    split is one column-wise groupby over a single contribution matrix.

    The obvious implementation instead masks the matrix once per label. With 35
    sectors over a 698 x 2015 panel that is 35 passes over 1.4M cells, and it
    measured 3.3 seconds against 0.1 for this version. On a page that recomputes
    on every edit, that difference is the difference between a tool and a wait.
    """
    g = group.reindex_like(weights)
    contrib = (weights * ret).fillna(0.0)
    gross_total = weights.abs().sum().sum()

    first = g.iloc[0]
    constant = bool((g.eq(first, axis=1) | g.isna()).all().all())

    out: dict[str, pd.Series] = {}
    if constant:
        labels = first.astype(str)
        shares = weights.abs().sum().groupby(labels).sum()
        pnl = contrib.T.groupby(labels).sum().T
        for label in pnl.columns:
            if label in ("nan", "None", "—"):
                continue
            share = shares.get(label, 0.0) / max(gross_total, 1e-9)
            out[f"{label} ({share:.0%})"] = pnl[label].cumsum()
        return out

    # labels that genuinely change over time: fall back to masking
    for label in pd.unique(g.to_numpy().ravel()):
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        m = (g == label)
        share = weights.where(m).abs().sum().sum() / max(gross_total, 1e-9)
        out[f"{label} ({share:.0%})"] = contrib.where(m, 0.0).sum(axis=1).cumsum()
    return out


def top_names(weights: pd.DataFrame, ret: pd.DataFrame, n: int = 5,
              side: str = "long") -> dict:
    """Cumulative P&L of the n names that contributed most on one side.

    Worth looking at because a cross-sectional strategy is supposed to be a
    statement about many names. If five of them account for most of the result,
    it is a handful of stock picks wearing a factor's clothes, and the Sharpe is
    far less meaningful than it looks.
    """
    w = weights.clip(lower=0) if side == "long" else weights.clip(upper=0)
    contrib = (w * ret).fillna(0.0)
    total = contrib.sum().sort_values(ascending=(side == "short"))
    picks = list(total.index[:n])
    return {str(t): contrib[t].cumsum() for t in picks}


def concentration(weights: pd.DataFrame) -> dict:
    """How concentrated the book is, two ways.

    ``max_weight`` is the largest single position as a share of gross, and
    ``top10_share`` the ten largest together. Both are the question "would one
    name's bad week undo this?", which the headline statistics never answer.
    """
    g = weights.abs().sum(axis=1).replace(0, np.nan)
    aw = weights.abs()
    return {
        "max_weight": (aw.max(axis=1) / g).fillna(0.0),
        "top10_share": (aw.apply(lambda r: r.nlargest(10).sum(), axis=1)
                        / g).fillna(0.0),
    }


def rolling_sharpe(net: pd.Series, window: int = 252) -> pd.Series:
    """Trailing one-year Sharpe.

    A single full-sample Sharpe hides the shape. This shows whether the edge was
    present throughout or arrived once and decayed, which is the difference
    between a strategy and an episode.
    """
    s = pd.Series(net).dropna()
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std(ddof=0)
    return (mu / sd.where(sd > 0) * np.sqrt(TRADING_DAYS)).dropna()


def cost_sweep(sig: pd.DataFrame, px: pd.DataFrame,
               cost: "xcosts.TWEquityCost", factors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
               **kw) -> pd.DataFrame:
    """Net Sharpe as the discretionary costs are scaled.

    The statutory 證交稅 does not move, so even factor 0 is not free. A strategy
    whose Sharpe crosses zero between 1.0 and 1.5 is not a strategy: the
    realistic cost estimate and a mildly pessimistic one disagree about whether
    it exists.
    """
    rows = []
    for f in factors:
        try:
            r = simulate(sig, px, cost=cost.scaled(f), **kw)
        except Exception:                                # noqa: BLE001
            continue
        rows.append({"factor": f, "cost_bps": cost.scaled(f).round_trip_bps,
                     "sharpe": r.sharpe, "mean": r.net.mean()})
    return pd.DataFrame(rows)


def hold_sweep(sig: pd.DataFrame, px: pd.DataFrame, holds=(1, 2, 5, 10, 20),
               **kw) -> pd.DataFrame:
    """Net Sharpe when the signal is held for h days instead of rebalanced daily.

    Holding longer cuts turnover, and at 62 bps a round trip turnover is usually
    what kills a Taiwanese cross-sectional strategy. A signal that improves as
    the hold lengthens was being destroyed by its own trading, not by its view.
    """
    rows = []
    for h in holds:
        held = sig if h <= 1 else sig.rolling(h, min_periods=1).mean()
        try:
            r = simulate(held, px, **kw)
        except Exception:                                # noqa: BLE001
            continue
        rows.append({"hold": h, "sharpe": r.sharpe, "turnover": r.avg_turnover,
                     "gross_sharpe": r.gross_sharpe})
    return pd.DataFrame(rows)


def factor_exposures(weights: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """Book exposure to each factor over time, as a weighted average of ranks.

    Exposure is computed from the holdings rather than by regressing the P&L,
    which is the honest version: it says what the book WAS, not what explains
    its returns after the fact.

    Each factor is cross-sectionally ranked and centred, so an exposure of zero
    means the book is factor-neutral and +0.5 means it sits at the top of the
    cross-section. Reading it is the quickest way to find out that your clever
    signal is a size bet.
    """
    from .ops import Rank
    out = {}
    g = weights.abs().sum(axis=1).replace(0, np.nan)
    for name, mx in factors.items():
        r = Rank(mx.reindex_like(weights)) - 0.5
        out[name] = ((weights * r).sum(axis=1) / g).fillna(0.0)
    return pd.DataFrame(out)


def entry_dates(weights: pd.DataFrame, side: str = "long",
                quantile: float = 0.8, min_count: int = 1) -> pd.DatetimeIndex:
    """Sessions where the book opened unusually many new positions on one side.

    An event axis that needs no external data: the signal defines its own
    events, and unlike earnings dates it is always available.

    Two things this is NOT, both of which were tried first.

    It is not "the number of names held rose". A cross-sectional book holds a
    roughly constant count while rotating which names it holds, so that
    definition found one event in six hundred sessions on a book that was
    trading every day.

    It is not "any name was opened" either. A daily-rebalanced book opens
    something almost every session, and an event study where every day is an
    event is just the unconditional mean with extra steps.

    So: count names going from flat to held, and keep the sessions where that
    count sits in the top ``quantile`` of its own distribution. Those are the
    days the signal actually did something, which is the question an event
    study is for.
    """
    w = weights.clip(lower=0) if side == "long" else weights.clip(upper=0)
    live = w != 0
    opened = (live & ~live.shift(1, fill_value=False)).sum(axis=1)
    opened = opened[opened >= max(min_count, 1)]
    if opened.empty:
        return pd.DatetimeIndex([])
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")
    thr = opened.quantile(quantile)
    hits = opened[opened >= thr]
    # a threshold that selects everything is not selecting
    if len(hits) > 0.6 * len(weights):
        hits = opened.nlargest(max(10, int(0.2 * len(opened))))
    return pd.DatetimeIndex(sorted(hits.index))


def event_caar(pnl: pd.Series, dates, before: int = 10, after: int = 10) -> dict:
    """CAAR of a strategy's own P&L around a set of dates.

    Applied to the P&L rather than to raw returns, so it answers "does this
    strategy behave differently around these dates" instead of "does the market".
    """
    from .events import caar
    return caar(pnl, dates, before, after)
