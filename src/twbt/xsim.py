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

from . import xcosts
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

    def verdict(self) -> list[str]:
        """Plain-language findings, written to be unflattering where true."""
        out = list(self.notes)
        if self.exec_shift == 0:
            out.append("SAME-CLOSE EXECUTION: this is not tradable. A signal from "
                       "today's close cannot be executed at today's close.")
        if self.gross.mean() <= 0:
            out.append("FAILS: no edge even before costs.")
        elif self.net.mean() <= 0:
            out.append(f"FAILS on cost: gross is positive but costs take "
                       f"{self.cost.mean()/self.gross.mean():.0%} of it. "
                       f"Turnover is {self.avg_turnover:.1%} of gross per day.")
        else:
            out.append(f"Net positive: Sharpe {self.sharpe:.2f}, t={self.t_stat:+.2f}, "
                       f"costs eat {self.cost_share:.0%} of gross.")

        yr = self.by_year()
        if len(yr) > 1:
            pos = int((yr["net_ret"] > 0).sum())
            out.append(f"{pos}/{len(yr)} calendar years net positive.")
            if pos <= 1 and len(yr) >= 3:
                out.append("A result carried by one year is one observation, "
                           "not several.")

        lng, sht = self.long_pnl.sum(), self.short_pnl.sum()
        if lng != 0 or sht != 0:
            dom = "long" if abs(lng) > abs(sht) * 3 else (
                "short" if abs(sht) > abs(lng) * 3 else None)
            if dom:
                out.append(f"Almost all P&L comes from the {dom} side "
                           f"(long {lng:+.3f} vs short {sht:+.3f}). Check it is not "
                           f"a market-direction bet.")
        return out


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

    if enforce_tradability:
        before = held.copy()
        # lag >= 1: the rules must be assessed with prices known when the trade
        # was placed, never over the period the position earns its return
        held = xcosts.apply_tradability(held, px, lag=max(exec_shift, 1))
        blocked = (held - before).abs().sum().sum() / max(before.abs().sum().sum(), 1e-9)
        if blocked > 0.001:
            notes.append(f"Tradability rules (漲停/跌停, 平盤下不得放空) changed "
                         f"{blocked:.1%} of intended book weight.")

    gross, lng, sht = PnlParts(held, ret.fillna(0.0))
    dw = held.diff().fillna(held)
    trade_cost = cost.trade_cost(dw)
    borrow = cost.borrow_cost(held)
    total_cost = (trade_cost + borrow).reindex(gross.index).fillna(0.0)

    return XSimResult(
        gross=gross, net=gross - total_cost, long_pnl=lng, short_pnl=sht,
        cost=total_cost, turnover=dw.abs().sum(axis=1) / 2.0,
        coverage=(held != 0).sum(axis=1), weights=held,
        exec_shift=exec_shift, notes=notes)
