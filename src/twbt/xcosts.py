"""Costs for a Taiwan cash-equity book.

This is a different animal from the futures model in ``costs.py``, and the
difference is the whole reason it is a separate module: **equity costs are
asymmetric**. A futures round trip is symmetric — no borrow, no locate, no
tax-on-one-side — so a long and a short cost the same. In Taiwan cash equities
they do not, and by a wide margin.

The components, per leg:

    手續費   brokerage     0.1425% of notional, both sides, before any discount.
                          Retail discounts of 30-60% are normal, so the default
                          here is 0.0600% rather than the list price.
    證交稅   transaction   0.30% of notional, **on the sell side only**. This is
             tax          the single largest cost in the book and the reason a
                          daily-rebalanced equity strategy in Taiwan is usually
                          dead before it starts. It is 0.15% for same-day
                          day trades, which this model does not assume.
    借券費   borrow fee    for shorts, an annualised rate accrued per day held.
                          Varies enormously by name; 1.5%/yr is a plausible
                          general-collateral default and badly wrong for a
                          hard-to-borrow name.

Two frictions that are not prices and are modelled separately, because treating
them as a cost understates how much they hurt:

    平盤下不得放空   many names cannot be shorted at or below the previous close.
    漲停/跌停       a name locked limit-up cannot be bought and one locked
                    limit-down cannot be sold, so the intended trade simply does
                    not happen and yesterday's position carries forward.

A round trip that buys and later sells therefore pays roughly
``2 x 0.06% + 0.30% = 0.42%`` — **42 basis points** — before any spread. For a
signal whose edge is measured in single-digit bps per day, that number decides
everything, which is why it is stated here in full rather than folded into a
single "cost" parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# All rates are fractions of notional, per leg, unless stated otherwise.
BROKERAGE_LIST = 0.001425          # 手續費 list price, each side
BROKERAGE_TYPICAL = 0.000600       # after a normal retail discount
TAX_SELL = 0.003000                # 證交稅, sell side only
TAX_SELL_DAYTRADE = 0.001500       # 當沖, sell side only
BORROW_ANNUAL = 0.015              # 借券費, annualised, general collateral
TRADING_DAYS = 252


@dataclass(frozen=True)
class TWEquityCost:
    """Per-leg costs for a Taiwan cash-equity book.

    Separate buy and sell rates, because the transaction tax falls on one side
    only. A model with a single symmetric "cost per trade" cannot express that,
    and silently halves the real cost of a long round trip.
    """

    brokerage: float = BROKERAGE_TYPICAL
    tax_sell: float = TAX_SELL
    borrow_annual: float = BORROW_ANNUAL
    half_spread: float = 0.0010        # 0.10%; one tick is ~0.05-0.1% mid-cap
    day_trade: bool = False

    @property
    def _tax(self) -> float:
        return TAX_SELL_DAYTRADE if self.day_trade else self.tax_sell

    @property
    def buy_rate(self) -> float:
        """Cost of buying 1 unit of notional: brokerage + half spread."""
        return self.brokerage + self.half_spread

    @property
    def sell_rate(self) -> float:
        """Cost of selling 1 unit: brokerage + tax + half spread."""
        return self.brokerage + self._tax + self.half_spread

    @property
    def round_trip(self) -> float:
        """Buy then sell, as a fraction of notional."""
        return self.buy_rate + self.sell_rate

    @property
    def round_trip_bps(self) -> float:
        return self.round_trip * 1e4

    def trade_cost(self, dw: pd.DataFrame) -> pd.Series:
        """Cost per date from a matrix of weight *changes*.

        A positive change is a buy and a negative one a sell, whichever side of
        the book it is on: covering a short is a buy and pays the buy rate, while
        opening a short is a sell and pays the tax. Netting the two into
        ``|dw| * one_rate`` is the mistake this signature exists to prevent.
        """
        buys = dw.clip(lower=0).sum(axis=1)
        sells = (-dw.clip(upper=0)).sum(axis=1)
        return buys * self.buy_rate + sells * self.sell_rate

    def borrow_cost(self, w: pd.DataFrame) -> pd.Series:
        """Daily accrual on the short side of the book."""
        short_gmv = (-w.clip(upper=0)).sum(axis=1)
        return short_gmv * self.borrow_annual / TRADING_DAYS

    def scaled(self, factor: float) -> TWEquityCost:
        """Scale only the discretionary parts. Tax is statutory and does not move."""
        return replace(self, brokerage=self.brokerage * factor,
                       half_spread=self.half_spread * factor)

    def describe(self) -> str:
        return (f"buy {self.buy_rate*1e4:.1f} bps / sell {self.sell_rate*1e4:.1f} bps "
                f"(手續費 {self.brokerage*1e4:.1f}, 證交稅 {self._tax*1e4:.1f} sell-only, "
                f"half-spread {self.half_spread*1e4:.1f}), round trip "
                f"{self.round_trip_bps:.1f} bps; 借券 {self.borrow_annual*100:.2f}%/yr")

    def breakdown(self) -> pd.DataFrame:
        """Where a round trip's cost actually goes. Usually a surprise."""
        rows = [
            ("手續費 brokerage (x2)", 2 * self.brokerage * 1e4),
            ("證交稅 tax (sell only)", self._tax * 1e4),
            ("spread (half x2)", 2 * self.half_spread * 1e4),
        ]
        total = sum(v for _, v in rows)
        return pd.DataFrame(
            [{"component": k, "bps": v, "share": v / total if total else np.nan}
             for k, v in rows]
            + [{"component": "TOTAL round trip", "bps": total, "share": 1.0}])


def shortable(px: pd.DataFrame, enforce: bool = True,
              lag: int = 1) -> pd.DataFrame:
    """Boolean matrix: could this name have been shorted when the trade was placed?

    Models 平盤下不得放空 - for most names a short sale may not be made at or
    below the previous close.

    ``lag`` is the critical argument and defaults to 1. The test must use prices
    known at EXECUTION time, not prices from the period the position is held.
    Comparing today's close with yesterday's is the same statement as today's
    return being positive:

        shortable_t = (px_t > px_{t-1})  ==  (ret_t > 0)

    Filtering a book by that deletes precisely the shorts that were about to make
    money. Measured on random data it gave a +0.80 correlation between the rule
    and the return it gated, and dragged a Sharpe of -0.49 down to -15.05 with a
    3.5% hit rate. It is look-ahead pointing the unusual way - it makes results
    look WORSE - but it is look-ahead all the same, and a rule that can move a
    Sharpe fifteen points in either direction is not a detail.

    Daily data cannot see whether an intraday uptick was available, so this stays
    an approximation. It is deliberately a strict one: it refuses some shorts an
    uptick would have allowed rather than allowing ones that were impossible.
    """
    if not enforce:
        return pd.DataFrame(True, index=px.index, columns=px.columns)
    known = px.shift(lag)
    return known > known.shift(1)


def limit_locked(px: pd.DataFrame, limit: float = 0.0998,
                 lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where a name was locked limit-up or limit-down when the trade was placed.

    Taiwan's daily band is +/-10%, approximated at 9.98% for tick rounding.
    Returns (locked_up, locked_down): you cannot buy the first, cannot sell the
    second.

    ``lag`` matches ``shortable`` and for the same reason: assessed at lag 0 the
    rule reads the return it is gating.
    """
    r = px.shift(lag).pct_change()
    return r >= limit, r <= -limit


def apply_tradability(target_w: pd.DataFrame, px: pd.DataFrame,
                      enforce_short_rule: bool = True,
                      enforce_limits: bool = True,
                      lag: int = 1) -> pd.DataFrame:
    """Turn intended weights into weights you could actually have held.

    Three rules, applied in order, each carrying yesterday's weight forward where
    the intended trade was impossible:

    1. a name locked limit-up cannot be **bought**, so an increase is blocked;
    2. a name locked limit-down cannot be **sold**, so a decrease is blocked;
    3. a name at or below its previous close cannot be **shorted**, so a new or
       larger short is blocked.

    Carrying the prior weight is the honest outcome — the trade did not happen —
    and it is why this returns weights rather than a mask. A backtest that simply
    drops the blocked names instead quietly rebalances into the ones that were
    tradable, which is a different and much better strategy than the one tested.

    ``lag`` is passed to both rules and must be at least 1; see ``shortable``.

    The resulting book has gross exposure below target, because a blocked trade
    means holding less, and it is deliberately not re-scaled. Re-scaling would
    concentrate the book into whatever happened to be tradable, which is the
    same error as dropping the names.
    """
    tgt = target_w.fillna(0.0)
    up, dn = limit_locked(px, lag=lag)
    can_short = shortable(px, enforce=enforce_short_rule, lag=lag)
    up, dn = up.reindex_like(tgt).fillna(False), dn.reindex_like(tgt).fillna(False)
    can_short = can_short.reindex_like(tgt).fillna(False)

    held = np.zeros(tgt.shape[1])
    out = np.empty(tgt.shape)
    tgt_v, up_v, dn_v, cs_v = (tgt.to_numpy(), up.to_numpy(),
                               dn.to_numpy(), can_short.to_numpy())

    for i in range(len(tgt_v)):
        want = tgt_v[i].copy()
        if enforce_limits:
            blocked_buy = up_v[i] & (want > held)
            blocked_sell = dn_v[i] & (want < held)
            want = np.where(blocked_buy | blocked_sell, held, want)
        if enforce_short_rule:
            new_short = (want < 0) & (want < held) & (~cs_v[i])
            want = np.where(new_short, np.minimum(held, 0.0), want)
        out[i] = want
        held = want

    return pd.DataFrame(out, index=tgt.index, columns=tgt.columns)
