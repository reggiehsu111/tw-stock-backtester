"""What a round trip actually costs.

The convention, which is the whole point of this module:

    a round trip crosses the spread ONCE in total

You buy at the ask — half a spread above mid — and sell at the bid, half below.
Charging a full spread on each leg double-counts, and during the research this came
from, that error alone moved a result from "clearly tradable" to "below cost". It is
easy to commit because "spread per trade" is ambiguous about whether a trade is a
leg or a round trip.

    cost = spread_entry/2 + spread_exit/2 + fees_entry + fees_exit

Per-leg spreads are kept separate rather than averaged, because the first and last
minutes of a session are routinely much wider than the middle. Averaging them makes
an open-entry strategy look better than it is, which is again the dangerous
direction.

Two components, two different units, for a reason measured rather than assumed:

* the **spread** is held constant in **basis points**, because the quoted width
  scales with the index level — it tripled in points over 2024-2026 while the index
  doubled, leaving bps nearly flat;
* the **fees** are held constant in **index points**, because they are a fixed NT$
  amount, and so genuinely do shrink as a share of notional when the index rises.

Better than either: pass the **measured** per-day spread. ``panel.build`` derives a
``spread_bps`` column from the closing bid/ask that the fetcher already downloads,
and the engine uses it automatically when present. That matters beyond accuracy on
average, because the spread is wider on exactly the days a signal is most likely to
fire: regressing it on the day's range gives t = +5.8, and high-range days are 0.56
bps more expensive than low-range days. A constant understates cost precisely when
you trade.

The cost-sensitivity curve exists so no conclusion rests on any of these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# TAIFEX contract specifications. Point values are NT$ per index point.
CONTRACTS = {
    "TXF": {"name": "臺股期貨 (large)", "point_value": 200, "tick": 1.0},
    "MXF": {"name": "小型臺指 (mini)", "point_value": 50, "tick": 1.0},
    "TMF": {"name": "微型臺指 (micro)", "point_value": 10, "tick": 1.0},
}

# Fees are quoted in INDEX POINTS per side: exchange fee + clearing + broker
# commission + futures transaction tax, as a fixed NT$ amount divided by the
# contract's point value. They are genuinely fixed in NT$, so they shrink as a
# fraction of notional when the index rises -- the conversion to bps divides by
# price, and that is correct for this component.
#
# Dividing by a smaller point value is also why the smaller contracts cost more
# per index point.
FEES_PTS_PER_SIDE = {"TXF": 0.35, "MXF": 0.40, "TMF": 0.45}

# The spread is quoted in BASIS POINTS, and this is the opposite convention to
# the fees above, deliberately.
#
# Measured on the TX front month, 2024-2026 (the bid/ask columns the fetcher
# already downloads and nothing used to read):
#
#     year   median close   median spread   median spread
#     2024         21,897        2.0 pts         1.12 bps
#     2025         23,344        3.0 pts         1.37 bps
#     2026         43,204        6.0 pts         1.50 bps
#
# The index roughly doubled and the point spread tripled, so the spread in POINTS
# is the unstable unit and the spread in bps is close to flat. An earlier version
# of this module held the spread fixed in points and argued the reverse. It was
# wrong, and wrong in the flattering direction: by 2026 it charged 0.92 bps where
# the measured cost was 2.33, a 2.5x understatement that grows over the sample.
#
# Prefer the measured per-day spread (panel column `spread_bps`) over any of
# these. They are the fallback for a panel without quote data.
MEDIAN_SPREAD_BPS = {"TXF": 1.35, "MXF": 1.35, "TMF": 1.80}

# The opening and closing minutes are wider than the middle of the session.
SESSION_WIDENING = {"open": 1.5, "close": 1.2, "mid": 1.0}


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost for one contract, in basis points of notional.

    ``spread_bps`` is the FULL quoted width, not half of it. The halving happens
    inside ``round_trip_bps``, once, so a caller cannot halve it twice or not at
    all. ``spread_pts`` is accepted as an override for a caller who really does
    mean a fixed point width, and is converted using the price.
    """

    contract: str = "TXF"
    spread_bps: float | None = None          # full width in bps; the default unit
    spread_pts: float | None = None          # full width in points; overrides bps
    fees_pts_per_side: float | None = None
    entry_widening: float = 1.0              # multiplier on the entry leg's spread
    exit_widening: float = 1.0

    def __post_init__(self) -> None:
        if self.contract not in CONTRACTS:
            raise ValueError(f"unknown contract {self.contract!r}; "
                             f"have {sorted(CONTRACTS)}")
        if self.spread_bps is not None and self.spread_pts is not None:
            raise ValueError("give spread_bps or spread_pts, not both")

    @property
    def _fees(self) -> float:
        return (FEES_PTS_PER_SIDE[self.contract] if self.fees_pts_per_side is None
                else self.fees_pts_per_side)

    @property
    def _widening(self) -> float:
        """Total spread multiplier for a round trip: half on each leg."""
        return (self.entry_widening + self.exit_widening) / 2.0

    def _spread_bps(self, price, measured=None):
        """The round trip's spread cost in bps: one full width, however specified.

        Precedence is measured, then an explicit point width, then an explicit bps
        width, then the contract default. Measured wins because it is the only one
        that moves with the market instead of with an assumption.
        """
        if measured is not None:
            base = pd.Series(measured) if not isinstance(measured, pd.Series) else measured
        elif self.spread_pts is not None:
            base = self.spread_pts / _as_price(price) * 1e4
        elif self.spread_bps is not None:
            base = self.spread_bps
        else:
            base = MEDIAN_SPREAD_BPS[self.contract]
        return base * self._widening

    def _fees_bps(self, price):
        """Fees in bps. Divided by price because they are a fixed NT$ amount."""
        return 2.0 * self._fees / _as_price(price) * 1e4

    def round_trip_bps(self, price, measured_spread_bps=None):
        """Total cost of entering and exiting one contract, in bps of notional.

        ``price`` may be a scalar or a Series. ``measured_spread_bps`` is the
        observed full width per date, and takes precedence over every default.
        """
        return self._spread_bps(price, measured_spread_bps) + self._fees_bps(price)

    def round_trip_pts(self, price=None) -> float:
        """The same cost in index points, for a given index level.

        Points are not the natural unit here -- the spread is held in bps because
        that is what is stable -- so this needs a price. It defaults to a round
        number purely so ``describe()`` has something to print.
        """
        px = 20_000.0 if price is None else float(price)
        return float(self.round_trip_bps(px)) * px / 1e4

    def at_session(self, entry: str = "mid", exit: str = "mid") -> CostModel:
        """A copy with the per-leg widening set from session position."""
        for k in (entry, exit):
            if k not in SESSION_WIDENING:
                raise ValueError(f"session must be one of {sorted(SESSION_WIDENING)}")
        return replace(self, entry_widening=SESSION_WIDENING[entry],
                       exit_widening=SESSION_WIDENING[exit])

    def scaled(self, factor: float) -> CostModel:
        """A copy with the spread multiplied -- the knob the sensitivity curve turns.

        Implemented through the widening multipliers rather than by rewriting the
        width, so it composes with a measured spread as well as with a constant.
        """
        return replace(self, entry_widening=self.entry_widening * factor,
                       exit_widening=self.exit_widening * factor)

    def describe(self, price: float = 20_000.0) -> str:
        c = CONTRACTS[self.contract]
        if self.spread_pts is not None:
            w = f"{self.spread_pts:.2f} pts"
        elif self.spread_bps is not None:
            w = f"{self.spread_bps:.2f} bps"
        else:
            w = f"{MEDIAN_SPREAD_BPS[self.contract]:.2f} bps (default)"
        return (f"{self.contract} ({c['name']}): spread {w} "
                f"(x{self.entry_widening:.2f} in / x{self.exit_widening:.2f} out), "
                f"fees {self._fees:.2f} pts/side, round trip "
                f"{float(self.round_trip_bps(price)):.2f} bps at {price:,.0f}")


def _as_price(price):
    """Price as something safe to divide by: non-positive values become NaN."""
    if isinstance(price, pd.Series):
        return price.where(price > 0)
    p = float(price)
    return p if p > 0 else np.nan


def sensitivity(gross_bps: pd.Series, price: pd.Series, model: CostModel,
                turnover: pd.Series | None = None,
                measured_spread_bps: pd.Series | None = None,
                factors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """Net mean and t-stat as the spread assumption is scaled up.

    This is the table behind the tearsheet's cost curve. A strategy whose t-stat
    crosses zero between factors 1.0 and 1.5 is not a strategy — the realistic
    estimate and a mildly pessimistic one disagree about whether it exists.

    ``turnover`` is the number of round trips per period (default: one per period
    with a non-zero return). Scaling by it matters for strategies that hold.
    """
    from .stats import performance            # local: avoid an import cycle

    gross_bps = pd.Series(gross_bps).dropna()
    price = pd.Series(price).reindex(gross_bps.index)
    if turnover is None:
        turnover = pd.Series(1.0, index=gross_bps.index)
    turnover = pd.Series(turnover).reindex(gross_bps.index).fillna(0.0)

    msb = (pd.Series(measured_spread_bps).reindex(gross_bps.index)
           if measured_spread_bps is not None else None)

    rows = []
    for f in factors:
        cost = model.scaled(f).round_trip_bps(price, msb) * turnover
        net = (gross_bps - cost).dropna()
        p = performance(net)
        rows.append({
            "spread_factor": f,
            "cost_bps": float(np.nanmean(cost)),
            "net_mean_bps": p.mean,
            "net_t": p.t,
            "net_sharpe": p.sharpe,
            "n": p.n,
        })
    return pd.DataFrame(rows)


def breakeven_bps(gross_bps) -> float:
    """The round-trip cost at which the mean gross edge disappears.

    Reported next to the realistic estimate so the margin is explicit rather than
    something the reader has to compute from two other numbers.
    """
    s = pd.Series(gross_bps).dropna()
    return float(s.mean()) if len(s) else float("nan")
