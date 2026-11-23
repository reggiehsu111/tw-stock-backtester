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

Fee figures are TAIFEX's published exchange fee plus a representative broker
commission. They are defaults, not measurements of anyone's actual schedule — the
tearsheet's cost-sensitivity curve exists so the conclusion does not rest on them.
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

# Representative all-in cost per side, in index points, excluding the spread:
# exchange fee + clearing + broker commission + the futures transaction tax.
# Quoted per side because fees are charged per side, unlike the spread.
FEES_PTS_PER_SIDE = {"TXF": 0.35, "MXF": 0.40, "TMF": 0.45}

# Observed median bid-ask width in index points. The large contract is the tightest
# in percentage terms; the micro is quoted in the same ticks but far less of the
# time, so its effective width is wider.
MEDIAN_SPREAD_PTS = {"TXF": 3.0, "MXF": 1.0, "TMF": 2.0}

# The opening and closing minutes are wider than the middle of the session.
SESSION_WIDENING = {"open": 1.5, "close": 1.2, "mid": 1.0}


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost for one contract, expressed in basis points of notional.

    ``spread_pts`` is the FULL quoted width, not half of it. The halving happens
    inside ``round_trip``, once, so a caller cannot accidentally halve it twice or
    not at all.
    """

    contract: str = "TXF"
    spread_pts: float | None = None          # full width; default = observed median
    fees_pts_per_side: float | None = None
    entry_widening: float = 1.0              # multiplier on the entry leg's spread
    exit_widening: float = 1.0

    def __post_init__(self) -> None:
        if self.contract not in CONTRACTS:
            raise ValueError(f"unknown contract {self.contract!r}; "
                             f"have {sorted(CONTRACTS)}")

    @property
    def _spread(self) -> float:
        return (MEDIAN_SPREAD_PTS[self.contract] if self.spread_pts is None
                else self.spread_pts)

    @property
    def _fees(self) -> float:
        return (FEES_PTS_PER_SIDE[self.contract] if self.fees_pts_per_side is None
                else self.fees_pts_per_side)

    def round_trip_pts(self) -> float:
        """Total cost of entering and exiting one contract, in index points."""
        half_in = self._spread * self.entry_widening / 2.0
        half_out = self._spread * self.exit_widening / 2.0
        return half_in + half_out + 2.0 * self._fees

    def round_trip_bps(self, price) -> pd.Series | float:
        """The same cost as basis points of the index level.

        Takes the price because a fixed point cost is a shrinking percentage as the
        index rises: 3 points was 2.0 bps at 15,000 and is 1.1 bps at 27,000.
        Holding the bps figure constant across a multi-year backtest understates
        cost early and overstates it late.
        """
        pts = self.round_trip_pts()
        if isinstance(price, pd.Series):
            return pts / price.where(price > 0) * 1e4
        return pts / float(price) * 1e4

    def at_session(self, entry: str = "mid", exit: str = "mid") -> CostModel:
        """A copy with the per-leg widening set from session position."""
        for k in (entry, exit):
            if k not in SESSION_WIDENING:
                raise ValueError(f"session must be one of {sorted(SESSION_WIDENING)}")
        return replace(self, entry_widening=SESSION_WIDENING[entry],
                       exit_widening=SESSION_WIDENING[exit])

    def scaled(self, factor: float) -> CostModel:
        """A copy with the spread multiplied — the knob the sensitivity curve turns."""
        return replace(self, spread_pts=self._spread * factor)

    def describe(self) -> str:
        c = CONTRACTS[self.contract]
        return (f"{self.contract} ({c['name']}): spread {self._spread:.2f} pts "
                f"(x{self.entry_widening:.2f} in / x{self.exit_widening:.2f} out), "
                f"fees {self._fees:.2f} pts/side, "
                f"round trip {self.round_trip_pts():.2f} pts")


def sensitivity(gross_bps: pd.Series, price: pd.Series, model: CostModel,
                turnover: pd.Series | None = None,
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

    rows = []
    for f in factors:
        cost = model.scaled(f).round_trip_bps(price) * turnover
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
