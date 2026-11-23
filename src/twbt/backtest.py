"""The engine: signal -> position -> P&L, and the checks that go with it.

Two pieces of arithmetic live here and nowhere else, because each is a place where
a plausible-looking mistake flatters the result.

PUBLICATION LAG. A strategy's ``signal`` is computed on columns dated by when the
data was *published*, not when it could be traded. TAIFEX releases 三大法人 and
大額交易人 at 15:00 TPE, after the day session has closed, so the earliest tradable
moment is the following session. The engine shifts, the strategy does not: a lag
that each strategy applied for itself would be applied twice in some and zero times
in others, and a zero-lag version of a flow strategy looks wonderful.

TURNOVER. Cost is charged per leg, as ``round_trip x |delta position| / 2``. Over a
full 0 -> 1 -> 0 cycle that sums to exactly one round trip. Charging a round trip on
every non-zero day instead would overstate a holding strategy by its holding period,
and charging one per entry only would understate it by half.

``run()`` evaluates every contract it is given rather than one, because the single
most useful check available is whether the result survives on sibling instruments —
see docs/METHOD.md section 5. Two findings in the research behind this project
passed every statistical test and died there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from . import costs as costs_mod
from . import stats


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Strategy:
    """Base class. Subclasses implement ``signal`` and may declare a ``grid``.

    ``publication_lag`` is in trading sessions and must be at least 1 for anything
    built from published data. The engine refuses 0 for a strategy that touches a
    published column, which is checked in ``run``.

    Subclasses are frozen dataclasses so ``with_params`` can produce variants
    without the walk-forward accidentally mutating a shared object.
    """

    name: str = "unnamed"
    publication_lag: int = 1

    def signal(self, panel: pd.DataFrame) -> pd.Series:
        """Target position per date, in units of one contract. Sign is direction."""
        raise NotImplementedError

    def grid(self) -> list[dict[str, Any]]:
        """Parameter combinations for the walk-forward. Empty = no tuning."""
        return []

    def with_params(self, **kw: Any) -> Strategy:
        return replace(self, **kw)

    # columns the strategy reads that are published after the close; used by run()
    # to verify the lag is non-zero
    published_columns: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class ContractResult:
    """One strategy on one contract, in both of the views that matter.

    ``daily_*`` run over the full index with flat days at zero and every leg
    charged, which is what the equity curve must be drawn from.

    ``trades`` has one row per holding period and is what the headline statistics
    are computed on, because that is the unit results are quoted in and the unit
    where a round trip lands whole on the trade that incurred it.
    """

    contract: str
    daily_gross: pd.Series
    daily_net: pd.Series
    daily_cost: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    gross_perf: stats.Performance
    net_perf: stats.Performance
    sensitivity: pd.DataFrame

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def equity(self) -> pd.Series:
        """Cumulative net P&L in bps, over the full index."""
        return self.daily_net.cumsum()

    @property
    def breakeven_bps(self) -> float:
        """Gross edge per trade -- the cost level at which the result disappears."""
        return (costs_mod.breakeven_bps(self.trades["gross_bps"])
                if len(self.trades) else float("nan"))

    @property
    def avg_cost_bps(self) -> float:
        return (float(self.trades["cost_bps"].mean())
                if len(self.trades) else float("nan"))


@dataclass
class BacktestResult:
    """A strategy across contracts, plus the cross-instrument verdict."""

    strategy: str
    primary: str
    contracts: dict[str, ContractResult] = field(default_factory=dict)
    walk_forward: pd.DataFrame | None = None
    noise_bar_95: float = float("nan")
    n_variants_tested: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def main(self) -> ContractResult:
        return self.contracts[self.primary]

    def consistency(self) -> pd.DataFrame:
        """Net t-stat per contract. The table that kills most findings."""
        rows = [{"contract": c, "n": r.net_perf.n, "net_mean_bps": r.net_perf.mean,
                 "net_t": r.net_perf.t, "net_sharpe": r.net_perf.sharpe}
                for c, r in self.contracts.items()]
        return pd.DataFrame(rows)

    def verdict(self) -> list[str]:
        """Plain-language findings. Written to be unflattering where that is true."""
        out = list(self.notes)
        m = self.main.net_perf

        if np.isfinite(self.noise_bar_95) and self.n_variants_tested > 1:
            if abs(m.t) < self.noise_bar_95:
                out.append(
                    f"FAILS multiple-testing check: net t={m.t:+.2f} is below the "
                    f"{self.noise_bar_95:.2f} bar for {self.n_variants_tested} variants.")
            else:
                out.append(
                    f"Clears the noise bar: net t={m.t:+.2f} vs {self.noise_bar_95:.2f} "
                    f"for {self.n_variants_tested} variants.")

        cons = self.consistency()
        if len(cons) > 1:
            signs = np.sign(cons["net_mean_bps"].dropna())
            if len(set(signs)) > 1:
                out.append(
                    "FAILS the cross-instrument check: sibling contracts disagree on "
                    "sign (" + ", ".join(f"{r.contract} {r.net_mean_bps:+.2f}"
                                         for r in cons.itertuples()) + "). No mechanism "
                    "produces that; independent noise draws do.")
            else:
                out.append("Consistent in sign across " + ", ".join(cons["contract"]) + ".")

        sens = self.main.sensitivity
        if len(sens):
            if not (m.mean > 0):
                # "survives every cost assumption" is not praise for something that
                # loses money at zero cost -- the cost curve is only meaningful once
                # there is an edge for cost to eat
                out.append("Cost-robustness is not applicable: the strategy is not "
                           "profitable before costs.")
            else:
                dies = sens.loc[sens["net_t"] <= 0, "spread_factor"]
                if len(dies):
                    f = dies.iloc[0]
                    verb = "FAILS" if f <= 1.5 else "survives"
                    out.append(f"{verb} cost-robustness: net t reaches zero at "
                               f"{f:.1f}x the spread estimate.")
                else:
                    out.append("Survives every spread assumption tested "
                               f"(up to {sens['spread_factor'].max():.1f}x).")

        if self.walk_forward is not None and len(self.walk_forward):
            wf = self.walk_forward
            pos = int((wf["net_mean_bps"] > 0).sum())
            out.append(f"Walk-forward: {pos}/{len(wf)} out-of-sample years positive, "
                       f"{int(wf['n'].sum())} trades total.")
            if int(wf["n"].sum()) < 100:
                out.append("Walk-forward trade count is too low to conclude anything.")

        return out


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #

def _positions(signal: pd.Series, lag: int) -> pd.Series:
    """Shift the signal into tradable alignment. This is the only shift in the run."""
    if lag < 0:
        raise ValueError(f"publication_lag must be >= 0; got {lag}")
    return signal.shift(lag).fillna(0.0)


def _legs(position: pd.Series) -> pd.Series:
    """Round-trip-equivalents traded per period: |delta position| / 2.

    A 0 -> 1 -> 0 cycle sums to exactly 1.0 across its three days, including the
    day the position returns to flat. That last day carries half the cost and none
    of the P&L, which is why the daily series below is kept over the full index:
    masking it to days with a non-zero position silently drops the exit leg and
    charges half a round trip.
    """
    return position.diff().fillna(position).abs() / 2.0


def _trades(position: pd.Series, gross: pd.Series, cost: pd.Series,
            cost_per_leg: pd.Series) -> pd.DataFrame:
    """Collapse the daily series into one row per contiguous holding period.

    Per-trade is the unit results are quoted in ("175 trades, net 5.04 bps"), and it
    is also the unit where cost accounting is unambiguous: a trade's cost is every
    leg from its entry up to and including its exit, so the round trip lands whole
    on the trade that incurred it.

    Flat days are not trades and are excluded, rather than being counted as
    zero-return observations that deflate the mean and inflate n.
    """
    held = position != 0.0
    if not held.any():
        return pd.DataFrame(columns=["entry", "exit", "days", "direction",
                                     "gross_bps", "cost_bps", "net_bps",
                                     "open_at_end"])

    # a new block starts wherever the position changes to or from flat, or flips sign
    block = (np.sign(position) != np.sign(position).shift(1)).cumsum()

    rows = []
    for _, idx in position.groupby(block).groups.items():
        idx = pd.DatetimeIndex(idx) if not isinstance(idx, pd.DatetimeIndex) else idx
        if position.loc[idx[0]] == 0.0:
            continue
        # the exit leg falls on the first flat day after the block; include it
        loc_end = position.index.get_loc(idx[-1])
        span = position.index[position.index.get_loc(idx[0]):min(loc_end + 2,
                                                                 len(position.index))]
        c = float(cost.loc[span].sum())

        # A trade still open when the sample ends has no exit day to carry its
        # closing leg, so the sum above is short by half a round trip. The position
        # has to be closed in reality, and leaving the cost out understates the
        # result -- always the dangerous direction -- so it is charged here.
        if loc_end >= len(position.index) - 1:
            c += float(cost_per_leg.loc[idx[-1]]) * abs(float(position.loc[idx[-1]])) / 2.0

        rows.append({
            "entry": idx[0],
            "exit": idx[-1],
            "days": len(idx),
            "direction": float(np.sign(position.loc[idx[0]])),
            "gross_bps": float(gross.loc[idx].sum()),
            "cost_bps": c,
            "open_at_end": bool(loc_end >= len(position.index) - 1),
        })

    tr = pd.DataFrame(rows)
    tr["net_bps"] = tr["gross_bps"] - tr["cost_bps"]
    return tr.set_index("entry", drop=False)


def run(panel: pd.DataFrame, strategy: Strategy,
        cost_model: costs_mod.CostModel | None = None,
        contracts: tuple[str, ...] = ("TXF", "MXF", "TMF"),
        n_variants_tested: int = 1,
        return_col: str = "ret", price_col: str = "close") -> BacktestResult:
    """Run one strategy across contracts and assemble the verdict.

    The daily panel is shared across contracts -- they track the same index, so the
    return series is the same and only the cost differs. That is the point: if a
    result survives on TXF and not on TMF, the difference is purely cost, which is a
    statement about tradability. If the *signs* differ, the panel cannot explain it
    and the result is noise.

    Each ContractResult carries both views: ``daily`` for the equity curve (full
    index, flat days zero, every leg charged) and ``trades`` for the headline
    statistics (one row per holding period).
    """
    if strategy.publication_lag < 1 and strategy.published_columns:
        raise ValueError(
            f"{strategy.name} reads published columns {strategy.published_columns} "
            "but has publication_lag=0. TAIFEX publishes these at 15:00, after the "
            "day session closes, so the earliest tradable session is the next one.")

    sig = strategy.signal(panel).reindex(panel.index)
    pos = _positions(sig, strategy.publication_lag)
    ret = panel[return_col]
    px = panel[price_col]
    legs = _legs(pos)

    gross = (pos * ret).fillna(0.0).rename("gross_bps")
    base_cost = cost_model or costs_mod.CostModel()

    out = BacktestResult(strategy=strategy.name, primary=contracts[0],
                         n_variants_tested=n_variants_tested)

    for c in contracts:
        # swap only the contract; an explicitly supplied spread or fee is kept, so
        # "charge me 5 points everywhere" means that, while the default (None) picks
        # up each contract's own observed width
        cm = replace(base_cost, contract=c)
        cost = (cm.round_trip_bps(px) * legs).fillna(0.0).rename("cost_bps")
        net = (gross - cost).rename("net_bps")
        tr = _trades(pos, gross, cost, cm.round_trip_bps(px))

        out.contracts[c] = ContractResult(
            contract=c,
            daily_gross=gross, daily_net=net, daily_cost=cost, position=pos,
            trades=tr,
            gross_perf=stats.performance(tr["gross_bps"] if len(tr) else pd.Series(dtype=float)),
            net_perf=stats.performance(tr["net_bps"] if len(tr) else pd.Series(dtype=float)),
            sensitivity=(costs_mod.sensitivity(
                gross[pos != 0.0], px[pos != 0.0], cm, turnover=legs[pos != 0.0])
                if (pos != 0.0).any() else pd.DataFrame()),
        )

    if len(out.main.trades) == 0:
        out.notes.append("Strategy never took a position over this sample.")
    elif n_variants_tested > 1 and len(out.main.trades) > 40:
        out.noise_bar_95 = stats.noise_bar(out.main.trades["net_bps"],
                                           n_variants_tested, iters=1500)

    return out


def walk_forward(panel: pd.DataFrame, strategy: Strategy,
                 cost_model: costs_mod.CostModel | None = None,
                 contract: str = "TXF", min_train_years: int = 2,
                 return_col: str = "ret", price_col: str = "close") -> pd.DataFrame:
    """Re-pick parameters at the start of each year using prior data only.

    For year Y, every combination in ``strategy.grid()`` is scored on data strictly
    before Y, the best by net mean is selected, and that one choice is applied to Y.
    The reported P&L is therefore out-of-sample in the only sense that counts:
    nothing used to choose a parameter was dated after the trade.

    A strategy with an empty grid is still walked -- the result is then simply the
    fixed strategy's performance year by year, which is the honest comparison.
    """
    grid = strategy.grid() or [{}]
    years = sorted({int(y) for y in panel.index.year})
    rows: list[dict[str, Any]] = []

    for y in years:
        train = panel[panel.index.year < y]
        test = panel[panel.index.year == y]
        if len(train) < min_train_years * 200 or len(test) < 40:
            continue

        best, best_mean = grid[0], -np.inf
        for params in grid:
            cand = strategy.with_params(**params) if params else strategy
            try:
                r = run(train, cand, cost_model, contracts=(contract,),
                        return_col=return_col, price_col=price_col)
            except Exception:                              # noqa: BLE001
                continue
            m = r.main.net_perf.mean
            if np.isfinite(m) and m > best_mean:
                best, best_mean = params, m

        chosen = strategy.with_params(**best) if best else strategy
        r = run(test, chosen, cost_model, contracts=(contract,),
                return_col=return_col, price_col=price_col)
        p = r.main.net_perf
        rows.append({
            "year": y, "params": str(best) if best else "fixed", "n": p.n,
            "net_mean_bps": p.mean, "net_t": p.t, "net_total_bps": p.total,
            "train_mean_bps": best_mean if np.isfinite(best_mean) else np.nan,
        })

    return pd.DataFrame(rows)
