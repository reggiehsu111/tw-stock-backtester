"""The wide matrices the operators run on, and the universe filter.

``build()`` pivots the long per-(date, ticker) cache into one matrix per field —
rows are trading days, columns are tickers — which is the only shape the
operators in ``ops`` accept.

The universe filter is the part that deserves attention, because it is where a
cross-sectional backtest usually goes wrong without saying so.

SURVIVORSHIP. The columns are every ticker that appeared *at any point* in the
sample, and a name that listed in 2025 is simply NaN before then. That is the
correct shape: dropping names that do not span the whole window would keep only
the survivors and quietly delete every delisting, which is the single largest
free lunch available to a careless equity backtest.

LIQUIDITY, POINT IN TIME. A universe of 1,850 names includes several hundred that
barely trade, where a cross-sectional rank is noise and a fill is fiction. The
filter is a **trailing** median turnover, so membership on any date is decided by
data from before it. Picking "the 300 most liquid names" over the full sample
would be look-ahead of the worst kind — it selects for names that went on to
become liquid, which correlates with having gone up.
"""

from __future__ import annotations

import pandas as pd

from . import cache

# Fields pivoted into matrices. Keys are cache columns, values are matrix names.
FIELDS = {
    "close": "close", "open": "open", "high": "high", "low": "low",
    "shares": "volume", "amount": "turnover", "trades": "trades",
    "shares_out": "shares_out",
}

DEFAULT_UNIVERSE_SIZE = 300
DEFAULT_LIQUIDITY_WINDOW = 60
MIN_PRICE = 5.0            # NT$; below this the tick is a large share of price


def build(root=None, fixture: bool = False,
          start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Pivot the cached cross-section into wide matrices, one per field."""
    kw = {"fixture": fixture} if fixture else {"root": root}
    df = cache.load("equities", **kw)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("no equity rows in the selected range")

    out: dict[str, pd.DataFrame] = {}
    for col, name in FIELDS.items():
        if col not in df.columns:
            continue
        m = df.pivot_table(index="date", columns="ticker", values=col,
                           aggfunc="last")
        out[name] = m.sort_index()

    # align every matrix to the same index and columns, so an operator can never
    # silently compare two differently-shaped frames
    ref = out["close"]
    for k in out:
        out[k] = out[k].reindex(index=ref.index, columns=ref.columns)

    if "board" in df.columns:
        out["board"] = (df.pivot_table(index="date", columns="ticker",
                                       values="board", aggfunc="last")
                        .reindex(index=ref.index, columns=ref.columns))
    return out


def liquid_universe(mats: dict[str, pd.DataFrame],
                    n: int = DEFAULT_UNIVERSE_SIZE,
                    window: int = DEFAULT_LIQUIDITY_WINDOW,
                    min_price: float = MIN_PRICE) -> pd.DataFrame:
    """Boolean matrix: is this name in the tradable universe on this date?

    Membership is the top ``n`` names by **trailing** median turnover, plus a
    minimum price. Trailing is the whole point — a full-sample liquidity screen
    selects names that went on to become liquid, and that correlates with having
    gone up.

    Names enter and leave as their liquidity changes, which is what a real
    universe does and which makes turnover in the backtest honest: rebalancing
    into a name that just entered costs money.
    """
    turn = mats.get("turnover")
    px = mats.get("close")
    if turn is None or px is None:
        raise ValueError("need 'turnover' and 'close' matrices")

    trailing = turn.shift(1).rolling(window, min_periods=max(10, window // 3)).median()
    rank = trailing.rank(axis=1, ascending=False, method="first")
    in_univ = rank <= n

    if min_price:
        in_univ &= px.shift(1) >= min_price
    # a name must actually have traded today to be holdable
    in_univ &= turn.notna() & (turn > 0)
    return in_univ.fillna(False)


def describe(mats: dict[str, pd.DataFrame],
             universe: pd.DataFrame | None = None) -> dict:
    """Summary of the panel, for the page header."""
    px = mats["close"]
    info = {
        "sessions": int(len(px)),
        "tickers": int(px.shape[1]),
        "start": str(px.index.min().date()),
        "end": str(px.index.max().date()),
        "avg_names_per_day": float(px.notna().sum(axis=1).mean()),
    }
    if universe is not None:
        info["universe_per_day"] = float(universe.sum(axis=1).mean())
        info["universe_names_ever"] = int(universe.any().sum())
    return info
