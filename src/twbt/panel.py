"""The aligned daily panel, and the only normalisation functions in the project.

Two jobs.

``build()`` turns the raw TAIFEX downloads into one row per trading day: front-month
futures OHLCV plus institutional and large-trader columns, joined on date.

``zscore`` / ``trailing_quantile`` / ``trailing_rank`` normalise a series. Each one
compares day t against days t-1 and earlier only, never against t itself or later:
the first two ``.shift(1)`` before ``.rolling(...)``; ``trailing_rank`` instead uses a
kernel one observation wider and ranks its last element against the rest, which is
the same guarantee by a different route. There is deliberately no
full-sample equivalent anywhere in the API: a ``series.quantile(1/3)`` is one
character away from a ``series.rolling(250).quantile(1/3)`` and reads almost the
same in a diff, which is how that bug survives review. Not providing it is cheaper
than catching it.

``tests/test_no_lookahead.py`` corrupts the back half of a series and asserts the
front half's signals are bit-identical.
"""

from __future__ import annotations

import pandas as pd

from . import cache

# TAIFEX's own labels. The front-month TAIEX future is 臺股期貨 / TX; the identity
# column in three_majors reads 外資及陸資 on the live endpoint -- not the bare 外資
# that appears in older documentation and in some internal tables.
PRODUCT_TXF = "臺股期貨"
IDENTITY = {"dealer": "自營商", "trust": "投信", "foreign": "外資及陸資"}

# large_trader encodes two of its columns as NUMBERS, not labels, which is not
# obvious from the Chinese headers and produces an empty join if you filter on text.
EXPIRY_ALL_MONTHS = 999999      # every contract month combined
EXPIRY_WEEKLIES = 666666        # the weekly contracts only
TRADER_ALL = 0                  # 所有交易人
TRADER_SPECIFIC = 1             # 特定法人
LT_PRODUCT_TX = "TX"            # 臺股期貨(TX+MTX/4+TMF/20), already TAIEX-equivalent

MIN_WINDOW = 120          # a trailing stat needs this much history before it is used
DEFAULT_WINDOW = 250      # ~one year of sessions


# --------------------------------------------------------------------------- #
# trailing-only normalisation
# --------------------------------------------------------------------------- #

def zscore(s: pd.Series, window: int = DEFAULT_WINDOW,
           min_periods: int = MIN_WINDOW) -> pd.Series:
    """(x_t - trailing mean) / trailing sd, using data strictly before t."""
    base = s.shift(1)
    mu = base.rolling(window, min_periods=min_periods).mean()
    sd = base.rolling(window, min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd.where(sd > 0)


def trailing_quantile(s: pd.Series, q: float, window: int = DEFAULT_WINDOW,
                      min_periods: int = MIN_WINDOW) -> pd.Series:
    """The q-th quantile of the trailing window, as of the previous day.

    Compare the raw series against this, not against ``s.quantile(q)``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")
    return s.shift(1).rolling(window, min_periods=min_periods).quantile(q)


def trailing_rank(s: pd.Series, window: int = DEFAULT_WINDOW,
                  min_periods: int = MIN_WINDOW) -> pd.Series:
    """Where x_t sits within its trailing window, in [0, 1].

    Rank rather than z-score when the series is skewed or fat-tailed — volume is
    the usual case, where a single expiry day can dominate a rolling standard
    deviation for a year.

    The kernel is ``window + 1`` wide and compares its last element against the
    ``window`` before it, so day t is ranked against days t-1 and earlier only.
    """

    def _pos(w) -> float:
        return float((w[:-1] < w[-1]).mean())

    return s.rolling(window + 1, min_periods=min_periods + 1).apply(_pos, raw=True)


def realised_vol(ret: pd.Series, window: int = 20,
                 min_periods: int = 10) -> pd.Series:
    """Trailing standard deviation of returns, as of the previous day."""
    return ret.shift(1).rolling(window, min_periods=min_periods).std(ddof=1)


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #

def _front_month(fut: pd.DataFrame) -> pd.DataFrame:
    """One row per date: the front-month TX day-session contract.

    'Front month' is the nearest expiry with real volume. Picking by expiry label
    alone breaks on expiry day itself, when the expiring contract still exists but
    has stopped trading; picking the most-traded contract handles the roll without
    a calendar.
    """
    df = fut[fut["product"].astype(str).str.contains("TX|臺股期貨", regex=True, na=False)].copy()
    if "session" in df.columns:
        # 一般 = regular day session; the after-hours file is a separate row
        night = df["session"].astype(str).str.contains("盤後|夜", na=False)
        if night.any() and (~night).any():
            df = df[~night]
    if "expiry" in df.columns:
        # weekly contracts carry a 'W' in the expiry label; monthlies are the series
        # the strategies trade, and mixing them makes volume ranking meaningless
        df = df[~df["expiry"].astype(str).str.contains("W", na=False)]
    if df.empty:
        raise ValueError("no TX day-session rows in futures_daily")
    df = df.sort_values(["date", "volume"], ascending=[True, False])
    return df.groupby("date", as_index=False).first()


def _majors_wide(maj: pd.DataFrame) -> pd.DataFrame:
    """三大法人 long -> wide: one column per institution per measure."""
    df = maj[maj["product"].astype(str).str.strip() == PRODUCT_TXF].copy()
    if df.empty:                       # some vintages label it differently
        df = maj.copy()
    df["identity"] = df["identity"].astype(str).str.strip()
    out = []
    for key, label in IDENTITY.items():
        part = df[df["identity"] == label]
        if part.empty:
            continue
        cols = [c for c in ("net_lots", "oi_net_lots") if c in part.columns]
        w = part.set_index("date")[cols].rename(columns={c: f"{key}_{c}" for c in cols})
        out.append(w[~w.index.duplicated(keep="first")])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1).sort_index()


def _trader_wide(lt: pd.DataFrame) -> pd.DataFrame:
    """大額交易人 -> daily concentration measures for the TAIEX futures complex.

    Three filters, each of which silently empties the frame if written as a text
    match, because TAIFEX encodes these as numbers:

        product    "TX"    -- already the combined TX + MTX/4 + TMF/20 series
        expiry     999999  -- all contract months (666666 is weeklies only)
        trader_type 0      -- all large traders (1 is specific institutions only)

    ``top5_net_frac`` is the top five's net position as a share of total open
    interest, and ``top5_conc`` their gross share. Both are already normalised by
    open interest, so unlike raw lot counts they are comparable across years.
    """
    df = lt.copy()
    for c in ("expiry", "trader_type", "top5_buy", "top5_sell",
              "top10_buy", "top10_sell", "total_oi"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "product" in df.columns:
        df = df[df["product"].astype(str).str.strip() == LT_PRODUCT_TX]
    if "expiry" in df.columns:
        df = df[df["expiry"] == EXPIRY_ALL_MONTHS]
    if "trader_type" in df.columns:
        df = df[df["trader_type"] == TRADER_ALL]

    cols = [c for c in ("top5_buy", "top5_sell", "top10_buy", "top10_sell", "total_oi")
            if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame()

    g = df.groupby("date")[cols].last().sort_index()
    if {"top5_buy", "top5_sell", "total_oi"} <= set(cols):
        oi = g["total_oi"].where(g["total_oi"] > 0)
        g["top5_net_frac"] = (g["top5_buy"] - g["top5_sell"]) / oi
        g["top5_conc"] = (g["top5_buy"] + g["top5_sell"]) / (2 * oi)
    if {"top10_buy", "top10_sell", "total_oi"} <= set(cols):
        oi = g["total_oi"].where(g["total_oi"] > 0)
        g["top10_net_frac"] = (g["top10_buy"] - g["top10_sell"]) / oi
    return g


def build(root=None, fixture: bool = False) -> pd.DataFrame:
    """Assemble the daily panel from cached (or fixture) TAIFEX data.

    Returns a date-indexed frame. Derived columns:

        ret          close-to-close return, in basis points
        ret_open     previous close to today's open, in bps (the overnight gap)
        ret_intra    today's open to today's close, in bps
        range_bps    (high - low) / close, in bps
        vol_20       trailing 20-day sd of ret, as of yesterday
        spread_bps   measured closing bid-ask width, in bps (NaN where implausible)

    Institutional columns are left on the date they were *published*. Shifting them
    into tradable alignment is the backtest engine's job, not the panel's — see
    ``backtest.run``, which applies an explicit publication lag so the lag appears
    once, in one place, where it can be tested.
    """
    kw = {"fixture": fixture} if fixture else {"root": root}
    fut = _front_month(cache.load("futures_daily", **kw))

    p = fut.set_index("date").sort_index()
    for c in ("open", "high", "low", "close", "volume", "settlement", "open_interest"):
        if c not in p.columns:
            p[c] = float("nan")

    p["ret"] = p["close"].pct_change() * 1e4
    p["ret_open"] = (p["open"] / p["close"].shift(1) - 1.0) * 1e4
    p["ret_intra"] = (p["close"] / p["open"] - 1.0) * 1e4
    p["range_bps"] = (p["high"] - p["low"]) / p["close"] * 1e4
    p["vol_20"] = realised_vol(p["ret"])
    p["oi_chg"] = p["open_interest"].diff()

    # Measured closing spread, from the bid/ask columns TAIFEX already ships.
    # This is one snapshot per session at the close rather than a time-weighted
    # average, so it is an estimate and a slightly conservative one -- the close
    # is wider than mid-session. It is still far better than a constant, because
    # the spread is systematically wider on volatile days, which are exactly the
    # days most signals fire. Implausible values (crossed quotes, or a width over
    # 30 bps) are dropped rather than clipped, so the cost model falls back to its
    # default for those sessions instead of charging a parse error.
    if {"bid", "ask"} <= set(p.columns):
        sp = (p["ask"] - p["bid"]) / p["close"].where(p["close"] > 0) * 1e4
        p["spread_bps"] = sp.where((sp >= 0) & (sp < 30))

    for name, fn in (("three_majors", _majors_wide), ("large_trader", _trader_wide)):
        try:
            extra = fn(cache.load(name, **kw))
        except FileNotFoundError:
            continue
        if len(extra):
            p = p.join(extra, how="left")

    p.index.name = "date"
    return p


# --------------------------------------------------------------------------- #
# normalisation, as an explicit choice
# --------------------------------------------------------------------------- #

NORMALISATIONS = {
    "zscore": "(x − trailing mean) / trailing sd",
    "rank": "position within the trailing window, 0 to 1",
    "robust": "(x − trailing median) / trailing IQR",
    "minmax": "scaled to the trailing range, 0 to 1",
    "none": "raw value, no normalisation",
}


def robust_zscore(s: pd.Series, window: int = DEFAULT_WINDOW,
                  min_periods: int = MIN_WINDOW) -> pd.Series:
    """Median and IQR instead of mean and sd.

    Worth having as an option because one limit-up day can dominate a rolling
    standard deviation for a year, which makes everything after it look calm.
    The IQR does not care how far the outlier went.
    """
    base = s.shift(1).rolling(window, min_periods=min_periods)
    med = base.median()
    iqr = base.quantile(0.75) - base.quantile(0.25)
    return (s - med) / iqr.where(iqr > 0)


def trailing_minmax(s: pd.Series, window: int = DEFAULT_WINDOW,
                    min_periods: int = MIN_WINDOW) -> pd.Series:
    """Where x sits in the trailing range, 0 to 1. Bounded, and very outlier-sensitive."""
    base = s.shift(1).rolling(window, min_periods=min_periods)
    lo, hi = base.min(), base.max()
    span = (hi - lo)
    return (s - lo) / span.where(span > 0)


def normalize(s: pd.Series, method: str = "zscore",
              window: int = DEFAULT_WINDOW,
              min_periods: int = MIN_WINDOW) -> pd.Series:
    """Normalise a series by the named method, always trailing-only.

    Exists so the choice is a parameter rather than a line of code buried in a
    strategy. Which one you pick changes results more than most people expect --
    a z-score and a rank disagree sharply whenever the distribution is skewed,
    which for volume and turnover is always -- and a strategy that only works
    under one of them is telling you something.

    Every method here shifts before it looks back. There is no option that uses
    the current observation to set the scale it is measured against.
    """
    if method not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {method!r}; "
                         f"choose from {sorted(NORMALISATIONS)}")
    if method == "zscore":
        return zscore(s, window, min_periods)
    if method == "rank":
        return trailing_rank(s, window, min_periods)
    if method == "robust":
        return robust_zscore(s, window, min_periods)
    if method == "minmax":
        return trailing_minmax(s, window, min_periods)
    return s
