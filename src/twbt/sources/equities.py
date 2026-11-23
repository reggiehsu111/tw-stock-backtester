"""Public daily cross-section of Taiwan listed equities.

TAIFEX gives one instrument. The ``f``-style operators this project mirrors are
cross-sectional — ``Rank``, ``Demean``, ``DemeanBySector`` are meaningless on a
single column — so they need a panel of names. Two free endpoints provide one, no
account required:

    TWSE  上市  /rwd/zh/afterTrading/MI_INDEX   ~1,250 names per day
    TPEX  上櫃  /www/zh-tw/afterTrading/otc       ~920 names per day

Both are **one request per trading day**, which is the main cost: three years is
roughly 1,500 requests. The cache makes that a one-time price.

Things these two share with TAIFEX, and a few they do not:

DATES ARE 民國 ON ONE SIDE. TWSE takes an ISO-ish ``20240902``; TPEX takes
``2024/09/02`` but *returns* ``113/09/02``. Republic-of-China years are +1911.

A NON-TRADING DAY RETURNS HTTP 200 WITH NO ROWS, not an error. That is normal and
is skipped, which is also why a missing interior date is not something the cache
tries to patch.

NUMBERS ARE STRINGS WITH COMMAS, and a suspended or untraded name carries ``--``
rather than a null. Both become NaN.

THE TABLE IS NOT ALWAYS THE FIRST ONE. TWSE returns several tables per response
(indices, then the quote table); the right one is found by looking for 證券代號 in
its fields rather than by position, which moves between dates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

UA = ("Mozilla/5.0 (compatible; tw-stock-backtester/0.1; "
      "+https://github.com/reggiehsu111/tw-stock-backtester)")
PACE_SEC = 1.1
RETRIES = 3

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"

# Chinese header -> our name. The two venues disagree on spelling, spacing and
# column order, and TPEX pads several of its headers with trailing spaces.
TWSE_COLS = {
    "證券代號": "ticker", "證券名稱": "name", "開盤價": "open", "最高價": "high",
    "最低價": "low", "收盤價": "close", "成交股數": "shares", "成交金額": "amount",
    "成交筆數": "trades",
}
TPEX_COLS = {
    "代號": "ticker", "名稱": "name", "開盤": "open", "最高": "high",
    "最低": "low", "收盤": "close", "成交股數": "shares", "成交金額(元)": "amount",
    "成交筆數": "trades", "發行股數": "shares_out",
}

NUMERIC = ("open", "high", "low", "close", "shares", "amount", "trades",
           "shares_out")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str)
                     .str.replace(",", "", regex=False)
                     .str.replace("--", "", regex=False)
                     .str.strip(),
                errors="coerce")
    for c in ("ticker", "name"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _get(url: str, params: dict, session: requests.Session) -> dict | None:
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, headers={"User-Agent": UA},
                            timeout=45)
            r.raise_for_status()
            return r.json()
        except Exception as exc:                        # noqa: BLE001
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(PACE_SEC * (2 ** attempt))
    raise RuntimeError(f"{url} failed after {RETRIES} attempts: {last}")


def _pick_table(payload: dict, key_field: str) -> dict | None:
    """The quote table, found by its fields rather than by position."""
    for t in (payload.get("tables") or []):
        if key_field in [str(f).strip() for f in (t.get("fields") or [])]:
            return t
    return None


def fetch_twse_day(d: date, session: requests.Session) -> pd.DataFrame:
    """上市 quotes for one date. Empty frame on a non-trading day."""
    j = _get(TWSE_URL, {"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999",
                        "response": "json"}, session)
    if not j or j.get("stat") != "OK":
        return pd.DataFrame()
    t = _pick_table(j, "證券代號")
    if not t or not t.get("data"):
        return pd.DataFrame()
    raw = pd.DataFrame(t["data"], columns=[str(f).strip() for f in t["fields"]])
    keep = {k: v for k, v in TWSE_COLS.items() if k in raw.columns}
    out = _clean(raw[list(keep)].rename(columns=keep))
    out["board"] = "TSE"
    out["date"] = pd.Timestamp(d)
    return out


def fetch_tpex_day(d: date, session: requests.Session) -> pd.DataFrame:
    """上櫃 quotes for one date. Empty frame on a non-trading day."""
    j = _get(TPEX_URL, {"date": d.strftime("%Y/%m/%d"), "type": "EW",
                        "response": "json"}, session)
    if not j:
        return pd.DataFrame()
    t = _pick_table(j, "代號")
    if not t or not t.get("data"):
        return pd.DataFrame()
    fields = [str(f).replace("<br>", "").strip() for f in t["fields"]]
    rows = [r[:len(fields)] for r in t["data"]]
    raw = pd.DataFrame(rows, columns=fields)
    keep = {k: v for k, v in TPEX_COLS.items() if k in raw.columns}
    out = _clean(raw[list(keep)].rename(columns=keep))
    out["board"] = "OTC"
    out["date"] = pd.Timestamp(d)
    return out


def _is_equity(ticker: str) -> bool:
    """Ordinary shares only: a 4-digit code.

    Excludes ETFs (0050, 00878), warrants (6-digit), TDRs, preferred shares and
    the rest. They trade differently enough that leaving them in the universe
    makes a cross-sectional rank mean something different on different days.
    """
    t = str(ticker).strip()
    return len(t) == 4 and t.isdigit() and not t.startswith("00")


def fetch_range(start: date, end: date, session: requests.Session | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Daily cross-section over [start, end], both boards, ordinary shares only.

    Returns long format: one row per (date, ticker). Weekends are skipped without
    a request; other non-trading days cost one request each and return nothing,
    which is the cheapest way to discover the calendar from a source that does
    not publish one.
    """
    own = session is None
    session = session or requests.Session()
    out: list[pd.DataFrame] = []
    cur = start
    n_days = 0
    try:
        while cur <= end:
            if cur.weekday() >= 5:
                cur += timedelta(days=1)
                continue
            parts = []
            for fn in (fetch_twse_day, fetch_tpex_day):
                try:
                    parts.append(fn(cur, session))
                except Exception as exc:                # noqa: BLE001
                    if verbose:
                        print(f"    {cur} {fn.__name__}: {exc}")
                time.sleep(PACE_SEC)
            day = pd.concat([p for p in parts if len(p)], ignore_index=True) \
                if any(len(p) for p in parts) else pd.DataFrame()
            if len(day):
                day = day[day["ticker"].map(_is_equity)]
                out.append(day)
                n_days += 1
                if verbose and n_days % 20 == 0:
                    print(f"    {cur}: {n_days} trading days, "
                          f"{sum(len(d) for d in out):,} rows")
            cur += timedelta(days=1)
    finally:
        if own:
            session.close()

    if not out:
        return pd.DataFrame(columns=["date", "ticker"])
    return (pd.concat(out, ignore_index=True)
            .drop_duplicates(subset=["date", "ticker"])
            .sort_values(["date", "ticker"])
            .reset_index(drop=True))
