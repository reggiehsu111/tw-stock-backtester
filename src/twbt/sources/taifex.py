"""Clients for TAIFEX's public CSV download endpoints.

These are the same files anyone can download from taifex.com.tw with no account.
Three datasets are used:

    futures_daily  TAIEX futures OHLCV, settlement, open interest
    three_majors   三大法人 — daily net position by institution type
    large_trader   大額交易人 — top-5 / top-10 trader open interest

Two things about this source that the code has to respect:

ENCODING. The CSVs are Big5, not UTF-8, and a handful of rows carry characters
outside the common Big5 range. Decoding with ``errors="replace"`` keeps a bad
character from killing an otherwise good month of data; the affected text is in
Chinese product names we immediately map to English anyway.

RATE. TAIFEX is a public service with no documented quota, and it slows down under
load rather than returning an error. Requests are paced and retried with backoff.
A slow response is the expected failure mode, not a 429.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://www.taifex.com.tw/cht"
UA = "Mozilla/5.0 (compatible; tw-stock-backtester/0.1; +https://github.com/reggiehsu111/tw-stock-backtester)"

# TAIFEX caps each download at roughly a season of rows, and rejects wider ranges
# silently by returning only the head of the period. Chunking keeps every request
# inside that window.
MAX_SPAN_DAYS = 60
PACE_SEC = 1.5
RETRIES = 4


@dataclass(frozen=True)
class Endpoint:
    """One TAIFEX download: its path, its form fields, and how to name its columns."""

    name: str
    path: str
    extra: dict
    columns: dict          # Chinese header -> snake_case name
    date_col: str


FUTURES_DAILY = Endpoint(
    name="futures_daily",
    path="3/futDataDown",
    extra={"down_type": "1", "commodity_id": "TX"},
    columns={
        "交易日期": "date", "契約": "product", "到期月份(週別)": "expiry",
        "開盤價": "open", "最高價": "high", "最低價": "low", "收盤價": "close",
        "成交量": "volume", "結算價": "settlement", "未沖銷契約數": "open_interest",
        "最後最佳買價": "bid", "最後最佳賣價": "ask", "交易時段": "session",
    },
    date_col="交易日期",
)

THREE_MAJORS = Endpoint(
    name="three_majors",
    path="3/futContractsDateDown",
    extra={"down_type": "1", "commodity_id": "TXF"},
    columns={
        "日期": "date", "商品名稱": "product", "身份別": "identity",
        "多方交易口數": "long_lots", "空方交易口數": "short_lots",
        "多空交易口數淨額": "net_lots",
        "多方未平倉口數": "oi_long_lots", "空方未平倉口數": "oi_short_lots",
        "多空未平倉口數淨額": "oi_net_lots",
    },
    date_col="日期",
)

LARGE_TRADER = Endpoint(
    name="large_trader",
    path="3/largeTraderFutDown",
    extra={"down_type": "1"},
    columns={
        "日期": "date", "商品(契約)": "product", "到期月份(週別)": "expiry",
        "交易人類別": "trader_type",
        "前五大交易人買方": "top5_buy", "前五大交易人賣方": "top5_sell",
        "前十大交易人買方": "top10_buy", "前十大交易人賣方": "top10_sell",
        "全市場未沖銷部位數": "total_oi",
    },
    date_col="日期",
)

ENDPOINTS = {e.name: e for e in (FUTURES_DAILY, THREE_MAJORS, LARGE_TRADER)}


def _post(path: str, data: dict, session: requests.Session) -> bytes:
    """POST with backoff. TAIFEX degrades by getting slow, so timeouts are retried."""
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.post(f"{BASE}/{path}", data=data,
                             headers={"User-Agent": UA}, timeout=45)
            r.raise_for_status()
            if len(r.content) < 200:
                raise ValueError(f"suspiciously short response ({len(r.content)} bytes)")
            return r.content
        except Exception as exc:                       # noqa: BLE001
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(PACE_SEC * (2 ** attempt))
    raise RuntimeError(f"TAIFEX {path} failed after {RETRIES} attempts: {last}")


def _parse(raw: bytes, ep: Endpoint) -> pd.DataFrame:
    text = raw.decode("big5", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    keep = {k: v for k, v in ep.columns.items() if k in df.columns}
    missing = set(ep.columns) - set(df.columns)
    if ep.date_col not in df.columns:
        raise ValueError(f"{ep.name}: date column {ep.date_col!r} absent; "
                         f"got {list(df.columns)[:6]}")
    df = df[list(keep)].rename(columns=keep)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                                format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c in ("date", "product", "identity", "expiry", "trader_type", "session"):
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if missing:
        df.attrs["missing_columns"] = sorted(missing)
    return df.reset_index(drop=True)


def fetch_range(name: str, start: date, end: date,
                session: requests.Session | None = None) -> pd.DataFrame:
    """Download one dataset over [start, end], chunked to TAIFEX's per-request span.

    Returns a tidy frame with a ``date`` column. Chunks that come back empty are
    skipped rather than raising: TAIFEX returns an empty body for a window that
    contains no trading days, which is normal around new year.
    """
    if name not in ENDPOINTS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(ENDPOINTS)}")
    ep = ENDPOINTS[name]
    own = session is None
    session = session or requests.Session()
    out, cur = [], start
    try:
        while cur <= end:
            stop = min(cur + timedelta(days=MAX_SPAN_DAYS - 1), end)
            data = {
                "queryStartDate": cur.strftime("%Y/%m/%d"),
                "queryEndDate": stop.strftime("%Y/%m/%d"),
                **ep.extra,
            }
            raw = _post(ep.path, data, session)
            try:
                chunk = _parse(raw, ep)
            except Exception as exc:                   # noqa: BLE001
                raise RuntimeError(f"{name} {cur}..{stop}: {exc}") from exc
            if len(chunk):
                out.append(chunk)
            cur = stop + timedelta(days=1)
            time.sleep(PACE_SEC)
    finally:
        if own:
            session.close()
    if not out:
        return pd.DataFrame(columns=["date"])
    return (pd.concat(out, ignore_index=True)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True))
