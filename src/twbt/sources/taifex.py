"""Clients for TAIFEX's public CSV download endpoints.

These are the same files anyone can download from taifex.com.tw with no account.
Three datasets are used:

    futures_daily  TAIEX futures OHLCV, settlement, open interest
    three_majors   三大法人 — daily net position by institution type
    large_trader   大額交易人 — top-5 / top-10 trader open interest

Four things about this source that the code has to respect. All four were found by
running it, not by reading documentation, and each one fails in a way that is easy
to misread.

FAILURE MODE. An invalid request returns **HTTP 200** with a Big5 HTML page
containing a JavaScript ``alert()`` and a redirect — not a 4xx, not an empty body.
Checking the status code tells you nothing. ``_post`` inspects the body and raises
with the alert's own message, so "DateTime error" surfaces as "DateTime error"
rather than as a pandas parse failure sixty lines later.

SPAN LIMITS DIFFER PER ENDPOINT, AND ARE NOT COUNTED IN DAYS. ``futDataDown``
enforces **one calendar month**, not a fixed number of days: 2024-02-01..2024-03-01
is accepted and 2024-02-01..2024-03-02 is rejected, while 2024-01-01..2024-01-31 —
the same 31 days — is fine. A 31-day chunker therefore works for ten months of the
year and fails in February, which is a delightful bug to debug in March. The chunk
size is 28 days, the shortest month, so every window is inside one calendar month
whatever the start date. The other two endpoints are looser; the limit lives on the
Endpoint rather than in a shared constant.

HISTORY DEPTH DIFFERS PER ENDPOINT. ``futures_daily`` and ``large_trader`` go back
years. ``three_majors`` serves only a **rolling ~3-year window** and returns the
DateTime error outside it. That window moves, so it is read from the query form's
own hidden ``firstDate`` / ``lastDate`` fields at runtime rather than hardcoded.

PARAMETER NAMES ARE INCONSISTENT. ``futDataDown`` wants ``commodity_id``;
``futContractsDateDown`` wants ``commodityId``. There is no pattern to infer.

Encoding: the CSVs are Big5 and a few rows carry characters outside the common
range, so decoding uses ``errors="replace"``. The affected text is in Chinese
product names that are mapped to English immediately.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import requests

BASE = "https://www.taifex.com.tw/cht"
UA = ("Mozilla/5.0 (compatible; tw-stock-backtester/0.1; "
      "+https://github.com/reggiehsu111/tw-stock-backtester)")

PACE_SEC = 1.5
RETRIES = 4

# Text columns: anything here is forced to string before it reaches parquet.
# large_trader's trader_type is blank on some rows, which pandas reads as float
# NaN and arrow then rejects as a mixed-type column.
TEXT_COLS = ("product", "identity", "expiry", "trader_type", "session")


@dataclass(frozen=True)
class Endpoint:
    """One TAIFEX download: path, form fields, column mapping, and its limits."""

    name: str
    path: str
    view: str                  # the query form, for discovering the date window
    extra: dict
    columns: dict              # Chinese header -> snake_case name
    date_col: str
    max_span_days: int         # measured, not documented
    rolling_window: bool = False   # True if only recent history is served


FUTURES_DAILY = Endpoint(
    name="futures_daily",
    path="3/futDataDown",
    view="3/futDataDown",
    extra={"down_type": "1", "commodity_id": "TX"},     # snake_case here
    columns={
        "交易日期": "date", "契約": "product", "到期月份(週別)": "expiry",
        "開盤價": "open", "最高價": "high", "最低價": "low", "收盤價": "close",
        "成交量": "volume", "結算價": "settlement", "未沖銷契約數": "open_interest",
        "最後最佳買價": "bid", "最後最佳賣價": "ask", "交易時段": "session",
    },
    date_col="交易日期",
    # one CALENDAR month, so 28 days is the largest provably-safe fixed window
    max_span_days=28,
)

THREE_MAJORS = Endpoint(
    name="three_majors",
    path="3/futContractsDateDown",
    view="3/futContractsDateView",
    extra={"down_type": "1", "commodityId": "TXF"},     # camelCase here
    columns={
        "日期": "date", "商品名稱": "product", "身份別": "identity",
        "多方交易口數": "long_lots", "空方交易口數": "short_lots",
        "多空交易口數淨額": "net_lots",
        "多方未平倉口數": "oi_long_lots", "空方未平倉口數": "oi_short_lots",
        "多空未平倉口數淨額": "oi_net_lots",
    },
    date_col="日期",
    max_span_days=90,
    rolling_window=True,                                # only ~3 years are served
)

LARGE_TRADER = Endpoint(
    name="large_trader",
    path="3/largeTraderFutDown",
    view="3/largeTraderFutView",
    extra={"down_type": "1"},
    columns={
        "日期": "date", "商品(契約)": "product", "到期月份(週別)": "expiry",
        "交易人類別": "trader_type",
        "前五大交易人買方": "top5_buy", "前五大交易人賣方": "top5_sell",
        "前十大交易人買方": "top10_buy", "前十大交易人賣方": "top10_sell",
        "全市場未沖銷部位數": "total_oi",
    },
    date_col="日期",
    # responses run to ~1.4 MB per 28 days across all products; 31 keeps each
    # request to something a slow connection finishes inside the timeout
    max_span_days=31,
)

ENDPOINTS = {e.name: e for e in (FUTURES_DAILY, THREE_MAJORS, LARGE_TRADER)}


class TaifexError(RuntimeError):
    """A request TAIFEX rejected, carrying its own explanation where there is one."""


def _is_error_page(body: bytes) -> bool:
    return b"DOCTYPE" in body[:200] or b"<html" in body[:200].lower()


def _alert_text(body: bytes) -> str:
    """Pull the message out of the JavaScript alert on an error page.

    The error page is UTF-8 -- its own meta tag says so -- while the CSV payloads
    are Big5. Decoding the error with the payload's encoding turns a perfectly
    clear Chinese message into mojibake, which is how "查詢日期區間錯誤" first
    reached the terminal as a row of replacement characters.
    """
    txt = body.decode("utf-8", errors="replace")
    m = re.search(r'alert\("([^"]*)"\)', txt)
    return m.group(1).strip() if m else "request rejected (no message)"


def _post(path: str, data: dict, session: requests.Session) -> bytes:
    """POST with backoff, treating TAIFEX's HTML error page as the failure it is."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = session.post(f"{BASE}/{path}", data=data,
                             headers={"User-Agent": UA}, timeout=90)
            r.raise_for_status()
            if _is_error_page(r.content):
                # a rejected request, not a flaky one -- retrying cannot help
                raise TaifexError(f"TAIFEX rejected the request: "
                                  f"{_alert_text(r.content)}")
            if len(r.content) < 120:
                raise ValueError(f"suspiciously short response "
                                 f"({len(r.content)} bytes)")
            return r.content
        except TaifexError:
            raise
        except Exception as exc:                       # noqa: BLE001
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(PACE_SEC * (2 ** attempt))
    raise TaifexError(f"TAIFEX {path} failed after {RETRIES} attempts: {last}")


_WINDOW_CACHE: dict[str, tuple[date, date]] = {}


def available_window(name: str,
                     session: requests.Session | None = None) -> tuple[date, date] | None:
    """The date range this endpoint actually serves, read from its own query form.

    The form carries hidden ``firstDate`` / ``lastDate`` fields that drive its
    client-side validation. Reading them is how a rolling window gets discovered
    rather than guessed, and it keeps working when the window moves.

    Returns None if the form does not advertise a window (the deep-history
    endpoints do not).
    """
    if name in _WINDOW_CACHE:
        return _WINDOW_CACHE[name]
    ep = ENDPOINTS[name]
    if not ep.rolling_window:
        return None
    own = session is None
    session = session or requests.Session()
    try:
        r = session.get(f"{BASE}/{ep.view}", headers={"User-Agent": UA}, timeout=45)
        txt = r.content.decode("utf-8", errors="replace")
        got = {}
        for key in ("firstDate", "lastDate"):
            m = re.search(rf'name="{key}"[^>]*value="([^"]+)"', txt)
            if m:
                got[key] = datetime.strptime(m.group(1)[:10], "%Y-%m-%d").date()
        if len(got) == 2:
            _WINDOW_CACHE[name] = (got["firstDate"], got["lastDate"])
            return _WINDOW_CACHE[name]
    except Exception:                                  # noqa: BLE001
        pass
    finally:
        if own:
            session.close()
    return None


def _parse(raw: bytes, ep: Endpoint) -> pd.DataFrame:
    text = raw.decode("big5", errors="replace")
    # index_col=False is load-bearing. These CSVs end each row with a trailing
    # comma, so pandas sees more fields than headers and silently promotes the
    # FIRST column to the index, shifting every name one place left. The date
    # column then holds the product code, every numeric column holds the one to
    # its left, and nothing raises -- the frame just comes back empty after the
    # date parse fails on "TX".
    df = pd.read_csv(io.StringIO(text), index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    if ep.date_col not in df.columns:
        raise TaifexError(f"{ep.name}: date column {ep.date_col!r} absent; "
                          f"got {list(df.columns)[:6]}")
    keep = {k: v for k, v in ep.columns.items() if k in df.columns}
    missing = set(ep.columns) - set(df.columns)
    df = df[list(keep)].rename(columns=keep)

    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                                format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c == "date":
            continue
        if c in TEXT_COLS:
            # force to string: a column that is blank on some rows reads as float
            # NaN, and a mixed object column cannot be written to parquet
            df[c] = df[c].fillna("").astype(str).str.strip()
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if missing:
        df.attrs["missing_columns"] = sorted(missing)
    return df.reset_index(drop=True)


def fetch_range(name: str, start: date, end: date,
                session: requests.Session | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """Download one dataset over [start, end], chunked to that endpoint's limit.

    The range is clamped to what the endpoint actually serves. Asking
    ``three_majors`` for 2020 is not an error on the caller's part — it is a
    reasonable request for data TAIFEX stopped publishing — so the range is
    narrowed and the caller is told, rather than the whole fetch failing.
    """
    if name not in ENDPOINTS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(ENDPOINTS)}")
    ep = ENDPOINTS[name]
    own = session is None
    session = session or requests.Session()

    try:
        win = available_window(name, session)
        if win:
            lo, hi = win
            if start < lo or end > hi:
                clamped_start, clamped_end = max(start, lo), min(end, hi)
                if clamped_start > clamped_end:
                    if verbose:
                        print(f"    {name}: serves only {lo}..{hi}; "
                              f"nothing in {start}..{end}")
                    return pd.DataFrame(columns=["date"])
                if verbose:
                    print(f"    {name}: serves only {lo}..{hi}; "
                          f"narrowing to {clamped_start}..{clamped_end}")
                start, end = clamped_start, clamped_end

        out, cur = [], start
        while cur <= end:
            stop = min(cur + timedelta(days=ep.max_span_days - 1), end)
            data = {
                "queryStartDate": cur.strftime("%Y/%m/%d"),
                "queryEndDate": stop.strftime("%Y/%m/%d"),
                **ep.extra,
            }
            raw = _post(ep.path, data, session)
            try:
                chunk = _parse(raw, ep)
            except TaifexError:
                raise
            except Exception as exc:                   # noqa: BLE001
                raise TaifexError(f"{name} {cur}..{stop}: {exc}") from exc
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
