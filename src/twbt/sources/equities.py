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

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

UA = ("Mozilla/5.0 (compatible; tw-stock-backtester/0.1; "
      "+https://github.com/reggiehsu111/tw-stock-backtester)")
PACE_SEC = 1.1
# TWSE needs its own, slower pace. 1.1s with six workers got the client blocked
# across half the range; 3.0s serially completed 347 sessions with zero
# failures. 2.5s is that measurement with a little room, not a guess.
PACE_TWSE = 2.5
RETRIES = 3

# Per-venue concurrency, because the two services tolerate very different loads.
#
# TWSE is SERIAL. Six workers against it looked fine -- every request either
# returned data or raised -- and it quietly got the client WAF-blocked for half
# the range, which arrived as HTTP 307 to a page reading 因為安全性考量，您所執行
# 的頁面無法呈現. Nothing in the response is a 4xx or 5xx, so raise_for_status()
# is silent and only the JSON decode fails. Going fast here is how the cache
# ended up missing 350 of 698 sessions.
#
# TPEX handled the same concurrency without complaint and is left parallel.
WORKERS_TWSE = 1
WORKERS_TPEX = 4
WORKERS = 4                       # kept for callers that pass nothing


class BlockedError(RuntimeError):
    """The venue is refusing us, and retrying sooner will not help."""

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


def _is_block(r: requests.Response) -> bool:
    """Is this TWSE's rate-limit page rather than data?

    It answers with a 307 and an HTML body, so the status code is not an error
    and raise_for_status() says nothing. Detecting it explicitly is the
    difference between "back off" and "retry three times, then record a failure
    and keep hammering".
    """
    if r.status_code in (301, 302, 307, 308):
        return True
    ctype = r.headers.get("content-type", "")
    return "json" not in ctype and b"\xe5\xae\x89\xe5\x85\xa8\xe6\x80\xa7" in r.content[:600]


def _get(url: str, params: dict, session: requests.Session) -> dict | None:
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, headers={"User-Agent": UA},
                            timeout=45)
            if _is_block(r):
                raise BlockedError(
                    f"{url.rsplit('/', 1)[-1]} is refusing requests "
                    f"(HTTP {r.status_code}). Wait before retrying; fetching "
                    "faster is what causes this.")
            r.raise_for_status()
            return r.json()
        except BlockedError:
            raise                                       # retrying makes it worse
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


def _fetch_venue(fn, days, workers: int, verbose: bool, label: str,
                 checkpoint=None, checkpoint_every: int = 40):
    """Fetch one venue over a list of dates, at that venue's own concurrency.

    Returns (frames, failures). A ``BlockedError`` stops the pass immediately:
    the service is telling us to go away, and the correct response is to stop,
    not to work through the remaining three hundred dates collecting failures.
    """
    out, failed, done = [], [], 0
    local = threading.local()

    def worker(d: date):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        time.sleep(PACE_SEC * (0.5 + random.random()))
        return d, fn(d, local.session)

    if workers <= 1:                       # serial, and deliberately so for TWSE
        pace = PACE_TWSE if label == "TSE" else PACE_SEC
        sess = requests.Session()
        for d in days:
            try:
                frame = fn(d, sess)
            except BlockedError as exc:
                if verbose:
                    print(f"    {label}: BLOCKED at {d} -- {exc}", flush=True)
                failed.append(d)
                break
            except Exception:              # noqa: BLE001
                failed.append(d)
                frame = None
            if frame is not None and len(frame):
                out.append(frame)
            done += 1
            if verbose and done % 25 == 0:
                print(f"    {label}: {done}/{len(days)} days, "
                      f"{sum(len(x) for x in out):,} rows, "
                      f"{len(failed)} failed", flush=True)
            if checkpoint and out and done % checkpoint_every == 0:
                checkpoint(_combine(out))
            time.sleep(pace)
        sess.close()
        return out, failed

    blocked = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, d): d for d in days}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                _, frame = fut.result()
            except BlockedError as exc:
                if not blocked and verbose:
                    print(f"    {label}: BLOCKED -- {exc}", flush=True)
                blocked = True
                failed.append(d)
                continue
            except Exception:              # noqa: BLE001
                failed.append(d)
                continue
            if len(frame):
                out.append(frame)
            done += 1
            if verbose and done % 25 == 0:
                print(f"    {label}: {done}/{len(days)} days, "
                      f"{sum(len(x) for x in out):,} rows, "
                      f"{len(failed)} failed", flush=True)
            if checkpoint and out and done % checkpoint_every == 0:
                checkpoint(_combine(out))
    return out, failed


def fetch_range(start: date, end: date, session: requests.Session | None = None,
                verbose: bool = True, workers: int | None = None,
                checkpoint=None, checkpoint_every: int = 40) -> pd.DataFrame:
    """Daily cross-section over [start, end], both boards, ordinary shares only.

    Returns long format: one row per (date, ticker).

    The two venues are fetched in separate passes at their own concurrency --
    TWSE serially, TPEX in parallel -- because they tolerate very different
    loads and a shared setting has to be as slow as the slower one or as broken
    as the faster one. See WORKERS_TWSE for what "broken" looked like.

    ``checkpoint`` is called with the frame so far every ``checkpoint_every``
    days of each pass. A long fetch that writes nothing until the end has no
    progress to show and nothing to keep if it stops, which has now cost this
    project two runs.

    Weekends are skipped without a request. Other non-trading days cost one
    request and return nothing, which is the cheapest way to discover the
    calendar from a source that does not publish one.
    """
    days = [d for d in _daterange(start, end) if d.weekday() < 5]
    if not days:
        return pd.DataFrame(columns=["date", "ticker"])

    collected: list[pd.DataFrame] = []

    def chained(partial):
        if checkpoint:
            checkpoint(_combine(collected + [partial]) if collected else partial)

    frames_tse, failed_tse = _fetch_venue(
        fetch_twse_day, days, WORKERS_TWSE, verbose, "TSE",
        chained, checkpoint_every)
    collected += frames_tse

    frames_otc, failed_otc = _fetch_venue(
        fetch_tpex_day, days, workers or WORKERS_TPEX, verbose, "OTC",
        chained, checkpoint_every)
    collected += frames_otc

    if verbose:
        for label, failed in (("TSE", failed_tse), ("OTC", failed_otc)):
            if failed:
                print(f"    WARNING: {label} failed on {len(failed)}/{len(days)} "
                      f"days, first {failed[0]}", flush=True)

    if not collected:
        return pd.DataFrame(columns=["date", "ticker"])
    combined = _combine(collected)
    combined = combined[combined["ticker"].map(_is_equity)]
    _warn_on_gaps(combined, verbose)
    return combined.reset_index(drop=True)


def _warn_on_gaps(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Report sessions where a venue is entirely absent.

    A partial fetch is indistinguishable from a quiet market unless something
    counts it, and the shape of this particular failure -- one venue missing on
    half the days -- produced a universe that silently excluded TSMC from 63% of
    sessions while every chart still rendered.
    """
    if df.empty or "board" not in df.columns:
        return {}
    per = df.groupby(["date", "board"]).size().unstack(fill_value=0)
    gaps = {str(b): int((per[b] == 0).sum()) for b in per.columns}
    if verbose:
        for board, n in sorted(gaps.items()):
            if n:
                print(f"    WARNING: {board} missing on {n}/{len(per)} sessions",
                      flush=True)
    return gaps


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["date", "ticker"])
            .sort_values(["date", "ticker"])
            .reset_index(drop=True))


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)
