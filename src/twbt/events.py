"""Event calendars for the futures tearsheet: expiries, and whatever you supply.

Two kinds of event, and they differ in how much you should trust them.

EXPIRY is derived from the data. A TAIFEX monthly contract settles on the third
Wednesday, but that is a rule about a calendar, not about a market: when the
third Wednesday is a holiday the contract trades on. February 2026 settled on
the 23rd rather than the 18th because Lunar New Year closed the exchange.
Checking the rule against observed last-trading-days over this sample, 34 of 41
contracts matched and the misses were that holiday plus contracts that have not
expired yet.

So ``expiry_dates`` reads the last session each contract actually traded, and
falls back to the rule only for months with no data. Derived-from-data is exact
where data exists; the rule is an estimate everywhere else, and the two are
labelled differently in the result so a chart cannot quietly mix them.

USER EVENTS are whatever you put in a CSV. This exists because the obvious thing
to overlay -- TSMC's earnings dates -- is **not available from any public
endpoint I could reach**. TWSE's open data publishes the latest quarter, the
latest month and today's announcements; MOPS does not serve a conference-date
archive. Rather than derive dates from TSMC's usual mid-January / mid-April /
mid-July / mid-October rhythm and present a guess as a fact, the tearsheet takes
a file:

    date,label
    2024-01-18,TSMC Q4 2023
    2024-04-18,TSMC Q1 2024

Put it at ``data/events/<name>.csv`` and it appears as an overlay and an event
study. A derived earnings date would be wrong by a day or two in a way that
matters enormously for a one-day event study and not at all for a chart
annotation, which is precisely the kind of error worth refusing to make.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "events"


def third_wednesday(year: int, month: int) -> pd.Timestamp:
    """The TAIFEX settlement rule, before holidays are taken into account."""
    weeks = calendar.monthcalendar(year, month)
    weds = [w[calendar.WEDNESDAY] for w in weeks if w[calendar.WEDNESDAY]]
    return pd.Timestamp(year, month, weds[2])


@dataclass
class ExpiryCalendar:
    """Settlement dates, with their provenance attached.

    ``observed`` holds contracts whose last trading day is in the data.
    ``estimated`` holds the rule applied to months with none. Keeping them apart
    stops an event study silently averaging exact dates with guesses.
    """

    observed: pd.Series          # expiry label -> last trading date
    estimated: pd.Series

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            sorted(set(self.observed.tolist()) | set(self.estimated.tolist())))

    @property
    def trusted_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(set(self.observed.tolist())))


def expiry_dates(futures: pd.DataFrame, sessions: pd.DatetimeIndex | None = None
                 ) -> ExpiryCalendar:
    """Settlement dates for TAIEX monthly futures.

    Weekly contracts (a ``W`` in the expiry label) and calendar-spread rows (a
    ``/``) are excluded: a spread is two contracts quoted as one instrument and
    has no settlement date of its own.
    """
    df = futures.copy()
    if "product" in df.columns:
        df = df[df["product"].astype(str).str.contains("TX|臺股期貨", regex=True,
                                                       na=False)]
    if "session" in df.columns:
        night = df["session"].astype(str).str.contains("盤後|夜", na=False)
        if (~night).any():
            df = df[~night]

    exp = df["expiry"].astype(str).str.strip()
    monthly = exp.str.fullmatch(r"\d{6}")
    df = df[monthly.fillna(False)]
    if df.empty:
        return ExpiryCalendar(pd.Series(dtype="datetime64[ns]"),
                              pd.Series(dtype="datetime64[ns]"))

    last = df.groupby(df["expiry"].astype(str).str.strip())["date"].max()

    # A contract whose last observed session is the final session in the sample
    # has not expired; we simply stopped watching. Treat it as unknown rather
    # than as a settlement on the last day of the data.
    end = df["date"].max()
    observed = last[last < end]
    unresolved = last[last >= end]

    est = {}
    for label in unresolved.index:
        y, m = int(label[:4]), int(label[4:])
        est[label] = third_wednesday(y, m)
    estimated = pd.Series(est, dtype="datetime64[ns]")

    if sessions is not None and len(estimated):
        # roll a holiday forward to the next session that exists
        idx = pd.DatetimeIndex(sessions)
        estimated = estimated.map(
            lambda d: idx[idx >= d][0] if (idx >= d).any() else d)

    return ExpiryCalendar(observed.sort_values(), estimated.sort_values())


def days_to_expiry(sessions: pd.DatetimeIndex,
                   expiries: pd.DatetimeIndex) -> pd.Series:
    """Trading sessions until the next settlement, 0 on settlement day itself.

    Counted in sessions rather than calendar days, because the thing a
    settlement effect runs on is trading time. A week containing a holiday is
    four days of positioning, not five.
    """
    sessions = pd.DatetimeIndex(sessions)
    expiries = pd.DatetimeIndex(sorted(expiries))
    if not len(expiries):
        return pd.Series(np.nan, index=sessions)

    pos = np.arange(len(sessions))
    out = np.full(len(sessions), np.nan)
    exp_pos = sessions.get_indexer(expiries, method="bfill")
    for ep in exp_pos:
        if ep < 0:
            continue
        reach = max(0, ep - 40)
        seg = slice(reach, ep + 1)
        cand = pos[ep] - pos[seg]
        cur = out[seg]
        out[seg] = np.where(np.isnan(cur), cand, np.minimum(cur, cand))
    return pd.Series(out, index=sessions, name="days_to_expiry")


def load_events(name: str, directory: Path | None = None) -> pd.DataFrame:
    """Read a user-supplied event file: a ``date`` column and a ``label``.

    Returns an empty frame when the file is absent, because a missing overlay
    should leave a panel saying so rather than failing the whole page.
    """
    d = directory or EVENTS_DIR
    path = (d / name) if str(name).endswith(".csv") else (d / f"{name}.csv")
    if not path.exists():
        return pd.DataFrame(columns=["date", "label"])
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"{path.name}: needs a 'date' column, got {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[cols["date"]], errors="coerce")})
    out["label"] = (df[cols["label"]].astype(str) if "label" in cols
                    else path.stem)
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def available_events(directory: Path | None = None) -> list[str]:
    d = directory or EVENTS_DIR
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def event_window(ret: pd.Series, dates, before: int = 10, after: int = 10
                 ) -> pd.DataFrame:
    """Returns aligned on an event-relative session axis.

    One row per event, one column per offset from -before to +after. Events too
    close to the ends of the sample are dropped rather than padded, because a
    half-window averaged with full ones silently reweights the tails.
    """
    r = pd.Series(ret).dropna()
    idx = r.index
    rows, keep = [], []
    for d in pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates))):
        hits = idx.get_indexer([d], method="bfill")
        i = hits[0]
        if i < 0 or i - before < 0 or i + after >= len(idx):
            continue
        rows.append(r.iloc[i - before:i + after + 1].to_numpy())
        keep.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=pd.DatetimeIndex(keep),
                        columns=range(-before, after + 1))


def caar(ret: pd.Series, dates, before: int = 10, after: int = 10) -> dict:
    """Cumulative average return around an event, with a t-stat per offset.

    The t-stat is the one to read. A CAAR line that drifts upward looks like a
    finding and usually is not: with twenty events the standard error is large
    enough that almost any shape is consistent with noise.
    """
    w = event_window(ret, dates, before, after)
    if w.empty:
        return {}
    mean = w.mean()
    se = w.std(ddof=1) / np.sqrt(len(w))
    return {
        "n": len(w),
        "aar": mean,
        "caar": mean.cumsum(),
        "t": (mean / se.where(se > 0)),
    }
