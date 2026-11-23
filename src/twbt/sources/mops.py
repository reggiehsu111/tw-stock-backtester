"""法人說明會 (investor conference) dates from MOPS, the public filing system.

This is where earnings dates come from. Any listed or OTC company, back to the
mid-2000s, no account:

    https://mopsov.twse.com.tw/mops/web/ajax_t100sb02_1

**The host matters.** ``mops.twse.com.tw`` answers that path with a 833-byte
error page; ``mopsov.twse.com.tw`` — the mirror — returns the data. An earlier
version of this project concluded from the first host that earnings dates were
not publicly available at all. They are, and the only difference was two
letters in a hostname.

CLASSIFICATION IS THE WHOLE JOB. A company files under this heading for two
quite different things, and mixing them ruins an event study:

    公布本公司2023年第4季財務報告及2024年第1季業績展望
        -> the quarterly results call. New information. This is the event.

    本公司受邀參加摩根大通證券所舉辦之『J.P. Morgan Taiwan CEO-CFO Conference』,
    會中就本公司1/18法說會已公開之財務數字、經營績效等相關資訊進行說明
        -> a broker conference, explicitly restating what was already
           disclosed on 1/18. No new information.

TSMC filed 17 of these in 2024 and only four were results calls. An event study
run on all seventeen would be averaging four real announcements with thirteen
non-events and reporting the result as an earnings effect.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

import pandas as pd

ENDPOINT = "https://mopsov.twse.com.tw/mops/web/ajax_t100sb02_1"
UA = "python-requests/2.31.0"
PACE_SEC = 1.5
RETRIES = 3

# A results call announces figures. A broker conference restates them. Both file
# under the same heading, so the description text is the only thing separating
# them.
RESULTS = re.compile(r"公布.*?(財務報告|財報|營運結果|業績)")
INVITED = re.compile(r"受邀參加|應邀參加|參加.*?(Conference|conference|論壇|研討會)")


class MopsError(RuntimeError):
    """MOPS refused or returned something that is not a results table."""


@dataclass(frozen=True)
class Conference:
    date: pd.Timestamp
    kind: str            # "earnings" | "conference" | "other"
    description: str

    @property
    def is_earnings(self) -> bool:
        return self.kind == "earnings"


def _decode(body: bytes) -> str:
    """MOPS serves UTF-8 on this path and Big5 on others. Try both."""
    for enc in ("utf-8", "cp950", "big5"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _roc_to_date(s: str) -> pd.Timestamp | None:
    """民國 YYY/MM/DD -> Timestamp. A range takes its first day.

    Multi-day conferences file as "113/03/05 至 113/03/07". The first day is the
    event: by the second, whatever was said is public.
    """
    m = re.search(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", str(s))
    if not m:
        return None
    y, mo, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    try:
        return pd.Timestamp(y, mo, d)
    except ValueError:
        return None


def _post(co_id: str, roc_year: int) -> str:
    data = {
        "encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1",
        "keyword4": "", "code1": "", "TYPEK2": "", "checkbtn": "",
        "queryName": "co_id", "inpuType": "co_id", "TYPEK": "all",
        "co_id": str(co_id), "year": str(roc_year),
    }
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                ENDPOINT, urllib.parse.urlencode(data).encode(),
                {"User-Agent": UA,
                 "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = _decode(r.read())
            if "Overrun" in text or "請稍後再試" in text:
                raise MopsError("MOPS rate limit (Overrun); slow down")
            return text
        except MopsError:
            raise
        except Exception as exc:                        # noqa: BLE001
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(PACE_SEC * (2 ** attempt))
    raise MopsError(f"MOPS failed after {RETRIES} attempts: {last}")


def _rows(text: str) -> list[list[str]]:
    out = []
    for raw in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c))
                 .replace("　", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", raw, re.S)]
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        if len(cells) >= 3:
            out.append(cells)
    return out


def classify(description: str) -> str:
    """Which kind of filing this is. Order matters.

    A results call that a broker also hosted would match both patterns, and it
    is still a results call -- the figures are new whoever is in the room. So
    the results test is applied first.
    """
    d = str(description)
    if RESULTS.search(d):
        return "earnings"
    if INVITED.search(d):
        return "conference"
    return "other"


def conferences(co_id: str = "2330", years: range | list | None = None,
                verbose: bool = False) -> list[Conference]:
    """Every 法人說明會 filing for one company, classified.

    ``years`` are 民國 years; the default covers the window the rest of this
    project uses. MOPS is paced, because it answers an impatient client with an
    "Overrun" page rather than a 429.
    """
    if years is None:
        this_roc = date.today().year - 1911
        years = range(this_roc - 3, this_roc + 1)

    out: list[Conference] = []
    for y in years:
        try:
            text = _post(co_id, int(y))
        except MopsError as exc:
            if verbose:
                print(f"    ROC {y}: {exc}")
            continue
        for cells in _rows(text):
            if str(cells[0]).strip() != str(co_id):
                continue                            # header or another company
            when = _roc_to_date(cells[2] if len(cells) > 2 else "")
            if when is None:
                continue
            # Classify on everything after the time column, not on the first
            # long cell. The columns run
            #   代號 | 名稱 | 日期 | 時間 | 地點 | 擇要訊息 | ...
            # and the venue is routinely longer than 20 characters, so picking
            # the first long cell classified "台北君悅酒店三樓 凱悅廳一區 (台北市
            # 信義區松壽路2號)" instead of the filing text -- which silently
            # dropped every January and July results call.
            blob = " ".join(cells[4:])
            desc = max(cells[4:], key=len, default="") if len(cells) > 4 else ""
            out.append(Conference(when, classify(blob), desc[:300]))
        time.sleep(PACE_SEC)

    seen, uniq = set(), []
    for c in sorted(out, key=lambda x: x.date):
        if c.date not in seen:
            seen.add(c.date)
            uniq.append(c)
    return uniq


def to_frame(items: list[Conference], earnings_only: bool = True) -> pd.DataFrame:
    """The event-file shape: date, label.

    ``earnings_only`` defaults to True because that is almost always what is
    wanted. TSMC filed 17 conferences in 2024 and four were results calls; an
    event study over all seventeen averages four announcements with thirteen
    restatements of them.
    """
    rows = [c for c in items if (c.is_earnings or not earnings_only)]
    if not rows:
        return pd.DataFrame(columns=["date", "label"])
    return pd.DataFrame({
        "date": [c.date.date() for c in rows],
        "label": [("earnings call" if c.is_earnings else c.kind) for c in rows],
    })
