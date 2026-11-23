"""Industry classification for TW listed names, from the public MOPS open data.

One request per venue, no account:

    TWSE  openapi.twse.com.tw/v1/opendata/t187ap03_L      ~1,100 names
    TPEX  tpex.org.tw/openapi/v1/mopsfin_t187ap03_O         ~900 names

A caveat that matters more than the plumbing: **this is the CURRENT
classification, not a point-in-time one.** A company that changed industry, or
one that has delisted since, is labelled by where it sits today. For a
sector-attribution chart that is tolerable — reclassification is rare and
sector P&L is a diagnostic rather than a signal input — but it would NOT be
acceptable to build a signal on, because today's label is partly a consequence
of what the company went on to become. ``sector_map`` says so in its return
value rather than leaving it to be discovered.

The two venues disagree on field names and on the code set: TPEX uses codes
above 31 that TWSE does not, and the same number does not always mean the same
industry across the two. They are therefore namespaced by board before being
joined, which is uglier than one shared mapping and is the honest shape.
"""

from __future__ import annotations

import pandas as pd
import requests

UA = ("Mozilla/5.0 (compatible; tw-stock-backtester/0.1; "
      "+https://github.com/reggiehsu111/tw-stock-backtester)")

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

# TWSE 產業別 codes. The numbering is the exchange's own and is stable.
TWSE_SECTORS = {
    "01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織", "05": "電機機械",
    "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙", "10": "鋼鐵", "11": "橡膠",
    "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅", "17": "金融保險",
    "18": "貿易百貨", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
    "24": "半導體", "25": "電腦及週邊", "26": "光電", "27": "通信網路",
    "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子",
    "32": "文化創意", "33": "農業科技", "34": "電子商務", "35": "綠能環保",
    "36": "數位雲端", "37": "運動休閒", "38": "居家生活",
}


def _get(url: str, session: requests.Session) -> list:
    r = session.get(url, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"{url}: expected a list, got {type(data).__name__}")
    return data


def fetch(session: requests.Session | None = None) -> pd.DataFrame:
    """One row per ticker: board, industry code, industry name."""
    own = session is None
    session = session or requests.Session()
    rows = []
    try:
        for item in _get(TWSE_URL, session):
            code = str(item.get("公司代號", "")).strip()
            ind = str(item.get("產業別", "")).strip()
            if code:
                rows.append({"ticker": code, "board": "TSE", "sector_code": ind,
                             "sector": TWSE_SECTORS.get(ind, f"TSE-{ind}")})
        for item in _get(TPEX_URL, session):
            code = str(item.get("SecuritiesCompanyCode", "")).strip()
            ind = str(item.get("SecuritiesIndustryCode", "")).strip()
            if code:
                rows.append({"ticker": code, "board": "OTC", "sector_code": ind,
                             "sector": TWSE_SECTORS.get(ind, f"OTC-{ind}")})
    finally:
        if own:
            session.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # a code may appear on both boards after a transfer; keep the first
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def sector_matrix(tickers, index, table: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the per-ticker labels into a matrix shaped like the panel.

    Constant down the date axis, because the source is a snapshot. That is a
    statement about the data, not a modelling choice, and a sector matrix that
    varied over time would imply a point-in-time history this source does not
    have.
    """
    if table is None or table.empty:
        return pd.DataFrame("—", index=index, columns=list(tickers))
    m = table.set_index("ticker")["sector"]
    row = pd.Series([m.get(str(t), "—") for t in tickers], index=list(tickers))
    return pd.DataFrame([row.to_numpy()] * len(index),
                        index=index, columns=list(tickers))
