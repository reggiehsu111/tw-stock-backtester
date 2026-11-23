"""Build the committed offline fixture from the local cache.

The fixture exists so a reviewer can run everything with no network and get the
same numbers. It is checked in, so it has to be small: large_trader alone is
~900k rows across every product TAIFEX lists, of which the panel reads the TX
all-months all-traders row and nothing else.

Filtering here rather than shipping the raw download is the difference between a
2 MB repository and a 60 MB one. The filters are deliberately the same ones
panel.py applies, so the fixture cannot disagree with the live path about what a
row means.

Run from the repo root after `twbt fetch`:

    python scripts/make_fixture.py
"""

from pathlib import Path

import pandas as pd

from twbt import cache, equity_panel, panel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_UNIVERSE = 150
OUT = ROOT / "data" / "fixture"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0

    fut = cache.load("futures_daily")
    fut = fut[fut["product"].astype(str).str.strip() == "TX"]

    maj = cache.load("three_majors")
    maj = maj[maj["product"].astype(str).str.strip() == panel.PRODUCT_TXF]

    lt = cache.load("large_trader")
    for c in ("expiry", "trader_type"):
        lt[c] = pd.to_numeric(lt[c], errors="coerce")
    lt = lt[(lt["product"].astype(str).str.strip() == panel.LT_PRODUCT_TX)
            & (lt["expiry"] == panel.EXPIRY_ALL_MONTHS)
            & (lt["trader_type"] == panel.TRADER_ALL)]

    frames = [("futures_daily", fut), ("three_majors", maj), ("large_trader", lt)]

    # The equity cross-section is ~1,850 names a day, which is far too much to
    # commit. The fixture keeps only names that were ever in the liquid universe,
    # using the same point-in-time screen the app applies, so the committed sample
    # cannot disagree with the live path about who is tradable.
    try:
        eq = cache.load("equities")
    except FileNotFoundError:
        eq = None
    if eq is not None and len(eq):
        # A smaller universe than the app's default 300. The full cross-section
        # is ~1,850 names a day and 9 MB even after screening, which is too much
        # to put in a git repository someone has to clone. 150 names is still a
        # real cross-section -- enough for Rank and Demean to mean something --
        # and the app fetches the full panel when asked.
        mats = equity_panel.build()
        univ = equity_panel.liquid_universe(mats, n=FIXTURE_UNIVERSE)
        keep = set(univ.columns[univ.any()])
        eq = eq[eq["ticker"].isin(keep)]
        # `name` is the Chinese company name and nothing reads it
        eq = eq.drop(columns=[c for c in ("name",) if c in eq.columns])
        frames.append(("equities", eq))

    for name, df in frames:
        path = OUT / f"{name}.parquet"
        df.reset_index(drop=True).to_parquet(path, index=False,
                                             compression="zstd")
        kb = path.stat().st_size / 1024
        total += kb
        print(f"  {name:15s} {len(df):>7,} rows  {kb:>7.1f} KB  "
              f"{df['date'].min().date()}..{df['date'].max().date()}")

    print(f"  {'total':15s} {'':>7s}       {total:>7.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
