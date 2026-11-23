"""Fill in sessions where one venue is missing from the equity cache.

Separate from `twbt fetch-equities` because it is a repair, not a fetch: it
looks at what the cache already has, works out which sessions are missing a
venue entirely, and asks only for those.

Serial and checkpointed, both for the same reason. TWSE blocks a client that
asks too fast, and an uncheckpointed run that gets blocked three hundred days
in has nothing to show for the three hundred.

    python scripts/backfill_tse.py [--board TSE] [--max-days N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twbt import cache, equity_panel                       # noqa: E402
from twbt.sources import equities as eq                    # noqa: E402

FETCHERS = {"TSE": eq.fetch_twse_day, "OTC": eq.fetch_tpex_day}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="TSE", choices=sorted(FETCHERS))
    ap.add_argument("--max-days", type=int, default=0,
                    help="stop after this many (0 = all)")
    ap.add_argument("--pace", type=float, default=2.5,
                    help="seconds between requests; raise it if you get blocked")
    args = ap.parse_args()

    path = ROOT / "data" / "cache" / "equities.parquet"
    df = cache.load("equities")
    cov = equity_panel.coverage(df)
    if args.board not in cov.columns:
        print(f"{args.board} never appears in the cache at all", file=sys.stderr)
        return 2

    missing = [d.date() for d in cov.index[cov[args.board] == 0]]
    if args.max_days:
        missing = missing[: args.max_days]
    if not missing:
        print(f"{args.board}: nothing missing")
        return 0

    print(f"{args.board}: {len(missing)} sessions missing, "
          f"{args.pace}s apart (~{len(missing) * args.pace / 60:.0f} min)\n")

    fetch = FETCHERS[args.board]
    sess = requests.Session()
    got: list[pd.DataFrame] = []
    failed = 0

    def save() -> int:
        """Merge what we have into the cache. Called as we go, not at the end."""
        if not got:
            return 0
        add = pd.concat(got, ignore_index=True)
        add = add[add["ticker"].map(eq._is_equity)]
        merged = (pd.concat([df, add], ignore_index=True)
                  .drop_duplicates(subset=["date", "ticker"])
                  .sort_values(["date", "ticker"])
                  .reset_index(drop=True))
        merged.to_parquet(path, index=False)
        return len(merged)

    try:
        for i, d in enumerate(missing, 1):
            try:
                frame = fetch(d, sess)
                if len(frame):
                    got.append(frame)
            except eq.BlockedError as exc:
                print(f"\n  BLOCKED at {d}: {exc}")
                print("  Stopping. Wait a while and re-run; progress is saved.")
                break
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                if failed <= 3:
                    print(f"  {d}: {str(exc)[:90]}")
            if i % 20 == 0:
                total = save()
                print(f"  {i}/{len(missing)}  recovered={len(got)}  "
                      f"failed={failed}  cache={total:,} rows", flush=True)
            time.sleep(args.pace)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        sess.close()
        total = save()

    cov2 = equity_panel.coverage(cache.load("equities"))
    still = int((cov2[args.board] == 0).sum()) if args.board in cov2 else -1
    print(f"\nrecovered {len(got)} sessions, {failed} failed, "
          f"cache now {total:,} rows")
    print(f"{args.board} still missing on {still} sessions")
    return 0 if still == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
