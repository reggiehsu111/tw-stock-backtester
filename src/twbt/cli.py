"""Command line: fetch, run, web, list.

Three commands is the whole interface. The design constraint is that someone who
has never seen this repository should get from `git clone` to a tearsheet in their
browser without reading anything, so every command has a sensible default and says
what it is doing while it does it.

    twbt fetch --start 2024-01-01     download public TAIFEX futures data
    twbt fetch-equities               download the TWSE/TPEX daily cross-section
    twbt fetch-events --co-id 2330    investor-conference dates (TSMC earnings)
    twbt run --strategy momentum      backtest, print the verdict, save the result
    twbt web                          serve the tearsheet and the signal builder

Add --fixture to `run` and `web` to work entirely offline from the committed
sample, skipping `fetch`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

from . import cache

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
DATASETS = ("futures_daily", "three_majors", "large_trader")


def _date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}") from None


# --------------------------------------------------------------------------- #

def cmd_fetch(args) -> int:
    start, end = args.start, args.end or date.today()
    if start >= end:
        print(f"error: --start {start} is not before --end {end}", file=sys.stderr)
        return 2

    print(f"Fetching TAIFEX public data, {start} to {end}")
    print("This hits a free public service and is paced deliberately; "
          "expect a few minutes.\n")
    failed = []
    for name in DATASETS:
        try:
            df = cache.update(name, start, end)
            print(f"  {name}: {len(df):,} rows cached\n")
        except Exception as exc:                           # noqa: BLE001
            print(f"  {name}: FAILED -- {exc}\n", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"Finished with {len(failed)} failure(s): {', '.join(failed)}.",
              file=sys.stderr)
        print("Re-running `twbt fetch` will retry only the missing dates.",
              file=sys.stderr)
        return 1
    print("Done. Now run:  twbt run --strategy institutional_flow")
    return 0


def cmd_fetch_equities(args) -> int:
    """The cross-section the operator vocabulary needs.

    One request per venue per trading day, so this is slower than the futures
    fetch by a wide margin -- roughly half an hour for three years. It is cached,
    so it is a one-time cost, and re-running only pulls what is missing.
    """
    start, end = args.start, args.end or date.today()
    if start >= end:
        print(f"error: --start {start} is not before --end {end}", file=sys.stderr)
        return 2

    from .sources import equities as _eq

    days = (end - start).days * 5 // 7
    # TWSE is fetched serially on purpose -- see WORKERS_TWSE -- so it sets the
    # clock. Quoting the real number beats a cheerful estimate that is wrong by
    # a factor of five.
    mins = (days * _eq.PACE_TWSE + days * _eq.PACE_SEC / _eq.WORKERS_TPEX) / 60
    print(f"Fetching TWSE + TPEX daily quotes, {start} to {end}")
    print(f"About {days:,} trading days. TWSE is fetched one at a time to avoid "
          f"being rate-limited, so expect roughly {mins:.0f} minutes.")
    print("Progress is checkpointed, so stopping and re-running loses nothing.")
    print("Cached to data/cache/equities.parquet; re-running pulls only gaps.\n")
    try:
        df = cache.update("equities", start, end)
    except Exception as exc:                               # noqa: BLE001
        print(f"  FAILED -- {exc}", file=sys.stderr)
        return 1
    if len(df):
        print(f"\n  {len(df):,} rows, {df['date'].nunique():,} sessions, "
              f"{df['ticker'].nunique():,} tickers")
    # one request per venue, so it rides along with the equity fetch
    try:
        sec = cache.update_snapshot("sectors")
        print(f"  sectors: {len(sec):,} tickers classified")
    except Exception as exc:                               # noqa: BLE001
        print(f"  sectors: FAILED -- {exc}", file=sys.stderr)

    print("\nDone. Now run:  twbt web   and open the signal builder.")
    return 0


def cmd_fetch_events(args) -> int:
    """Investor-conference dates from MOPS, classified and written as an event file."""
    from .events import EVENTS_DIR
    from .sources import mops

    roc_now = date.today().year - 1911
    years = range(roc_now - args.years + 1, roc_now + 1)
    print(f"Fetching 法人說明會 filings for {args.co_id}, ROC {years[0]}..{years[-1]}")
    try:
        items = mops.conferences(args.co_id, years=years, verbose=True)
    except Exception as exc:                               # noqa: BLE001
        print(f"  FAILED -- {exc}", file=sys.stderr)
        return 1

    kinds: dict[str, int] = {}
    for c in items:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    print(f"  {len(items)} filings: " +
          ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    df = mops.to_frame(items, earnings_only=not args.all)
    if df.empty:
        print("  nothing to write", file=sys.stderr)
        return 1

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVENTS_DIR / f"{args.name or args.co_id}.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {len(df)} events to {out.name}")
    print("\n  A results call announces new figures; a broker conference restates")
    print("  them. Only the former is an event, which is why --all is not the")
    print("  default: it would average real announcements with restatements.")
    print("\nDone. The event file now appears in the builder and futures pages.")
    return 0


def cmd_run(args) -> int:
    # imported here so `twbt fetch` does not pay for pandas/fastapi import time
    from . import backtest, costs, panel, strategies

    try:
        p = panel.build(fixture=args.fixture)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.start:
        p = p[p.index.date >= args.start]
    if args.end:
        p = p[p.index.date <= args.end]
    if len(p) < 250:
        print(f"error: only {len(p)} sessions in the panel; fetch a longer range",
              file=sys.stderr)
        return 2

    strat = strategies.get(args.strategy)()
    n_variants = getattr(strat, "n_variants", 1)
    cm = costs.CostModel(args.contract)

    print(f"Panel: {len(p):,} sessions, {p.index.min().date()} to {p.index.max().date()}")
    print(f"Strategy: {strat.name}")
    print(f"  {strat.describe()}")
    print(f"Costs: {cm.describe()}\n")

    res = backtest.run(p, strat, cm, n_variants_tested=n_variants)
    if not args.no_walk_forward:
        res.walk_forward = backtest.walk_forward(p, strat, cm,
                                                 contract=args.contract)

    main = res.main
    print(f"Trades: {main.n_trades}")
    if main.n_trades:
        g, n = main.gross_perf, main.net_perf
        print(f"  gross  mean {g.mean:+8.2f} bps   t {g.t:+6.2f}   "
              f"Sharpe {g.sharpe:+6.2f}")
        print(f"  net    mean {n.mean:+8.2f} bps   t {n.t:+6.2f}   "
              f"Sharpe {n.sharpe:+6.2f}   hit {n.hit_rate:.1%}")
        print(f"  cost   mean {main.avg_cost_bps:8.2f} bps/trade   "
              f"breakeven {main.breakeven_bps:+.2f} bps\n")
        print("Across contracts:")
        print("  " + res.consistency().to_string(index=False).replace("\n", "\n  "))

    print("\nVerdict:")
    for v in res.verdict():
        prefix = "  !! " if v.startswith("FAILS") else "   - "
        print(prefix + v)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{strat.name}.pkl"
    with out.open("wb") as fh:
        pickle.dump(res, fh)
    (RESULTS_DIR / f"{strat.name}.json").write_text(json.dumps({
        "strategy": strat.name,
        "description": strat.describe(),
        "costs": cm.describe(),
        "sessions": len(p),
        "start": str(p.index.min().date()),
        "end": str(p.index.max().date()),
        "n_trades": main.n_trades,
        "net": main.net_perf.as_dict(),
        "gross": main.gross_perf.as_dict(),
        "verdict": res.verdict(),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=str))

    print(f"\nSaved to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    print("View it with:  twbt web")
    return 0


def cmd_web(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed. Run: pip install -e \".[dev]\"",
              file=sys.stderr)
        return 2

    from .web.app import create_app

    if not RESULTS_DIR.exists() or not list(RESULTS_DIR.glob("*.pkl")):
        # not fatal: the signal builder does not depend on saved results
        print("\n  No saved futures results yet, so the results page will be empty.")
        print("  The signal builder works regardless. To fill the results page:")
        print("    twbt run --strategy institutional_flow"
              + (" --fixture" if args.fixture else ""))

    # Checking first turns uvicorn's OSError traceback into one actionable line.
    # The usual cause is an earlier `twbt web` still running, which is not
    # obvious from "[Errno 48] error while attempting to bind".
    import socket
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((args.host, args.port))
        except OSError:
            print(f"error: port {args.port} is already in use on {args.host}.",
                  file=sys.stderr)
            print(f"  Another `twbt web` is probably still running. Either stop "
                  f"it, or pick another port:\n    twbt web --port {args.port + 1}"
                  + (" --fixture" if args.fixture else ""), file=sys.stderr)
            return 2

    app = create_app(fixture=args.fixture)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Signal builder (equities)   {url}")
    print(f"  Futures backtest            {url}/futures")
    print(f"  Saved futures runs          {url}/results\n")
    print("  Open any of those in a browser. Ctrl-C to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_list(args) -> int:
    from . import strategies

    print("Strategies:")
    for name in strategies.available():
        s = strategies.get(name)()
        print(f"  {name}\n      {s.describe()}")
    print("\nCached datasets:")
    for name in DATASETS + ("equities",):
        cov = cache.covered(name)
        print(f"  {name}: " + (f"{cov[0]} to {cov[1]}" if cov else "not fetched"))
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="twbt",
        description="Backtest Taiwan index futures on free public TAIFEX data.",
        epilog="Start with:  twbt fetch --start 2020-01-01",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download public TAIFEX data into the cache")
    f.add_argument("--start", type=_date, default=_date("2020-01-01"),
                   help="first date, YYYY-MM-DD (default 2020-01-01)")
    f.add_argument("--end", type=_date, default=None, help="last date (default today)")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run a backtest and save the result")
    r.add_argument("--strategy", default="institutional_flow",
                   help="strategy name; see `twbt list`")
    r.add_argument("--contract", default="TXF", choices=("TXF", "MXF", "TMF"),
                   help="contract whose costs to charge (default TXF)")
    r.add_argument("--start", type=_date, default=None)
    r.add_argument("--end", type=_date, default=None)
    r.add_argument("--fixture", action="store_true",
                   help="use the committed offline sample instead of the cache")
    r.add_argument("--no-walk-forward", action="store_true",
                   help="skip the walk-forward (faster)")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("web", help="serve the tearsheet")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--fixture", action="store_true")
    w.set_defaults(func=cmd_web)

    fe = sub.add_parser("fetch-equities",
                        help="download the TWSE/TPEX daily cross-section")
    fe.add_argument("--start", type=_date, default=_date("2024-01-01"),
                    help="first date, YYYY-MM-DD (default 2024-01-01)")
    fe.add_argument("--end", type=_date, default=None)
    fe.set_defaults(func=cmd_fetch_equities)

    fv = sub.add_parser("fetch-events",
                        help="investor-conference dates from MOPS (e.g. earnings)")
    fv.add_argument("--co-id", default="2330", help="company code (default 2330 TSMC)")
    fv.add_argument("--years", type=int, default=4, help="how many ROC years back")
    fv.add_argument("--name", default=None, help="event file name (default: co-id)")
    fv.add_argument("--all", action="store_true",
                    help="keep broker conferences too, not just results calls")
    fv.set_defaults(func=cmd_fetch_events)

    ls = sub.add_parser("list", help="show strategies and what data is cached")
    ls.set_defaults(func=cmd_list)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
