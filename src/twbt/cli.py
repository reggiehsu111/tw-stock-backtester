"""Command line: fetch, run, web, list.

Three commands is the whole interface. The design constraint is that someone who
has never seen this repository should get from `git clone` to a tearsheet in their
browser without reading anything, so every command has a sensible default and says
what it is doing while it does it.

    twbt fetch --start 2020-01-01     download public TAIFEX data into data/cache
    twbt run --strategy momentum      backtest, print the verdict, save the result
    twbt web                          serve the tearsheet at 127.0.0.1:8000

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
        print("No saved results yet. Run a backtest first, for example:\n")
        print("  twbt run --strategy institutional_flow"
              + (" --fixture" if args.fixture else "") + "\n")
        return 2

    app = create_app(fixture=args.fixture)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Tearsheet running at  {url}\n")
    print("  Open that address in a browser. Ctrl-C to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_list(args) -> int:
    from . import strategies

    print("Strategies:")
    for name in strategies.available():
        s = strategies.get(name)()
        print(f"  {name}\n      {s.describe()}")
    print("\nCached datasets:")
    for name in DATASETS:
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
