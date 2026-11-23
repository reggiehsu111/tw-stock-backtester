# tw-stock-backtester

A small backtester for Taiwan index futures, built around the parts of backtesting that
usually go wrong: **look-ahead, overlapping windows, multiple testing, and costs.**

It fetches free public data from TAIFEX (the Taiwan Futures Exchange), runs a strategy,
and serves a web tearsheet. There are no credentials, no database and no vendor feed —
`pip install -e .` then one command.

## Why these four things

Most backtest bugs are not coding errors; they are statistical ones that produce a
*better-looking* result, so nothing alerts you. This project makes each of them hard to
commit by accident:

| failure | how it shows up | what this does about it |
|---|---|---|
| **look-ahead** | a threshold or z-score computed over the whole sample | every normalisation is trailing-only; `twbt.panel` refuses full-sample statistics |
| **overlapping windows** | a 30-day signal predicting 30-day returns, t-stat inflated ~√30 | `twbt.stats` reports OLS, Newey–West *and* disjoint-sample t together |
| **multiple testing** | the best of 40 variants reported as if it were the only one | `twbt.stats.noise_bar` simulates the max \|t\| for the number of tests actually run |
| **costs** | a flat "2 bps" that quietly doubles or halves the edge | `twbt.costs` uses a measured half-spread-per-leg convention, and the tearsheet shows a cost-sensitivity curve |

Those choices came out of real research where each one changed a conclusion. See
[`docs/METHOD.md`](docs/METHOD.md) for the worked examples.

## Quick start

```bash
git clone https://github.com/reggiehsu111/tw-stock-backtester.git
cd tw-stock-backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. fetch public TAIFEX data (a few minutes; cached to data/cache/)
twbt fetch --start 2024-01-01

# 2. run a backtest
twbt run --strategy institutional_flow

# 3. open the tearsheet, then visit the address it prints
twbt web            # -> http://127.0.0.1:8000
```

`twbt web` starts a local server and prints its URL; open that in a browser. It serves
whatever `twbt run` last saved, so run at least one backtest first. Use `--port` to move
it off 8000. `twbt list` shows the available strategies and what data you already have.

**Offline?** A committed fixture ships in `data/fixture/`. Skip step 1 and add
`--fixture` to steps 2 and 3; everything else is identical.

## Data

Everything comes from TAIFEX's public download endpoints — the same CSVs anyone can
download from their website, no account required:

| dataset | what it is | endpoint | history |
|---|---|---|---|
| `futures_daily` | TAIEX futures OHLCV, settlement, open interest | `3/futDataDown` | back to 2016 at least |
| `three_majors` | 三大法人 — daily net position by institution type | `3/futContractsDateDown` | **rolling ~3 years only** |
| `large_trader` | 大額交易人 — top-5 / top-10 trader open interest | `3/largeTraderFutDown` | back to 2016 at least |

`three_majors` is the binding constraint: TAIFEX serves only a rolling three-year window
for it, so any strategy using institutional positioning is limited to that span no matter
what `--start` you pass. The fetcher reads the available window from the query form at
runtime, narrows your request to fit, and tells you it did — rather than failing, or
worse, silently returning less than you asked for.

Fetches are cached to parquet and are idempotent; re-running `twbt fetch` only pulls dates
you do not already have. The client is paced to stay inside TAIFEX's tolerance.

A note on this source, since it shapes a lot of `sources/taifex.py`: an invalid request
returns **HTTP 200** with an HTML page containing a JavaScript `alert()`, so the status
code tells you nothing. Span limits are counted in calendar months rather than days, and
differ per endpoint. Parameter names are inconsistent between endpoints (`commodity_id`
vs `commodityId`). The CSVs have a trailing comma on every row, which makes pandas
silently promote the date column to the index and shift every other column one place
left. Each of those is handled, and commented where it is handled.

## What the strategies are (and are not)

Two are included, both operating on daily TAIFEX data:

- **`institutional_flow`** — trades the next session against dealer positioning and with
  foreign open-interest changes. Published positions are released at 15:00 TPE, so the
  earliest tradable moment is the following session; the engine enforces that lag.
- **`momentum`** — a plain trailing-return strategy, included as a reference point. It is
  *supposed* to be unremarkable: a backtester that cannot show you a mediocre result is
  not telling you the truth about the good one.

These are demonstrations of the engine, **not investment advice and not validated
signals**. `docs/METHOD.md` is explicit about what would be required before believing one.

## Layout

```
src/twbt/
  sources/taifex.py   public TAIFEX clients, one Endpoint per dataset
  cache.py            parquet cache, idempotent fetch
  panel.py            aligned daily panel; trailing-only normalisation
  costs.py            cost model (half-spread per leg + fees)
  stats.py            overlap-aware significance, noise bars
  backtest.py         the engine: signal -> position -> pnl, and the verdict
  strategies/         signal definitions
  cli.py              fetch / run / web / list
  web/charts.py       inline SVG, no plotting dependency and no CDN
  web/app.py          FastAPI tearsheet
tests/                unit tests, including look-ahead regression tests
docs/METHOD.md        why the statistics are done this way
```

## What the tearsheet shows

Four charts per strategy, each answering a question that the headline number cannot:

- **cumulative P&L**, gross and net together — the gap between the lines is the cost
- **by year**, one line per calendar year labelled with that year's Sharpe, plus the
  average path. This is where a result carried by a single good year stops hiding.
- **long / short decomposition** — a result that comes entirely from one side is usually
  a market-direction bet wearing a signal's clothes
- **cost sensitivity** — net t-stat as the spread assumption is scaled, with 1.5x marked

Then the statistics, the same signal priced on all three sibling contracts, and the
walk-forward year by year. Above all of it, a plain-language verdict that names the
checks the strategy **failed**, in its own output.

## Tests

```bash
pytest
```

The suite includes guard tests that fail if look-ahead is reintroduced — for example
`test_no_lookahead.py` feeds the panel a series whose future half is corrupted and asserts
that signals computed on the past half are unchanged.
