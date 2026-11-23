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
twbt fetch --start 2020-01-01

# 2. run a backtest
twbt run --strategy institutional_flow

# 3. open the tearsheet
twbt web            # -> http://127.0.0.1:8000
```

**Offline?** A committed fixture covering 2024–2025 ships in `data/fixture/`. Skip step 1
and add `--fixture` to steps 2 and 3; everything else is identical.

## Data

Everything comes from TAIFEX's public download endpoints — the same CSVs anyone can
download from their website, no account required:

| dataset | what it is | endpoint |
|---|---|---|
| `futures_daily` | TAIEX futures OHLCV, settlement, open interest | `/futures/download` |
| `three_majors` | 三大法人 — daily net position by institution type | `/futures/TradingAnalysis` |
| `large_trader` | 大額交易人 — top-5 / top-10 trader open interest | `/futures/largeTraderFutQry` |

Fetches are cached to parquet and are idempotent; re-running `twbt fetch` only pulls dates
you do not already have. The client is rate-limited to stay inside TAIFEX's tolerance.

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
  sources/taifex.py   public TAIFEX clients, one function per dataset
  cache.py            parquet cache, idempotent fetch
  panel.py            aligned daily panel; trailing-only normalisation
  costs.py            cost model (half-spread per leg + fees)
  stats.py            overlap-aware significance, noise bars
  backtest.py         the engine: weights -> positions -> pnl
  strategies/         signal definitions
  web/                FastAPI tearsheet
tests/                unit tests, including look-ahead regression tests
docs/METHOD.md        why the statistics are done this way
```

## Tests

```bash
pytest
```

The suite includes guard tests that fail if look-ahead is reintroduced — for example
`test_no_lookahead.py` feeds the panel a series whose future half is corrupted and asserts
that signals computed on the past half are unchanged.
