# tw-stock-backtester

A backtester for Taiwan markets, built around the parts of backtesting that usually go
wrong: **look-ahead, overlapping windows, multiple testing, and costs.**

Two halves, sharing one discipline:

- a **futures** engine for single-instrument time-series strategies on TAIFEX data;
- an **interactive signal builder** for cross-sectional equity strategies, where you
  compose matrix operators in the browser and get a tearsheet back.

Everything runs on free public data — TAIFEX, TWSE and TPEX. No credentials, no
database, no vendor feed: `pip install -e .` and one command.

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

`twbt web` starts a local server and prints two URLs: `/` for saved futures results and
`/build` for the interactive signal builder. Use `--port` to move it off 8000, and
`twbt list` to see what data you already have.

For the signal builder you also need the equity cross-section, which is a slower fetch
because it is one request per venue per trading day:

```bash
twbt fetch-equities --start 2024-01-01     # ~35 min for three years; checkpointed
twbt web                                   # -> http://127.0.0.1:8000
```

**You probably do not need that fetch.** The committed fixture already contains three
years, and `twbt web --fixture` serves the whole builder from it. The live fetch is slow
because TWSE rate-limits aggressively and is therefore requested one session at a time —
going faster is what got an earlier version of this client blocked across half the range.
Progress is checkpointed, so stopping and re-running costs nothing.

The page opens on a worked example — 12-1 momentum, already run, with all its plots and a
verdict that says it loses money after costs. That is a more honest landing page than a
blank form or a curated winner.

**Offline?** A committed fixture ships in `data/fixture/` — three years of futures data
plus a 150-name equity cross-section, about 6 MB. Skip the fetches and add `--fixture`;
everything else is identical. The full fetch gives a 300-name universe, the fixture 150,
so numbers shift slightly while the conclusions do not.

## Data

Everything comes from TAIFEX's public download endpoints — the same CSVs anyone can
download from their website, no account required:

| dataset | what it is | endpoint | history |
|---|---|---|---|
| `futures_daily` | TAIEX futures OHLCV, settlement, open interest | `3/futDataDown` | back to 2016 at least |
| `three_majors` | 三大法人 — daily net position by institution type | `3/futContractsDateDown` | **rolling ~3 years only** |
| `large_trader` | 大額交易人 — top-5 / top-10 trader open interest | `3/largeTraderFutDown` | back to 2016 at least |
| `equities` | TWSE 上市 + TPEX 上櫃 daily OHLCV, ~1,850 ordinary shares | `MI_INDEX`, `afterTrading/otc` | years, one request per venue per day |

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

## The signal builder

Cross-sectional operators — `Rank`, `Demean`, `Neutralize` — are meaningless on a single
futures contract, so the builder runs on a panel of ~1,850 TW listed names with a
point-in-time liquid universe of 300.

Two ways to build a signal, and they share one representation.

**Blocks.** Drag operators from the palette onto the canvas, drag the boxes to reorder,
edit window parameters in place. Double-click a palette entry to append it. Hovering any
operator shows its signature and a one-line description of what it does. The canvas draws
a linear chain, which covers most signals.

**Code.** Type the expression directly:

```
GrossScale(LongShortBalance(Rank(Neg(InstSum(5, Returns(px))))))
Rank(Div(Returns(px), InstStdev(20, Returns(px))))
LongShortBalance(Rank(Returns(px) / InstStdev(20, Returns(px))))
```

Arithmetic (`+ - * /`) works on matrices, so expressions can **branch** — a ratio has two
inputs. That is the one thing the canvas cannot draw, so a tree is code-only and the page
says so rather than silently dropping a branch. A chain typed as code syncs back to the
blocks; switching to the Code tab always works in the other direction.

The editor validates as you type, against names and arity rather than syntax alone:

```
Rnk(returns)   -> unknown operator 'Rnk' — did you mean 'Rank'? (at column 1)
Lag(returns)   -> Lag takes 2 argument(s), got 1. Signature: Lag(n, mx)
Rank(bananas)  -> unknown source 'bananas'. Available: amihud, close, high_low, px, …
```

**It is parsed, never `eval`'d.** The grammar has numbers, whitelisted names and five
arithmetic symbols — no attribute access, no indexing, no subscripting. There is nothing
to escape *to*, which is a stronger guarantee than any denylist of dangerous names.
`tests/test_expr.py` posts a dozen hostile strings, including
`().__class__.__bases__[0].__subclasses__()`, and asserts the grammar rejects each.

**35 operators**, 27 of which fit on the canvas:

| group | operators |
|---|---|
| time series | `Lag` `InstMean` `InstStdev` `InstSkew` `InstSum` `InstCount` `InstRank` `InstZscore` `InstBeta`\* `ForwardFill` `Returns` `LogReturns` |
| cross section | `Rank` `Demean` `Zscore` `Winsorize` `Neutralize`\* `DemeanByGroup`\* `GroupRank`\* |
| elementwise | `Abs` `Log` `Sqrt` `Sign` `Neg` `Clip` `Pow` `Finite` `Div`\* `And`\* `Or`\* `Mask`\* |
| portfolio | `LongShortBalance` `GrossScale` `TopN` |

\* takes two matrices, so code tab only. `Lead` exists for building the *target* of a
study and is refused inside a signal.

### Equity costs are asymmetric, and that is the point

A futures round trip is symmetric. A Taiwan equity round trip is not:

| component | bps | share |
|---|---|---|
| 手續費 brokerage, both sides | 12.0 | 19% |
| **證交稅 transaction tax, sell side only** | **30.0** | **48%** |
| spread, half each side | 20.0 | 32% |
| **total round trip** | **62.0** | |

Half the cost is a tax that falls on one side only, so buys and sells are charged from
the *signed* weight change — `|dw| x one_rate` would halve the real cost of a long round
trip. Shorts additionally accrue 借券費 per day held.

Two frictions are modelled as mechanics rather than prices, because pricing them
understates them: **漲停/跌停** locks mean a blocked trade simply does not happen, and
**平盤下不得放空** means many names cannot be shorted. When a trade is blocked the prior
weight carries forward — dropping the name instead would silently rebalance into whatever
*was* tradable, which is a different and much better strategy than the one you tested.

### What the bundled presets actually do

Measured over 2024-01 to 2026-11, 698 sessions, ~290 names a day:

| preset | gross Sharpe | net Sharpe | turnover/day | names |
|---|---|---|---|---|
| short-term reversal | −0.57 | −4.99 | 26.6% | 282 |
| **12-1 momentum** | **+0.47** | **−0.03** | 3.2% | 252 |
| low volatility | −0.37 | −0.68 | 2.4% | 288 |
| Amihud illiquidity | −0.94 | −1.82 | 4.0% | 288 |
| concentrated reversal | −1.48 | −4.66 | 54.0% | 12 |

Momentum is the instructive row: a genuine gross edge, and costs take 107% of it — it
ends up a hair below break-even. On the chart the gap between the gross and net lines
*is* the tax. Nothing here is a tradable strategy, and three years is too short to
conclude much in either direction; the point is that the tool says so in its own output
rather than leaving you to notice.

At 62 bps a round trip, most daily-rebalanced cross-sectional signals in Taiwan are dead
before they start. The builder is designed to show you that in seconds.

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
  cli.py              fetch / fetch-equities / run / web / list
  ops.py              26 matrix operators; the cross-sectional vocabulary
  equity_panel.py     wide matrices + point-in-time liquidity universe
  xcosts.py           asymmetric TW equity costs and tradability rules
  xsim.py             cross-sectional simulator
  pipeline.py         a signal as a validated list of steps
  web/charts.py       inline SVG, no plotting dependency and no CDN
  web/app.py          FastAPI tearsheet
  web/builder.py      interactive signal builder routes
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
