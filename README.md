# tw-stock-backtester

A backtester for Taiwan markets, built around the parts of backtesting that usually go
wrong: **look-ahead, overlapping windows, multiple testing, and costs.**

Everything runs on free public data — TAIFEX, TWSE, TPEX and MOPS. No credentials, no
database, no vendor feed.

```bash
git clone https://github.com/reggiehsu111/tw-stock-backtester.git
cd tw-stock-backtester
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 290 tests, no network needed
twbt web --fixture        # -> http://127.0.0.1:8000
```

That is the whole setup. A three-year sample ships in the repository, so nothing needs
fetching to see the tool work.

### Prerequisites

| | |
|---|---|
| **Python** | **3.11 verified.** `pyproject.toml` allows ≥3.10; 3.12 and 3.13 are untested, and the pandas/pyarrow wheels are where trouble would appear. |
| **OS** | developed and verified on macOS. Nothing is platform-specific — pure Python plus wheels. |
| **Credentials** | none, for anything. No API keys, no database, no vendor account. |
| **Network** | **not needed** for `pytest` or `twbt web --fixture`. Only the `twbt fetch-*` commands reach the internet. |
| **Disk** | ~18 MB cloned; ~250 MB with the virtualenv. |
| **Time** | install ~15 s, tests ~20 s. |

Activate the virtualenv in the same shell before running `twbt` — the entry point installs
into `.venv/bin`, and "command not found" is the usual symptom of skipping it. If port 8000
is busy, `twbt web --fixture --port 8001`.

**Start here:** [`docs/SUMMARY.md`](docs/SUMMARY.md) — problem, intended user, scope and
known limitations. [`docs/METHOD.md`](docs/METHOD.md) — why each statistical choice was
made, with the worked example that forced it. [`docs/AI_NOTES.md`](docs/AI_NOTES.md) —
how this was built with Claude Code, and the six things it got wrong.
[`docs/CONVERSATION_EXCERPTS.md`](docs/CONVERSATION_EXCERPTS.md) — seven exchanges from the
session, verbatim, with redactions disclosed.

## How to read this repository

The core is small and is the part worth reviewing:

| file | lines | what it is |
|---|---|---|
| `stats.py` | ~220 | overlap-aware significance, bootstrapped noise bars |
| `backtest.py` | ~290 | signal → position → P&L, and the verdict |
| `costs.py` / `xcosts.py` | ~230 each | futures and equity cost models |
| `panel.py` / `ops.py` | ~190 / ~460 | trailing-only normalisation, 35 matrix operators |

Everything else — two web pages, five data sources, i18n, the expression parser — is a
layer on top of that. If you read four files, read those.

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

Two columns, because they differ and you will see the first one. `--fixture` uses the
committed 150-name universe; the full panel after `twbt fetch-equities` uses 300.

| preset | fixture gross / net | full gross / net |
|---|---|---|
| short-term reversal | −1.00 / −5.33 | −0.57 / −4.99 |
| **12-1 momentum** | **+0.70 / +0.21** | **+0.47 / −0.03** |
| low volatility | −0.88 / −1.19 | −0.37 / −0.68 |
| Amihud illiquidity | −0.09 / −0.86 | −0.94 / −1.82 |
| concentrated reversal | −2.01 / −6.22 | −1.48 / −4.66 |

Momentum is the instructive row twice over. It has a genuine gross edge that costs very
nearly eat — and **its net Sharpe changes sign** between the two universes, +0.21 against
−0.03. Nothing about the strategy changed; only how many names it could choose from.

That is the honest headline for this whole project. A result that flips sign when the
universe goes from 150 names to 300 is not a result, and three years is not enough sample
to say otherwise. The tool is built to make that visible rather than to produce a number
you could put in a deck.

At 62 bps a round trip, most daily-rebalanced cross-sectional signals in Taiwan are dead
before they start. The builder is designed to show you that in seconds.

## The futures tearsheet

`/futures` is the same kind of tool for the single-instrument side: pick a strategy,
turn the dials, get twelve panels back in about 150 ms. Three things live here that the
cross-sectional page cannot have, because a futures contract has a settlement date and a
cross-section does not.

**Expiry, derived from the data.** A TAIFEX monthly contract settles on the third
Wednesday — except when that Wednesday is a holiday. February 2026 settled on the **23rd**
because Lunar New Year closed the 18th. Checking the rule against observed last-trading
days, 34 of 41 contracts matched and the misses were that holiday plus contracts that
have not expired yet. So settlement dates are read from the last session each contract
actually traded, and the rule is used only for months with no data — the two are labelled
separately so a study cannot quietly average exact dates with estimates.

**Normalisation as a control, not a hardcoded choice.** `zscore`, `rank`, `robust`
(median/IQR) and `minmax`, all trailing-only. The choice matters more than most people
expect: on a skewed series a rank and a z-score correlate at **0.13**, and the bundled
momentum strategy returns anywhere from +23 to +713 bps net depending purely on which one
you pick — none of them significant. A panel shows all four side by side, because a
result that exists under one method and not the others is telling you the method did the
work.

**Event studies**, on both pages. Three sources:

| event | where from |
|---|---|
| settlement | derived from contract data, exact |
| signal entries | the signal's own busiest entry sessions — needs no external data |
| anything dated | a CSV you drop in `data/events/` |

Each renders as a CAAR with its **t-statistic dashed on top and ±2 marked**, because a
drifting CAAR looks like a finding and with twenty events usually is not.

### Where the earnings dates come from

```bash
twbt fetch-events --co-id 2330 --years 4 --name tsmc_earnings
```

MOPS publishes 法人說明會 filings for every listed and OTC company, back to the
mid-2000s, with no account. Two things about it are worth knowing.

**The host.** `mopsov.twse.com.tw` serves the data; `mops.twse.com.tw` answers the same
path with an 833-byte error page. An earlier version of this project probed the second,
got the error page, and concluded in its own documentation that earnings dates were not
publicly available. They are.

**Classification is not optional.** A company files under this one heading both for
results calls (`公布本公司2023年第4季財務報告`) and for broker conferences that explicitly
restate what was already disclosed (`受邀參加…會中就本公司1/18法說會已公開之財務數字…`).
TSMC filed **55** over ROC 112–115: 16 results calls and 37 restatements. An event study
over all 55 would average real announcements with non-events. `fetch-events` keeps the
results calls; `--all` keeps everything and is not the default.

Twelve of those calls fall inside the futures sample. TXF CAAR drifts to +98 bps across
−5…+5, which looks like a finding, and no offset clears significance: max |t| is 1.54
against a 2.89 noise bar for eleven offsets. That is why the t-statistic is drawn on the
chart rather than left for the reader to compute.

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
