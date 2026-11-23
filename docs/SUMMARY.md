# Project summary

## The problem

Backtests mostly fail for statistical reasons rather than coding ones, and the
statistical failures are the dangerous kind because they produce a *better*
result. A bug that loses money announces itself; a bug that makes money gets
written up. The four that do most of the damage are look-ahead, overlapping
windows, multiple testing, and costs.

This project is a backtester for Taiwan markets built so those four are hard to
commit by accident, and so the tool says when a result fails them.

Taiwan rather than the US — the suggested theme — because its market mechanics
make the cost question unavoidable rather than an afterthought. The 證交稅 is
0.30% on sells only and is roughly half of a 62 bps round trip, so almost every
daily-rebalanced cross-sectional signal is dead before it starts. A backtester
that treats cost as a flat parameter cannot tell you that; this one leads with
it.

## Intended user

Someone evaluating a signal idea who wants to know quickly whether it is real,
and who would rather be told "no" in thirty seconds than find out in three
months. It assumes familiarity with cross-sectional equity research — the
operator vocabulary (`Rank`, `Demean`, `Neutralize`) will be either obvious or
opaque depending on background.

Not a trading system. Nothing here places orders or is intended to.

## Core functionality

Three things, in order of how much they matter.

**An engine that refuses to flatter a result.** Trailing-only normalisation with
no full-sample equivalent in the API; an execution lag the engine owns so a
strategy cannot apply it twice or not at all; asymmetric Taiwanese costs; 漲停/
跌停 and 平盤下不得放空 modelled as mechanics rather than prices; significance
reported as OLS, Newey–West *and* disjoint t together, with a bootstrapped
multiple-testing bar.

**An interactive signal builder** (`/`) for cross-sectional equity strategies.
Compose 35 matrix operators as drag-and-drop blocks or as a typed expression:

```
GrossScale(LongShortBalance(Rank(Neg(InstSum(5, Returns(px))))))
```

25 panels come back in about 2.6 s, including the diagnostics that answer
questions the equity curve cannot: execution-lag decay, signal quintiles, sector
attribution, holdings-based factor exposures, and the gap between the book you
intended and the one the price limits let you hold.

**An interactive futures tearsheet** (`/futures`) for single-instrument
strategies, with the things only a futures contract has — settlement dates
derived from the contract data, a days-to-expiry profile, and event studies.

Both pages switch between English and 繁體中文, have fully configurable costs,
and run offline from a committed fixture.

## Scope

**In scope.** Daily data. Taiwan listed equities (TWSE + TPEX, ~2,000 names) and
TAIEX index futures (TXF/MXF/TMF). Public sources only, no credentials. Three
years, 2024-01 to 2026-11.

**Out of scope, deliberately.** Intraday data. Order routing. Portfolio
optimisation. Risk models beyond the four obvious factors. Any claim that a
bundled strategy is tradable — the two futures strategies and five equity presets
exist to exercise the engine, and the tool reports that four of the five equity
presets have no edge even before costs.

## Known limitations

**Three years is short.** Enough for 12-1 momentum to produce a signal at all
(252 sessions of history before its first), not enough to conclude much. The
per-year panel exists partly to make that obvious.

**Sector classification is a current snapshot, not point-in-time.** MOPS
publishes today's industry code. Fine for attribution; it must not become a
signal input, because today's label is partly a consequence of what the company
went on to become. `sources/sectors.py` says so where someone would use it.

**Expiry dates are exact where the data covers them and estimated otherwise.**
A TAIFEX contract settles on the third Wednesday unless that is a holiday —
February 2026 settled on the 23rd because Lunar New Year closed the 18th. Dates
are read from the last session each contract actually traded; the rule is used
only for contracts that have not expired yet, and the two are kept apart so a
study cannot average exact dates with guesses.

**Costs are defaults, not your fills.** The spread is measured from TAIFEX quote
data for futures; the equity half-spread, brokerage and borrow are
representative. Every one is an editable input, and both pages carry a
cost-sensitivity panel so no conclusion rests on them.

**The equity fetch is slow and can be rate-limited.** TWSE blocks a client that
asks too fast — it returns HTTP 307 with an HTML page, not an error status — so
it is fetched one session at a time, about 35 minutes for three years. Progress
is checkpointed. **You do not need to run it:** the committed fixture holds the
same three years and `--fixture` serves everything offline.

**Equity shares-outstanding is TPEX-only.** The size factor falls back to price
where share counts are missing, which is most TWSE names.

**Fixture and full-panel results differ, and one preset changes sign.** The committed
fixture screens a 150-name universe; the full fetch uses 300. 12-1 momentum nets +0.21 on
the first and −0.03 on the second. Nothing about the strategy changed — only how many
names it could choose from, which is itself the clearest evidence that a marginal result
over three years should not be believed. The README tabulates both.

**No transaction-level fills.** Positions are weights, costs are charged on
weight changes, and there is no model of partial fills, market impact, or
size-dependence. It prices one contract or one unit of notional.

## Verifying it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 290 tests, no network needed
twbt web --fixture        # -> http://127.0.0.1:8000
```

Verified on Python 3.11 from a clean clone. No credentials are required for anything, and
neither `pytest` nor `--fixture` touches the network.

The suite includes guard tests that fail if the known failure modes return:
`test_no_lookahead.py` corrupts the future half of a series and asserts the past
half is bit-identical for every operator in the registry; `test_expr.py` posts a
dozen hostile strings at the expression parser; `test_end_to_end.py` asserts that
no venue is missing from whole sessions and that TSMC stays in a liquidity
universe.

`docs/METHOD.md` explains why each statistical choice was made, with the worked
example that forced it. `docs/AI_NOTES.md` covers how the project was built and
what the AI got wrong.
