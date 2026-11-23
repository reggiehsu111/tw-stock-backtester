# AI collaboration and design notes

Built with **Claude Code** (Opus), over a single long session. Essentially all
the code here was written by the model; my role was to set direction, decide
what to believe, and refuse answers that did not survive checking.

This document is about the second part, because that is the part worth
discussing. The interesting evidence is not that an AI can write a backtester —
it is the record of what it got **wrong**, how each error surfaced, and which
checks caught them. Every incident below is in the commit history with the test
that now prevents it.

---

## How the tool was used

A plain loop, repeated maybe forty times:

1. I state a requirement in a sentence or two.
2. The model proposes an approach; I accept, redirect, or push back.
3. It writes code **and** runs it — the loop always ends in execution, not in a
   diff. It has shell access, so it fetches real data, runs the backtest, prints
   numbers, and screenshots the page in a browser.
4. I read the numbers for plausibility. Anything implausible gets investigated
   before anything else proceeds.
5. The fix lands with a test that pins it, and a commit message that explains
   the failure rather than the change.

Step 4 is where the value is. Step 3 is what makes step 4 possible: a model that
only produces diffs cannot be checked this way, because there is nothing to be
implausible about.

**Verification techniques that actually caught things**, in rough order of how
much they earned:

| technique | what it catches |
|---|---|
| **reading output for implausibility** | the two worst bugs here, both invisible to tests |
| **adversarial tests** — corrupt the future, assert the past is bit-identical | look-ahead of any kind |
| **null-data tests** — feed noise, assert it reports nothing | miscalibrated statistics |
| **cross-instrument checks** | results that are one draw from noise |
| **pushing back on confident claims** | twice, the model was simply wrong |

---

## Six things the AI got wrong

### 1. Look-ahead inside a market-rules check

The model implemented Taiwan's 平盤下不得放空 rule — a short sale may not be made
at or below the previous close — as:

```python
shortable_t = px_t > px_{t-1}
```

which is literally `ret_t > 0`. It gated the book with the return that book was
about to earn, deleting precisely the shorts that were about to win.

Found by a number that could not be true: gross Sharpe **−15.05** on random
data with a 3.5% hit rate. Random data has no edge in either direction, so a
large negative Sharpe is as impossible as a large positive one.

Correlation between the rule and the return it gated: **+0.797**. After fixing
the alignment so the rule reads prices known at execution: **−0.002**, and the
Sharpe returns to −0.41.

Notable because the bug made results look **worse**. That is why it nearly
survived — a backtest getting less attractive reads as conservatism rather than
as a defect. Commit `c4a92ad`; the regression test asserts both the fixed and
the broken behaviour, so it cannot quietly stop proving anything.

### 2. A silent `except: pass` that lost half the data

Parallelising the equity fetch, the model wrote:

```python
except Exception:
    pass          # reported by the caller
```

The caller did not report it. TWSE answered roughly half the concurrent
requests with **HTTP 307** and a rate-limit page — not a 4xx, so
`raise_for_status()` was silent, and only the JSON decode failed. The cache
ended up with **350 of 698 sessions missing all TSE data**. Every panel still
rendered. Every statistic was internally consistent. All of it was computed on a
market missing its large caps half the time.

Found by implausibility again: **TSMC sat outside a top-300-by-turnover universe
on 63% of days**, and board attribution showed OTC at 84% of gross when TSE holds
~70% of market turnover.

The fix was three things, and the third generalises: per-venue concurrency
(TWSE serial), explicit block detection, and **counting what arrived rather than
what was requested**. `equity_panel.coverage()` now reports names per session per
board, and the page shows a warning banner. Commits `99f3b58`, `9fdb14f`,
documented as `docs/METHOD.md` §6.

An audit for the same pattern found **nine more** instances, including seven
tearsheet panels that rendered a silent em-dash on exception. They now display
the actual error.

### 3. Confidently asserting data did not exist

Asked for TSMC earnings dates, the model probed
`mops.twse.com.tw/.../ajax_t100sb02_1`, got an 833-byte error page, and
concluded that earnings dates were **not available from any public source**. It
wrote that conclusion into two README files and a module docstring, and built a
deliberate "we refuse to guess" feature around it.

The user asked: *"But we did get the TSMC earnings dates previously, how did we
get it?"*

The working host is **`mopsov`**.twse.com.tw — the MOPS mirror. Two letters. The
model had generalised from one failed request to a claim about the whole public
internet, and then documented it as established fact.

This is the single most useful exchange in the session, and the lesson is not
about MOPS. A model states a negative with the same fluency as a positive, and a
negative is much harder to check. Commit `ffce41f`.

### 4. A cost model that was wrong in the flattering direction

The futures cost model held the bid-ask spread fixed in **index points**, with a
docstring arguing that holding it fixed in basis points would misstate costs.
Measuring it settled the question the other way:

| year | median close | spread (pts) | spread (bps) |
|---|---|---|---|
| 2024 | 21,897 | 2.0 | 1.12 |
| 2026 | 43,204 | 6.0 | 1.50 |

The index doubled and the point spread tripled. Holding points constant charged
**0.92 bps in 2026 where the measured cost was 2.33** — an understatement that
*grows* across the sample, which is the worst shape an error can have because it
manufactures a trend out of a convention. Commit `30393d0`.

The data to check this was already being downloaded and discarded: TAIFEX ships
bid and ask in the same file as prices.

### 5. A Sharpe ratio that could not coexist with its own t-statistic

The momentum strategy reported **Sharpe 4.03 beside t = 1.02**. Both cannot be
true. `performance()` annualised with a hardcoded `sqrt(252)` while the series
was 16 multi-day *trades* over three years — five observations a year, not 252.
Annualisation now measures frequency from the index; the same number is **0.68**.
Commit `cf6b0a9`.

### 6. Two wrong definitions of the same concept

Asked for an event study around "when the signal enters", the model first
defined entries as *"the number of names held rose"* — which found **1 event in
600 sessions**, because a cross-sectional book rotates names at a constant count.
Second attempt, *"any name was opened"*, found nearly every session — and an
event study where every day is an event is the unconditional mean with extra
steps. Third definition — names going flat-to-held, keeping the busiest
sessions — works. Commit `4078a2e`.

---

## Design decisions worth defending

**Statistics that cannot be reported dishonestly.** `stats.regress()` returns
OLS, Newey–West and disjoint-sample t *together* and offers no way to get only
the first. On overlapping windows over pure noise, OLS has median |t| 3.25 and
clears 2 on 60% of seeds; the disjoint estimator reads 0.46 and 10%. If the
flattering number is available alone, it gets quoted alone.

**The engine owns the lag.** Strategies declare which columns are published;
`backtest.run` refuses `publication_lag=0` for them. A lag each strategy applied
itself would be applied twice in some and zero times in others — and the zero-lag
version of a flow strategy looks wonderful.

**Browser input is parsed, never evaluated.** The expression editor is a
hand-written tokenizer and recursive-descent parser. The grammar contains
numbers, whitelisted names and five arithmetic symbols — no attribute access, no
indexing — so there is nothing to escape *to*. That is a stronger guarantee than
a denylist, which is always one unthought-of name from defeat.
`tests/test_expr.py` posts `().__class__.__bases__[0].__subclasses__()` and
friends.

**Costs are asymmetric because the market is.** Taiwan's 證交稅 is 0.30% on sells
only — 48% of a 62 bps round trip. Buys and sells are charged from the *signed*
weight change; `|dw| × one_rate` would halve the true cost of a long round trip.

**Blocked trades carry the prior weight.** When 漲停/跌停 prevents a trade, the
position carries forward rather than being dropped. Dropping would silently
rebalance into whatever *was* tradable, which is a different and much better
strategy than the one under test.

**Point-in-time universe.** Membership is a *trailing* median-turnover screen.
Picking "the 300 most liquid names" over the full sample selects for names that
went on to become liquid, which correlates with having gone up.

---

## What the AI was reliably good at

Mechanical correctness at scale, and recall of its own conventions. Given "the
spread convention is one per round trip", it applied that consistently across
the engine, the tests and the docs without being reminded. It also wrote better
tests than I would have asked for — the adversarial look-ahead test is
parametrised over the operator registry, so a new operator is covered the moment
it is registered rather than whenever someone remembers.

It was weakest at exactly one thing: **knowing when it did not know.** Every
incident above except #4 and #5 is a confident assertion that happened to be
false. None of them were hedged. The practical consequence is that reviewing an
AI's *conclusions* matters more than reviewing its code, and the cheapest way to
review a conclusion is to look at a number it implies and ask whether that number
could be true.

---

## Disclosure

Per the assessment's note on editing conversation records: during this session
the model read source from a **private research repository of mine that has a
co-author**, in order to answer a question about how earnings dates had been
obtained previously. No code from it was copied into this project — the operator
vocabulary here shares names and argument order with that package but every
implementation is independent, written against public data.

Those excerpts are **redacted from the submitted conversation history**, along
with hostnames, IP addresses and credentials. The redactions are cuts, not
rewrites: nothing has been reworded to read better.

See [`CONVERSATION_EXCERPTS.md`](CONVERSATION_EXCERPTS.md) for the seven selected
exchanges and the full disclosure of what was selected, truncated and redacted.
