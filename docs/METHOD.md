# Method notes

Why this backtester is shaped the way it is. Each section is a mistake that produced a
*better-looking* result than the truth, which is what makes them dangerous — a bug that
loses money announces itself, a bug that makes money does not.

---

## 1. Look-ahead hides in thresholds, not just in prices

Everyone knows not to use tomorrow's price. The subtler version is using tomorrow's
*distribution*: a z-score, a quantile cut, a volatility estimate computed over the whole
sample. The signal value on day 1 then depends on data from day 900.

A worked example. A volume filter was defined as "bottom tercile of relative volume",
where the tercile came from the full sample:

```python
thr = rel_volume.quantile(1/3)        # WRONG: uses the whole history
quiet = rel_volume < thr
```

The fix is a trailing quantile:

```python
thr = rel_volume.shift(1).rolling(250, min_periods=120).quantile(1/3)
quiet = rel_volume < thr
```

In that case it happened not to matter much — the trailing threshold tracked the
full-sample one closely and classification agreed 97% of the time. But *that could only
be known after checking*, and the same bug elsewhere can be worth several multiples of
the edge.

**What this project does:** `panel.zscore()` and `panel.trailing_quantile()` take only
backward-looking windows and always `shift(1)` first. There is no full-sample equivalent
in the API. `tests/test_no_lookahead.py` corrupts the future half of a series and asserts
past-half signals are bit-identical.

---

## 2. Overlapping windows inflate t-stats by about √h

If a 30-day trailing average predicts a 30-day forward return, consecutive daily
observations share 29/30 of both series. Ordinary OLS treats them as independent and the
t-stat is overstated by roughly √h.

Measured on real data, the same specification gave:

| method | t |
|---|---|
| OLS | **−16.30** |
| Newey–West (lag = h) | −4.43 |
| disjoint sample (every h-th observation) | **−3.56** |

Three different stories: "overwhelming", "strong", "interesting". Only the last two are
defensible, and the disjoint one is the most convincing because it needs no kernel
assumption — it simply throws away the overlap.

Another specification in the same family collapsed from −11.79 to **−1.65**. You cannot
tell in advance which it will be.

How bad is it in general? `tests/test_stats.py` runs the construction on data with
*no relationship at all* — a trailing 30-day mean against a strictly forward 30-day
mean of the same white noise, two windows that never share a single observation.
Across 20 seeds:

| | median \|t\| | fraction over 2 |
|---|---|---|
| OLS | 3.25 | 60% |
| Newey–West | 0.88 | 20% |
| disjoint | 0.46 | 10% |

OLS declares significance on noise the majority of the time. One seed gives
t = +4.94 against a disjoint t of +0.45 — a result that would be written up.

**What this project does:** `stats.regress()` returns all three t-statistics side by side
and the tearsheet displays them together. There is no way to get only the OLS number.
`RegressionResult.honest_t` picks the disjoint value whenever overlap is declared, and
`.inflation` reports the ratio so you can see how much the overlap was worth.

---

## 3. The best of N tests is not a finding

Search forty variants of a strategy and the best one will show t ≈ 2.5 even if none has
any edge. The fix is to know what the maximum looks like under the null *for the number
of tests you actually ran*.

`stats.noise_bar(returns, n_tests)` bootstraps exactly that: it randomly relabels the
same return series into `n_tests` groups and records the largest |t| each time. The 95th
percentile of that distribution is the bar a real result has to clear.

Measured on a realistic sample (1,300 daily observations, fat-tailed):

| tests run | 95th-percentile \|t\| under the null |
|---|---|
| 1 | 1.97 |
| 5 | 2.55 |
| 12 | 2.86 |
| 20 | 3.06 |
| 40 | 3.36 |
| 100 | 4.24 |

The first row recovers the textbook 1.96, which is the sanity check that the simulation
is calibrated. The rest is the cost of searching. **A t of 2.8 from a forty-variant sweep
is below the noise floor** — and 40 variants is a small sweep.

Two buckets gives the same bar as one test, not two: each bucket is the other's
complement, so there is only one comparison. The function handles this rather than
counting it twice.

The related discipline: when splitting a result into buckets, the statistic that matters
is the bucket **versus its complement**, not the bucket's own t. A bucket can easily show
t = 3.4 on its own while being statistically indistinguishable from the other buckets.

---

## 4. Costs: one spread per round trip, and measure it

Two separate mistakes live here. The first is arithmetic, the second is an assumption
that survives because it sounds careful.

**One spread, not two.** A round trip crosses the spread **once in total** — you buy
at the ask, half a spread above mid, and sell at the bid, half below. Charging a full
spread at each leg double-counts.

```
cost = spread_entry/2 + spread_exit/2 + fees_entry + fees_exit
```

Getting this wrong by one spread flipped a result from "clearly tradable" to "below
cost" during development. The related trap is the *other* direction: masking the cost
series to days with a non-zero position drops the exit leg, because the position is
already flat on the day the exit is charged, and the backtest then pays half a round
trip. Both are invisible in a diff. `tests/test_backtest.py` pins the arithmetic:
`0 → 1 → 0` costs exactly one round trip, and so does `0 → 1 → 1 → 1 → 0`.

**Hold the spread in bps, not in points.** An earlier version of this project held the
spread fixed at 3.0 index points and argued that holding it fixed in bps would
"understate cost early and overstate it late". That argument was backwards, and
measuring it settled the question. On the TX front month:

| year | median close | median spread (pts) | median spread (bps) |
|---|---|---|---|
| 2024 | 21,897 | 2.0 | 1.12 |
| 2025 | 23,344 | 3.0 | 1.37 |
| 2026 | 43,204 | 6.0 | 1.50 |

The index roughly doubled and the point spread tripled. **Points are the unstable
unit.** The fixed-point model charged 0.92 bps in 2026 where the measured cost was
2.33 — a 2.5× understatement that *grows* across the sample, which is the worst shape
an error can take, because it manufactures a trend.

Fees go the other way. They are a fixed NT$ amount, so they genuinely do shrink as a
share of notional. The model therefore holds the spread constant in bps and the fees
constant in points, which is not an inconsistency but the measured behaviour of two
different things.

**Better still: do not assume at all.** TAIFEX ships the closing bid and ask in the
same file as the prices. `panel.build` turns them into a `spread_bps` column and the
engine uses it automatically. This matters beyond accuracy-on-average, because the
spread is **wider on exactly the days a signal fires**:

| day's range | median spread |
|---|---|
| Q1 (calmest) | 1.27 bps |
| Q3 | 1.27 bps |
| Q5 (wildest) | 1.85 bps |

Regressing spread on range gives t = +6.0; high-range days cost 0.56 bps more than
low-range days (t = +5.8). Any strategy that triggers on volatility — which is most of
them — is therefore systematically undercharged by a constant. Measured spreads raise
the realised cost of the bundled momentum strategy by 52%, from 2.61 to 3.98 bps per
trade. It is a daily closing snapshot rather than a time-weighted average, so it is an
estimate too, but it is an estimate that moves with the market instead of with a guess.

`costs.CostModel` accepts measured spreads, a bps constant, or a point constant, in
that order of preference, and the tearsheet plots **t-stat against cost** so you can
see how much slippage the result survives. A strategy that dies between 1.5 and 2.0
bps is not a strategy.

**What this model does not include**, and should not be read as including: market
impact, partial fills, and any size-dependence at all. It prices one contract. There
is also no short-side asymmetry, which is correct for index futures — no borrow, no
locate, no uptick rule — but would be badly wrong if this were ever pointed at cash
equities, where a Taiwan short carries both a borrow fee and the risk of being trapped
by a limit-up close.

## 5. The test that outranks all the statistics

Everything above asks *"could this be chance?"*. A different and more dangerous question
is *"is the stated mechanism true?"* — and the cheapest way to ask it is to run the
identical construction on instruments the explanation says must behave the same way.

Two results during development passed every statistical check in sections 1–4 —
monotone response, clean plateaus, surviving controls, held-out period working — and both
died here. Run on sibling contracts of the same index, they gave positive, zero, and
significantly negative. No mechanism produces that pattern; three independent draws from
noise do.

**What this project does:** `backtest.run()` accepts a list of instruments and the
tearsheet shows every one. A strategy that works on a single contract and not its
siblings is flagged in the output rather than quietly reported as a success.

---

## What would make a result believable here

For reference, the bar this project's own tooling is designed to measure against:

- a **parameter plateau**, not a spike — neighbouring settings should work almost as well
- a **walk-forward** that re-picks parameters on prior data only, positive in most years
- **cost-robustness** well past the realistic estimate
- **consistency across sibling instruments**
- an out-of-sample period that was never looked at while choosing anything

Neither bundled strategy clears all five. They are not meant to — they exist to exercise
the engine and to show what the output looks like when a result is merely interesting
rather than real.
