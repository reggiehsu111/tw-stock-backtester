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

**What this project does:** `stats.regress()` returns all three t-statistics side by side
and the tearsheet displays them together. There is no way to get only the OLS number.

---

## 3. The best of N tests is not a finding

Search forty variants of a strategy and the best one will show t ≈ 2.5 even if none has
any edge. The fix is to know what the maximum looks like under the null *for the number
of tests you actually ran*.

`stats.noise_bar(returns, n_tests)` bootstraps exactly that: it randomly relabels the
same return series into `n_tests` groups and records the largest |t| each time. The 95th
percentile of that distribution is the bar a real result has to clear.

Typical bars, for the sample sizes here: 12 tests → ≈3.2, 20 tests → ≈3.1, 40 tests
→ ≈3.5. A t of 2.8 from a forty-variant sweep is *below* the noise floor.

The related discipline: when splitting a result into buckets, the statistic that matters
is the bucket **versus its complement**, not the bucket's own t. A bucket can easily show
t = 3.4 on its own while being statistically indistinguishable from the other buckets.

---

## 4. Costs: one spread per round trip, not two

A round trip crosses the spread **once in total** — you buy at the ask (half a spread
above mid) and sell at the bid (half below). Charging a full spread at each leg
double-counts.

```
cost = spread_entry/2 + spread_exit/2 + fees
```

Getting this wrong by one spread flipped a result from "clearly tradable" to "below
cost" during development. The model also keeps per-leg spreads separate, because the
first minute of a session is routinely 50% wider than the rest — charging the average
makes an open-entry strategy look better than it is.

`costs.CostModel` takes measured spreads rather than a flat assumption, and the tearsheet
plots **t-stat against cost** so you can see how much slippage the result survives. A
strategy that dies between 1.5 and 2.0 bps is not a strategy.

---

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
