# Conversation excerpts

Seven exchanges from the Claude Code session that produced this repository,
taken verbatim from the session transcript.

**Why these seven.** The assessment asks how AI-generated output was evaluated
and validated. The useful evidence is not the model writing code well — it is
the record of it being wrong and that being caught. Six of the seven are
failures; the exception is the last, which is the model assessing its own work
against the brief.

Each is cross-referenced to the commit that resolved it, so a decision can be
followed from conversation → code → test.

**The pattern worth noticing.** The two most serious bugs (#2 and #3) had no
failing test and could not have had one. Both produced internally consistent
results from wrong inputs. Both were caught by reading a number and asking
whether it could be true:

- a gross Sharpe of −15 on random data, when random data has no edge in either
  direction
- TSMC outside a top-300-by-turnover universe on 63% of sessions, when it is the
  most traded stock in the market

That is the main thing I would take from this session. Reviewing an AI's
*conclusions* matters more than reviewing its code, and the cheapest way to
review a conclusion is to look at a number it implies.

---

## Editing and redaction, disclosed

Per the assessment's note on editing conversation records:

- **Selection.** Seven exchanges out of roughly 290 messages in the relevant
  portion of the session. The full transcript is available on request.
- **Truncation.** Long responses are cut at ~1,400 characters, marked
  `[… truncated …]`. Tool calls, file diffs and command output are omitted.
- **Redaction.** Paths to a private research repository, a private script name,
  database hostnames, IP addresses, SSH commands and a webhook URL are replaced
  with `[redacted: …]` markers.
- **No rewriting.** Nothing has been reworded, reordered, or improved after the
  fact. Every line inside a code fence is as it was written at the time.

One disclosure beyond redaction: during this session the model read source from
a private research repository of mine that has a co-author, to answer a question
about how earnings dates had previously been obtained. No code from it was
copied into this project — the operator vocabulary here shares names and
argument order with that package, but every implementation is independent and
written against public data. Those excerpts are not included here.

---

## 1. A confident claim that turned out to be false

*The model had written into two READMEs and a module docstring that TSMC earnings dates were not available from any public source, and built a deliberate refuse-to-guess feature around it. One question undid the whole thing: the working host differs by two letters (`mopsov` vs `mops`). A model states a negative as fluently as a positive, and a negative is far harder to check.*

**Commit:** `ffce41f`

**Prompt**

> But we did get the TSMC earnings dates previously, how did we get it?

**Response**

`````markdown
**I was wrong, and here's the difference:** I used `mops.twse.com.tw`; the working host is **`mopsov.twse.com.tw`** — the MOPS mirror. My probe hit an error page because of the hostname, and I concluded the data didn't exist.

Let me verify that endpoint live:
`````

`````markdown
**Confirmed — I was wrong.** The dates are public: **2024-01-18, 2024-04-18** are TSMC's quarterly calls, now sourced rather than guessed. My probe failed only because I used `mops.twse.com.tw` instead of `mopsov.twse.com.tw`.

Let me see the row structure so I can parse and classify it properly:
`````

---

## 2. Debugging from a number that could not be true

*No prompt — this surfaced while implementing the equity simulator. Random data has no edge in either direction, so a gross Sharpe of −15 is as impossible as +15. There was no failing test; the result was internally consistent and wrong.*

**Commit:** `c4a92ad`

**Prompt:** none — surfaced during implementation.

**Response**

`````markdown
**Found it — a genuine look-ahead bug, and the last line is the proof:**

```
corr(shortable_t, ret_t) = +0.797   <-- by construction
```

My `shortable()` rule is `px_t > px_{t−1}` — which *is* today's return being positive. So I was using day *t*'s return to decide whether a position that earns day *t*'s return may exist, systematically deleting exactly the shorts that were about to win. Hence the 3.5% hit rate.

The real 平盤下不得放空 test happens at **execution time**, using information available then. Fixing the alignment:
`````

`````markdown
The patch aborted before writing (assertion on an exact-match string), so the file is unchanged. Let me look at the actual text and patch it precisely:
`````

`````markdown
**Fixed.** Correlation +0.797 → **−0.002**, and gross Sharpe with tradability is now −0.41 against −0.49 without — both ~0 on noise, as they must be. Writing the regression test so it can't come back:
`````

---

## 3. An implausible number exposes half-missing data

*Surfaced while adding sector attribution. TSMC cannot leave a top-300-by-turnover universe. Working backwards from that one figure found 350 of 698 sessions with no TSE data at all, lost to a silent `except: pass` the model wrote when parallelising the fetch.*

**Commit:** `99f3b58 / 9fdb14f`

**Prompt**

> What about the by sector plots, SReturn plots etc. What other subplots are present in my f package but not present here

**Response**

`````markdown
**Real bug: TSMC (2330) is in the universe only 254 of 698 days.** The most liquid stock in Taiwan should never leave a top-300-by-turnover screen. And OTC has twice TSE's names despite TSE holding 70% of turnover. Investigating:
`````

`````markdown
The data itself is fine — 2330 tops the day at NT$177bn. But `TSE` has **1091 rows on the last day yet only 367,727 total**, which is ~337 days' worth out of 698. TSE data is missing for half the history:
`````

`````markdown
**Confirmed: 350 of 698 sessions have no TSE data at all.** OTC is complete. My `_fetch_one_day` swallowed every TWSE exception with `except Exception: pass` and a comment claiming the caller reports it — the caller does not. Classic silent no-op.

This means **every result I've shown was computed on a panel missing half the TSE sessions.** Fixing the swallowing first, so it can't recur:
`````

---

## 4. Measuring instead of arguing

*The futures cost model held the bid-ask spread fixed in index points, with a docstring arguing the opposite convention was wrong. Data the fetcher was already downloading and discarding settled it the other way — and the error grew across the sample, which is the worst shape an error can have.*

**Commit:** `30393d0`

**Prompt:** none — surfaced during implementation.

**Response**

`````markdown
This is the decisive table, and it shows **my backtester's cost model is wrong in direction**:

| year | measured | flat bps (published) | **flat pts (twbt)** |
|---|---|---|---|
| 2024 | 1.70 | 1.43 | 1.74 |
| 2025 | 1.87 | 1.43 | 1.56 |
| 2026 | 2.33 | 1.43 | **0.92** |

My `costs.py` docstring argues that holding *bps* constant is the error. The data says the opposite: the spread scales with the index, so **points** are the unstable unit. By 2026 my model charges 0.92 bps against a measured 2.33 — a 2.5× understatement, in the flattering direction. Fixing it:
`````

`````markdown
Now wiring the measured spread through `sensitivity`, the panel, and the engine:
`````

---

## 5. A statistic that contradicted itself

*Sharpe 4.03 printed beside t = 1.02. Both cannot be true: the annualisation assumed 252 independent observations a year where there were five.*

**Commit:** `cf6b0a9`

**Prompt:** none — surfaced during implementation.

**Response**

`````markdown
**Problem — Sharpe 4.03 next to t=1.02 is incoherent.** `performance()` annualises with `√252`, but these are 16 multi-day *trades*, not 252 daily observations. That's the same annualisation artifact that inflated the live SR audit. Annualising by the **observed** frequency instead:
`````

`````markdown
Sharpe 4.03 → 0.68, now coherent with t=1.02. Starting the web server and checking the page actually renders:
`````

---

## 6. Domain judgment the model did not have

*The model translated a section heading literally into casual Chinese. The correction is small; the bug it exposed is not. Failure lines were highlighted by matching English text prefixes, so rewording them in either language would have silently stopped failures being marked red.*

**Commit:** `a80f7cd`

**Prompt**

> "是不是真的" should be robustness check

**Response**

`````markdown
"是不是真的" is too colloquial for a section heading — the term is **穩健性檢定**. Let me fix that, and since you flagged it, check the rest of the Chinese for the same problem:
`````

`````markdown
Several are too colloquial. Fixing them — and removing a fragile coupling while I'm here: `_is_fail` detects failures by **string-prefix matching**, so any wording change silently breaks the red highlighting. That's exactly what rewording would have done.
`````

---

## 7. Assessing the work against its own brief

*Asked directly whether the project meets the assessment, the honest answer was that it overshoots the explicit scope guidance — which the brief states three times.*

**Commit:** `ef54f47`

**Prompt**

> Do you think this project right now fits the requirements mentioned in the pdf?

**Response**

`````markdown
**Partly — and the honest answer is that the biggest risk is it overshoots the brief.**

**What fits well.** Clean-environment reproducibility I just verified: fresh clone → venv → `pip install -e ".[dev]"` → **289 tests pass**, all three pages serve from the committed fixture with no network. Public data only. AI guided essentially all of it.

**What doesn't fit:**

**1. Size — this is the real problem.** The brief says it three times:

> "The goal is **not** to build a large or feature-rich system"
> "**Simple is completely fine**, but the logic should be sound"
> "Project size, number of features... are **not** evaluation criteria"

You have **9,746 lines, 27 modules, 37 panels, 2 asset classes, 5 data sources**, i18n, drag-and-drop, and a hand-written expression parser. That's not small-scale. And §4 says you must *"understand and be able to defend all submitted code"* and *"make a small change when given a new requirement"* — both get harder with size. A reviewer could reasonably read this as poor judgment about requirements.

**2. "AI collaboration and design notes" is missing entirely.** It's an explicit deliverable with named contents: key prompts, major design decisions, how you validated AI output, preserved conversation history. Nothing exists.

**3. Nothing is pushed.** `git ls-remote` returns zero heads — 24 commits, all local. The repo is empty.


*[… truncated …]*
`````

`````markdown
Now the project summary, structured exactly as the brief asks:
`````
