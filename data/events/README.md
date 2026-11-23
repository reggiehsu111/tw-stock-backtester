# Event files

Drop a CSV here and it appears in the futures tearsheet as an overlay and an
event study. Two columns:

```csv
date,label
2024-01-18,TSMC Q4 2023
2024-04-18,TSMC Q1 2024
```

The file name becomes the event set's name in the dropdown.

## Why there is no earnings file shipped here

The obvious thing to overlay is **TSMC's earnings dates** — it is roughly a
third of the TAIEX, so its results move the index more than any other single
announcement.

Those dates are not available from any public endpoint reachable from this
project. TWSE's open data publishes the *latest* quarter, the *latest* month
and *today's* announcements; none of them is a history, and MOPS does not serve
a conference-date archive.

TSMC's schedule is regular enough that the dates could be approximated — the
investor conference lands in mid-January, mid-April, mid-July and mid-October.
That approximation is deliberately **not** shipped. A date that is wrong by a
day or two barely matters for a chart annotation and matters enormously for a
one-day event study, which is exactly the kind of error worth refusing to make
on a user's behalf. Supply the real dates and the study is exact.

Expiry dates need no file: they are derived from the contract data itself, and
are exact wherever the data covers the settlement.
