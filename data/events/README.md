# Event files

Each CSV here becomes an event overlay and a CAAR study on both the builder and
the futures page. Two columns:

```csv
date,label
2024-01-18,TSMC Q4 2023 earnings call
```

The file name is what appears in the dropdown.

## tsmc_earnings.csv

Fetched, not hand-written:

```bash
twbt fetch-events --co-id 2330 --years 4 --name tsmc_earnings
```

The source is MOPS 法人說明會 filings, which are public for any listed or OTC
company back to the mid-2000s. Change `--co-id` for a different company.

### The host is the whole trick

```
https://mopsov.twse.com.tw/mops/web/ajax_t100sb02_1    <- returns the data
https://mops.twse.com.tw/mops/web/ajax_t100sb02_1      <- 833-byte error page
```

An earlier version of this project probed the second one, got the error page,
and concluded that earnings dates were not publicly available. They are. The
difference was two letters.

### Classification is not optional

A company files under this one heading for two quite different things:

| filing text | what it is |
|---|---|
| 公布本公司2023年第4季財務報告及2024年第1季業績展望 | the results call — **new information** |
| 受邀參加…『J.P. Morgan Taiwan CEO-CFO Conference』，會中就本公司**1/18法說會已公開之**財務數字…進行說明 | a broker conference restating what was already disclosed |

Over ROC 112–115, TSMC filed **55** of these: 16 results calls, 37 broker
conferences, 2 unclassified. Running an event study on all 55 would average 16
announcements with 37 restatements of them and report the result as an earnings
effect.

`twbt fetch-events` keeps only the results calls. `--all` keeps everything, and
is not the default for that reason.

### What the study actually shows

Twelve TSMC earnings calls fall inside the 2024–2026 futures sample. TXF CAAR
drifts to **+98 bps** over the −5…+5 window, which looks like something. No
single offset reaches significance: max |t| is **1.54** against a noise bar of
**2.89** for eleven offsets. With twelve events it could hardly be otherwise —
which is the point of plotting the t-statistic on top of the CAAR.
