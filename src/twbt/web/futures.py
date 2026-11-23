"""The interactive futures tearsheet.

Parallel to the equity builder, for the single-instrument side. The futures page
used to be four static charts rendered from a pickle; this is the same shape of
tool as ``/``: pick a strategy, turn the dials, get the full panel set back in
under a second.

What is here that the equity side cannot have, because a single instrument has a
contract and a cross-section does not:

    expiry        settlement dates derived from the contract data, exact
                  wherever the data covers them, plus a days-to-expiry profile
                  and an event study around settlement
    events        a CAAR around any dated event you supply as a CSV
    seasonality   month and weekday, which on one instrument is a real question
                  and on 300 names is mostly a restatement of the market

And the normalisation control, which belongs here because the futures
strategies turn a raw series into a threshold and the method used to do that
changes the answer more than the threshold does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import backtest, costs, events as ev, i18n, panel as panel_mod, stats
from .. import strategies as strat_mod
from . import charts

DEFAULTS = {
    "strategy": "momentum",
    "normalization": "zscore",
    "contract": "TXF",
    "exec_shift": 1,
}


@dataclass
class FuturesState:
    panel: pd.DataFrame
    expiries: ev.ExpiryCalendar
    dte: pd.Series
    info: dict

    @classmethod
    def load(cls, fixture: bool = False) -> FuturesState:
        from .. import cache

        p = panel_mod.build(fixture=fixture)
        fut = cache.load("futures_daily", **({"fixture": True} if fixture else {}))
        cal = ev.expiry_dates(fut, sessions=p.index)
        dte = ev.days_to_expiry(p.index, cal.trusted_dates)
        info = {
            "sessions": int(len(p)),
            "start": str(p.index.min().date()),
            "end": str(p.index.max().date()),
            "expiries_observed": int(len(cal.observed)),
            "expiries_estimated": int(len(cal.estimated)),
        }
        return cls(panel=p, expiries=cal, dte=dte, info=info)


def _num(v, nd=2, sign=False, suffix=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return (f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}") + suffix


def register(app, templates, fixture: bool = False) -> None:
    router = APIRouter()
    state: dict = {"s": None}

    def st() -> FuturesState:
        if state["s"] is None:
            state["s"] = FuturesState.load(fixture=fixture)
        return state["s"]

    @router.get("/futures", response_class=HTMLResponse)
    def page(request: Request, lang: str = "en"):
        lang = i18n.pick(lang)
        try:
            s = st()
            info, err = s.info, None
        except Exception as exc:                        # noqa: BLE001
            info, err = {}, str(exc)
        return templates.TemplateResponse(request, "futures.html", {
            "lang": lang, "langs": i18n.LANGS, "wide": True,
            "T": lambda k: i18n.t(i18n.UI, k, lang),
            "CAP": lambda k: i18n.t(i18n.CAPTION, k, lang),
            "info": info,
            "load_error": err,
            "strategies": {n: strat_mod.get(n)().describe()
                           for n in strat_mod.available()},
            "normalisations": panel_mod.NORMALISATIONS,
            "event_sets": ev.available_events(),
            "defaults": DEFAULTS,
            "fixture": fixture,
            "cfg": {"lang": lang, "defaults": DEFAULTS,
                    "strings": {k: i18n.t(i18n.UI, k, lang)
                                for k in ("running", "done_in")}},
        })

    @router.post("/api/futures/run")
    async def run(request: Request):
        t0 = time.time()
        try:
            body = await request.json()
        except Exception:                               # noqa: BLE001
            return JSONResponse({"error": "malformed request body"}, status_code=400)

        lang = i18n.pick(str(body.get("lang") or "en"))
        C = lambda k: i18n.t(i18n.CHART, k, lang)       # noqa: E731

        try:
            s = st()
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"panel unavailable: {exc}"},
                                status_code=503)

        name = str(body.get("strategy") or DEFAULTS["strategy"])
        if name not in strat_mod.REGISTRY:
            return JSONResponse({"error": f"unknown strategy {name!r}"},
                                status_code=400)
        norm = str(body.get("normalization") or DEFAULTS["normalization"])
        if norm not in panel_mod.NORMALISATIONS:
            return JSONResponse({"error": f"unknown normalisation {norm!r}"},
                                status_code=400)

        try:
            contract = str(body.get("contract") or "TXF")
            exec_shift = int(body.get("exec_shift", 1))
            spread = body.get("spread_bps")
            cm = costs.CostModel(
                contract,
                spread_bps=(float(spread) if spread not in (None, "") else None),
                fees_pts_per_side=(float(body["fees_pts"])
                                   if body.get("fees_pts") not in (None, "") else None))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": f"bad option: {exc}"}, status_code=400)

        params = {"normalization": norm}
        for k in ("entry_z", "lookback", "hold", "window", "dealer_z"):
            if body.get(k) not in (None, ""):
                try:
                    params[k] = type(getattr(strat_mod.get(name)(), k, 1.0))(body[k])
                except Exception:                       # noqa: BLE001
                    pass
        try:
            strategy = strat_mod.get(name)(**params)
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"bad parameter: {exc}"}, status_code=400)

        p = s.panel
        try:
            res = backtest.run(p, strategy, cm,
                               n_variants_tested=getattr(strategy, "n_variants", 1))
            if body.get("walk_forward", True):
                res.walk_forward = backtest.walk_forward(p, strategy, cm,
                                                         contract=contract)
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=400)

        m = res.main
        why: dict[str, str] = {}

        def attempt(key, fn, default=None):
            try:
                return fn()
            except Exception as exc:                    # noqa: BLE001
                why[key] = f"{type(exc).__name__}: {exc}"
                return default

        ret = p["ret"]
        daily = m.daily_net

        # --- expiry ------------------------------------------------------- #
        def _dte_profile():
            d = pd.concat([ret, s.dte], axis=1).dropna()
            if d.empty:
                return None
            cut = pd.cut(d["days_to_expiry"], [-0.5, 0.5, 2.5, 5.5, 10.5, 1e9],
                         labels=["0", "1-2", "3-5", "6-10", "11+"])
            g = d.groupby(cut, observed=True)["ret"]
            mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
            return mean, se, g.count()

        dte_prof = attempt("dte_profile", _dte_profile)
        expiry_caar = attempt(
            "expiry_caar",
            lambda: ev.caar(ret, s.expiries.trusted_dates, 10, 10), {})

        # --- user events --------------------------------------------------- #
        ev_name = str(body.get("event_set") or "")
        user_caar, ev_n = {}, 0
        if ev_name:
            def _ue():
                tbl = ev.load_events(ev_name)
                return ev.caar(ret, tbl["date"], 10, 10), len(tbl)
            got = attempt("event_caar", _ue, ({}, 0))
            user_caar, ev_n = got if got else ({}, 0)

        # --- seasonality ---------------------------------------------------- #
        def _season(by):
            d = pd.DataFrame({"ret": ret}).dropna()
            key = d.index.month if by == "month" else d.index.dayofweek
            g = d.groupby(key)["ret"]
            mean = g.mean()
            se = g.std(ddof=1) / np.sqrt(g.count())
            names = ({i: pd.Timestamp(2024, i, 1).strftime("%b") for i in range(1, 13)}
                     if by == "month"
                     else dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])))
            mean.index = [names.get(i, str(i)) for i in mean.index]
            se.index = mean.index
            return mean, se

        months = attempt("months", lambda: _season("month"))
        weekdays = attempt("weekdays", lambda: _season("weekday"))

        # --- normalisation comparison ---------------------------------------- #
        def _norm_compare():
            out = {}
            for meth in panel_mod.NORMALISATIONS:
                if meth == "none":
                    continue
                cand = strat_mod.get(name)(**{**params, "normalization": meth})
                r = backtest.run(p, cand, cm, contracts=(contract,))
                out[meth] = r.main.net_perf.mean
            return pd.Series(out)

        norm_cmp = attempt("norm_compare", _norm_compare)

        rolling = attempt("rolling",
                          lambda: daily.rolling(252, min_periods=120).mean()
                          / daily.rolling(252, min_periods=120).std(ddof=0)
                          * np.sqrt(252))

        wf = res.walk_forward
        stats_rows = [
            ("trades", f"{m.n_trades:,}"),
            ("gross mean", _num(m.gross_perf.mean, 2, True, " bps")),
            ("net mean", _num(m.net_perf.mean, 2, True, " bps")),
            ("net t", _num(m.net_perf.t, 2, True)),
            ("net Sharpe", _num(m.net_perf.sharpe, 2, True)),
            ("hit rate", _num(m.net_perf.hit_rate * 100, 1, suffix="%")),
            ("avg cost", _num(m.avg_cost_bps, 2, suffix=" bps/trade")),
            ("breakeven", _num(m.breakeven_bps, 2, True, " bps")),
            ("max drawdown", _num(m.net_perf.max_drawdown, 0, suffix=" bps")),
        ]
        if np.isfinite(res.noise_bar_95):
            stats_rows.append(("noise bar (95%)",
                               _num(res.noise_bar_95, 2)
                               + f" for {res.n_variants_tested} variants"))

        return JSONResponse({
            "describe": strategy.describe(),
            "costs": cm.describe(),
            "elapsed_ms": int((time.time() - t0) * 1000),
            "verdict": [{"text": v, "fail": v.startswith("FAILS")}
                        for v in res.verdict()],
            "stats": stats_rows,
            "consistency": [[r.contract, _num(r.net_mean_bps, 2, True),
                             _num(r.net_t, 2, True), _num(r.net_sharpe, 2, True)]
                            for r in res.consistency().itertuples()],
            "walk_forward": ([[int(r.year), str(r.params), int(r.n),
                               _num(r.net_mean_bps, 2, True), _num(r.net_t, 2, True)]
                              for r in wf.itertuples()] if wf is not None and len(wf)
                             else []),
            "event_n": ev_n,
            "charts": {
                "equity": charts.equity(
                    {C("l_gross"): m.daily_gross.cumsum(),
                     C("l_net"): daily.cumsum()}, title=C("cum_pnl")),
                "by_year": charts.by_year(daily),
                "decomp": charts.decomposition(daily, m.position),
                "drawdown": charts.drawdown(daily, title=C("drawdown")),
                "roll_sharpe": (charts.series_panel(
                    C("y_sharpe"), rolling, C("roll_sharpe"), C("y_sharpe"))
                    if rolling is not None and rolling.notna().any()
                    else charts.empty(why.get("rolling", "—"))),
                "cost_curve": charts.cost_curve(m.sensitivity),
                "expiry_profile": (charts.profile(
                    dte_prof[0], dte_prof[1], C("dte_profile"), "bps",
                    C("x_dte")) if dte_prof else
                    charts.empty(why.get("dte_profile", "—"))),
                "expiry_caar": (charts.caar_panel(expiry_caar, C("expiry_caar"))
                                if expiry_caar
                                else charts.empty(why.get("expiry_caar", "—"))),
                "event_caar": (charts.caar_panel(user_caar, C("event_caar"))
                               if user_caar else charts.empty(
                                   why.get("event_caar",
                                           i18n.t(i18n.UI, "no_events", lang)))),
                "months": (charts.profile(months[0], months[1], C("months"), "bps")
                           if months else charts.empty(why.get("months", "—"))),
                "weekdays": (charts.profile(weekdays[0], weekdays[1],
                                            C("weekdays"), "bps")
                             if weekdays else charts.empty(why.get("weekdays", "—"))),
                "norm_compare": (charts.exposure_bars(
                    norm_cmp, C("norm_compare"), "bps")
                    if norm_cmp is not None and len(norm_cmp)
                    else charts.empty(why.get("norm_compare", "—"))),
            },
        })

    app.include_router(router)
