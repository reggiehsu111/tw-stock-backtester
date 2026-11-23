"""Routes for the interactive signal builder.

The panel is loaded once at startup and kept in memory. A pipeline run is a few
matrix operations over a few hundred columns, which is fast enough to feel
immediate, so the page posts JSON and gets a rendered result back rather than
needing any job machinery.

Nothing from the browser is ever evaluated as code. A posted pipeline is a list
of operator *names* resolved against the registry, with parameters coerced to
declared types — see ``twbt.pipeline``. The attack this avoids is obvious once
stated: a research page that accepted a Python expression would be remote code
execution with a nice stylesheet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import equity_panel, expr as ex, i18n, ops, pipeline as pl, xcosts, xsim
from . import charts


# What the page shows before you have built anything. A real factor rather than
# a syntax demonstration, so the first thing a visitor sees is a result with a
# verdict attached -- including the fact that it loses money after costs.
EXAMPLE = {
    "name": "12-1 momentum",
    "expression": ("GrossScale(LongShortBalance(Rank(\n"
                   "    InstSum(231, LogReturns(Lag(21, px)))\n"
                   ")))"),
}


@dataclass
class PanelState:
    """The loaded panel, built once and shared by every request."""

    mats: dict
    sources: dict
    universe: pd.DataFrame
    info: dict

    @classmethod
    def load(cls, fixture: bool = False, universe_size: int = 300) -> PanelState:
        mats = equity_panel.build(fixture=fixture)
        univ = equity_panel.liquid_universe(mats, n=universe_size)
        sources = pl.build_sources(
            px=mats["close"], vol=mats.get("volume"), amount=mats.get("turnover"),
            high=mats.get("high"), low=mats.get("low"),
            shares_out=mats.get("shares_out"))
        # `px` is the conventional name for the price matrix in this vocabulary,
        # and every example uses it; alias rather than rename so both work
        sources["px"] = sources["close"]
        return cls(mats=mats, sources=sources, universe=univ,
                   info=equity_panel.describe(mats, univ))


def _rate(value, default: float, cap: float = 0.5) -> float:
    """A cost rate from the form: a fraction, clamped, falling back to default.

    Clamped rather than validated because a typo in a cost box should give a
    silly-but-finite answer you can see on the chart, not a 400 that loses the
    rest of the form.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v) or v < 0:
        return default
    return min(v, cap)


def _fmt(v, nd=2, pct=False, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if pct:
        return f"{v*100:+.{nd}f}%" if sign else f"{v*100:.{nd}f}%"
    return f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"


def register(app, templates, fixture: bool = False) -> None:
    """Attach the builder routes to an existing app."""
    router = APIRouter()
    state: dict[str, PanelState | None] = {"panel": None}

    def panel() -> PanelState:
        if state["panel"] is None:
            state["panel"] = PanelState.load(fixture=fixture)
        return state["panel"]

    @router.get("/", response_class=HTMLResponse)
    @router.get("/build", response_class=HTMLResponse)   # kept so old links work
    def build_page(request: Request, lang: str = "en"):
        lang = i18n.pick(lang)
        try:
            p = panel()
            info, err = p.info, None
        except Exception as exc:                        # noqa: BLE001
            info, err = {}, str(exc)

        cat = ops.catalogue(max_matrix_args=1)
        every = ops.catalogue()
        for c in cat + every:
            c["hint"] = i18n.op_hint(c["name"], c["hint"], lang)

        return templates.TemplateResponse(request, "builder.html", {
            "lang": lang,
            "langs": i18n.LANGS,
            "wide": True,
            "T": lambda k: i18n.t(i18n.UI, k, lang),
            "CAP": lambda k: i18n.t(i18n.CAPTION, k, lang),
            "example": EXAMPLE,
            # Jinja has no dict comprehension, so the strings the browser needs
            # are assembled here rather than in the template
            "cfg": {
                "lang": lang,
                "example": EXAMPLE,
                "defaults": {
                    "brokerage": xcosts.BROKERAGE_TYPICAL * 1e4,
                    "tax_sell": xcosts.TAX_SELL * 1e4,
                    "half_spread": 10.0,
                    "borrow_annual": xcosts.BORROW_ANNUAL * 100,
                },
                "strings": {k: i18n.t(i18n.UI, k, lang) for k in (
                    "running", "done_in", "add_operator", "write_expr",
                    "valid_blocks", "valid_tree", "no_portfolio", "steps",
                    "round_trip")},
            },
            # the canvas draws a linear chain, so it only offers single-matrix
            # operators; the code editor gets the full registry
            "catalogue": cat,
            "all_ops": every,
            "sources": pl.SOURCES,
            "presets": pl.PRESETS,
            "info": info,
            "load_error": err,
            "cost": xcosts.TWEquityCost().describe(),
            "cost_rows": xcosts.TWEquityCost().breakdown().to_dict("records"),
            "defaults": {
                "brokerage": xcosts.BROKERAGE_TYPICAL * 1e4,
                "tax_sell": xcosts.TAX_SELL * 1e4,
                "half_spread": 10.0,
                "borrow_annual": xcosts.BORROW_ANNUAL * 100,
            },
            "fixture": fixture,
        })

    @router.post("/api/parse")
    async def parse_only(request: Request):
        """Validate an expression without running it, and flatten it if it is a chain.

        The editor calls this as you type. It is cheap -- parsing never touches
        the data -- so the page can tell you about a typo before you run anything.
        """
        try:
            body = await request.json()
            src = str(body.get("expression", ""))
        except Exception:                               # noqa: BLE001
            return JSONResponse({"error": "malformed request body"}, status_code=400)

        if not src.strip():
            return JSONResponse({"ok": False, "error": "the expression is empty"})
        try:
            node = ex.parse(src)
            # names and arity too: parsing alone would call Rnk(returns) valid
            ex.validate(node, set(panel().sources) if state["panel"] is not None
                        else set(pl.SOURCES) | {"px"})
        except ex.ExprError as exc:
            return JSONResponse({"ok": False, "error": str(exc), "pos": exc.pos})
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"ok": False, "error": str(exc)})

        chain = ex.to_chain(node)
        return JSONResponse({
            "ok": True,
            "normalised": ex.to_source(node),
            "chain": ({"source": chain.source, "steps": chain.steps}
                      if chain else None),
        })

    @router.post("/api/run")
    async def run(request: Request):
        t0 = time.time()
        try:
            body = await request.json()
        except Exception:                               # noqa: BLE001
            return JSONResponse({"error": "malformed request body"}, status_code=400)

        lang = i18n.pick(str(body.get("lang") or "en"))
        C = lambda k: i18n.t(i18n.CHART, k, lang)          # noqa: E731
        expression = str(body.get("expression") or "").strip()
        pipe = None
        if not expression:
            try:
                pipe = pl.Pipeline.from_json(body)
            except pl.PipelineError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

        try:
            p = panel()
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"panel unavailable: {exc}"},
                                status_code=503)

        opts = body.get("options") or {}
        try:
            exec_shift = int(opts.get("exec_shift", 1))
            # every component of the cost model is a free number, because the
            # right values depend on your broker and the names you trade, and a
            # fixed dropdown quietly asserts otherwise
            cost = xcosts.TWEquityCost(
                brokerage=_rate(opts.get("brokerage"), xcosts.BROKERAGE_TYPICAL),
                tax_sell=_rate(opts.get("tax_sell"), xcosts.TAX_SELL),
                half_spread=_rate(opts.get("half_spread"), 0.0010),
                borrow_annual=_rate(opts.get("borrow_annual"),
                                    xcosts.BORROW_ANNUAL, cap=1.0),
                day_trade=bool(opts.get("day_trade", False)))
            tradability = bool(opts.get("tradability", True))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": f"bad option: {exc}"}, status_code=400)

        try:
            if expression:
                sig = ex.run(expression, p.sources)
                shown = ex.to_source(ex.parse(expression))
                res = pl.simulate_signal(
                    sig, p.mats["close"], exec_shift=exec_shift, cost=cost,
                    enforce_tradability=tradability, universe=p.universe)
            else:
                sig, res = pl.run_pipeline(
                    pipe, p.sources, p.mats["close"], exec_shift=exec_shift,
                    cost=cost, enforce_tradability=tradability,
                    universe=p.universe)
                shown = pipe.expression()
        except (pl.PipelineError, ex.ExprError) as exc:
            return JSONResponse({"error": str(exc),
                                 "pos": getattr(exc, "pos", None)},
                                status_code=400)
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=400)

        s = res.summary()
        by_year = res.by_year()

        # The extra panels. Each is a re-run or a regrouping of what we already
        # have, and the whole set costs well under a second on this panel, so
        # they are computed every time rather than hidden behind a button.
        lags, bkt, boards = {}, {}, {}
        try:
            lags = xsim.lag_sweep(sig, p.mats["close"], cost=cost,
                                  enforce_tradability=tradability)
        except Exception:                               # noqa: BLE001
            pass
        try:
            bkt = xsim.bucket_pnl(sig, res.returns, exec_shift=exec_shift)
        except Exception:                               # noqa: BLE001
            pass
        try:
            if "board" in p.mats:
                boards = xsim.group_pnl(res.weights, res.returns, p.mats["board"])
        except Exception:                               # noqa: BLE001
            pass

        return JSONResponse({
            "expression": shown,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "verdict": [{"text": v["text"], "fail": v["fail"]}
                        for v in res.verdict_items(lang)],
            "stats": [
                [i18n.t(i18n.STAT, k, lang), v] for k, v in (
                    ("net Sharpe", _fmt(s["sharpe"], 2, sign=True)),
                    ("gross Sharpe", _fmt(s["gross_sharpe"], 2, sign=True)),
                    ("t-stat", _fmt(s["t_stat"], 2, sign=True)),
                    ("annualised return", _fmt(s["ann_return"], 2, pct=True, sign=True)),
                    ("max drawdown", _fmt(s["max_drawdown"], 2, pct=True)),
                    ("hit rate", _fmt(s["hit_rate"], 1, pct=True)),
                    ("turnover / day", _fmt(s["avg_turnover"], 1, pct=True)),
                    ("cost / gross edge", _fmt(s["cost_share"], 0, pct=True)),
                    ("names held", _fmt(s["avg_names"], 0)),
                    ("sessions", f"{s['days']:,}"),
                )
            ],
            "cost_bps": round(cost.round_trip_bps, 1),
            "by_year": [[int(r["year"]), f"{r['gross_ret']*100:+.2f}%",
                         f"{r['net_ret']*100:+.2f}%", f"{r['sharpe']:+.2f}",
                         f"{r['hit']*100:.0f}%"] for _, r in by_year.iterrows()],
            "charts": {
                "equity": charts.equity(
                    {C("l_gross"): res.gross.cumsum(), C("l_net"): res.net.cumsum()},
                    title=C("cum_pnl"), ylab=C("y_cum_return")),
                "decomp": charts.equity(
                    {C("l_total"): res.net.cumsum(),
                     C("l_long"): res.long_pnl.cumsum(),
                     C("l_short"): res.short_pnl.cumsum()},
                    title=C("long_short"), ylab=C("y_cum_return")),
                "by_year": charts.by_year(res.net, title=C("by_year"),
                                          ylab=C("y_cum_return")),
                "avg_path": charts.avg_path(
                    res.net, res.long_pnl, res.short_pnl, title=C("avg_path"),
                    labels=(C("l_total"), C("l_long"), C("l_short")),
                    ylab=C("y_cum_return"), xlab=C("x_session")),

                "lag_decay": charts.lag_decay(lags, title=C("lag_decay"),
                                              ylab=C("y_cum_return"),
                                              lag_label=C("l_lag")),
                "buckets": charts.buckets(bkt, title=C("buckets"),
                                          ylab=C("y_cum_return")),
                "drawdown": charts.drawdown(res.net, title=C("drawdown"),
                                            label=C("l_drawdown"),
                                            ylab=C("y_cum_return")),
                "boards": (charts.groups(boards, title=C("boards"),
                                         ylab=C("y_cum_return"))
                           if boards else charts.empty("—")),

                "cost": charts.equity(
                    {C("l_gross"): res.gross.cumsum(), C("l_net"): res.net.cumsum(),
                     C("l_cost"): res.cost.cumsum()},
                    title=C("cost"), ylab=C("y_cum_return")),
                "turnover": charts.series_panel(
                    C("l_turnover"), res.turnover, C("turnover"),
                    C("y_fraction"), window=20),
                "counts": charts.counts(res.long_count, res.short_count,
                                        title=C("counts"),
                                        labels=(C("l_long"), C("l_short")),
                                        ylab=C("y_names")),
                "limit_drag": (charts.equity(
                    {C("l_intended"): res.raw_gross.cumsum(),
                     C("l_tradable"): res.gross.cumsum()},
                    title=C("limit_drag"), ylab=C("y_cum_return"))
                    if res.raw_gross is not None else charts.empty("—")),

                "blocked": (charts.counts(
                    res.blocked.get("blocked_buy"), res.blocked.get("blocked_sell"),
                    title=C("blocked"), labels=("漲停", "跌停"), ylab=C("y_names"))
                    if res.blocked else charts.empty("—")),
                "untraded": (charts.series_panel(
                    C("l_untraded"), res.blocked["untraded_gmv"], C("untraded"),
                    C("y_fraction"), window=20)
                    if res.blocked else charts.empty("—")),
            },
        })

    app.include_router(router)
