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

from .. import equity_panel, ops, pipeline as pl, xcosts, xsim
from . import charts


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
        return cls(mats=mats, sources=sources, universe=univ,
                   info=equity_panel.describe(mats, univ))


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

    @router.get("/build", response_class=HTMLResponse)
    def build_page(request: Request):
        try:
            p = panel()
            info, err = p.info, None
        except Exception as exc:                        # noqa: BLE001
            info, err = {}, str(exc)
        return templates.TemplateResponse(request, "builder.html", {
            "catalogue": ops.catalogue(),
            "sources": pl.SOURCES,
            "presets": pl.PRESETS,
            "info": info,
            "load_error": err,
            "cost": xcosts.TWEquityCost().describe(),
            "cost_rows": xcosts.TWEquityCost().breakdown().to_dict("records"),
            "fixture": fixture,
        })

    @router.post("/api/run")
    async def run(request: Request):
        t0 = time.time()
        try:
            body = await request.json()
        except Exception:                               # noqa: BLE001
            return JSONResponse({"error": "malformed request body"}, status_code=400)

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
            cost = xcosts.TWEquityCost(
                brokerage=float(opts.get("brokerage", xcosts.BROKERAGE_TYPICAL)),
                half_spread=float(opts.get("half_spread", 0.0010)),
                day_trade=bool(opts.get("day_trade", False)))
            tradability = bool(opts.get("tradability", True))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": f"bad option: {exc}"}, status_code=400)

        try:
            sig, res = pl.run_pipeline(
                pipe, p.sources, p.mats["close"], exec_shift=exec_shift,
                cost=cost, enforce_tradability=tradability, universe=p.universe)
        except pl.PipelineError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:                        # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=400)

        s = res.summary()
        by_year = res.by_year()

        return JSONResponse({
            "expression": pipe.expression(),
            "elapsed_ms": int((time.time() - t0) * 1000),
            "verdict": [{"text": v, "fail": v.startswith(("FAILS", "SAME-CLOSE"))}
                        for v in res.verdict()],
            "stats": [
                ["net Sharpe", _fmt(s["sharpe"], 2, sign=True)],
                ["gross Sharpe", _fmt(s["gross_sharpe"], 2, sign=True)],
                ["t-stat", _fmt(s["t_stat"], 2, sign=True)],
                ["annualised return", _fmt(s["ann_return"], 2, pct=True, sign=True)],
                ["max drawdown", _fmt(s["max_drawdown"], 2, pct=True)],
                ["hit rate", _fmt(s["hit_rate"], 1, pct=True)],
                ["turnover / day", _fmt(s["avg_turnover"], 1, pct=True)],
                ["cost / gross edge", _fmt(s["cost_share"], 0, pct=True)],
                ["names held", _fmt(s["avg_names"], 0)],
                ["sessions", f"{s['days']:,}"],
            ],
            "by_year": [[int(r["year"]), f"{r['gross_ret']*100:+.2f}%",
                         f"{r['net_ret']*100:+.2f}%", f"{r['sharpe']:+.2f}",
                         f"{r['hit']*100:.0f}%"] for _, r in by_year.iterrows()],
            "charts": {
                "equity": charts.equity(
                    {"gross": res.gross.cumsum(), "net": res.net.cumsum()},
                    title="Cumulative P&L", ylab="cumulative return"),
                "by_year": charts.by_year(res.net),
                "decomp": charts.equity(
                    {"total": res.net.cumsum(),
                     "long": res.long_pnl.cumsum(),
                     "short": res.short_pnl.cumsum()},
                    title="Long / short decomposition",
                    ylab="cumulative return"),
                "turnover": charts.equity(
                    {"cumulative cost": res.cost.cumsum()},
                    title="Cumulative cost", ylab="cumulative cost"),
            },
        })

    app.include_router(router)
