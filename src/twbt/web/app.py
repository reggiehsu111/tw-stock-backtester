"""The tearsheet server.

One page per saved backtest, plus an index. It reads the pickles written by
``twbt run`` rather than re-running anything, so the page you look at is exactly
the result that was printed to your terminal — a server that re-ran the backtest on
each request could quietly disagree with the CLI.

Charts are server-rendered SVG with no JavaScript and no CDN, so this works with no
network connection.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from . import builder, charts, futures

RESULTS_DIR = Path(__file__).resolve().parents[3] / "data" / "results"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC_DIR = Path(__file__).parent / "static"


def _load_all() -> dict:
    out = {}
    if not RESULTS_DIR.exists():
        return out
    for p in sorted(RESULTS_DIR.glob("*.pkl")):
        try:
            with p.open("rb") as fh:
                out[p.stem] = pickle.load(fh)
        except Exception:                                  # noqa: BLE001
            continue
    return out


def _fmt(v, nd: int = 2, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"
    return s


def _table(df: pd.DataFrame, numeric_nd: int = 2) -> list:
    """A DataFrame as (header, rows) with numbers already formatted."""
    if df is None or len(df) == 0:
        return []
    rows = []
    for rec in df.to_dict("records"):
        rows.append([(_fmt(v, numeric_nd) if isinstance(v, (int, float, np.number))
                      and not isinstance(v, bool) else str(v))
                     for v in rec.values()])
    return [list(df.columns), rows]


def create_app(fixture: bool = False) -> FastAPI:
    app = FastAPI(title="twbt tearsheet", docs_url=None, redoc_url=None)
    # served locally rather than from a CDN, so the page works with no network
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/results", response_class=HTMLResponse)
    def index(request: Request):
        results = _load_all()
        cards = []
        for name, r in results.items():
            m = r.main.net_perf
            fails = [v for v in r.verdict() if v.startswith("FAILS")]
            cards.append({
                "name": name,
                "trades": r.main.n_trades,
                "net_mean": _fmt(m.mean, 2, sign=True),
                "net_t": _fmt(m.t, 2, sign=True),
                "sharpe": _fmt(m.sharpe, 2, sign=True),
                "fails": len(fails),
            })
        return TEMPLATES.TemplateResponse(
            request, "index.html",
            {"cards": cards, "fixture": fixture, "lang": "en", "langs": None})

    @app.get("/s/{name}", response_class=HTMLResponse)
    def tearsheet(request: Request, name: str):
        results = _load_all()
        if name not in results:
            return HTMLResponse(
                f"<p>No saved result called <code>{name}</code>. "
                f"Run <code>twbt run --strategy {name}</code> first.</p>",
                status_code=404)
        r = results[name]
        m = r.main

        verdict = [{"text": v, "fail": v.startswith("FAILS")} for v in r.verdict()]

        wf = r.walk_forward
        wf_table = _table(wf[["year", "params", "n", "net_mean_bps", "net_t"]]) \
            if wf is not None and len(wf) else []

        stats_rows = [
            ("trades", f"{m.n_trades:,}"),
            ("gross mean", _fmt(m.gross_perf.mean, 2, True) + " bps"),
            ("net mean", _fmt(m.net_perf.mean, 2, True) + " bps"),
            ("net t", _fmt(m.net_perf.t, 2, True)),
            ("net Sharpe", _fmt(m.net_perf.sharpe, 2, True)),
            ("hit rate", _fmt(m.net_perf.hit_rate * 100, 1) + "%"),
            ("avg cost", _fmt(m.avg_cost_bps, 2) + " bps/trade"),
            ("breakeven", _fmt(m.breakeven_bps, 2, True) + " bps"),
            ("max drawdown", _fmt(m.net_perf.max_drawdown, 0) + " bps"),
        ]
        if np.isfinite(r.noise_bar_95):
            stats_rows.append(("noise bar (95%)",
                               _fmt(r.noise_bar_95, 2)
                               + f" for {r.n_variants_tested} variants"))

        return TEMPLATES.TemplateResponse(request, "tearsheet.html", {
            "lang": "en", "langs": None,
            "name": name,
            "verdict": verdict,
            "stats_rows": stats_rows,
            "consistency": _table(r.consistency()),
            "sensitivity": _table(m.sensitivity),
            "wf_table": wf_table,
            "chart_equity": charts.equity(
                {"gross": m.daily_gross.cumsum(), "net": m.daily_net.cumsum()},
                title=f"{name}: cumulative P&L"),
            "chart_by_year": charts.by_year(m.daily_net),
            "chart_decomp": charts.decomposition(m.daily_net, m.position),
            "chart_cost": charts.cost_curve(m.sensitivity),
        })

    # registered last so its "/" is added after the named routes above; FastAPI
    # matches in definition order and these paths do not overlap, but keeping the
    # catch-all-looking route last avoids any future shadowing
    futures.register(app, TEMPLATES, fixture=fixture)
    builder.register(app, TEMPLATES, fixture=fixture)
    return app
