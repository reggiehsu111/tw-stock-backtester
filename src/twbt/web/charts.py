"""Inline SVG charts, with no plotting dependency.

Charts are generated server-side as SVG strings and embedded directly in the page.
No matplotlib, no CDN, no JavaScript charting library: the tearsheet renders
identically on a machine with no network, which matters because a reviewer running
this for the first time should not need anything beyond `pip install -e .`.

Colours come from CSS custom properties so the page's light and dark themes both
work without regenerating the SVG.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd

W, H = 760, 320
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 16, 34

# A categorical ramp that stays distinguishable in both themes and does not rely on
# red/green, which carry an unwanted meaning in a P&L chart.
SERIES_COLOURS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#eeca3b",
                  "#72b7b2", "#ff9da6", "#9d755d", "#bab0ac", "#e45756"]


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


@dataclass
class Scale:
    lo: float
    hi: float
    px0: float
    px1: float

    def __call__(self, v: float) -> float:
        if self.hi == self.lo:
            return (self.px0 + self.px1) / 2
        return self.px0 + (v - self.lo) / (self.hi - self.lo) * (self.px1 - self.px0)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(abs(raw)))
    step = min([m * mag for m in (1, 2, 2.5, 5, 10)],
               key=lambda s: abs(s - raw))
    start = np.ceil(lo / step) * step
    return list(np.arange(start, hi + step / 2, step))


def _tick_decimals(ticks: list[float]) -> int:
    """Decimal places that keep adjacent ticks distinguishable.

    Formatting with a fixed ``.0f`` is fine for a P&L axis in the thousands and
    useless for a t-stat axis spanning 1.01 to 1.03, where every label rounds to
    the same number and the axis reads "1, 1, 1, 0, 0, -0".
    """
    vals = [t for t in ticks if np.isfinite(t)]
    if len(vals) < 2:
        return 0
    step = min(abs(b - a) for a, b in zip(vals, vals[1:]) if b != a) \
        if any(b != a for a, b in zip(vals, vals[1:])) else 0
    if step <= 0:
        return 0
    return int(min(4, max(0, np.ceil(-np.log10(step)) + 1)))


def _frame(x: Scale, y: Scale, xlab: str, ylab: str,
           xticks: list, yticks: list, xfmt=None) -> str:
    """Axes, gridlines and labels. Shared by every chart below."""
    out = []
    nd = _tick_decimals(list(yticks))
    for t in yticks:
        py = y(t)
        # a tick that rounds to zero prints as "-0.000" without this
        t = 0.0 if abs(t) < 10 ** -(nd + 3) else t
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{py:.1f}" '
                   f'x2="{W - PAD_R}" y2="{py:.1f}"/>')
        out.append(f'<text class="tick ytick" x="{PAD_L - 8}" y="{py + 3.5:.1f}">'
                   f'{t:,.{nd}f}</text>')
    for t in xticks:
        px = x(t)
        label = xfmt(t) if xfmt else f"{t:g}"
        out.append(f'<line class="grid" x1="{px:.1f}" y1="{PAD_T}" '
                   f'x2="{px:.1f}" y2="{H - PAD_B}"/>')
        out.append(f'<text class="tick xtick" x="{px:.1f}" y="{H - PAD_B + 16}">'
                   f'{_esc(label)}</text>')
    out.append(f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" '
               f'x2="{W - PAD_R}" y2="{H - PAD_B}"/>')
    out.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" '
               f'x2="{PAD_L}" y2="{H - PAD_B}"/>')
    if ylab:
        out.append(f'<text class="axlab" transform="translate(13,{H/2}) rotate(-90)">'
                   f'{_esc(ylab)}</text>')
    if xlab:
        out.append(f'<text class="axlab" x="{(PAD_L + W - PAD_R)/2}" y="{H - 4}">'
                   f'{_esc(xlab)}</text>')
    return "".join(out)


def _path(xs, ys, x: Scale, y: Scale) -> str:
    pts = [f"{x(a):.2f},{y(b):.2f}" for a, b in zip(xs, ys)
           if np.isfinite(a) and np.isfinite(b)]
    return "M" + "L".join(pts) if pts else ""


def _svg(body: str, title: str, desc: str = "") -> str:
    return (f'<svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{_esc(title)}" preserveAspectRatio="xMidYMid meet">'
            f'<title>{_esc(title)}</title>'
            + (f'<desc>{_esc(desc)}</desc>' if desc else "")
            + body + '</svg>')


def empty(message: str) -> str:
    """A placeholder that says why there is nothing to draw."""
    return _svg(
        f'<text class="empty" x="{W/2}" y="{H/2}" text-anchor="middle">'
        f'{_esc(message)}</text>', message)


# --------------------------------------------------------------------------- #
# the three charts the research workflow uses
# --------------------------------------------------------------------------- #

def equity(series: dict[str, pd.Series], title: str = "Cumulative P&L",
           ylab: str = "cumulative bps") -> str:
    """Cumulative P&L lines against calendar time.

    Takes a dict so gross and net can be drawn together — the gap between them is
    the cost, which is the single most useful thing to be able to see at a glance.
    """
    series = {k: pd.Series(v).dropna() for k, v in series.items()}
    series = {k: v for k, v in series.items() if len(v) > 1}
    if not series:
        return empty("no P&L to plot")

    all_x = np.concatenate([v.index.values.astype("datetime64[D]").astype(int)
                            for v in series.values()])
    all_y = np.concatenate([v.to_numpy(dtype=float) for v in series.values()])
    lo_y, hi_y = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = (hi_y - lo_y) * 0.08 or 1.0
    x = Scale(float(all_x.min()), float(all_x.max()), PAD_L, W - PAD_R)
    y = Scale(lo_y - pad, hi_y + pad, H - PAD_B, PAD_T)

    yticks = _nice_ticks(lo_y - pad, hi_y + pad)
    xt = np.linspace(all_x.min(), all_x.max(), 6)
    xfmt = lambda v: pd.Timestamp(int(v), unit="D").strftime("%Y-%m")  # noqa: E731

    body = [_frame(x, y, "", ylab, list(xt), yticks, xfmt)]
    if lo_y < 0 < hi_y:
        body.append(f'<line class="zero" x1="{PAD_L}" y1="{y(0):.1f}" '
                    f'x2="{W - PAD_R}" y2="{y(0):.1f}"/>')

    legend = []
    for i, (name, s) in enumerate(series.items()):
        c = SERIES_COLOURS[i % len(SERIES_COLOURS)]
        xs = s.index.values.astype("datetime64[D]").astype(int)
        body.append(f'<path class="line" stroke="{c}" d="'
                    f'{_path(xs, s.to_numpy(dtype=float), x, y)}"/>')
        legend.append((name, c, float(s.iloc[-1])))

    for i, (name, c, last) in enumerate(legend):
        ly = PAD_T + 14 + i * 16
        body.append(f'<rect class="swatch" x="{W - PAD_R - 190}" y="{ly - 8}" '
                    f'fill="{c}"/>')
        body.append(f'<text class="legend" x="{W - PAD_R - 176}" y="{ly}">'
                    f'{_esc(name)} <tspan class="num">{last:+,.0f}</tspan></text>')

    return _svg("".join(body), title)


def by_year(daily_net: pd.Series, title: str = "Cumulative P&L by year") -> str:
    """One line per calendar year, each labelled with that year's Sharpe.

    This is the view that separates "it works" from "2022 happened". A strategy
    whose equity curve looks like a staircase usually has one year carrying it, and
    that is immediately visible here and invisible in the cumulative chart.

    The heavy dashed line is AvgPath: the average cumulative path across years,
    i.e. what a typical year looked like.
    """
    s = pd.Series(daily_net).dropna()
    if len(s) < 20 or not isinstance(s.index, pd.DatetimeIndex):
        return empty("not enough data to split by year")

    paths: dict[int, pd.Series] = {}
    sharpes: dict[int, float] = {}
    for yr, g in s.groupby(s.index.year):
        if len(g) < 5:
            continue
        paths[int(yr)] = g.cumsum().reset_index(drop=True)
        sd = g.std(ddof=1)
        sharpes[int(yr)] = (g.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    if not paths:
        return empty("not enough data to split by year")

    n_max = max(len(v) for v in paths.values())
    all_y = np.concatenate([v.to_numpy() for v in paths.values()])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = (hi - lo) * 0.08 or 1.0
    x = Scale(0, n_max - 1, PAD_L, W - PAD_R)
    y = Scale(lo - pad, hi + pad, H - PAD_B, PAD_T)

    body = [_frame(x, y, "session of year", "cumulative bps",
                   list(np.linspace(0, n_max - 1, 5)),
                   _nice_ticks(lo - pad, hi + pad))]
    if lo < 0 < hi:
        body.append(f'<line class="zero" x1="{PAD_L}" y1="{y(0):.1f}" '
                    f'x2="{W - PAD_R}" y2="{y(0):.1f}"/>')

    for i, (yr, path) in enumerate(sorted(paths.items())):
        c = SERIES_COLOURS[i % len(SERIES_COLOURS)]
        body.append(f'<path class="line thin" stroke="{c}" d="'
                    f'{_path(np.arange(len(path)), path.to_numpy(), x, y)}"/>')
        ly = PAD_T + 14 + i * 15
        body.append(f'<rect class="swatch" x="{W - PAD_R - 150}" y="{ly - 8}" '
                    f'fill="{c}"/>')
        sh = sharpes.get(yr, np.nan)
        sh_txt = f"SR {sh:+.2f}" if np.isfinite(sh) else "SR n/a"
        body.append(f'<text class="legend" x="{W - PAD_R - 136}" y="{ly}">'
                    f'{yr} <tspan class="num">{sh_txt}</tspan></text>')

    # AvgPath: mean across years at each session index
    avg = (pd.DataFrame({k: v for k, v in paths.items()})
           .mean(axis=1, skipna=True))
    body.append(f'<path class="line avg" d="'
                f'{_path(np.arange(len(avg)), avg.to_numpy(), x, y)}"/>')
    ly = PAD_T + 14 + len(paths) * 15
    body.append(f'<text class="legend avglab" x="{W - PAD_R - 150}" y="{ly}">'
                f'AvgPath <tspan class="num">{avg.iloc[-1]:+,.0f}</tspan></text>')

    return _svg("".join(body), title)


def decomposition(daily_net: pd.Series, position: pd.Series,
                  title: str = "Long / short decomposition") -> str:
    """Total, long-only and short-only cumulative P&L.

    A strategy whose entire result comes from one side is a different object from
    one that works both ways — usually it is a market-direction bet wearing a
    signal's clothes. Splitting it is the cheapest way to find that out.
    """
    s = pd.Series(daily_net).dropna()
    pos = pd.Series(position).reindex(s.index).fillna(0.0)
    if len(s) < 5:
        return empty("not enough data to decompose")

    parts = {
        "total": s.cumsum(),
        "long": s.where(pos > 0, 0.0).cumsum(),
        "short": s.where(pos < 0, 0.0).cumsum(),
    }
    if (pos < 0).sum() == 0:
        parts.pop("short")
    if (pos > 0).sum() == 0:
        parts.pop("long")
    return equity(parts, title=title)


def cost_curve(sens: pd.DataFrame, title: str = "Cost sensitivity") -> str:
    """Net t-stat against the spread assumption, with the zero line marked.

    Reading this chart: find where the line crosses zero. If that happens at or
    below 1.5x, the realistic cost estimate and a mildly pessimistic one disagree
    about whether the strategy exists, and it does not.
    """
    if sens is None or len(sens) == 0:
        return empty("no cost sensitivity available")
    df = sens.dropna(subset=["net_t"])
    if len(df) < 2:
        return empty("no cost sensitivity available")

    xs = df["spread_factor"].to_numpy(dtype=float)
    ys = df["net_t"].to_numpy(dtype=float)
    lo, hi = float(min(ys.min(), 0)), float(max(ys.max(), 0))
    pad = (hi - lo) * 0.15 or 1.0
    x = Scale(xs.min(), xs.max(), PAD_L, W - PAD_R)
    y = Scale(lo - pad, hi + pad, H - PAD_B, PAD_T)

    body = [_frame(x, y, "spread assumption (x the estimate)", "net t-stat",
                   list(xs), _nice_ticks(lo - pad, hi + pad))]
    body.append(f'<line class="zero" x1="{PAD_L}" y1="{y(0):.1f}" '
                f'x2="{W - PAD_R}" y2="{y(0):.1f}"/>')
    # the 1.5x line: the point past which "it depends on the cost assumption"
    if xs.min() <= 1.5 <= xs.max():
        body.append(f'<line class="marker" x1="{x(1.5):.1f}" y1="{PAD_T}" '
                    f'x2="{x(1.5):.1f}" y2="{H - PAD_B}"/>')
        body.append(f'<text class="marklab" x="{x(1.5) + 5:.1f}" y="{PAD_T + 11}">'
                    f'1.5x</text>')
    body.append(f'<path class="line" stroke="{SERIES_COLOURS[0]}" '
                f'd="{_path(xs, ys, x, y)}"/>')
    for a, b in zip(xs, ys):
        body.append(f'<circle class="dot" cx="{x(a):.1f}" cy="{y(b):.1f}" r="3.2" '
                    f'fill="{SERIES_COLOURS[0]}"/>')
    return _svg("".join(body), title)
