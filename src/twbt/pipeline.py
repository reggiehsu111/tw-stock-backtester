"""A signal as a list of steps, so it can come from a web form safely.

The interactive builder posts a pipeline as JSON:

    {"source": "close",
     "steps": [{"op": "Returns"},
               {"op": "InstRank", "params": {"n": 20}},
               {"op": "Demean"},
               {"op": "Rank"},
               {"op": "LongShortBalance"}]}

``run_pipeline`` applies them in order to the chosen source matrix and hands the
result to the simulator. Nothing is ``eval``'d: every step is looked up in the
operator registry by name and its parameters are coerced to declared types, so a
pipeline from a browser can only ever reach the functions in that registry with
arguments of the right shape. A page that accepted a Python expression would be
a remote code execution hole wearing a research tool's clothes.

Three validation rules, checked before anything runs:

    * every ``op`` must exist in the registry;
    * no step may be a look-ahead operator;
    * the pipeline must end somewhere usable as weights, which in practice means
      it should finish with a portfolio step. This is a WARNING rather than an
      error, because the simulator normalises anyway and refusing would stop
      people exploring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ops, xcosts, xsim

# Source matrices the builder offers, by key. Every one is a wide matrix on the
# same index and columns, derived in equities_panel.build.
SOURCES = {
    "close": "adjusted close price",
    "returns": "daily return",
    "volume": "shares traded",
    "turnover": "traded value in NT$",
    "amihud": "illiquidity: |return| / turnover",
    "high_low": "daily range as a fraction of close",
    "shares_out": "shares outstanding (TPEX names only)",
}

MAX_STEPS = 24


class PipelineError(ValueError):
    """A pipeline that cannot be run, with a reason fit to show a user."""


@dataclass
class Step:
    op: str
    params: dict = field(default_factory=dict)

    def label(self) -> str:
        if not self.params:
            return self.op
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.op}({inner})"


@dataclass
class Pipeline:
    source: str = "returns"
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: dict) -> Pipeline:
        src = str(payload.get("source", "returns"))
        if src not in SOURCES:
            raise PipelineError(f"unknown source {src!r}; "
                                f"choose from {sorted(SOURCES)}")
        raw = payload.get("steps") or []
        if not isinstance(raw, list):
            raise PipelineError("steps must be a list")
        if len(raw) > MAX_STEPS:
            raise PipelineError(f"too many steps ({len(raw)}); max {MAX_STEPS}")

        steps = []
        for i, s in enumerate(raw, 1):
            if not isinstance(s, dict) or "op" not in s:
                raise PipelineError(f"step {i} must be an object with an 'op'")
            name = str(s["op"])
            if name not in ops.ALL_OPS:
                raise PipelineError(f"step {i}: unknown operator {name!r}")
            if ops.ALL_OPS[name].lookahead:
                raise PipelineError(
                    f"step {i}: {name} looks into the future and cannot be part "
                    "of a signal.")
            params = s.get("params") or {}
            if not isinstance(params, dict):
                raise PipelineError(f"step {i}: params must be an object")
            steps.append(Step(name, params))
        return cls(source=src, steps=steps)

    def expression(self) -> str:
        """The pipeline written as the nested call it is equivalent to.

        Shown in the UI so what you built is also something you could paste into
        a script -- the page is a way to write code, not a replacement for it.
        """
        expr = {"close": "px", "returns": "Returns(px)"}.get(
            self.source, f"{self.source}")
        for s in self.steps:
            args = ", ".join(str(v) for v in s.params.values())
            expr = f"{s.op}({args}, {expr})" if args else f"{s.op}({expr})"
        return expr

    def warnings(self) -> list[str]:
        out = []
        names = [s.op for s in self.steps]
        if not names:
            out.append(("empty", {}))
            return out
        portfolio = {n for n, o in ops.ALL_OPS.items() if o.group == "portfolio"}
        if not set(names) & portfolio:
            out.append(("no_portfolio", {}))
        # LongShortBalance does not centre. Fed a non-negative matrix -- which is
        # exactly what Rank produces -- it yields a long-only book with net +0.5,
        # and nothing else in the pipeline complains.
        nonneg = {"Rank", "GroupRank", "InstRank", "Abs", "InstCount", "Sqrt"}
        if "LongShortBalance" in names:
            i = names.index("LongShortBalance")
            before = names[:i]
            if before and before[-1] in nonneg and "Demean" not in before:
                out.append(("lsb_uncentred", {"op": before[-1]}))
        if names.count("GrossScale") > 1:
            out.append(("double_gross", {}))
        return out


def build_sources(px: pd.DataFrame, vol: pd.DataFrame | None = None,
                  amount: pd.DataFrame | None = None,
                  high: pd.DataFrame | None = None,
                  low: pd.DataFrame | None = None,
                  shares_out: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Assemble the source matrices the builder can start from."""
    out: dict[str, pd.DataFrame] = {"close": px, "returns": px.pct_change()}
    if vol is not None:
        out["volume"] = vol
    if amount is not None:
        out["turnover"] = amount
        # Amihud illiquidity: how much the price moves per dollar traded. A
        # genuine, widely used factor and a good default thing to rank on.
        out["amihud"] = (px.pct_change().abs() / amount.replace(0, np.nan)) * 1e9
    if high is not None and low is not None:
        out["high_low"] = (high - low) / px.where(px > 0)
    if shares_out is not None:
        out["shares_out"] = shares_out
    return out


def simulate_signal(sig: pd.DataFrame, px: pd.DataFrame, *,
                    exec_shift: int = 1,
                    cost: xcosts.TWEquityCost | None = None,
                    enforce_tradability: bool = True,
                    universe: pd.DataFrame | None = None) -> xsim.XSimResult:
    """Mask, validate and simulate a signal matrix.

    Shared by the block canvas and the expression editor so the two cannot drift
    apart on universe masking or on what counts as an unusable signal.
    """
    sig = sig.replace([np.inf, -np.inf], np.nan)
    if universe is not None:
        sig = sig.where(universe.reindex_like(sig).fillna(False))
    if sig.notna().sum().sum() == 0:
        raise PipelineError(
            "The pipeline produced no usable values. A long trailing window on a "
            "short sample is the usual cause.")
    return xsim.simulate(sig, px, exec_shift=exec_shift, cost=cost,
                         enforce_tradability=enforce_tradability)


def run_pipeline(pipe: Pipeline, sources: dict[str, pd.DataFrame],
                 px: pd.DataFrame, *,
                 exec_shift: int = 1,
                 cost: xcosts.TWEquityCost | None = None,
                 enforce_tradability: bool = True,
                 universe: pd.DataFrame | None = None) -> tuple[pd.DataFrame, xsim.XSimResult]:
    """Apply the steps, then simulate. Returns (signal, result).

    ``universe`` is an optional boolean mask of tradable names per date. It is
    applied to the *signal*, before normalisation, so a name outside the universe
    contributes nothing rather than being dropped after the book is scaled.
    """
    if pipe.source not in sources:
        raise PipelineError(
            f"source {pipe.source!r} is not available in this panel; "
            f"have {sorted(sources)}")

    mx = sources[pipe.source]
    for i, step in enumerate(pipe.steps, 1):
        try:
            mx = ops.apply_op(step.op, mx, step.params)
        except Exception as exc:                        # noqa: BLE001
            raise PipelineError(f"step {i} ({step.label()}): {exc}") from exc
        if not isinstance(mx, pd.DataFrame):
            raise PipelineError(f"step {i} ({step.label()}) did not return a matrix")

    res = simulate_signal(mx, px, exec_shift=exec_shift, cost=cost,
                          enforce_tradability=enforce_tradability,
                          universe=universe)
    res.notes = pipe.warnings() + res.notes
    sig = mx.replace([np.inf, -np.inf], np.nan)
    if universe is not None:
        sig = sig.where(universe.reindex_like(sig).fillna(False))
    return sig, res


# Pipelines offered in the UI as starting points. Each is a real, recognisable
# factor rather than a demonstration of the syntax, and none of them is expected
# to survive 62 bps of round-trip cost -- which is the lesson.
PRESETS: dict[str, dict] = {
    "short-term reversal": {
        "source": "returns",
        "steps": [{"op": "InstSum", "params": {"n": 5}}, {"op": "Neg"},
                  {"op": "Rank"}, {"op": "Demean"}, {"op": "LongShortBalance"},
                  {"op": "GrossScale"}],
        "note": "Buy last week's losers. The best-known cross-sectional effect "
                "in equities, and the most thoroughly killed by turnover.",
    },
    "12-1 momentum": {
        "source": "close",
        "steps": [{"op": "Lag", "params": {"n": 21}},
                  {"op": "LogReturns"}, {"op": "InstSum", "params": {"n": 231}},
                  {"op": "Rank"}, {"op": "Demean"}, {"op": "LongShortBalance"},
                  {"op": "GrossScale"}],
        "note": "Twelve-month return skipping the most recent month, which is "
                "skipped precisely because short-term reversal lives there.",
    },
    "low volatility": {
        "source": "returns",
        "steps": [{"op": "InstStdev", "params": {"n": 60}}, {"op": "Neg"},
                  {"op": "Rank"}, {"op": "Demean"}, {"op": "LongShortBalance"},
                  {"op": "GrossScale"}],
        "note": "Long the calm names, short the wild ones. Low turnover, which "
                "in Taiwan matters more than the raw edge.",
    },
    "Amihud illiquidity": {
        "source": "amihud",
        "steps": [{"op": "InstMean", "params": {"n": 20}}, {"op": "Rank"},
                  {"op": "Demean"}, {"op": "LongShortBalance"},
                  {"op": "GrossScale"}],
        "note": "Long illiquid names. Note the irony: the premium is paid for "
                "illiquidity, and the cost model charges you for trading it.",
    },
    "concentrated reversal": {
        "source": "returns",
        "steps": [{"op": "InstSum", "params": {"n": 5}}, {"op": "Neg"},
                  {"op": "Rank"}, {"op": "Demean"}, {"op": "LongShortBalance"},
                  {"op": "TopN", "params": {"n": 25}},
                  {"op": "GrossScale"}],
        "note": "The same reversal in 50 names rather than 1,800. Higher mean, "
                "much higher variance, same turnover problem.",
    },
}
