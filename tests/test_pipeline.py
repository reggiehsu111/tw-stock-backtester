"""Tests for the pipeline the web builder posts.

Two groups. The first is validation: a pipeline arriving from a browser is
untrusted input, and the only functions it may reach are the ones in the
operator registry, with arguments coerced to declared types. The second is that
the thing it builds is the thing the expression string claims it built.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import ops, pipeline as pl, xcosts

IDX = pd.bdate_range("2023-01-02", periods=400)
COLS = [f"{i:04d}" for i in range(1101, 1181)]


def _px(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, (len(IDX), len(COLS))), axis=0)),
        index=IDX, columns=COLS)


@pytest.fixture
def env():
    px = _px()
    amount = pd.DataFrame(1e7, index=IDX, columns=COLS)
    sources = pl.build_sources(px=px, amount=amount,
                               high=px * 1.01, low=px * 0.99)
    return px, sources


# --------------------------------------------------------------------------- #
# untrusted input
# --------------------------------------------------------------------------- #

def test_only_registered_operators_can_be_reached():
    """The point of the registry: a posted name is looked up, never evaluated."""
    for payload in (
        {"steps": [{"op": "__import__"}]},
        {"steps": [{"op": "eval"}]},
        {"steps": [{"op": "os.system"}]},
        {"steps": [{"op": "Rank.__class__"}]},
    ):
        with pytest.raises(pl.PipelineError, match="unknown operator"):
            pl.Pipeline.from_json(payload)


def test_lookahead_operators_are_rejected_at_parse_time():
    with pytest.raises(pl.PipelineError, match="future"):
        pl.Pipeline.from_json({"steps": [{"op": "Lead", "params": {"n": 1}}]})


def test_unknown_source_is_rejected():
    with pytest.raises(pl.PipelineError, match="unknown source"):
        pl.Pipeline.from_json({"source": "/etc/passwd", "steps": []})


def test_malformed_payloads_are_rejected_with_a_usable_message():
    for payload, match in (
        ({"steps": "Rank"}, "must be a list"),
        ({"steps": ["Rank"]}, "must be an object"),
        ({"steps": [{"params": {}}]}, "must be an object"),
        ({"steps": [{"op": "Lag", "params": 5}]}, "params must be an object"),
    ):
        with pytest.raises(pl.PipelineError, match=match):
            pl.Pipeline.from_json(payload)


def test_pipeline_length_is_bounded():
    long = {"steps": [{"op": "Abs"}] * (pl.MAX_STEPS + 1)}
    with pytest.raises(pl.PipelineError, match="too many steps"):
        pl.Pipeline.from_json(long)


def test_bad_parameter_types_fail_cleanly_not_with_a_traceback(env):
    px, sources = env
    pipe = pl.Pipeline.from_json(
        {"source": "returns", "steps": [{"op": "Lag", "params": {"n": "banana"}}]})
    with pytest.raises(pl.PipelineError, match="step 1"):
        pl.run_pipeline(pipe, sources, px)


# --------------------------------------------------------------------------- #
# what it builds
# --------------------------------------------------------------------------- #

def test_expression_matches_what_is_actually_computed(env):
    px, sources = env
    pipe = pl.Pipeline.from_json({
        "source": "returns",
        "steps": [{"op": "InstRank", "params": {"n": 20}},
                  {"op": "Demean"}, {"op": "LongShortBalance"}]})
    assert pipe.expression() == \
        "LongShortBalance(Demean(InstRank(20, Returns(px))))"

    sig, _ = pl.run_pipeline(pipe, sources, px)
    manual = ops.LongShortBalance(ops.Demean(ops.InstRank(20, sources["returns"])))
    pd.testing.assert_frame_equal(sig, manual.replace([np.inf, -np.inf], np.nan),
                                  check_dtype=False)


def test_steps_apply_in_order(env):
    px, sources = env
    a = pl.Pipeline.from_json({"source": "returns",
                               "steps": [{"op": "Rank"}, {"op": "Neg"}]})
    b = pl.Pipeline.from_json({"source": "returns",
                               "steps": [{"op": "Neg"}, {"op": "Rank"}]})
    sa, _ = pl.run_pipeline(a, sources, px)
    sb, _ = pl.run_pipeline(b, sources, px)
    assert not sa.equals(sb)


def test_universe_mask_is_applied_before_normalisation(env):
    """A name outside the universe must contribute nothing, not be dropped late."""
    px, sources = env
    univ = pd.DataFrame(True, index=IDX, columns=COLS)
    univ.iloc[:, :40] = False
    pipe = pl.Pipeline.from_json({
        "source": "returns",
        "steps": [{"op": "Rank"}, {"op": "LongShortBalance"}, {"op": "GrossScale"}]})
    sig, res = pl.run_pipeline(pipe, sources, px, universe=univ)
    assert sig.iloc[:, :40].notna().sum().sum() == 0
    assert res.weights.iloc[:, :40].abs().sum().sum() == 0
    # and the remaining book is still fully scaled
    g = res.weights.abs().sum(axis=1)
    assert g.max() <= 1.0 + 1e-9


def test_an_all_nan_result_is_an_error_with_an_explanation(env):
    px, sources = env
    pipe = pl.Pipeline.from_json({
        "source": "returns",
        "steps": [{"op": "InstMean", "params": {"n": 100000}}]})
    with pytest.raises(pl.PipelineError, match="no usable values"):
        pl.run_pipeline(pipe, sources, px)


def test_empty_pipeline_runs_but_warns(env):
    px, sources = env
    pipe = pl.Pipeline.from_json({"source": "returns", "steps": []})
    _, res = pl.run_pipeline(pipe, sources, px)
    assert any("Empty pipeline" in n for n in res.notes)


def test_missing_portfolio_step_warns(env):
    px, sources = env
    pipe = pl.Pipeline.from_json({"source": "returns", "steps": [{"op": "Rank"}]})
    assert any("No portfolio step" in w for w in pipe.warnings())


def test_warnings_reach_the_result(env):
    px, sources = env
    pipe = pl.Pipeline.from_json({
        "source": "returns",
        "steps": [{"op": "GrossScale"}, {"op": "GrossScale"}]})
    _, res = pl.run_pipeline(pipe, sources, px)
    assert any("GrossScale more than once" in n for n in res.notes)


# --------------------------------------------------------------------------- #
# sources and presets
# --------------------------------------------------------------------------- #

def test_build_sources_derives_amihud(env):
    px, sources = env
    assert "amihud" in sources
    a = sources["amihud"].dropna(how="all")
    assert len(a) > 100 and (a.stack() >= 0).all()


def test_requesting_a_source_the_panel_lacks_fails_clearly(env):
    px, _ = env
    bare = {"close": px, "returns": px.pct_change()}
    pipe = pl.Pipeline.from_json({"source": "amihud", "steps": [{"op": "Rank"}]})
    with pytest.raises(pl.PipelineError, match="not available"):
        pl.run_pipeline(pipe, bare, px)


@pytest.mark.parametrize("name", sorted(pl.PRESETS))
def test_every_preset_parses_and_runs(env, name):
    px, sources = env
    preset = pl.PRESETS[name]
    pipe = pl.Pipeline.from_json(preset)
    sig, res = pl.run_pipeline(pipe, sources, px, cost=xcosts.TWEquityCost())
    assert sig.notna().sum().sum() > 0
    assert np.isfinite(res.net.sum())
    assert preset["note"], f"{name} should explain itself in the UI"


@pytest.mark.parametrize("name", sorted(pl.PRESETS))
def test_every_preset_ends_with_a_portfolio_step(env, name):
    pipe = pl.Pipeline.from_json(pl.PRESETS[name])
    portfolio = {n for n, o in ops.ALL_OPS.items() if o.group == "portfolio"}
    assert pipe.steps[-1].op in portfolio, f"{name} does not produce weights"
