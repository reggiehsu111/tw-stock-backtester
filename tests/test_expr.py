"""Tests for the expression language.

The first group is the one that matters. The code editor accepts a string from a
browser, so the only defence against it being remote code execution is that the
string is *parsed* rather than evaluated, and that the parser can express
nothing but numbers, whitelisted names, and five arithmetic symbols.
"""

import numpy as np
import pandas as pd
import pytest

from twbt import expr, ops

IDX = pd.bdate_range("2023-01-02", periods=400)
COLS = [f"{i:04d}" for i in range(1101, 1161)]


@pytest.fixture
def sources():
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(IDX), len(COLS))), axis=0)),
        index=IDX, columns=COLS)
    return {"px": px, "close": px, "returns": px.pct_change(),
            "turnover": pd.DataFrame(1e7, index=IDX, columns=COLS)}


# --------------------------------------------------------------------------- #
# nothing to escape to
# --------------------------------------------------------------------------- #

HOSTILE = [
    "__import__('os').system('id')",
    "().__class__.__bases__[0].__subclasses__()",
    "eval('1+1')",
    "open('/etc/passwd').read()",
    "px.__class__",
    "px.values",
    "px[0]",
    "lambda: 1",
    "exec('x=1')",
    "globals()",
    "Rank.__globals__",
    "px;import os",
]


@pytest.mark.parametrize("src", HOSTILE)
def test_hostile_input_cannot_parse_or_evaluate(sources, src):
    """There is no attribute access, no indexing and no name outside the registry.

    These are rejected by the grammar itself rather than by a denylist, which is
    the point -- a denylist of dangerous names is defeated by the next name
    nobody thought of.
    """
    with pytest.raises(expr.ExprError):
        expr.run(src, sources)


def test_the_tokenizer_has_no_token_for_attribute_access(sources):
    for ch in ".[]{}:;=!<>&|^%@'\"\\`~?#$":
        with pytest.raises(expr.ExprError):
            expr.tokenize(f"Rank(px{ch})")


def test_only_registry_operators_are_callable(sources):
    with pytest.raises(expr.ExprError, match="unknown operator"):
        expr.run("print(px)", sources)
    with pytest.raises(expr.ExprError, match="unknown operator"):
        expr.run("Op(px)", sources)


def test_lookahead_is_refused(sources):
    with pytest.raises(expr.ExprError, match="future"):
        expr.run("Lead(1, returns)", sources)


@pytest.mark.parametrize("src,limit", [
    ("Rank(" * 60 + "returns" + ")" * 60, "deeply"),
    # 191 atoms + 190 operators = 381 nodes, but only 1,907 characters, so this
    # trips the node limit rather than the length limit
    ("returns" + " + returns" * 190, "too many"),
    ("Rank(" + "x" * 3000 + ")", "too long"),
])
def test_resource_limits(sources, src, limit):
    with pytest.raises(expr.ExprError, match=limit):
        expr.run(src, sources)


# --------------------------------------------------------------------------- #
# errors a person can act on
# --------------------------------------------------------------------------- #

def test_a_misspelled_operator_suggests_the_right_one(sources):
    with pytest.raises(expr.ExprError, match="did you mean 'Rank'"):
        expr.run("Rnk(returns)", sources)


def test_an_unknown_source_lists_what_is_available(sources):
    with pytest.raises(expr.ExprError, match="Available:"):
        expr.run("Rank(bananas)", sources)


def test_wrong_arity_reports_the_signature(sources):
    with pytest.raises(expr.ExprError, match=r"Signature: Lag\(n, mx\)"):
        expr.run("Lag(returns)", sources)


def test_errors_carry_a_column(sources):
    try:
        expr.run("Rank(Rnk(returns))", sources)
    except expr.ExprError as e:
        assert e.pos is not None and e.pos > 0
        assert "column" in str(e)
    else:
        pytest.fail("should have raised")


def test_unbalanced_parentheses_say_so(sources):
    with pytest.raises(expr.ExprError, match="expected"):
        expr.run("Rank(returns", sources)


def test_validate_catches_names_without_touching_data():
    """Parsing alone calls Rnk(returns) well-formed; validation is separate."""
    node = expr.parse("Rnk(returns)")          # parses fine
    with pytest.raises(expr.ExprError, match="unknown operator"):
        expr.validate(node, {"returns"})


# --------------------------------------------------------------------------- #
# what it computes
# --------------------------------------------------------------------------- #

def test_integer_literals_stay_integers(sources):
    """5.0 reaches pandas rolling() as a float and is rejected; 5 must stay 5."""
    assert isinstance(expr.parse("Lag(5, returns)").args[0].value, int)
    assert isinstance(expr.parse("Pow(returns, 2.5)").args[1].value, float)
    expr.run("InstSum(5, returns)", sources)   # must not raise


def test_a_chain_matches_calling_the_operators_directly(sources):
    got = expr.run("GrossScale(LongShortBalance(Rank(Neg(InstSum(5, Returns(px))))))",
                   sources)
    want = ops.GrossScale(ops.LongShortBalance(
        ops.Rank(ops.Neg(ops.InstSum(5, ops.Returns(sources["px"]))))))
    pd.testing.assert_frame_equal(got, want)


def test_arithmetic_on_matrices(sources):
    got = expr.run("Returns(px) / InstStdev(20, Returns(px))", sources)
    r = sources["px"].pct_change()
    want = r / ops.InstStdev(20, r).replace(0, np.nan)
    pd.testing.assert_frame_equal(got, want)


def test_division_by_zero_becomes_nan_not_inf(sources):
    zeros = pd.DataFrame(0.0, index=IDX, columns=COLS)
    out = expr.evaluate(expr.parse("returns / z"), {**sources, "z": zeros})
    assert not np.isinf(out.to_numpy()).any()


def test_two_plain_numbers_are_refused(sources):
    with pytest.raises(expr.ExprError, match="plain numbers"):
        expr.run("1 + 2", sources)


def test_an_expression_must_produce_a_matrix(sources):
    with pytest.raises(expr.ExprError):
        expr.run("5", sources)


def test_precedence_and_parentheses(sources):
    a = expr.to_source(expr.parse("1 + 2 * 3"))
    b = expr.to_source(expr.parse("(1 + 2) * 3"))
    assert a == "(1 + (2 * 3))"
    assert b == "((1 + 2) * 3)"


def test_two_matrix_operators_work_in_the_editor(sources):
    """Div, Neutralize and friends have no block form but must be typable."""
    for src in ("Div(returns, turnover)",
                "Neutralize(Rank(returns), Rank(turnover))",
                "Mask(Rank(returns), InstMean(20, turnover))"):
        out = expr.run(src, sources)
        assert isinstance(out, pd.DataFrame)


# --------------------------------------------------------------------------- #
# round-tripping with the block canvas
# --------------------------------------------------------------------------- #

def test_a_chain_round_trips_through_blocks():
    src = "GrossScale(LongShortBalance(Rank(Neg(InstSum(5, Returns(px))))))"
    chain = expr.to_chain(expr.parse(src))
    assert chain is not None
    assert chain.source == "px"
    assert [s["op"] for s in chain.steps] == [
        "Returns", "InstSum", "Neg", "Rank", "LongShortBalance", "GrossScale"]
    assert chain.steps[1]["params"] == {"n": 5}
    assert expr.from_chain(chain) == src


def test_a_tree_has_no_block_form():
    """Returning None beats approximating: a canvas that dropped one side of a
    ratio would be lying about what runs."""
    for src in ("Rank(Div(Returns(px), InstStdev(20, Returns(px))))",
                "Rank(Returns(px) / InstStdev(20, Returns(px)))",
                "Rank(-returns * 2)"):
        assert expr.to_chain(expr.parse(src)) is None


def test_matrix_first_operators_round_trip_in_the_right_order():
    """GrossScale(mx, gmv) vs Lag(n, mx): the renderer must know the difference."""
    chain = expr.to_chain(expr.parse("GrossScale(Rank(returns))"))
    assert expr.from_chain(chain) == "GrossScale(Rank(returns))"
    chain2 = expr.to_chain(expr.parse("TopN(25, Rank(returns))"))
    assert expr.from_chain(chain2) == "TopN(25, Rank(returns))"
