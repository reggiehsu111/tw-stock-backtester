"""Tests for the two-language surface.

The useful assertion is completeness: every key must exist in both languages,
and every operator must have a Chinese hint. A half-translated page is worse
than an English one, because the missing half looks like a bug in the tool
rather than a gap in the translation.
"""

import pytest

from twbt import i18n, ops

TABLES = {"UI": i18n.UI, "CHART": i18n.CHART, "CAPTION": i18n.CAPTION,
          "STAT": i18n.STAT, "VERDICT": i18n.VERDICT}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_every_string_exists_in_both_languages(name):
    missing = [k for k, v in TABLES[name].items()
               if not v.get("en") or not v.get("zh")]
    assert not missing, f"{name} incomplete: {missing}"


def test_every_operator_has_a_chinese_hint():
    missing = [n for n in ops.ALL_OPS if n not in i18n.OP_HINT_ZH]
    assert not missing, f"no 繁體中文 hint for: {missing}"


def test_no_operator_hint_is_just_the_docstring():
    """The palette hint is written for someone choosing a block, not reading source."""
    for name, op in ops.ALL_OPS.items():
        assert op.hint, name
        assert i18n.OP_HINT_ZH[name].strip(), name


def test_hints_are_short_enough_for_a_tooltip():
    for name, op in ops.ALL_OPS.items():
        assert len(op.hint) < 190, f"{name} English hint is too long"
        assert len(i18n.OP_HINT_ZH[name]) < 110, f"{name} Chinese hint is too long"


def test_every_operator_exposes_its_signature():
    for name, op in ops.ALL_OPS.items():
        assert op.signature.startswith(name + "(")
        assert op.signature.endswith(")")


def test_unknown_language_falls_back_rather_than_failing():
    assert i18n.pick("klingon") == "en"
    assert i18n.t(i18n.UI, "operators", "klingon") == "Operators"
    assert i18n.t(i18n.UI, "operators", "zh") == "運算子"


def test_an_unknown_key_returns_itself_rather_than_raising():
    assert i18n.t(i18n.UI, "no_such_key", "zh") == "no_such_key"


def test_verdict_templates_all_format():
    """A template with a typo in a placeholder would only fail at render time."""
    params = {
        "cost_kills": {"share": 0.5, "turn": 0.2},
        "net_positive": {"sharpe": 1.0, "t": 2.0, "share": 0.3},
        "years": {"pos": 2, "n": 3},
        "one_side": {"side": "long", "lng": 0.1, "sht": 0.2},
        "tradability": {"moved": 0.05},
    }
    for key, row in i18n.VERDICT.items():
        for lang in ("en", "zh"):
            row[lang].format(**params.get(key, {}))


def test_market_terms_stay_in_chinese_in_both_languages():
    """漲停, 證交稅 and friends are names of rules, not words to translate."""
    for term, key in (("證交稅", "tax_note"), ("漲停", "tradability")):
        assert term in i18n.UI[key]["en"], f"{term} should stay in the English text"


def test_failure_lines_are_flagged_by_key_not_by_wording():
    """Prefix-matching on "FAILS" breaks the moment anything is reworded.

    It also cannot survive translation at all: a Chinese verdict never starts
    with "FAILS", so under the old scheme every failure in 繁體中文 rendered as
    an ordinary line.
    """
    import numpy as np
    import pandas as pd

    from twbt import ops, xsim

    idx = pd.bdate_range("2023-01-02", periods=400)
    cols = [f"{i:04d}" for i in range(1101, 1141)]
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), len(cols))), axis=0)),
        index=idx, columns=cols)
    sig = ops.LongShortBalance(ops.Rank(-ops.InstSum(5, ops.Returns(px))))

    res = xsim.simulate(sig, px)
    en = res.verdict_items("en")
    zh = res.verdict_items("zh")

    assert [i["key"] for i in en] == [i["key"] for i in zh]
    assert [i["fail"] for i in en] == [i["fail"] for i in zh], (
        "a failure must be flagged identically in both languages")
    assert any(i["fail"] for i in zh), "this signal should fail something"
    # and the flag must not depend on the text starting with an English word
    for i in zh:
        if i["fail"]:
            assert not i["text"].startswith("FAILS")
