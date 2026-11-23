"""A tiny expression language for signals, parsed rather than evaluated.

This is what the code editor in the builder accepts:

    GrossScale(LongShortBalance(Rank(Neg(InstSum(5, Returns(px))))))
    Rank(Div(Returns(px), InstStdev(20, Returns(px))))
    LongShortBalance(Rank(-InstSum(5, Returns(px)) * InstRank(60, turnover)))

Three reasons it is a hand-written parser and not ``eval``.

SAFETY. ``eval`` on a string from a browser is remote code execution, and no
amount of regex blacklisting fixes that — ``().__class__.__bases__`` gets you out
of any denylist. Here the only things that exist are numbers, the source matrices,
the operators in the registry, and five arithmetic symbols. There is no attribute
access, no indexing, no subscripting, no names beyond the whitelist, so there is
nothing to escape *to*.

ERRORS WORTH READING. A parser knows where it is. "unknown operator 'Rnk' at
column 18 — did you mean 'Rank'?" is a different experience from a NameError.

TREES, NOT JUST CHAINS. The block canvas builds a linear chain, which covers most
signals. An expression can branch, and the ratio of two matrices is a perfectly
ordinary thing to want. The two editors share this representation: a chain
round-trips to blocks, a tree is shown as code only, and the UI says which.

Grammar, standard precedence:

    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := '-' factor | atom
    atom   := NUMBER | NAME | NAME '(' [expr (',' expr)*] ')' | '(' expr ')'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from inspect import Parameter, signature

import numpy as np
import pandas as pd

from . import ops

MAX_LEN = 2000
MAX_DEPTH = 40
MAX_NODES = 200


class ExprError(ValueError):
    """A parse or evaluation failure, with a position where one is known."""

    def __init__(self, message: str, pos: int | None = None):
        super().__init__(message)
        self.message = message
        self.pos = pos

    def __str__(self) -> str:
        return (f"{self.message} (at column {self.pos + 1})"
                if self.pos is not None else self.message)


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #

TOKEN_RE = re.compile(r"""
    (?P<space>\s+)
  | (?P<number>\d+\.\d+|\.\d+|\d+)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>[-+*/(),])
""", re.VERBOSE)


@dataclass
class Token:
    kind: str
    text: str
    pos: int


def tokenize(src: str) -> list[Token]:
    if len(src) > MAX_LEN:
        raise ExprError(f"expression is too long ({len(src)} characters, "
                        f"max {MAX_LEN})")
    out, i = [], 0
    while i < len(src):
        m = TOKEN_RE.match(src, i)
        if not m:
            raise ExprError(f"unexpected character {src[i]!r}", i)
        kind = m.lastgroup
        if kind != "space":
            out.append(Token(kind, m.group(), i))
        i = m.end()
    return out


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

@dataclass
class Num:
    value: float | int
    pos: int = 0


@dataclass
class Ref:
    """A source matrix, by name."""
    name: str
    pos: int = 0


@dataclass
class Call:
    name: str
    args: list
    pos: int = 0


@dataclass
class BinOp:
    op: str
    left: object
    right: object
    pos: int = 0


@dataclass
class Neg:
    operand: object
    pos: int = 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

class Parser:
    def __init__(self, tokens: list[Token], src: str):
        self.toks = tokens
        self.src = src
        self.i = 0
        self.nodes = 0

    def _peek(self) -> Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, text: str | None = None) -> Token:
        t = self._peek()
        if t is None:
            raise ExprError(f"expected {text!r} but the expression ended"
                            if text else "the expression ended unexpectedly",
                            len(self.src))
        if text is not None and t.text != text:
            raise ExprError(f"expected {text!r}, found {t.text!r}", t.pos)
        self.i += 1
        return t

    def _count(self, pos: int) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ExprError(f"expression has too many terms (max {MAX_NODES})", pos)

    def parse(self):
        if not self.toks:
            raise ExprError("the expression is empty")
        node = self.expr(0)
        extra = self._peek()
        if extra is not None:
            raise ExprError(f"unexpected {extra.text!r} after the expression",
                            extra.pos)
        return node

    def expr(self, depth: int):
        if depth > MAX_DEPTH:
            raise ExprError(f"expression is nested too deeply (max {MAX_DEPTH})")
        node = self.term(depth + 1)
        while (t := self._peek()) and t.text in "+-":
            self._eat()
            self._count(t.pos)
            node = BinOp(t.text, node, self.term(depth + 1), t.pos)
        return node

    def term(self, depth: int):
        if depth > MAX_DEPTH:
            raise ExprError(f"expression is nested too deeply (max {MAX_DEPTH})")
        node = self.factor(depth + 1)
        while (t := self._peek()) and t.text in "*/":
            self._eat()
            self._count(t.pos)
            node = BinOp(t.text, node, self.factor(depth + 1), t.pos)
        return node

    def factor(self, depth: int):
        t = self._peek()
        if t and t.text == "-":
            self._eat()
            self._count(t.pos)
            return Neg(self.factor(depth + 1), t.pos)
        return self.atom(depth + 1)

    def atom(self, depth: int):
        if depth > MAX_DEPTH:
            raise ExprError(f"expression is nested too deeply (max {MAX_DEPTH})")
        t = self._peek()
        if t is None:
            raise ExprError("the expression ended where a value was expected",
                            len(self.src))
        self._count(t.pos)

        if t.text == "(":
            self._eat("(")
            node = self.expr(depth + 1)
            self._eat(")")
            return node

        if t.kind == "number":
            self._eat()
            # keep an integer literal an int. Window arguments go to pandas
            # rolling(), which rejects 5.0 with "window must be an integer" --
            # a confusing error for someone who typed 5.
            v = float(t.text) if "." in t.text else int(t.text)
            return Num(v, t.pos)

        if t.kind == "name":
            self._eat()
            nxt = self._peek()
            if nxt and nxt.text == "(":
                self._eat("(")
                args = []
                if (p := self._peek()) and p.text != ")":
                    args.append(self.expr(depth + 1))
                    while (p := self._peek()) and p.text == ",":
                        self._eat(",")
                        args.append(self.expr(depth + 1))
                self._eat(")")
                return Call(t.text, args, t.pos)
            return Ref(t.text, t.pos)

        raise ExprError(f"unexpected {t.text!r}", t.pos)


def parse(src: str):
    """Parse an expression string into an AST."""
    return Parser(tokenize(src), src).parse()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def _suggest(name: str, options) -> str:
    near = get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f" — did you mean {near[0]!r}?" if near else ""


def _bind(fn, name: str, args: list, pos: int):
    """Match positional arguments to the operator's actual signature.

    Operators differ in argument order on purpose — ``Lag(n, mx)`` puts the
    window first, ``GrossScale(mx, gmv)`` puts the matrix first — because that is
    the research vocabulary. Binding against the real signature means the editor
    accepts exactly what the function accepts, rather than a second, slightly
    different convention that has to be kept in sync.
    """
    sig = signature(fn)
    params = [p for p in sig.parameters.values()
              if p.kind in (Parameter.POSITIONAL_ONLY,
                            Parameter.POSITIONAL_OR_KEYWORD)]
    required = [p for p in params if p.default is Parameter.empty]
    if len(args) < len(required) or len(args) > len(params):
        shape = ", ".join(
            p.name if p.default is Parameter.empty else f"{p.name}={p.default!r}"
            for p in params)
        raise ExprError(
            f"{name} takes {len(required)}"
            + (f" to {len(params)}" if len(params) != len(required) else "")
            + f" argument(s), got {len(args)}. Signature: {name}({shape})", pos)
    return args


def evaluate(node, sources: dict[str, pd.DataFrame]):
    """Evaluate an AST against the available source matrices."""
    if isinstance(node, Num):
        return node.value

    if isinstance(node, Ref):
        if node.name not in sources:
            raise ExprError(
                f"unknown source {node.name!r}{_suggest(node.name, sources)}. "
                f"Available: {', '.join(sorted(sources))}", node.pos)
        return sources[node.name]

    if isinstance(node, Neg):
        v = evaluate(node.operand, sources)
        return -v

    if isinstance(node, BinOp):
        a = evaluate(node.left, sources)
        b = evaluate(node.right, sources)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            raise ExprError("arithmetic between two plain numbers does not "
                            "produce a signal", node.pos)
        try:
            if node.op == "+":
                return a + b
            if node.op == "-":
                return a - b
            if node.op == "*":
                return a * b
            # division guards a zero denominator, because inf propagates
            # silently through every later operator and only surfaces as a
            # blank chart
            den = b.replace(0, np.nan) if isinstance(b, pd.DataFrame) else (
                np.nan if b == 0 else b)  # noqa: PLR2004
            return a / den
        except Exception as exc:                        # noqa: BLE001
            raise ExprError(f"cannot apply {node.op!r}: {exc}", node.pos) from exc

    if isinstance(node, Call):
        if node.name not in ops.ALL_OPS:
            raise ExprError(
                f"unknown operator {node.name!r}{_suggest(node.name, ops.ALL_OPS)}",
                node.pos)
        op = ops.ALL_OPS[node.name]
        if op.lookahead:
            raise ExprError(
                f"{node.name} looks into the future and cannot be part of a "
                "signal. It exists for building the target of a study.", node.pos)
        args = [evaluate(a, sources) for a in node.args]
        _bind(op.fn, node.name, args, node.pos)
        try:
            result = op.fn(*args)
        except ExprError:
            raise
        except Exception as exc:                        # noqa: BLE001
            raise ExprError(f"{node.name}: {type(exc).__name__}: {exc}", node.pos) from exc
        if not isinstance(result, pd.DataFrame):
            raise ExprError(f"{node.name} did not return a matrix", node.pos)
        return result

    raise ExprError(f"cannot evaluate {type(node).__name__}")


def validate(node, source_names) -> None:
    """Check names and arity without touching data.

    Parsing alone is not validation: ``Rnk(returns)`` is a perfectly well-formed
    call, and a parser that only checks syntax will tell you a typo is fine. The
    editor calls this as you type, so a misspelled operator is reported
    immediately rather than when you press Run.
    """
    if isinstance(node, Num):
        return
    if isinstance(node, Ref):
        if node.name not in source_names:
            raise ExprError(
                f"unknown source {node.name!r}{_suggest(node.name, source_names)}. "
                f"Available: {', '.join(sorted(source_names))}", node.pos)
        return
    if isinstance(node, Neg):
        validate(node.operand, source_names)
        return
    if isinstance(node, BinOp):
        validate(node.left, source_names)
        validate(node.right, source_names)
        return
    if isinstance(node, Call):
        if node.name not in ops.ALL_OPS:
            raise ExprError(
                f"unknown operator {node.name!r}{_suggest(node.name, ops.ALL_OPS)}",
                node.pos)
        op = ops.ALL_OPS[node.name]
        if op.lookahead:
            raise ExprError(
                f"{node.name} looks into the future and cannot be part of a "
                "signal. It exists for building the target of a study.", node.pos)
        _bind(op.fn, node.name, node.args, node.pos)
        for a in node.args:
            validate(a, source_names)
        return
    raise ExprError(f"cannot validate {type(node).__name__}")


def run(src: str, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Parse and evaluate, returning the signal matrix."""
    result = evaluate(parse(src), sources)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("the expression must produce a matrix, not a number")
    return result


# --------------------------------------------------------------------------- #
# round-tripping with the block canvas
# --------------------------------------------------------------------------- #

def to_source(node) -> str:
    """Render an AST back to source. Used to normalise what the editor shows."""
    if isinstance(node, Num):
        return str(node.value) if isinstance(node.value, int) else f"{node.value:g}"
    if isinstance(node, Ref):
        return node.name
    if isinstance(node, Neg):
        return f"-{to_source(node.operand)}"
    if isinstance(node, BinOp):
        return f"({to_source(node.left)} {node.op} {to_source(node.right)})"
    if isinstance(node, Call):
        return f"{node.name}({', '.join(to_source(a) for a in node.args)})"
    raise ExprError(f"cannot render {type(node).__name__}")


@dataclass
class Chain:
    """A linear pipeline: one source, then operators applied in order."""

    source: str
    steps: list = field(default_factory=list)   # [{"op": str, "params": {...}}]


def to_chain(node) -> Chain | None:
    """Flatten an AST to a linear chain, or None if it branches.

    The block canvas can only draw a chain. Returning None rather than
    approximating is deliberate: a canvas that silently dropped one side of a
    ratio would be lying about what runs.
    """
    steps: list[dict] = []
    cur = node
    while True:
        if isinstance(cur, Ref):
            steps.reverse()
            return Chain(cur.name, steps)
        if not isinstance(cur, Call):
            return None                      # arithmetic, a bare number, a Neg
        matrix_args = [a for a in cur.args
                       if isinstance(a, (Call, Ref, BinOp, Neg))]
        if len(matrix_args) != 1:
            return None                      # branches, or no matrix input
        scalars = [a for a in cur.args if isinstance(a, Num)]
        op = ops.ALL_OPS.get(cur.name)
        if op is None:
            return None
        names = [p[0] for p in op.params]
        params = {n: (int(s.value) if t is int else t(s.value))
                  for (n, t, _), s in zip(op.params, scalars)} if names else {}
        steps.append({"op": cur.name, "params": params})
        cur = matrix_args[0]


def from_chain(chain: Chain) -> str:
    """Render a chain as an expression, the inverse of ``to_chain``."""
    expr = chain.source
    for step in chain.steps:
        op = ops.ALL_OPS.get(step["op"])
        params = step.get("params") or {}
        if op and op.name in ops.MATRIX_FIRST:
            args = [expr] + [f"{v:g}" for v in params.values()]
        else:
            args = [f"{v:g}" for v in params.values()] + [expr]
        expr = f"{step['op']}({', '.join(args)})"
    return expr
