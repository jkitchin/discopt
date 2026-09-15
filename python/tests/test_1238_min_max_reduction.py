"""Vector ``min`` / ``max`` reduction: issue #1238.

Before this, "the smallest element of this vector" had no spelling.
``dm.minimum`` / ``dm.maximum`` took exactly two operands, a shaped ``Variable``
had no ``.min()`` / ``.max()``, and ``np.min(xs)`` is refused by
``Expression.__array_ufunc__ = None``. The only route was a hand-written chain
``dm.minimum(dm.minimum(a, b), c)``, whose depth grows 1:1 with the element
count -- the same left-deep fold #1235 removed for ``sum``, and the one
``performance-plan.md`` §54 records raising ``RecursionError`` out of the
LP/MPS/GAMS writers at n >= 1000.

What ships is the *balanced* fold, not an n-ary ``Min``/``Max`` IR node. #1238's
entry experiment measured the n-ary envelope against the balanced fold over 288
fixed-box comparisons and found them bit-identical (performance-plan §64), so the
issue's own kill criterion fires: the fold needs no flag, no IR change and no
corpus panel, and it is the only one of the two that cannot reintroduce the
silent ``args[2:]`` drop that ``nl_parser.rs`` folds ``o11``/``o12`` to binary to
avoid (C-8).
"""

from __future__ import annotations

import math
import sys

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
import pytest
from discopt import Model


def _depth(expr, _memo=None) -> int:
    """Longest root-to-leaf path through an expression DAG (mirrors #1235's)."""
    memo = {} if _memo is None else _memo
    key = id(expr)
    if key in memo:
        return memo[key]
    kids = []
    for attr in ("left", "right", "operand", "base", "args", "terms"):
        value = getattr(expr, attr, None)
        if value is None:
            continue
        kids.extend(value if isinstance(value, (list, tuple)) else [value])
    depth = 1 + max(
        (_depth(k, memo) for k in kids if isinstance(k, core.Expression)),
        default=0,
    )
    memo[key] = depth
    return depth


def _count_calls(expr, fname: str) -> int:
    """Number of distinct ``FunctionCall(fname, ...)`` nodes in the DAG."""
    seen: set[int] = set()
    found: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, core.FunctionCall) and node.func_name == fname:
            found.add(id(node))
        for attr in ("left", "right", "operand", "base", "args", "terms"):
            value = getattr(node, attr, None)
            if value is None:
                continue
            for k in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(k, core.Expression):
                    stack.append(k)
    return len(found)


def _left_deep(fname: str, terms):
    """The spelling #1238 replaces: ``min(min(min(a, b), c), d)``."""
    acc = terms[0]
    for t in terms[1:]:
        acc = core.FunctionCall(fname, acc, t)
    return acc


@pytest.fixture
def model():
    return Model("m1238")


# ─────────────────────────────────────────────────────────────
# the gap itself: every row of the issue's table now has an answer
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_shaped_expression_has_the_reduction_method(model, name):
    """``xs.min()`` / ``xs.max()`` exist -- the AttributeError in the issue."""
    xs = model.continuous(f"xs_{name}", shape=(6,), lb=0, ub=1)
    node = getattr(xs, name)()
    assert isinstance(node, core.Expression)
    assert _count_calls(node, name) == 5, "n - 1 binary nodes, same count as the fold"


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_numpy_reduction_protocol_reaches_the_method(model, name):
    """``np.min(xs)`` used to raise on ``__array_ufunc__ = None``."""
    xs = model.continuous(f"nx_{name}", shape=(4,), lb=0, ub=1)
    via_numpy = getattr(np, name)(xs)
    via_method = getattr(xs, name)()
    assert repr(via_numpy) == repr(via_method)


@pytest.mark.parametrize("public,fname", [("minimum", "min"), ("maximum", "max")])
def test_1238_module_function_takes_more_than_two_operands(model, public, fname):
    """``dm.minimum(x, y, z)`` -- the TypeError in the issue."""
    xs = model.continuous(f"v_{fname}", shape=(3,), lb=0, ub=1)
    node = getattr(dm, public)(xs[0], xs[1], xs[2])
    assert _count_calls(node, fname) == 2


@pytest.mark.parametrize("public,fname", [("minimum", "min"), ("maximum", "max")])
def test_1238_two_operand_call_is_byte_identical_to_before(model, public, fname):
    """The binary call is untouched: one node, same shape cache, same repr."""
    xs = model.continuous(f"b_{fname}", shape=(2,), lb=0, ub=1)
    node = getattr(dm, public)(xs[0], xs[1])
    assert isinstance(node, core.FunctionCall)
    assert node.func_name == fname
    assert len(node.args) == 2
    assert repr(node) == repr(core.FunctionCall(fname, xs[0], xs[1]))


# ─────────────────────────────────────────────────────────────
# the depth claim, which is the point of the issue
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [4, 16, 64, 256, 1024])
@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_reduction_depth_is_logarithmic_not_linear(model, n, name):
    """``ceil(log2(n))`` deep, against the left-deep fold's ``n - 1``.

    The two extra levels below the fold are the ``IndexExpression`` and the
    ``Variable`` leaf, which both spellings carry.
    """
    v = model.continuous(f"v{name}{n}", shape=(n,), lb=0, ub=1)
    balanced = getattr(v, name)()
    assert _depth(balanced) == math.ceil(math.log2(n)) + 2

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * n + 1000))
    left = _left_deep(name, [v[i] for i in range(n)])
    assert _depth(left) == n + 1, "the hand-written chain is still left-deep"
    assert _depth(balanced) < _depth(left)


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_module_function_fold_is_balanced_too(model, name):
    """``dm.minimum(*terms)`` folds balanced, not left -- same guarantee."""
    public = "minimum" if name == "min" else "maximum"
    v = model.continuous(f"mf{name}", shape=(64,), lb=0, ub=1)
    terms = [v[i] for i in range(64)]
    assert _depth(getattr(dm, public)(*terms)) == math.ceil(math.log2(64)) + 2


@pytest.mark.parametrize("n", [3, 5, 7, 9, 17])
@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_odd_element_counts_stay_balanced(model, n, name):
    """An odd term rides one level up; the tree stays within one level of ideal."""
    v = model.continuous(f"o{name}{n}", shape=(n,), lb=0, ub=1)
    assert _depth(getattr(v, name)()) == math.ceil(math.log2(n)) + 2


# ─────────────────────────────────────────────────────────────
# numerical agreement -- the fold must BE the min, not resemble it
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_agrees_numerically_with_the_explicit_fold(name):
    """Same model, two spellings, same certified objective and node count."""

    def build(vectorised):
        m = Model(f"agree_{name}")
        v = m.continuous("v", shape=(5,), lb=0.5, ub=3.0)
        m.subject_to(v.sum() == 7.0, name="budget")
        w = getattr(v, name)() if vectorised else _left_deep(name, [v[i] for i in range(5)])
        # minimise the worst case / maximise the best case -- either way the
        # nonsmooth node is what the objective is made of.
        m.minimize(w if name == "max" else -w)
        return m.solve()

    flat, folded = build(True), build(False)
    assert flat.status == folded.status
    assert flat.objective == pytest.approx(folded.objective, abs=1e-9)
    # Five variables in [0.5, 3] summing to 7: the worst case is minimised, and
    # the best case maximised, by the flat point 1.4.
    assert abs(flat.objective) == pytest.approx(1.4, rel=1e-6)


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_reduction_evaluates_as_the_reduction(name):
    """Evaluate the built DAG at sampled points against numpy's reduction.

    This is the check that a *wrong* fold (a dropped term, a mis-ordered pair)
    would fail while every structural assertion above still passed.
    """
    rng = np.random.default_rng(1238)
    n = 7
    m = Model(f"eval_{name}")
    v = m.continuous("v", shape=(n,), lb=-2.0, ub=2.0)
    m.subject_to(getattr(v, name)() == 0.0, name="pin")
    body = m._constraints[0].body
    checked = 0
    for _ in range(50):
        point = rng.uniform(-2.0, 2.0, size=n)
        got = _eval_expr(body, {"v": point})
        want = (np.min if name == "min" else np.max)(point)
        assert got == pytest.approx(want, abs=1e-12)
        checked += 1
    assert checked == 50, "the probe must actually have compared something"


def _eval_expr(node, values):
    """Tiny recursive evaluator over the node kinds this file builds."""
    if isinstance(node, core.Constant):
        return float(np.asarray(node.value).reshape(()))
    if isinstance(node, core.IndexExpression):
        base = node.base
        assert isinstance(base, core.Variable), type(base)
        idx = node.index
        arr = np.asarray(values[base.name])
        return float(arr[idx])
    if isinstance(node, core.FunctionCall):
        args = [_eval_expr(a, values) for a in node.args]
        if node.func_name == "min":
            return min(args)
        if node.func_name == "max":
            return max(args)
        raise AssertionError(f"unexpected call {node.func_name}")
    if isinstance(node, core.BinaryOp):
        left = _eval_expr(node.left, values)
        right = _eval_expr(node.right, values)
        return {"+": lambda: left + right, "-": lambda: left - right}[node.op]()
    raise AssertionError(f"unexpected node {type(node).__name__}")


# ─────────────────────────────────────────────────────────────
# refusals -- every one names a spelling that works
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_axis_reduction_is_refused_not_faked(model, name):
    """No array-valued min/max node exists, so ``axis=`` refuses loudly."""
    X = model.continuous(f"X{name}", shape=(2, 3), lb=0, ub=1)
    with pytest.raises(NotImplementedError, match=r"axis"):
        getattr(X, name)(axis=1)
    # ... and the spelling the message names does work.
    per_row = getattr(X[0, :], name)()
    assert _count_calls(per_row, name) == 2
    unpacked = getattr(dm, "minimum" if name == "min" else "maximum")(*X[0, :])
    assert _count_calls(unpacked, name) == 2


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_unknown_shape_is_refused_with_the_fix_named(model, name):
    """A matmul result carries no static shape, so there is nothing to fold."""
    A = np.ones((3, 4))
    y = model.continuous(f"y{name}", shape=(4,), lb=0, ub=1)
    with pytest.raises(TypeError, match=r"statically known shape"):
        getattr(A @ y, name)()


@pytest.mark.parametrize("public", ["minimum", "maximum"])
def test_1238_single_operand_points_at_the_reduction(model, public):
    """``dm.minimum(xs)`` is elementwise-with-one-operand, i.e. a mistake."""
    xs = model.continuous(f"s_{public}", shape=(4,), lb=0, ub=1)
    with pytest.raises(TypeError, match=r"at least two operands"):
        getattr(dm, public)(xs)
    with pytest.raises(TypeError, match=r"at least two operands"):
        getattr(dm, public)()


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_numpy_only_keywords_are_refused(model, name):
    """``out=`` / ``keepdims=`` cannot be honoured by a DAG node (the #1235 rule)."""
    xs = model.continuous(f"k{name}", shape=(3,), lb=0, ub=1)
    with pytest.raises(TypeError, match=r"out="):
        getattr(xs, name)(out=np.zeros(1))
    with pytest.raises(TypeError, match=r"keepdims"):
        getattr(xs, name)(keepdims=True)


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_scalar_reduces_to_itself(model, name):
    """``min`` of one element is that element, not a 1-argument call.

    A ``FunctionCall(fname, a)`` would be the dangerous form: every consumer
    reads ``args[0], args[1]`` and a one-argument call has no ``args[1]``.
    """
    xs = model.continuous(f"sc{name}", shape=(3,), lb=0, ub=1)
    assert getattr(xs[0], name)() is xs[0]
    one = model.continuous(f"one{name}", shape=(1,), lb=0, ub=1)
    node = getattr(one, name)()
    assert not (isinstance(node, core.FunctionCall) and node.func_name == name)


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_empty_reduction_is_refused(model, name):
    """The min of no elements is undefined -- do not silently return a constant."""
    empty = model.continuous(f"e{name}", shape=(0,), lb=0, ub=1)
    with pytest.raises(ValueError, match=r"empty reduction"):
        getattr(empty, name)()


# ─────────────────────────────────────────────────────────────
# the bound claim from the entry experiment, pinned
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["min", "max"])
def test_1238_balanced_fold_is_bound_neutral_against_the_left_deep_fold(name):
    """Re-nesting an associative fold must not move the relaxation bound.

    This is the in-repo pin of #1238's entry experiment (performance-plan §64):
    over the default per-node engine, left-deep, balanced and n-ary spellings of
    the same ``min``/``max`` all give the *same* LP bound, because
    ``uniform_relax._build_multivar`` emits the exact convex-hull facets and the
    intermediate auxes project straight back out.
    """
    from discopt._relax.model_utils import flat_variable_bounds
    from discopt._relax.uniform_relax import build_uniform_relaxation

    def bound(vectorised):
        m = Model(f"bn_{name}")
        v = m.continuous("v", shape=(6,), lb=-1.0, ub=2.0)
        terms = [(0.5 + 0.25 * i) * v[i] - 0.1 * i for i in range(6)]
        w = (
            getattr(v, name)()
            if vectorised is None
            else (
                _left_deep(name, terms)
                if vectorised
                else core.FunctionCall(name, *terms)  # the n-ary form, unshipped
            )
        )
        m.minimize(w)
        flb, fub = flat_variable_bounds(m)
        rel = build_uniform_relaxation(m, box=(flb, fub))
        res = rel.model.solve(backend="simplex")
        assert res.objective is not None, res.status
        return rel.obj_sense_sign * (float(res.objective) + rel.obj_offset)

    left, nary = bound(True), bound(False)
    balanced = _balanced_bound(name)
    assert balanced == pytest.approx(left, abs=1e-12)
    assert balanced == pytest.approx(nary, abs=1e-12)


def _balanced_bound(name: str) -> float:
    from discopt._relax.model_utils import flat_variable_bounds
    from discopt._relax.uniform_relax import build_uniform_relaxation

    m = Model(f"bal_{name}")
    v = m.continuous("v", shape=(6,), lb=-1.0, ub=2.0)
    terms = [(0.5 + 0.25 * i) * v[i] - 0.1 * i for i in range(6)]
    m.minimize(dm.minimum(*terms) if name == "min" else dm.maximum(*terms))
    flb, fub = flat_variable_bounds(m)
    rel = build_uniform_relaxation(m, box=(flb, fub))
    res = rel.model.solve(backend="simplex")
    assert res.objective is not None, res.status
    return rel.obj_sense_sign * (float(res.objective) + rel.obj_offset)
