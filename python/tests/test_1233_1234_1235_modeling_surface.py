"""Modeling-surface regressions: issues #1233, #1234, #1235.

Three gaps in the Python modeling API, fixed together because they share one
cause -- ``Expression.__array_ufunc__ = None`` (``modeling/core.py``). That
NEP-13 opt-out is deliberate and stays: it is why ``x * np.array([1.0, 2.0])``
collapses to a single ``BinaryOp`` instead of an object array of two. Its
side-effect is that NumPy habits do not transfer, so every route it closes has
to be opened explicitly on discopt's own surface:

* #1233 -- ``np.tanh(x)`` is refused, and ``tanh`` was not exported from the
  top-level namespace either, leaving no discoverable way to write it.
* #1234 -- ``arr * x`` reflects into ``Expression.__rmul__``, which tried to
  build one ``Constant`` from an array of expressions.
* #1235 -- ``np.sum(xs)`` is refused, and there was no ``.sum()`` method, so the
  builtin ``sum()`` was the only route and it folds left.
"""

from __future__ import annotations

import sys

import discopt
import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
import pytest
from discopt import Model

# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

#: Names in ``modeling/core.py`` that are symbolic intrinsic functions -- the set
#: a user reaches for when writing a model by hand. Derived by introspection in
#: :func:`_intrinsic_names` rather than hardcoded, so a function added to
#: ``core.py`` and to ``discopt.modeling`` but forgotten at the top level fails
#: this file instead of drifting (the #1233 failure mode).
_NON_INTRINSIC = frozenset(
    {
        # aggregations and constructors, not elementwise intrinsics
        "sum",
        "prod",
        "norm",
        "concatenate",
        "stack",
        "custom",
        "udf",
        "if_else",
        "piecewise",  # adds an output variable and rows, like if_else (#1482)
        "model_from_repr",
        "bulk_construction_gc",
        # logical / GDP helpers
        "land",
        "lor",
        "lnot",
        "atleast",
        "atmost",
        "exactly",
        # importers
        "from_pyomo",
        "from_nl",
        "from_gams",
        "from_description",
    }
)


def _intrinsic_names() -> list[str]:
    """Every public elementwise intrinsic defined in ``modeling.core``."""
    out = []
    for name, value in vars(core).items():
        if name.startswith("_") or name in _NON_INTRINSIC:
            continue
        if not callable(value) or isinstance(value, type):
            continue
        if getattr(value, "__module__", None) != "discopt.modeling.core":
            continue
        if name not in dm.__all__ and name != "abs_":
            continue
        out.append(name)
    return sorted(out)


def _depth(expr, _memo=None) -> int:
    """Longest root-to-leaf path through an expression DAG."""
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


@pytest.fixture
def model():
    return Model("surface")


# ─────────────────────────────────────────────────────────────
# #1233 -- intrinsics reachable from the top-level namespace
# ─────────────────────────────────────────────────────────────


def test_1233_every_core_intrinsic_is_reachable_from_the_top_level():
    """The two lists cannot drift apart again -- enumerated, not hardcoded."""
    names = _intrinsic_names()
    # §6: prove the probe fired. An introspection bug that found nothing would
    # otherwise make this test a green no-op.
    assert len(names) >= 19, f"introspection found only {len(names)} intrinsics: {names}"
    missing = [n for n in names if not hasattr(discopt, n)]
    assert missing == [], f"not reachable as `from discopt import ...`: {missing}"


def test_1233_top_level_intrinsics_are_the_same_objects_as_the_modeling_ones():
    """Re-export, not a reimplementation that could drift in behaviour."""
    checked = 0
    for name in _intrinsic_names():
        modeling_name = "abs" if name == "abs_" else name
        assert getattr(discopt, name) is getattr(dm, modeling_name), name
        checked += 1
    assert checked >= 19


def test_1233_the_reported_thirteen_are_present():
    """The names issue #1233 listed explicitly."""
    for name in ("abs_,acos,acosh,asin,asinh,atan,atanh,cosh,log10,log1p,log2,sinh,tanh").split(
        ","
    ):
        assert hasattr(discopt, name), name


def test_1233_tanh_builds_a_function_call(model):
    x = model.continuous("x", lb=0.1, ub=2.0)
    assert isinstance(discopt.tanh(x), core.FunctionCall)


def test_1233_abs_keeps_its_underscore_and_does_not_shadow_the_builtin():
    """``from discopt import *`` must not replace the builtin ``abs``.

    ``discopt.modeling`` exports it as ``dm.abs`` because that namespace is
    always reached through the ``dm.`` prefix; the top level is not.
    """
    assert hasattr(discopt, "abs_")
    assert getattr(discopt, "abs", abs) is abs


def test_1233_numpy_ufuncs_are_still_refused(model):
    """The NEP-13 opt-out is the reason these exports are needed; it stays."""
    x = model.continuous("x", lb=0.1, ub=2.0)
    assert core.Expression.__array_ufunc__ is None
    with pytest.raises(TypeError, match="ufunc"):
        np.tanh(x)


# ─────────────────────────────────────────────────────────────
# #1234 -- object-dtype ndarray against a scalar Expression
# ─────────────────────────────────────────────────────────────

_BINOPS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
    ("**", lambda a, b: a**b),
]


@pytest.mark.parametrize("op,fn", _BINOPS, ids=[o for o, _ in _BINOPS])
@pytest.mark.parametrize("array_first", [True, False], ids=["arr_op_x", "x_op_arr"])
def test_1234_object_array_and_scalar_expression_go_elementwise(model, op, fn, array_first):
    """Both operand orders, all five arithmetic operators (the issue's ask)."""
    x = model.continuous("x", lb=0.1, ub=2.0)
    y = model.continuous("y", lb=0.1, ub=2.0)
    t = model.continuous("t", lb=0.1, ub=2.0)
    arr = np.array([x, y], dtype=object)

    out = fn(arr, t) if array_first else fn(t, arr)

    assert isinstance(out, np.ndarray)
    assert out.dtype == object
    assert out.shape == (2,)
    for i, element in enumerate(arr):
        node = out[i]
        assert isinstance(node, core.BinaryOp)
        assert node.op == op
        # The scalar is broadcast against the array, in the written order.
        if array_first:
            assert node.left is element and node.right is t
        else:
            assert node.left is t and node.right is element


def test_1234_the_reported_valueerror_is_gone(model):
    """Verbatim from the issue's reproduction block."""
    x = model.continuous("x", lb=0.1, ub=2.0)
    y = model.continuous("y", lb=0.1, ub=2.0)
    arr = np.array([x, y], dtype=object)
    for expr in (lambda: arr * x, lambda: x * arr, lambda: arr + x):
        result = expr()
        assert isinstance(result, np.ndarray) and result.dtype == object


def test_1234_two_dimensional_object_arrays_keep_their_shape(model):
    x = model.continuous("x", shape=(6,), lb=0.1, ub=2.0)
    t = model.continuous("t", lb=0.1, ub=2.0)
    arr = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]]], dtype=object)
    out = arr * t
    assert out.shape == (2, 3)
    assert all(isinstance(out[i, j], core.BinaryOp) for i in range(2) for j in range(3))


def test_1234_object_array_of_mixed_numbers_and_expressions(model):
    """Elements need not all be expressions; a number reflects as usual."""
    x = model.continuous("x", lb=0.1, ub=2.0)
    t = model.continuous("t", lb=0.1, ub=2.0)
    out = np.array([x, 2.0], dtype=object) * t
    assert isinstance(out[0], core.BinaryOp)
    assert isinstance(out[1], core.BinaryOp)


def test_1234_an_empty_object_array_stays_empty(model):
    t = model.continuous("t", lb=0.1, ub=2.0)
    out = np.array([], dtype=object) * t
    assert isinstance(out, np.ndarray) and out.shape == (0,)


def test_1234_a_shaped_expression_operand_is_refused_not_broadcast(model):
    """The ambiguous case gets an error, not a nine-row model for a three-row ask.

    ``arr * xs`` with both of length 3 reads as elementwise pairing; treating
    ``xs`` as a scalar operand would give a length-3 array of length-3 elements.
    Neither reading can be inferred, so it refuses.
    """
    xs = model.continuous("xs", shape=(3,), lb=0.5, ub=4.0)
    arr = np.array([xs[0], xs[1], xs[2]], dtype=object)
    for expr in (lambda: arr * xs, lambda: xs * arr, lambda: arr + xs, lambda: xs - arr):
        with pytest.raises(TypeError, match="ambiguous"):
            expr()


def test_1234_the_dispatch_sentinel_is_a_valueerror_for_wraps_other_callers(model):
    """``_wrap`` is not only called by the operators, so its escape must read well.

    The elementwise branch is reached by catching ``_ObjectArrayOperand`` raised
    from ``_wrap``, which keeps the type test off the scalar-literal path (it is
    ~3 of the 35 calls a constraint row makes, and an ``isinstance`` there
    measured +2.2% at 3.3 sigma). But ``dm.sum``, ``dm.prod`` and constraint
    bodies call ``_wrap`` too, and an uncaught escape from one of those must
    still look like the ``ValueError`` numpy used to raise -- with a better
    message, not a stranger type.
    """
    assert issubclass(core._ObjectArrayOperand, ValueError)
    with pytest.raises(ValueError, match="object-dtype array"):
        core._wrap(np.array([model.continuous("q", lb=0, ub=1)], dtype=object))


def test_1234_a_shape_mismatch_is_still_rejected_not_reshaped(model):
    """The ``try`` must not swallow ``BinaryOp``'s own shape guard.

    ``BinaryOp`` raises ``ValueError`` from ``_broadcast_shapes`` on incompatible
    operands. If the ``try`` wrapped the construction rather than just ``_wrap``,
    that rejection would be caught and rerouted into the elementwise branch,
    turning a model the layer correctly refuses into a silently reshaped one.
    """
    a = model.continuous("a", shape=(3,), lb=0, ub=1)
    with pytest.raises(ValueError):
        a * np.ones((4,))


def test_1234_the_neighbouring_cases_still_behave_as_before(model):
    """The table in the issue -- these worked and must keep working.

    In particular a *numeric* ndarray must still collapse into ONE ``BinaryOp``
    rather than fanning out elementwise: that collapse is the entire point of
    ``__array_ufunc__ = None`` and is what the array-valued-body idiom rests on.
    """
    x = model.continuous("x", lb=0.1, ub=2.0)
    y = model.continuous("y", lb=0.1, ub=2.0)
    arr = np.array([x, y], dtype=object)
    checks = 0

    assert isinstance(x * np.array([1.0, 2.0]), core.BinaryOp)
    assert isinstance(np.array([1.0, 2.0]) * x, core.BinaryOp)
    checks += 2

    for neighbour in (arr * arr, arr + 1.0, np.eye(2) @ arr):
        assert isinstance(neighbour, np.ndarray) and neighbour.dtype == object
        checks += 1

    assert isinstance(np.sum(arr), core.Expression)
    checks += 1
    assert checks == 6


def test_1234_comparisons_are_deliberately_not_covered(model):
    """``arr <= x`` stays unsupported because ``arr <= 1.0`` cannot work either.

    NumPy coerces each element of a comparison result to ``bool`` and a
    ``Constraint`` has no truth value, so there is no working neighbour here for
    the elementwise branch to be consistent with. Pinned so a later change that
    makes one of the two work has to think about the other.
    """
    x = model.continuous("x", lb=0.1, ub=2.0)
    y = model.continuous("y", lb=0.1, ub=2.0)
    arr = np.array([x, y], dtype=object)
    with pytest.raises(TypeError, match="truth value"):
        _ = arr <= 1.0


def test_1234_an_object_array_result_is_usable_in_a_model(model):
    """End to end: the fix has to produce a model that actually solves."""
    x = model.continuous("x", shape=(3,), lb=0.5, ub=4.0)
    t = model.continuous("t", lb=0.5, ub=2.0)
    terms = np.array([x[i] for i in range(3)], dtype=object)
    scaled = terms * t  # the comprehension idiom from the issue
    for i, body in enumerate(scaled):
        model.subject_to(body <= 6.0, name=f"c{i}")
    model.minimize(dm.sum([-b for b in scaled]))
    result = model.solve()
    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(-18.0, rel=1e-4)


# ─────────────────────────────────────────────────────────────
# #1235 -- reductions on a shaped expression
# ─────────────────────────────────────────────────────────────


def test_1235_sum_exists_and_is_a_single_n_ary_node(model):
    xs = model.continuous("xs", shape=(200,), lb=0, ub=1)
    node = xs.sum()
    assert isinstance(node, core.SumExpression)
    assert node.operand is xs and node.axis is None


@pytest.mark.parametrize("n", [4, 50, 200, 1000])
def test_1235_sum_depth_does_not_grow_with_n(model, n):
    """The depth claim, which is the point of the issue.

    The builtin ``sum()`` folds left: depth grows 1:1 with the vector length
    (measured 202 for n=200 in the issue). ``.sum()`` must stay flat.
    """
    v = model.continuous(f"v{n}", shape=(n,), lb=0, ub=1)
    assert _depth(v.sum()) == 2
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * n + 1000))
    assert _depth(sum(v)) == n + 2, "builtin sum is still the left-deep fold"


def test_1235_sum_agrees_numerically_with_the_builtin_fold():
    """Same model, two spellings, same certified objective."""

    def build(vectorised):
        m = Model("agree")
        v = m.continuous("v", shape=(5,), lb=0.5, ub=3.0)
        m.subject_to(v.sum() <= 7.0 if vectorised else sum(v) <= 7.0, name="cap")
        m.minimize(-(v.sum() if vectorised else sum(v)))
        return m.solve()

    flat, folded = build(True), build(False)
    assert flat.status == folded.status
    assert flat.objective == pytest.approx(folded.objective, abs=1e-9)
    assert flat.objective == pytest.approx(-7.0, rel=1e-6)


def test_1235_axis_reduction_keeps_the_other_axes(model):
    """``axis=k`` is one row per surviving element, not one fused row (#1160)."""
    X = model.continuous("X", shape=(2, 3), lb=0.5, ub=4.0)
    node = X.sum(axis=1)
    assert isinstance(node, core.SumExpression) and node.axis == 1
    model.subject_to(node <= 9.0, name="c")
    model.minimize(X.sum())
    assert sum(c.size if hasattr(c, "size") else 1 for c in model._constraints) >= 1
    # The row count is what distinguishes an axis reduction from a full one.
    from discopt._relax.scalarize import sum_is_full_reduction, sum_result_shape

    assert sum_result_shape(node) == (2,)
    assert not sum_is_full_reduction(node)
    assert sum_is_full_reduction(X.sum())


def test_1235_method_matches_the_module_level_function(model):
    """``xs.sum()`` is ``dm.sum(xs)`` -- one node type, not a second dialect."""
    xs = model.continuous("xs", shape=(4,), lb=0, ub=1)
    by_method, by_function = xs.sum(), dm.sum(xs)
    assert type(by_method) is type(by_function)
    assert by_method.operand is by_function.operand
    assert by_method.axis == by_function.axis


def test_1235_prod_and_mean_exist(model):
    xs = model.continuous("xs", shape=(4,), lb=0.5, ub=2.0)
    assert isinstance(xs.prod(), core.FunctionCall)
    assert xs.prod().func_name == "prod"
    mean = xs.mean()
    assert isinstance(mean, core.BinaryOp) and mean.op == "/"
    assert isinstance(mean.left, core.SumExpression)
    assert float(mean.right.value) == 4.0


def test_1235_mean_divides_by_the_reduced_count_only(model):
    X = model.continuous("X", shape=(2, 3), lb=0.5, ub=4.0)
    assert float(X.mean().right.value) == 6.0
    assert float(X.mean(axis=0).right.value) == 2.0
    assert float(X.mean(axis=1).right.value) == 3.0
    assert float(X.mean(axis=(0, 1)).right.value) == 6.0


def test_1235_mean_solves_to_the_right_number():
    m = Model("mean")
    v = m.continuous("v", shape=(4,), lb=0.5, ub=3.0)
    m.subject_to(v.mean() <= 2.0, name="cap")
    m.minimize(-v.sum())
    r = m.solve()
    assert r.status in ("optimal", "feasible")
    assert r.objective == pytest.approx(-8.0, rel=1e-6)


def test_1235_numpy_reductions_now_dispatch_to_the_methods(model):
    """``np.sum(xs)`` was refused by the ufunc opt-out; the method opens it.

    numpy's ``_wrapreduction`` looks for a same-named method on a non-ndarray, so
    defining ``.sum`` is what makes the NumPy habit transfer.
    """
    xs = model.continuous("xs", shape=(4,), lb=0, ub=1)
    assert isinstance(np.sum(xs), core.SumExpression)
    assert isinstance(np.prod(xs), core.FunctionCall)
    assert isinstance(np.mean(xs), core.BinaryOp)


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda v: v.sum(out=np.zeros(1)), "no storage"),
        (lambda v: v.sum(dtype=np.float32), "no accumulation dtype"),
        (lambda v: np.sum(v, keepdims=True), "unsupported keyword"),
        (lambda v: v.prod(axis=0), "full reduction"),
    ],
)
def test_1235_unsupportable_numpy_keywords_are_refused_by_name(model, call, match):
    """Refuse loudly rather than accept-and-ignore.

    ``out=`` asks for a write into a buffer and ``dtype=`` for an accumulation
    type; an expression has neither, so honouring them is impossible and
    ignoring them would hand back a wrong answer, not a slow one.
    """
    v = model.continuous("v", shape=(4,), lb=0, ub=1)
    with pytest.raises((TypeError, NotImplementedError), match=match):
        call(v)


def test_1235_mean_refuses_an_unknown_shape_rather_than_guessing(model):
    """``n`` has to be a literal in the DAG; a matmul result has no static shape."""
    xs = model.continuous("xs", shape=(4,), lb=0, ub=1)
    with pytest.raises(TypeError, match="statically known shape"):
        (np.eye(4) @ xs).mean()


def test_1235_mean_validates_the_axis(model):
    X = model.continuous("X", shape=(2, 3), lb=0, ub=1)
    with pytest.raises(np.exceptions.AxisError):
        X.mean(axis=5)
    with pytest.raises(ValueError, match="duplicate axis"):
        X.mean(axis=(0, 0))


def test_1235_reshape_stays_absent_by_design(model):
    """Holding the documented position, not an oversight.

    ``docs/notebooks/modeling_guide.ipynb`` tells the reader to declare the shape
    they want *because* ``Variable`` has no ``.reshape``; issue #1235 raised the
    shape manipulators as a "consider", and reversing a documented design
    position is not part of a bug fix. Pinned here next to the reductions so the
    two decisions are read together.
    """
    x = model.continuous("x", shape=(6,), lb=0.5, ub=4.0)
    for absent in ("reshape", "flatten", "T"):
        assert not hasattr(x, absent), absent
