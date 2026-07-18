"""Regression tests for the flow-aware integer-multilinear reformulation
(issue #707, ``DISCOPT_INTEGER_MULTILINEAR_REFORM``).

The pass exact-linearizes products of >=3 variable factors where every factor
but at most one is integer/binary-valued (declared or implied) — e.g. ex1252's
objective ``(c + 1800*x15)*x0*x3*x18`` with integer flow factors ``x0,x3`` and a
0/1 indicator ``x18``. Each integer factor is binary-expanded and the resulting
binary product is lifted to its exact hull (an n-ary AND, plus one big-M product
for the single continuous factor), replacing the loose term-wise trilinear
McCormick envelope with the per-integer-level exact envelope.

The invariant under test is **soundness**: the rewrite is an exact algebraic
identity (value-preserving), so it can only tighten — never invalidate — the dual
bound. A separate slow test locks that it lifts ex1252 off its 5134 floor.
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.integer_product_reform import (
    extend_initial_point,
    has_integer_multilinear_reformulation_work,
    reformulate_integer_multilinear,
)
from discopt.modeling.core import BinaryOp, Constant, IndexExpression, UnaryOp, Variable

_DATA = Path(__file__).parent / "data" / "minlplib"
_EX1252_OPT = 128893.74


def _flat_offsets(model):
    off, o = {}, 0
    for v in model._variables:
        off[v._index] = o
        o += v.size
    return off, o


def _eval(expr, x, off):
    """Evaluate an expression DAG at a flat point ``x`` (offsets in ``off``)."""
    if isinstance(expr, Constant):
        return float(expr.value)
    if isinstance(expr, Variable):
        return float(x[off[expr._index]])
    if isinstance(expr, IndexExpression):
        v = getattr(expr, "variable", None) or getattr(expr, "var", None)
        idx = getattr(expr, "index", 0)
        return float(x[off[v._index] + (idx if isinstance(idx, int) else 0)])
    if isinstance(expr, UnaryOp):
        a = _eval(expr.operand, x, off)
        return {"neg": lambda z: -z, "exp": math.exp, "log": math.log,
                "sqrt": math.sqrt, "abs": abs}[expr.op](a)
    if isinstance(expr, BinaryOp):
        a, b = _eval(expr.left, x, off), _eval(expr.right, x, off)
        if expr.op == "+":
            return a + b
        if expr.op == "-":
            return a - b
        if expr.op == "*":
            return a * b
        if expr.op == "/":
            return a / b if b else math.inf
        if expr.op == "**":
            return a ** b
        raise ValueError(f"unhandled op {expr.op}")
    raise TypeError(type(expr))


@pytest.mark.correctness
def test_ex1252_multilinear_reform_is_value_preserving():
    """The reform must be an **exact algebraic identity** on the objective: for any
    point with integer flow factors and 0/1 indicators, the reformed objective
    (aux columns reconstructed from the originals) equals the original objective.
    This is the soundness bedrock — a mismatch would mean the rewrite changes the
    problem, not just its relaxation."""
    m = dm.from_nl(str(_DATA / "ex1252.nl"))
    r = reformulate_integer_multilinear(m)
    assert r is not m, "expected a rewritten model for ex1252"
    o_off, _ = _flat_offsets(m)
    r_off, _ = _flat_offsets(r)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(500):
        x = np.zeros(len(m._variables))
        for i in range(6):  # x0..x5 integer flow factors in {0,1,2,3}
            x[i] = rng.integers(0, 4)
        for i in range(15, 18):  # x15..x17 continuous cost slopes
            x[i] = rng.uniform(0, 80)
        for i in range(18, 21):  # x18..x20 indicators in {0,1}
            x[i] = rng.integers(0, 2)
        xf = extend_initial_point(r, x)
        assert xf is not None, "warm-start extension failed on an integer point"
        vo = _eval(m._objective.expression, x, o_off)
        vr = _eval(r._objective.expression, xf, r_off)
        max_err = max(max_err, abs(vo - vr))
    assert max_err < 1e-5, f"reform not value-preserving: max abs err {max_err}"


@pytest.mark.correctness
def test_ex1252_reform_eliminates_integer_multilinear_terms():
    """After the reform, ex1252 carries no trilinear/multilinear product terms —
    the integer-multilinear objective terms are gone. The residual nonlinearity is
    the genuine *continuous* cubic cost rows, which the pass correctly leaves for
    the spatial relaxation."""
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = dm.from_nl(str(_DATA / "ex1252.nl"))
    assert has_integer_multilinear_reformulation_work(m)
    r = reformulate_integer_multilinear(m)
    nl = classify_nonlinear_terms(r)
    assert not nl.trilinear, f"trilinear terms remain: {nl.trilinear}"
    assert not nl.multilinear, f"multilinear terms remain: {nl.multilinear}"


@pytest.mark.correctness
def test_reform_noop_on_pure_continuous_product():
    """A product with two continuous factors is a genuine continuous nonlinearity,
    not exact-linearizable — the pass must be a no-op (returns the same model) so it
    never bloats or perturbs models outside its class."""
    m = dm.Model("cont")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    z = m.continuous("z", lb=0.0, ub=5.0)
    m.minimize(x * y * z)
    assert not has_integer_multilinear_reformulation_work(m)
    assert reformulate_integer_multilinear(m) is m


@pytest.mark.correctness
def test_reform_generic_integer_trilinear_is_exact_and_sound():
    """General (non-ex1252) class check: a pure integer trilinear whose continuous
    McCormick relaxation is loose. The reform is value-preserving and the model
    solves to the true integer optimum, sound throughout.

    ``max a*b*c`` s.t. ``a+b+c <= 3`` over ``a,b,c in {0,1,2,3}`` has integer
    optimum ``1`` at ``(1,1,1)`` while the continuous hull reaches ``> 1`` — the
    gap the exact reform closes."""
    m = dm.Model("tri")
    a = m.integer("a", lb=0, ub=3)
    b = m.integer("b", lb=0, ub=3)
    c = m.integer("c", lb=0, ub=3)
    m.subject_to(a + b + c <= 3)
    m.minimize(-(a * b * c))  # maximize a*b*c

    assert has_integer_multilinear_reformulation_work(m)
    r = reformulate_integer_multilinear(m)
    assert r is not m

    # Value-preserving on the product over the integer box.
    o_off, _ = _flat_offsets(m)
    r_off, _ = _flat_offsets(r)
    for av in range(4):
        for bv in range(4):
            for cv in range(4):
                x = np.array([av, bv, cv], dtype=float)
                xf = extend_initial_point(r, x)
                assert xf is not None
                assert abs(_eval(m._objective.expression, x, o_off)
                           - _eval(r._objective.expression, xf, r_off)) < 1e-9

    os.environ["DISCOPT_INTEGER_MULTILINEAR_REFORM"] = "1"
    try:
        res = m.solve(time_limit=30)
    finally:
        os.environ.pop("DISCOPT_INTEGER_MULTILINEAR_REFORM", None)
    assert res.objective is not None
    assert res.objective >= -1.0 - 1e-4, f"incumbent {res.objective} below true optimum -1"
    if res.bound is not None and math.isfinite(res.bound):
        assert res.bound <= -1.0 + 1e-4, f"UNSOUND bound {res.bound} above true optimum -1"


@pytest.mark.slow
@pytest.mark.correctness
def test_ex1252_multilinear_reform_sound_and_lifts_bound():
    """End-to-end on ex1252: with the flag on, the dual bound is sound (<= the true
    optimum) and lifts well past the structural 5134 floor that the term-wise
    trilinear McCormick envelope plateaus at. Certification of the full instance
    additionally needs the residual continuous-cubic barrier to close (spatial
    branching); here we lock soundness + the bound lift, which is the #707 gain."""
    os.environ["DISCOPT_INTEGER_MULTILINEAR_REFORM"] = "1"
    try:
        m = dm.from_nl(str(_DATA / "ex1252.nl"))
        res = m.solve(time_limit=60)
    finally:
        os.environ.pop("DISCOPT_INTEGER_MULTILINEAR_REFORM", None)
    assert res.bound is not None and math.isfinite(res.bound)
    # Sound: a valid dual bound never exceeds the true optimum.
    assert res.bound <= _EX1252_OPT + 1e-2, (
        f"ex1252 multilinear-reform UNSOUND dual bound {res.bound} > optimum {_EX1252_OPT}"
    )
    # Any incumbent found is feasible (>= optimum for this minimize).
    if res.objective is not None and math.isfinite(res.objective):
        assert res.objective >= _EX1252_OPT - 1e-2
    # The tightening: the bound clears the 5134 floor the term-wise envelope stalls at.
    assert res.bound > 5134.5, (
        f"ex1252 multilinear-reform bound {res.bound} did not lift past the 5134 floor"
    )


@pytest.mark.correctness
def test_flag_with_unextendable_seed_does_not_crash():
    """A user warm start that cannot be extended across the big-M lift (e.g. an
    off-integer value on an expanded factor) must be dropped, not left as a
    length-mismatched vector against the grown model — which would crash the
    downstream integrality check on the spatial path."""
    m = dm.Model("tri")
    a = m.integer("a", lb=0, ub=3)
    b = m.integer("b", lb=0, ub=3)
    c = m.integer("c", lb=0, ub=3)
    m.subject_to(a + b + c <= 3)
    m.minimize(-(a * b * c))
    os.environ["DISCOPT_INTEGER_MULTILINEAR_REFORM"] = "1"
    try:
        # a=0.5 is fractional on an expanded integer factor → not extendable.
        res = m.solve(time_limit=20, initial_solution={a: 0.5, b: 1.0, c: 1.0})
    finally:
        os.environ.pop("DISCOPT_INTEGER_MULTILINEAR_REFORM", None)
    # Solve still completes soundly (seed simply dropped).
    assert res.objective is None or res.objective >= -1.0 - 1e-4


@pytest.mark.slow
@pytest.mark.correctness
def test_flag_does_not_regress_easy_instance():
    """The spatial-path blowup guard must keep the flag from *regressing* an
    already-tractable instance. ``nvs01`` (3 variables) certifies to 12.4697 on the
    plain spatial path; its integer-multilinear reform balloons it to ~200 columns,
    so the guard rejects the non-pure-MILP reform and leaves the original model —
    the flag-on solve must reach the same optimum, not stall below it."""
    nvs01_opt = 12.46966882
    os.environ["DISCOPT_INTEGER_MULTILINEAR_REFORM"] = "1"
    try:
        m = dm.from_nl(str(_DATA / "nvs01.nl"))
        res = m.solve(time_limit=30)
    finally:
        os.environ.pop("DISCOPT_INTEGER_MULTILINEAR_REFORM", None)
    # Sound in any case.
    if res.bound is not None and math.isfinite(res.bound):
        assert res.bound <= nvs01_opt + 1e-2, f"nvs01 UNSOUND bound {res.bound}"
    # No regression: the guard preserves the fast certification (bound reaches opt).
    assert res.bound is not None and res.bound >= nvs01_opt - 1e-2, (
        f"nvs01 regressed under the flag: bound {res.bound} < optimum {nvs01_opt}"
    )
