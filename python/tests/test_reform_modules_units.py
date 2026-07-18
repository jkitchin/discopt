"""Unit tests for the two reformulation passes (#87 coverage restoration).

Covers ``discopt._jax.binary_multilinear_reform`` (Fortet/Glover exact
linearization of pure-binary multilinear models, issue #187) and
``discopt._jax.gdp_reformulate`` (big-M / hull / mbigm GDP lowering).

Every test asserts a *semantic property* of the documented behavior rather
than replaying implementation details:

- expansion / substitution / coefficient extraction must reproduce the exact
  value of the original expression at sampled (or exhaustively enumerated)
  points;
- interval enclosures must contain every sampled value (soundness), with
  exactness asserted where the docstring promises it (even-power straddle);
- a reformulated model must have the same projected feasible set and the same
  objective value as the original at every enumerated integer point, and the
  same certified optimum under a full solve;
- validation paths must refuse loudly with the documented errors
  (unbounded big-M, infinite SOS bounds, non-literal logical leaves).

Models are tiny (1-6 scalar variables); feasibility checks enumerate all
binary assignments instead of solving, so almost everything is sub-second.
The few full solves are additionally marked ``smoke``.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax import binary_multilinear_reform as B
from discopt._jax import gdp_reformulate as G
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    IndexExpression,
    LogicalAnd,
    LogicalNot,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _wrap,
)

pytestmark = pytest.mark.unit

# The exact LP oracle and the MILP engine both need the Rust extension; under
# coverage instrumentation in some environments the binding cannot load, and
# the modules under test then take their documented fallback paths.
try:
    from discopt.solvers.lp_backend import get_exact_lp_solver

    _HAS_EXACT_LP = get_exact_lp_solver() is not None
except ImportError:  # pragma: no cover - defensive
    _HAS_EXACT_LP = False
needs_exact_lp = pytest.mark.skipif(
    not _HAS_EXACT_LP, reason="exact LP oracle (Rust simplex) unavailable"
)


# ---------------------------------------------------------------------------
# Shared evaluation helpers (reference semantics for expression DAGs)
# ---------------------------------------------------------------------------


def _eval(expr, env):
    """Reference numeric evaluation of an expression DAG at ``env`` (values by
    variable name). Independent of any solver path, so agreement between a
    reformulated body and the original body is a genuine semantic check."""
    if isinstance(expr, Constant):
        return np.asarray(expr.value, dtype=np.float64)
    if isinstance(expr, Parameter):
        return np.asarray(expr.value, dtype=np.float64)
    if isinstance(expr, Variable):
        return np.asarray(env[expr.name], dtype=np.float64)
    if isinstance(expr, IndexExpression):
        return np.asarray(_eval(expr.base, env))[expr.index]
    if isinstance(expr, BinaryOp):
        left, right = _eval(expr.left, env), _eval(expr.right, env)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return left**right
        raise AssertionError(f"unexpected binary op {expr.op!r}")
    if isinstance(expr, UnaryOp):
        val = _eval(expr.operand, env)
        if expr.op == "neg":
            return -val
        if expr.op == "abs":
            return np.abs(val)
        raise AssertionError(f"unexpected unary op {expr.op!r}")
    if isinstance(expr, FunctionCall):
        fn = getattr(np, expr.func_name)
        return fn(*[_eval(a, env) for a in expr.args])
    if isinstance(expr, SumExpression):
        return np.sum(_eval(expr.operand, env))
    if isinstance(expr, SumOverExpression):
        return sum(np.sum(_eval(t, env)) for t in expr.terms)
    raise AssertionError(f"unexpected node {type(expr).__name__}")


def _violation(con: Constraint, env) -> float:
    val = float(np.asarray(_eval(con.body, env)).reshape(()))
    rhs = float(con.rhs)
    if con.sense == "<=":
        return max(0.0, val - rhs)
    if con.sense == ">=":
        return max(0.0, rhs - val)
    return abs(val - rhs)


def _feasible(constraints, env, tol: float = 1e-7) -> bool:
    return all(_violation(c, env) <= tol for c in constraints)


def _env_from_flat(model, flat):
    """Map a flat point (all-scalar variable models) to a name -> value env."""
    assert all(v.size == 1 for v in model._variables)
    return {v.name: float(flat[i]) for i, v in enumerate(model._variables)}


def _poly_value(poly, ctx, env):
    """Evaluate a ``{monomial: coef}`` expansion; keys are (var_index, elem)."""
    by_key = {}
    for key, ref in ctx.refs.items():
        leaf = B._leaf_ref(ref)
        assert leaf is not None
        var, elem, _ = leaf
        by_key[key] = float(np.asarray(env[var.name]).flat[elem])
    total = 0.0
    for mono, coef in poly.items():
        term = coef
        for key in mono:
            term *= by_key[key]
        total += term
    return total


# ---------------------------------------------------------------------------
# binary_multilinear_reform: model builders
# ---------------------------------------------------------------------------


def build_autocorr(n, k_max, objvar_sense=None):
    """Bernasconi autocorrelation: minimize sum_k (sum_i s_i s_{i+k})**2 with
    s = 2b - 1 over {0,1}-bounded INTEGER b (from_nl's typing, issue #187)."""
    m = Model(name=f"ac{n}_{k_max}")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(n)]
    s = [2 * bi - 1 for bi in b]
    total = None
    for k in range(1, k_max + 1):
        ck = None
        for i in range(n - k):
            t = s[i] * s[i + k]
            ck = t if ck is None else ck + t
        term = ck * ck
        total = term if total is None else total + term
    if objvar_sense is None:
        m.minimize(total)
    else:
        tau = m.continuous("objvar")  # default bounds are the free sentinel
        m.subject_to(tau == total if objvar_sense == "==" else tau >= total)
        m.minimize(tau)
    return m, b


def autocorr_energy(bits, k_max):
    n = len(bits)
    s = [2 * x - 1 for x in bits]
    return float(sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, k_max + 1)))


# ---------------------------------------------------------------------------
# binary_multilinear_reform: syntactic gate + leaf helpers
# ---------------------------------------------------------------------------


def test_gate_witness_detection():
    """The gate fires exactly on a supported degree>=3 all-binary product and
    stays off for degree-2, non-binary factors, and unsupported structure
    (False must always be safe: the pass is simply skipped)."""
    m = Model("gate")
    b = [m.binary(f"b{i}") for i in range(3)]
    x = m.continuous("x", lb=0.0, ub=1.0)

    def with_obj(expr):
        mm = Model("g")
        mm._variables = list(m._variables)
        mm.minimize(expr)
        return mm

    assert B.has_binary_multilinear_work(with_obj(b[0] * b[1] * b[2]))
    assert B.has_binary_multilinear_work(with_obj(b[0] ** 3))  # b^3 has degree 3
    assert not B.has_binary_multilinear_work(with_obj(b[0] * b[1]))  # degree 2 only
    assert not B.has_binary_multilinear_work(with_obj(b[0] * b[1] * x))  # non-binary factor
    assert not B.has_binary_multilinear_work(with_obj(dm.sin(b[0] * b[1] * b[2])))  # unsupported
    assert not B.has_binary_multilinear_work(with_obj(b[0] ** b[1]))  # variable exponent
    assert not B.has_binary_multilinear_work(with_obj(b[0] ** 2.5))  # fractional exponent

    empty = Model("noobj")
    empty.binary("b")
    assert not B.has_binary_multilinear_work(empty)  # no objective -> nothing to gain


def test_gate_witness_in_constraint_body():
    """A degree>=3 binary product in a *constraint* is a witness too."""
    m = Model("gate_con")
    b = [m.binary(f"b{i}") for i in range(3)]
    m.minimize(b[0])
    m.subject_to(b[0] * b[1] * b[2] <= 0)
    assert B.has_binary_multilinear_work(m)


def test_scalar_constant_rejects_vector():
    with pytest.raises(B._Unsupported, match="non-scalar"):
        B._scalar_constant(Constant(np.array([1.0, 2.0])))
    assert B._scalar_constant(Constant(3.5)) == 3.5


def test_leaf_ref_matrix_element_and_vector():
    m = Model("leaf")
    w = m.binary("w", shape=(2, 2))
    z = m.binary("z")
    ref = B._leaf_ref(w[1, 1])
    # (1, 1) in a row-major (2, 2) block is flat element 3.
    assert ref is not None and ref[0] is w and ref[1] == 3
    assert B._leaf_ref(z) == (z, 0, z)
    assert B._leaf_ref(w) is None  # whole array is not one scalar element


# ---------------------------------------------------------------------------
# binary_multilinear_reform: expansion semantics
# ---------------------------------------------------------------------------


def test_expansion_matches_evaluation_at_all_integer_points():
    """The multilinear expansion (with b**2 == b collapse) is *exact* at every
    integer point of the box — checked against reference DAG evaluation for an
    expression exercising powers, division by constants, parameters, negation,
    SumExpression and SumOverExpression."""
    m = Model("exp")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    y = m.integer("y", lb=-2, ub=3)
    p = m.parameter("p", value=3.0)
    expr = (
        (b[0] + b[1]) ** 3
        - (b[2] * p) / 2.0
        + dm.sum([b[0] * b[1], -b[1], 1.5])
        + dm.sum(b[2])
        + y
        + b[0] ** 0  # k = 0 power expands to the constant 1
    )
    ctx = B._ExpandCtx()
    poly = B._expand_to_multilinear(expr, ctx)
    for bits in itertools.product([0, 1], repeat=3):
        for yv in (-2, 0, 3):
            env = {"b0": bits[0], "b1": bits[1], "b2": bits[2], "y": yv, "p": 3.0}
            expect = float(_eval(expr, env))
            assert _poly_value(poly, ctx, env) == pytest.approx(expect, abs=1e-12)


def test_expansion_rejects_inexact_structures():
    """Everything the pass cannot linearize *exactly* must raise (and hence
    fall back), never silently approximate: variable divisors/exponents,
    fractional or huge exponents, repeated non-binary factors (x**2 -> x would
    be a rewrite), and transcendental calls."""
    m = Model("rej")
    b0 = m.binary("b0")
    b1 = m.binary("b1")
    x = m.continuous("x", lb=0.0, ub=2.0)
    for bad in (
        b0 / b1,  # division by a non-constant
        b0 / 0.0,  # division by zero
        b0**b1,  # non-constant exponent
        b0**2.5,  # fractional exponent
        b0**20,  # exponent above _MAX_POW
        x * x,  # repeated non-binary factor
        dm.sin(b0),  # unsupported node
    ):
        with pytest.raises(B._Unsupported):
            B._expand_to_multilinear(bad, B._ExpandCtx())


def test_collect_addends_reconstructs_value():
    """_collect_addends splits a +/- spine into (const, [(coef, node)]) such
    that const + sum(coef * node) reproduces the expression exactly."""
    m = Model("add")
    b0, b1, b2 = (m.binary(f"b{i}") for i in range(3))
    expr = (
        2.0 * (b0 * b1)
        - 3.0 * b2
        + (b0 * b1) * 4.0
        + b1 / 2.0
        - (-b0)
        + dm.sum([b1, 0.5])
        + dm.sum(b2)
    )
    const, addends = B._collect_addends(expr)
    for bits in itertools.product([0, 1], repeat=3):
        env = {"b0": bits[0], "b1": bits[1], "b2": bits[2]}
        got = const + sum(coef * float(_eval(node, env)) for coef, node in addends)
        assert got == pytest.approx(float(_eval(expr, env)), abs=1e-12)


def test_square_base_detection():
    m = Model("sq")
    b0, b1 = m.binary("b0"), m.binary("b1")
    e = b0 + b1
    assert B._square_base(e**2) is e
    assert B._square_base(BinaryOp("*", e, e)) is e  # DAG-shared E*E
    assert B._square_base(e**3) is None
    assert B._square_base(e**2.5) is None
    assert B._square_base(b0 * b1) is None  # distinct factors
    # A non-scalar exponent constant is handled (returns None, no crash).
    assert B._square_base(BinaryOp("**", e, Constant(np.array([2.0, 2.0])))) is None


# ---------------------------------------------------------------------------
# binary_multilinear_reform: intervals, integer ranges, attainable grids
# ---------------------------------------------------------------------------


def _expand(expr):
    ctx = B._ExpandCtx()
    return B._expand_to_multilinear(expr, ctx), ctx


def test_poly_int_range_values_and_abstentions():
    m = Model("rng")
    b0, b1 = m.binary("b0"), m.binary("b1")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.integer("y", lb=-2, ub=3)
    yf = m.integer("yf", lb=0, ub=3e9)  # range overflows the 2**31 safety cap

    # 2*b0 + 3*b1 - 1 over b in {0,1}^2 attains exactly [-1, 4].
    poly, ctx = _expand(2 * b0 + 3 * b1 - 1)
    assert B._poly_int_range(poly, ctx) == (-1, 4)
    # Bounded general integer appearing linearly widens the range soundly.
    poly, ctx = _expand(y + b0)
    assert B._poly_int_range(poly, ctx) == (-2, 4)
    # Abstentions (return None -> caller falls back to flat expansion):
    poly, ctx = _expand(0.5 * b0)  # fractional coefficient
    assert B._poly_int_range(poly, ctx) is None
    poly, ctx = _expand(x + b0)  # continuous variable
    assert B._poly_int_range(poly, ctx) is None
    poly, ctx = _expand(yf + b0)  # interval magnitude above 2**31
    assert B._poly_int_range(poly, ctx) is None
    poly, ctx = _expand(y * b0 * b1)  # degree>=2 monomial with non-binary factor
    assert B._poly_int_range(poly, ctx) is None


def test_interval_of_dag_soundness():
    """The factored-DAG enclosure must contain every sampled value; the
    even-power branch is clamped to be nonnegative; unsupported structure
    degrades to (-inf, inf), never to a wrong finite bound."""
    m = Model("ivl")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    b = m.binary("b")
    exprs = [(2 * b - 1) * x, x - b, x / 2.0, x**2, x**3, -x]
    xs = np.linspace(-2.0, 3.0, 21)
    for expr in exprs:
        lo, hi = B._interval_of_dag(expr)
        for xv in xs:
            for bv in (0.0, 1.0):
                val = float(_eval(expr, {"x": xv, "b": bv}))
                assert lo - 1e-9 <= val <= hi + 1e-9
    # Even power over a straddling box is clamped at 0 from below.
    lo, hi = B._interval_of_dag(x**2)
    assert lo == 0.0 and hi >= 9.0
    # Division by a non-degenerate interval and unsupported calls -> top.
    assert B._interval_of_dag(x / x) == (-np.inf, np.inf)
    assert B._interval_of_dag(dm.sin(x)) == (-np.inf, np.inf)
    # inf * 0 products are NaN: they must degrade to top, never to a wrong
    # finite bound.
    z = m.continuous("z", lb=-np.inf, ub=np.inf)
    assert B._interval_of_dag(BinaryOp("*", Constant(0.0), z)) == (-np.inf, np.inf)
    lo, hi = B._interval_of_dag(z * b)
    assert lo == -np.inf and hi == np.inf


def test_attainable_grid_snapping():
    b0 = frozenset([(0, 0)])
    b1 = frozenset([(1, 0)])
    empty = frozenset()
    # Coefficients {2, 4}, constant 1: attainable values are odd, spacing 2;
    # -5 and 7 are both odd so the interval only gains the g=2 step.
    assert B._attainable_grid({b0: 2.0, b1: 4.0, empty: 1.0}, -5, 7) == (-5, 7, 2)
    # Even bounds snap inward to the odd grid.
    assert B._attainable_grid({b0: 2.0, b1: 4.0, empty: 1.0}, -4, 6) == (-3, 5, 2)
    # gcd 1: passthrough.
    assert B._attainable_grid({b0: 1.0}, 0, 1) == (0, 1, 1)
    assert B._attainable_grid({b0: 1.0}, 1, 0) is None  # empty interval
    # No attainable point inside [1, 1] for values ≡ 0 (mod 2) -> None.
    assert B._attainable_grid({b0: 2.0}, 1, 1) is None


# ---------------------------------------------------------------------------
# binary_multilinear_reform: objvar defining-row detection
# ---------------------------------------------------------------------------


def _objvar_model(sense, row, tau_bounds=(None, None), extra_row=False):
    m = Model("ov")
    b = m.integer("b", lb=0, ub=1)
    kwargs = {}
    if tau_bounds[0] is not None:
        kwargs["lb"] = tau_bounds[0]
    if tau_bounds[1] is not None:
        kwargs["ub"] = tau_bounds[1]
    tau = m.continuous("tau", **kwargs)
    m.subject_to(row(tau, b))
    if extra_row:
        m.subject_to(tau <= 10 * b)
    (m.minimize if sense == "min" else m.maximize)(tau)
    return m, tau, b


def test_detect_objvar_row_accepts_transmitting_row():
    """min tau with row ``tau >= g(b)`` bounds tau from below (its improving
    side) -> detected, with the constraint's normalized linear coefficient."""
    m, tau, b = _objvar_model("min", lambda t, b: t >= b)
    det = B._detect_objvar_row(m, 1.0)
    assert det is not None
    row_idx, a = det
    assert row_idx == 0 and a != 0.0
    # The reported coefficient must be tau's true linear coefficient in the row.
    body = m._constraints[0].body
    assert float(_eval(body, {"tau": 1.0, "b": 0.0}) - _eval(body, {"tau": 0.0, "b": 0.0})) == a


def test_detect_objvar_row_rejections():
    # Finite tau bounds could clip the surrogate -> refuse.
    m, _, _ = _objvar_model("min", lambda t, b: t >= b, tau_bounds=(-5.0, 5.0))
    assert B._detect_objvar_row(m, 1.0) is None
    # tau appearing in a second row -> refuse.
    m, _, _ = _objvar_model("min", lambda t, b: t >= b, extra_row=True)
    assert B._detect_objvar_row(m, 1.0) is None
    # Inequality on the non-improving side (min tau, tau <= g) -> refuse.
    m, _, _ = _objvar_model("min", lambda t, b: t <= b)
    assert B._detect_objvar_row(m, 1.0) is None
    # Same row under MAX (mu = -1) no longer transmits pressure -> refuse.
    m, _, _ = _objvar_model("max", lambda t, b: t >= b)
    assert B._detect_objvar_row(m, -1.0) is None
    # tau entering nonlinearly -> refuse.
    m, _, _ = _objvar_model("min", lambda t, b: t * t >= b)
    assert B._detect_objvar_row(m, 1.0) is None
    # Objective not a bare variable -> not the objvar convention.
    m = Model("nb")
    b = m.integer("b", lb=0, ub=1)
    tau = m.continuous("tau")
    m.subject_to(tau >= b)
    m.minimize(tau + b)
    assert B._detect_objvar_row(m, 1.0) is None


# ---------------------------------------------------------------------------
# binary_multilinear_reform: full reformulation exactness
# ---------------------------------------------------------------------------


def _cubic_model():
    m = Model("cubic")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    m.minimize(b[0] * b[1] * b[2] + 2 * b[0] - b[1])
    m.subject_to(b[0] + b[1] + b[2] <= 2)
    return m, b


def test_reformulation_vertex_exactness():
    """At every binary vertex, the extended point satisfies all auxiliary
    (Fortet) rows, the rebuilt constraint takes the original body's value, and
    the reformed objective equals the original objective — i.e. the MILP is an
    equivalent model, not a relaxation."""
    m, _ = _cubic_model()
    rm = B.reformulate_binary_multilinear(m)
    assert rm is not m
    orig_obj = m._objective.expression
    orig_con = m._constraints[0]
    for bits in itertools.product([0, 1], repeat=3):
        ext = B.extend_initial_point(rm, np.asarray(bits, dtype=float))
        assert ext is not None
        env = _env_from_flat(rm, ext)
        orig_env = {f"b{i}": bits[i] for i in range(3)}
        # All aux rows (Fortet z-rows) hold exactly at the extended point.
        assert _feasible(rm._constraints[1:], env, tol=1e-9)
        # The rebuilt original row is algebraically identical on the vertices.
        assert float(_eval(rm._constraints[0].body, env)) == pytest.approx(
            float(_eval(orig_con.body, orig_env)), abs=1e-9
        )
        assert rm._constraints[0].sense == orig_con.sense
        assert rm._constraints[0].rhs == orig_con.rhs
        # Objective value is preserved exactly.
        assert float(_eval(rm._objective.expression, env)) == pytest.approx(
            float(_eval(orig_obj, orig_env)), abs=1e-9
        )


def test_reformulation_objvar_secant_exactness():
    """`.nl` objvar form: setting tau to the true energy at each vertex yields
    a feasible point of the reformed MILP with objective exactly that energy —
    validity of the secant encoding (no original solution is cut, none is
    improved)."""
    n, k_max = 4, 2
    m, _ = build_autocorr(n, k_max, objvar_sense="==")
    rm = B.reformulate_binary_multilinear(m)
    assert rm is not m
    for bits in itertools.product([0, 1], repeat=n):
        e = autocorr_energy(bits, k_max)
        x0 = np.asarray(list(bits) + [e], dtype=float)  # vars: b0..b3, objvar
        ext = B.extend_initial_point(rm, x0)
        assert ext is not None
        env = _env_from_flat(rm, ext)
        assert _feasible(rm._constraints, env, tol=1e-9)
        assert float(_eval(rm._objective.expression, env)) == pytest.approx(e, abs=1e-9)


def test_reformulate_abstains_out_of_scope():
    """The pass must return the model object *unchanged* whenever it cannot
    fire exactly — never a partial rewrite."""
    # No objective.
    m = Model("noobj")
    m.binary("b")
    assert B.reformulate_binary_multilinear(m) is m
    # SOS structure would be silently dropped by a rebuild -> abstain.
    m, b = _cubic_model()
    m.sos1([b[0], b[1]])
    assert B.reformulate_binary_multilinear(m) is m
    # Degree-2-only binary models stay on the existing McCormick paths.
    m2 = Model("deg2")
    c = [m2.binary(f"b{i}") for i in range(2)]
    m2.minimize(c[0] * c[1])
    assert B.reformulate_binary_multilinear(m2) is m2
    # A continuous factor in a degree-3 monomial is not exactly linearizable.
    m3 = Model("cont")
    d = [m3.binary(f"b{i}") for i in range(2)]
    x = m3.continuous("x", lb=0.0, ub=1.0)
    m3.minimize(x * d[0] * d[1])
    assert B.reformulate_binary_multilinear(m3) is m3


def test_extend_initial_point_guards():
    m, _ = _cubic_model()
    rm = B.reformulate_binary_multilinear(m)
    # No reformulation metadata (plain model) -> None.
    assert B.extend_initial_point(m, np.zeros(3)) is None
    # Wrong length, non-finite, and fractional binaries -> None.
    assert B.extend_initial_point(rm, np.zeros(2)) is None
    assert B.extend_initial_point(rm, np.array([np.nan, 0.0, 0.0])) is None
    assert B.extend_initial_point(rm, np.array([0.4, 0.0, 0.0])) is None


# ---------------------------------------------------------------------------
# binary_multilinear_reform: heuristic incumbent
# ---------------------------------------------------------------------------


def test_heuristic_incumbent_finds_optimum_and_is_deterministic():
    """On an unconstrained pure-binary model the 1-flip heuristic point must
    (a) be reproducible, (b) satisfy every reformed row, and (c) reach the
    brute-force optimum on this 16-point instance."""
    n, k_max = 4, 3
    m, _ = build_autocorr(n, k_max)
    rm = B.reformulate_binary_multilinear(m)
    assert rm is not m
    pt = B.heuristic_incumbent(rm)
    assert pt is not None
    np.testing.assert_allclose(pt, B.heuristic_incumbent(rm))  # deterministic
    bits = [int(round(v)) for v in pt[:n]]
    assert all(v in (0, 1) for v in bits)
    env = _env_from_flat(rm, pt)
    assert _feasible(rm._constraints, env, tol=1e-9)
    best = min(autocorr_energy(cand, k_max) for cand in itertools.product([0, 1], repeat=n))
    assert autocorr_energy(bits, k_max) == pytest.approx(best)
    assert float(_eval(rm._objective.expression, env)) == pytest.approx(best, abs=1e-9)
    # No metadata on the original model -> the heuristic abstains.
    assert B.heuristic_incumbent(m) is None


def test_heuristic_incumbent_respects_fixed_bits():
    """A {0,1} variable pinned by its bounds (lb == ub == 1) must keep that
    value in the heuristic point, and the search still reaches the optimum of
    the free bits."""
    m = Model("fixed")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    bfix = m.integer("bfix", lb=1, ub=1)
    m.minimize(3 * b[0] * b[1] * b[2] * bfix - 2 * b[0] - b[1] + b[2])
    rm = B.reformulate_binary_multilinear(m)
    assert rm is not m
    pt = B.heuristic_incumbent(rm)
    assert pt is not None
    assert pt[3] == 1.0

    def obj(bits):
        return 3 * bits[0] * bits[1] * bits[2] * 1 - 2 * bits[0] - bits[1] + bits[2]

    best = min(obj(c) for c in itertools.product([0, 1], repeat=3))
    env = _env_from_flat(rm, pt)
    assert float(_eval(rm._objective.expression, env)) == pytest.approx(best, abs=1e-9)


# ---------------------------------------------------------------------------
# gdp_reformulate: interval bounds
# ---------------------------------------------------------------------------


def test_bound_expression_units_are_sound():
    """Interval enclosures for unary ops, function calls, and division must
    contain every sampled value; unknown structure must degrade to
    (-inf, inf), never to a wrong finite bound."""
    m = Model("bnd")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    finite_exprs = [
        UnaryOp("neg", x),
        UnaryOp("abs", x),
        UnaryOp("abs", y),  # one-signed interval: endpoint path
        x / y,
        dm.exp(x),
        dm.log(y),
        dm.sqrt(y),
        dm.sin(x),
        FunctionCall("abs", x),
    ]
    xs = np.linspace(-2.0, 3.0, 11)
    ys = np.linspace(1.0, 2.0, 5)
    for expr in finite_exprs:
        lo, hi = G._bound_expression(expr, m)
        for xv in xs:
            for yv in ys:
                val = float(_eval(expr, {"x": xv, "y": yv}))
                assert lo - 1e-9 <= val <= hi + 1e-9, expr
    # Exactness spot-checks for documented formulas.
    assert G._bound_expression(UnaryOp("abs", x), m) == (0.0, 3.0)
    assert G._bound_expression(x / y, m) == (-2.0, 3.0)
    # Divisor interval crossing zero, unknown ops, non-integer powers -> top.
    assert G._bound_expression(x / x, m) == (-np.inf, np.inf)
    assert G._bound_expression(UnaryOp("floor", x), m) == (-np.inf, np.inf)
    assert G._bound_expression(FunctionCall("tanh", x), m) == (-np.inf, np.inf)
    assert G._bound_expression(x**2.5, m) == (-np.inf, np.inf)
    # log of an interval touching <= 0 has no finite lower bound.
    lo, hi = G._bound_expression(dm.log(x), m)
    assert lo == -np.inf and hi == pytest.approx(np.log(3.0))
    # The "neg" function-call spelling mirrors the unary op: -[-2, 3] = [-3, 2].
    assert G._bound_expression(FunctionCall("neg", x), m) == (-3.0, 2.0)


def test_bound_expression_even_power_straddle_is_rigorous():
    """C-34/FR-1: for even p over a base straddling 0 the true range is
    [0, max(lb^p, ub^p)] — the endpoint-only range [lb^p, ub^p] misses the
    interior minimum at 0 and once produced invalid boxes."""
    m = Model("pow")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    assert G._bound_expression(x**4, m) == (0.0, 16.0)
    assert G._bound_expression(x**2, m) == (0.0, 4.0)
    # Sampled soundness: every value of x**4 on the box is inside the range.
    lo, hi = G._bound_expression(x**4, m)
    for xv in np.linspace(-1.0, 2.0, 31):
        assert lo - 1e-12 <= xv**4 <= hi + 1e-12
    # One-signed even power stays on the (exact) endpoint path.
    z = m.continuous("z", lb=1.0, ub=3.0)
    assert G._bound_expression(z**4, m) == (1.0, 81.0)
    # Odd power is monotone: endpoints are exact.
    assert G._bound_expression(x**3, m) == (-1.0, 8.0)


def test_compute_big_m_values_and_refusals():
    m = Model("bigm")
    x = m.continuous("x", lb=-4.0, ub=2.0)
    free = m.continuous("free")  # default ±9.999e19 sentinel bounds
    # '==' needs both directions: max(|hi|, |lo|) = 4, with the 1% margin.
    assert G._compute_big_m(Constraint(body=x, sense="==", rhs=0.0), m) == pytest.approx(4.04)
    # A tiny but valid bound is floored at 1e-8 before the margin.
    m2 = Model("tiny")
    t = m2.continuous("t", lb=0.0, ub=1e-12)
    assert G._compute_big_m(Constraint(body=t, sense="<=", rhs=0.0), m2) == pytest.approx(
        1e-8 * 1.01
    )
    # Sentinel-sized bounds are refused loudly in the needed direction:
    # a vacuous M would let the selector defeat the disjunction (GDP-2).
    with pytest.raises(ValueError, match="from below"):
        G._compute_big_m(Constraint(body=free, sense=">=", rhs=0.0), m)
    with pytest.raises(ValueError, match="from above"):
        G._compute_big_m(Constraint(body=free, sense="<=", rhs=0.0), m)


# ---------------------------------------------------------------------------
# gdp_reformulate: LP-tightened big-M (mbigm)
# ---------------------------------------------------------------------------


def _lp_model():
    m = Model("lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x <= 4)
    m.minimize(x)
    return m, x


def test_extract_body_coeffs_linear_property():
    m, x = _lp_model()
    w = m.continuous("w", lb=0.0, ub=7.0, shape=(2,))
    n_vars = 3  # x + the two w elements
    body = 2.0 * x - w[1] + 3.0 + (x + w[0]) * 0.5 - (-x)
    out = G._extract_body_coeffs(body, m, n_vars)
    assert out is not None
    c_vec, offset = out
    rng = np.random.default_rng(0)
    for _ in range(5):
        pt = rng.uniform(0.0, 5.0, size=n_vars)
        env = {"x": pt[0], "w": pt[1:]}
        assert c_vec @ pt + offset == pytest.approx(float(_eval(body, env)), abs=1e-12)
    # Nonlinear / non-scalar shapes are refused (fall back to intervals).
    assert G._extract_body_coeffs(x * w[0], m, n_vars) is None  # bilinear
    assert G._extract_body_coeffs(w, m, n_vars) is None  # whole vector
    assert G._extract_body_coeffs(dm.exp(x), m, n_vars) is None
    assert G._extract_body_coeffs(dm.sum(x), m, n_vars) is None
    assert G._extract_body_coeffs(UnaryOp("neg", dm.exp(x)), m, n_vars) is None


@needs_exact_lp
def test_compute_big_m_lp_tightens_but_stays_valid():
    """The LP-based M uses the row ``x <= 4``, so max(x - 1) = 3 (not the
    interval bound 9); the result must still over-estimate the body on the
    true feasible region (validity) while being tighter than intervals."""
    m, x = _lp_model()
    lp_data = G._precompute_lp_relaxation(m)
    assert lp_data is not None and lp_data[4] == 1
    con_le = x <= 1  # body x - 1
    m_lp = G._compute_big_m_lp(con_le, m, lp_data)
    assert m_lp == pytest.approx(3.0 * 1.01)
    assert m_lp < G._compute_big_m(con_le, m)  # strictly tighter than intervals
    assert m_lp >= max(v - 1 for v in (0.0, 4.0))  # valid on the feasible region
    # '>=' needs |min body|: min(x - 6) over the box is -6 (x = 0).
    assert G._compute_big_m_lp(x >= 6, m, lp_data) == pytest.approx(6.0 * 1.01)
    # '==' takes the larger direction: max(|3|, |-1|) = 3.
    assert G._compute_big_m_lp(x == 1, m, lp_data) == pytest.approx(3.0 * 1.01)
    # A nonlinear body falls back to the interval value exactly.
    nl = Constraint(body=x**2 - 1, sense="<=", rhs=0.0)
    assert G._compute_big_m_lp(nl, m, lp_data) == G._compute_big_m(nl, m)


def test_compute_big_m_lp_without_oracle_falls_back_to_intervals(monkeypatch):
    """With no exact LP oracle the mbigm path must degrade to the (sound)
    interval M — never to the inexact IPM (issue #145)."""
    import discopt.solvers.lp_backend as lp_backend

    monkeypatch.setattr(lp_backend, "get_exact_lp_solver", lambda: None)
    m, x = _lp_model()
    lp_data = G._precompute_lp_relaxation(m)
    con = x <= 1
    assert G._compute_big_m_lp(con, m, lp_data) == G._compute_big_m(con, m)


def test_indicator_mbigm_semantics():
    """method='mbigm' on an indicator: y = 1 enforces the constraint, y = 0
    relaxes it for every point of the true feasible region (the LP-tight M is
    still a valid over-estimate)."""
    m, x = _lp_model()
    y = m.binary("y")
    y2 = m.binary("y2")
    m.if_then(y, [x <= 1])
    # Explicit ">=" indicator row (the overloads normalize to "<=").
    m.if_then(y2, [Constraint(body=x - 2.0, sense=">=", rhs=0.0)])
    rm = G.reformulate_gdp(m, method="mbigm")
    assert rm is not m
    cons = rm._constraints
    assert len(cons) == 3  # regular row + two reformulated indicators
    assert _feasible(cons, {"x": 0.5, "y": 1.0, "y2": 0.0})  # active + satisfied
    assert not _feasible(cons, {"x": 3.0, "y": 1.0, "y2": 0.0})  # active + violated
    assert _feasible(cons, {"x": 3.0, "y": 0.0, "y2": 0.0})  # inactive -> relaxed
    assert _feasible(cons, {"x": 4.0, "y": 0.0, "y2": 0.0})  # relaxed at the edge
    assert _feasible(cons, {"x": 3.0, "y": 0.0, "y2": 1.0})  # >= indicator satisfied
    assert not _feasible(cons, {"x": 0.5, "y": 0.0, "y2": 1.0})  # >= indicator violated
    assert _feasible(cons, {"x": 0.5, "y": 1.0, "y2": 0.0})  # >= relaxed when off


# ---------------------------------------------------------------------------
# gdp_reformulate: disjunctions (big-M, nested, override, hull)
# ---------------------------------------------------------------------------


def _disj_ge_model():
    # The comparison overloads normalize every inequality to "<=", so genuine
    # ">=" rows (as produced by other passes) are built explicitly here.
    m = Model("disj")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.minimize(x)
    m.either_or(
        [
            [Constraint(body=x - 3.0, sense=">=", rhs=0.0)],
            [Constraint(body=x - 7.0, sense=">=", rhs=0.0)],
        ],
        name="modes",
    )
    return m, x


def test_disjunction_bigm_ge_semantics():
    """Feasible set of (x>=3) v (x>=7) is x >= 3: the reformed MILP must admit
    each x with the correct selector and reject every selector assignment for
    x below 3."""
    m, _ = _disj_ge_model()
    rm = G.reformulate_gdp(m)
    names = [v.name for v in rm._variables]
    assert names[0] == "x" and len(names) == 3
    s0, s1 = names[1], names[2]

    def env(x, a, b):
        return {"x": x, s0: a, s1: b}

    assert _feasible(rm._constraints, env(3.0, 1, 0))
    assert _feasible(rm._constraints, env(7.0, 0, 1))
    assert _feasible(rm._constraints, env(20.0, 1, 0))
    # x = 0 is infeasible under every selector assignment.
    for a, b in itertools.product([0, 1], repeat=2):
        assert not _feasible(rm._constraints, env(0.0, a, b))
    # And activating the wrong disjunct is rejected too.
    assert not _feasible(rm._constraints, env(3.0, 0, 1))


@pytest.mark.smoke
@needs_exact_lp
def test_disjunction_preserves_certified_optimum():
    """min x over (x>=3) v (x>=7) has optimum 3; big-M and hull lowerings of
    the same disjunction must both certify it."""
    m, _ = _disj_ge_model()
    r_bigm = G.reformulate_gdp(m, method="big-m").solve()
    assert r_bigm.status == "optimal"
    assert r_bigm.objective == pytest.approx(3.0, abs=1e-2)
    m2, _ = _disj_ge_model()
    r_hull = G.reformulate_gdp(m2, method="hull").solve()
    assert r_hull.status == "optimal"
    assert r_hull.objective == pytest.approx(3.0, abs=1e-2)


def test_nested_disjunction_semantics():
    """(y<=2) v [ (y>=8) v (y==5) ]: the projected feasible set is
    [0,2] u {5} u [8,10]. Enumerating all selector assignments proves points
    outside it are cut and points inside it survive."""
    m = Model("nest")
    y = m.continuous("y", lb=0.0, ub=10.0)
    inner = m.disjunction([[Constraint(body=y - 8.0, sense=">=", rhs=0.0)], [y == 5]], name="inner")
    m.either_or([[y <= 2], [inner]], name="outer")
    m.minimize(-y)
    rm = G.reformulate_gdp(m)
    names = [v.name for v in rm._variables]
    assert len(names) == 5  # y + 2 outer selectors + 2 inner selectors

    def some_selector_feasible(yv):
        for sel in itertools.product([0, 1], repeat=4):
            env = dict(zip(names, [yv, *sel]))
            if _feasible(rm._constraints, env):
                return True
        return False

    assert some_selector_feasible(1.0)  # first arm
    assert some_selector_feasible(5.0)  # inner equality arm
    assert some_selector_feasible(9.0)  # inner >= arm
    assert not some_selector_feasible(4.0)  # in no arm
    assert not some_selector_feasible(6.5)  # in no arm


def test_disjunction_method_override():
    """A disjunction-local method override wins over the solver-wide method,
    unless the caller opts out with respect_disjunction_methods=False."""
    m = Model("ovr")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 8]], name="modes")
    m._constraints[-1].method = "hull"
    rm = G.reformulate_gdp(m, method="big-m")
    assert any(c.name and c.name.startswith("_hull") for c in rm._constraints)
    rm2 = G.reformulate_gdp(m, method="big-m", respect_disjunction_methods=False)
    assert not any(c.name and c.name.startswith("_hull") for c in rm2._constraints)
    assert any(c.name and c.name.startswith("_gdp") for c in rm2._constraints)


def _hull_env(rm, x, y0, y1, v0, v1):
    names = [v.name for v in rm._variables]
    assert len(names) == 5
    return dict(zip(names, [x, y0, y1, v0, v1]))


def test_hull_equality_disjunct_semantics():
    """Hull of (x==2) v (x>=8): with one-hot selectors, the bound-linking rows
    pin the inactive disaggregated copy to 0 and aggregation pins the active
    one to x, so feasibility at each candidate point is fully determined —
    x=2 and x=9 survive, x=5 is cut under both selector choices."""
    m = Model("hulleq")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    # Explicit ">=" row so the hull's ge-sense lowering is exercised too.
    m.either_or([[x == 2], [Constraint(body=x - 8.0, sense=">=", rhs=0.0)]], name="modes")
    rm = G.reformulate_gdp(m, method="hull")
    cons = rm._constraints
    assert _feasible(cons, _hull_env(rm, 2.0, 1, 0, 2.0, 0.0))
    assert _feasible(cons, _hull_env(rm, 9.0, 0, 1, 0.0, 9.0))
    # x = 5: active copy violates its own arm in both cases.
    assert not _feasible(cons, _hull_env(rm, 5.0, 1, 0, 5.0, 0.0))
    assert not _feasible(cons, _hull_env(rm, 5.0, 0, 1, 0.0, 5.0))


def test_hull_nonlinear_perspective_semantics():
    """Hull of (x**2<=4) v (x>=8) routes the nonlinear arm through the
    eps-clamped perspective; within the solver feasibility tolerance the arm
    admits |x| <= 2 and rejects x = 3 under both selector choices, and the
    inactive arm's disaggregated 0 does not create a spurious violation."""
    m = Model("hullnl")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize(x)
    m.either_or([[x**2 <= 4], [x >= 8]], name="modes")
    rm = G.reformulate_gdp(m, method="hull")
    cons = rm._constraints
    tol = 1e-6  # absorbs the documented O(eps) perspective residual
    assert _feasible(cons, _hull_env(rm, 2.0, 1, 0, 2.0, 0.0), tol=tol)
    assert _feasible(cons, _hull_env(rm, -1.0, 1, 0, -1.0, 0.0), tol=tol)
    assert _feasible(cons, _hull_env(rm, 9.0, 0, 1, 0.0, 9.0), tol=tol)
    assert not _feasible(cons, _hull_env(rm, 3.0, 1, 0, 3.0, 0.0), tol=tol)
    assert not _feasible(cons, _hull_env(rm, 3.0, 0, 1, 0.0, 3.0), tol=tol)


def test_hull_linear_substitute_property():
    """For linear f(x) = a^T x + b the hull form is a^T v + b*y: evaluating
    the substituted expression must equal the homogeneous part on the
    disaggregated values plus the constants scaled by the selector."""
    m = Model("hls")
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    w = m.continuous("w", lb=0.0, ub=10.0, shape=(2,))
    vx = m.continuous("vx", lb=0.0, ub=10.0)
    vz = m.continuous("vz", lb=0.0, ub=10.0)
    vw = m.continuous("vw", lb=0.0, ub=10.0, shape=(2,))
    y = m.binary("y")
    expr = 2.0 * x - z / 2.0 + 1.5 - (-x) + w[0] * 3.0
    var_map = {"x": vx, "z": vz, "w": vw}
    out = G._hull_linear_substitute(expr, var_map, y)
    for a, b, c, t in [(1.0, 2.0, 3.0, 1.0), (4.0, 0.5, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)]:
        env = {"vx": a, "vz": b, "vw": [c, 0.0], "y": t, "x": 99.0, "z": 99.0, "w": [99.0, 99.0]}
        expect = 2.0 * a - b / 2.0 + 1.5 * t + a + 3.0 * c
        assert float(_eval(out, env)) == pytest.approx(expect, abs=1e-12)
    # Non-linear input falls through to plain substitution (no y-scaling).
    out2 = G._hull_linear_substitute(dm.exp(x), var_map, y)
    assert float(_eval(out2, {"vx": 1.5, "y": 0.0})) == pytest.approx(np.exp(1.5))


def test_is_linear_classification():
    m = Model("lin")
    x = m.continuous("x", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=1.0, ub=2.0)
    w = m.continuous("w", lb=0.0, ub=1.0, shape=(2,))
    for good in (x, 2.0 * x + 1.0, (x - z) / 2.0, -x, w[0]):
        assert G._is_linear(good)
    for bad in (x * z, x**2, dm.exp(x), x / z, UnaryOp("abs", x)):
        assert not G._is_linear(bad)


def test_substitute_vars_semantics_and_identity():
    m = Model("sub")
    x = m.continuous("x", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    w = m.continuous("w", lb=0.0, ub=1.0, shape=(2,))
    vx = m.continuous("vx", lb=0.0, ub=1.0)
    vw = m.continuous("vw", lb=0.0, ub=1.0, shape=(2,))
    expr = dm.exp(x) + w[1] * z - (-x)
    out = G._substitute_vars(expr, {"x": vx, "w": vw})
    env = {"vx": 0.3, "vw": [0.0, 0.7], "z": 0.5, "x": 99.0, "w": [99.0, 99.0]}
    assert float(_eval(out, env)) == pytest.approx(np.exp(0.3) + 0.7 * 0.5 + 0.3)
    # An empty substitution returns the very same nodes (no gratuitous copies).
    assert G._substitute_vars(expr, {}) is expr


def test_extract_disjunct_bounds_patterns():
    m = Model("db")
    w = m.continuous("w", lb=-10.0, ub=10.0)
    wv = m.continuous("wv", lb=0.0, ub=7.0, shape=(2,))
    # const - var <= 0  =>  var >= const.
    db = G._extract_disjunct_bounds([Constraint(body=_wrap(2.0) - w, sense="<=", rhs=0.0)], m)
    assert db == {"w": (2.0, 10.0)}
    # const - var >= 0  =>  var <= const.
    db = G._extract_disjunct_bounds([Constraint(body=_wrap(2.0) - w, sense=">=", rhs=0.0)], m)
    assert db == {"w": (-10.0, 2.0)}
    # Equality pins the variable.
    assert G._extract_disjunct_bounds([w == 3], m) == {"w": (3.0, 3.0)}
    # var - const pattern from the operator overloads.
    assert G._extract_disjunct_bounds([w <= 5], m) == {"w": (-10.0, 5.0)}
    # Indexed-variable body with explicit rhs.
    db = G._extract_disjunct_bounds([Constraint(body=wv[0], sense="<=", rhs=3.0)], m)
    assert db == {"wv": (0.0, 3.0)}


# ---------------------------------------------------------------------------
# gdp_reformulate: SOS
# ---------------------------------------------------------------------------


def test_sos_infinite_bound_refusal_and_single_variable():
    # Truly infinite lower bound: no valid finite linking M exists -> refuse.
    m = Model("sosinf")
    z = m.continuous("z", lb=-np.inf, ub=5.0)
    z2 = m.continuous("z2", lb=0.0, ub=1.0)
    m.minimize(z)
    m.sos1([z, z2], name="s")
    with pytest.raises(ValueError, match="from below"):
        G.reformulate_gdp(m)
    # Infinite upper bound refuses in the other direction.
    m2 = Model("sosinf2")
    u = m2.continuous("u", lb=0.0, ub=np.inf)
    u2 = m2.continuous("u2", lb=0.0, ub=1.0)
    m2.minimize(u)
    m2.sos1([u, u2], name="s")
    with pytest.raises(ValueError, match="from above"):
        G.reformulate_gdp(m2)
    # Single-variable SOS1: x can be nonzero only when its indicator is on.
    m3 = Model("sos1one")
    x = m3.continuous("x", lb=0.0, ub=5.0)
    m3.minimize(-x)
    m3.sos1([x], name="one")
    rm = G.reformulate_gdp(m3)
    zname = rm._variables[-1].name
    assert _feasible(rm._constraints, {"x": 5.0, zname: 1.0})
    assert not _feasible(rm._constraints, {"x": 3.0, zname: 0.0})
    assert _feasible(rm._constraints, {"x": 0.0, zname: 0.0})


# ---------------------------------------------------------------------------
# gdp_reformulate: logical constraints
# ---------------------------------------------------------------------------


def _truth(lexpr, env) -> bool:
    from discopt.modeling.core import (
        BooleanVar,
        LogicalAnd,
        LogicalAtLeast,
        LogicalAtMost,
        LogicalEquivalent,
        LogicalExactly,
        LogicalImplies,
        LogicalNot,
        LogicalOr,
    )

    if isinstance(lexpr, BooleanVar):
        var = lexpr.variable
        name = var.name if isinstance(var, Variable) else var.base.name
        return bool(env[name])
    if isinstance(lexpr, LogicalNot):
        return not _truth(lexpr.operand, env)
    if isinstance(lexpr, LogicalAnd):
        return _truth(lexpr.left, env) and _truth(lexpr.right, env)
    if isinstance(lexpr, LogicalOr):
        return _truth(lexpr.left, env) or _truth(lexpr.right, env)
    if isinstance(lexpr, LogicalImplies):
        return (not _truth(lexpr.antecedent, env)) or _truth(lexpr.consequent, env)
    if isinstance(lexpr, LogicalEquivalent):
        return _truth(lexpr.left, env) == _truth(lexpr.right, env)
    if isinstance(lexpr, LogicalAtLeast):
        return sum(_truth(o, env) for o in lexpr.operands) >= lexpr.k
    if isinstance(lexpr, LogicalAtMost):
        return sum(_truth(o, env) for o in lexpr.operands) <= lexpr.k
    if isinstance(lexpr, LogicalExactly):
        return sum(_truth(o, env) for o in lexpr.operands) == lexpr.k
    raise AssertionError(type(lexpr).__name__)


def _literals_only(lexpr) -> bool:
    from discopt.modeling.core import BooleanVar, LogicalNot

    if isinstance(lexpr, BooleanVar):
        return True
    if isinstance(lexpr, LogicalNot):
        return isinstance(lexpr.operand, BooleanVar)
    children = [
        getattr(lexpr, a)
        for a in ("left", "right", "antecedent", "consequent")
        if hasattr(lexpr, a)
    ]
    return all(_literals_only(c) for c in children)


def test_get_binary_var_rejects_non_leaf():
    m = Model("gbv")
    a, b = m.boolean("A"), m.boolean("B")
    with pytest.raises(TypeError, match="Expected BooleanVar"):
        G._get_binary_var(LogicalAnd(a, b))
    assert G._get_binary_var(a) is a.variable


def test_to_nnf_preserves_truth_tables():
    """NNF conversion (De Morgan, implication/equivalence elimination, double
    negation) must be a logical no-op: identical truth table on all 8
    assignments, with negations pushed onto literals."""
    m = Model("nnf")
    a, b, c = m.boolean("A"), m.boolean("B"), m.boolean("C")
    formulas = [
        ~(a & b),
        ~(a | b),
        ~(~a),
        ~(a.implies(b)),
        ~(a.equivalent_to(b)),
        a.equivalent_to(b),
        (a.implies(b)).implies(c),
        ~((a & b) | ~c),
    ]
    for f in formulas:
        nnf = G._to_nnf(f)
        assert _literals_only(nnf), f
        for bits in itertools.product([0, 1], repeat=3):
            env = {"A": bits[0], "B": bits[1], "C": bits[2]}
            assert _truth(nnf, env) == _truth(f, env), (f, bits)


def test_to_nnf_cardinality_and_de_morgan():
    # Since the #750 fix, NNF lowers a negated cardinality atom by De
    # Morgan for counts: not atleast(1, A, B) == atmost(0, A, B). A bare
    # atom passes through untouched.
    m = Model("card")
    a, b = m.boolean("A"), m.boolean("B")
    at = dm.atleast(1, a, b)
    assert G._to_nnf(at) is at
    lowered = G._to_nnf(LogicalNot(at))
    assert type(lowered).__name__ == "LogicalAtMost"
    assert lowered.k == 0
    assert list(lowered.operands) == [a, b]


def test_logical_tseitin_encoding_is_semantically_exact():
    """(A & B) | ~C forces a Tseitin auxiliary for the And inside the Or; the
    reformed rows must be satisfiable for exactly the satisfying assignments
    (for some value of the auxiliary) and unsatisfiable otherwise."""
    m = Model("ts")
    a, b, c = m.boolean("A"), m.boolean("B"), m.boolean("C")
    formula = (a & b) | ~c
    m.logical(formula)
    rm = G.reformulate_gdp(m)
    names = [v.name for v in rm._variables]
    assert len(names) == 4  # A, B, C + one Tseitin auxiliary
    for bits in itertools.product([0, 1], repeat=3):
        env0 = {"A": bits[0], "B": bits[1], "C": bits[2]}
        sat = any(
            _feasible(rm._constraints, {**dict(zip(names[:3], bits)), names[3]: aux})
            for aux in (0.0, 1.0)
        )
        assert sat == _truth(formula, env0), bits


def test_logical_cardinality_semantics():
    """Top-level atleast/atmost/exactly lower to single linear rows whose
    feasibility matches the cardinality condition on every assignment."""
    for maker, cond in [
        (lambda a, b, c: dm.atleast(2, a, b, c), lambda s: s >= 2),
        (lambda a, b, c: dm.atmost(1, a, b, c), lambda s: s <= 1),
        (lambda a, b, c: dm.exactly(2, a, b, c), lambda s: s == 2),
    ]:
        m = Model("cardsem")
        a, b, c = m.boolean("A"), m.boolean("B"), m.boolean("C")
        m.logical(maker(a, b, c))
        rm = G.reformulate_gdp(m)
        assert len(rm._constraints) == 1
        for bits in itertools.product([0, 1], repeat=3):
            env = {"A": bits[0], "B": bits[1], "C": bits[2]}
            assert _feasible(rm._constraints, env) == cond(sum(bits)), bits


def test_logical_cardinality_nested_in_and_is_not_dropped():
    m = Model("nested_card")
    a, b, c = m.boolean("A"), m.boolean("B"), m.boolean("C")
    m.logical(a & dm.atleast(1, b, c))
    rm = G.reformulate_gdp(m)
    names = [v.name for v in rm._variables]
    # A=1, B=C=0 violates atleast(1, B, C): no assignment of any auxiliaries
    # may make the reformed rows feasible.
    base = {"A": 1.0, "B": 0.0, "C": 0.0}
    aux_names = [n for n in names if n not in base]
    sat = any(
        _feasible(rm._constraints, {**base, **dict(zip(aux_names, aux))})
        for aux in itertools.product([0.0, 1.0], repeat=len(aux_names))
    )
    assert not sat
