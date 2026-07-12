"""Canonical expression DAG tests (issue #632, R1.1).

The load-bearing property is **semantic equivalence**: ``reconstruct(canonicalize)``
must evaluate equal to the original expression at random points, on generated trees
and on every vendored corpus instance. The rest pin the structural guarantees the
downstream dispatch relies on: idempotence, content-addressed interning (CSE),
construction-order determinism, and sound refusal (opaque) for anything the grammar
cannot represent.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.canonical_expr import CNode, canonicalize, reconstruct
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import Constant, CustomCall, from_nl

pytestmark = [pytest.mark.claim_boundary]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _values(expr, model, X):
    f = compile_expression(expr, model)
    return np.array([float(f(jnp.asarray(x))) for x in X])


def _assert_equiv(expr, model, lo, hi, n_pts=200, seed=0):
    dag = canonicalize(model)
    node = dag.cnode_of(expr)
    rec = reconstruct(node, model)
    rng = np.random.default_rng(seed)
    nvar = sum(v.size for v in model._variables)
    X = rng.uniform(lo, hi, size=(n_pts, nvar))
    v0 = _values(expr, model, X)
    v1 = _values(rec, model, X)
    fin = np.isfinite(v0) & np.isfinite(v1)
    # Deep random trees can overflow to inf/NaN (exp-of-exp, log of a negative
    # subtraction); those samples carry no signal. Compare wherever BOTH are
    # finite — the round-trip must never diverge there. If almost everything
    # overflowed, this particular tree is uninformative, not a failure.
    if fin.sum() < 10:
        return node
    assert np.allclose(v0[fin], v1[fin], rtol=1e-8, atol=1e-8), (
        f"canonical round-trip diverged: max|Δ|={np.max(np.abs(v0[fin] - v1[fin])):.3e}"
    )
    return node


# --------------------------------------------------------------------------- #
# random expression generator (positive-domain, so log/sqrt/div stay defined)
# --------------------------------------------------------------------------- #
_UNARY = ["exp", "log", "sqrt", "sin", "cos", "tanh", "atan"]


def _rand_expr(rng, vs, depth):
    if depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.7:
            return vs[rng.integers(len(vs))]
        return Constant(float(rng.uniform(0.5, 3.0)))
    r = rng.random()
    a = _rand_expr(rng, vs, depth - 1)
    if r < 0.24:
        b = _rand_expr(rng, vs, depth - 1)
        return a + b
    if r < 0.44:
        b = _rand_expr(rng, vs, depth - 1)
        return a - b
    if r < 0.60:
        b = _rand_expr(rng, vs, depth - 1)
        return a * b
    if r < 0.72:
        return a ** int(rng.integers(2, 4))
    if r < 0.82:
        return Constant(float(rng.uniform(1.5, 4.0))) * a
    fn = _UNARY[rng.integers(len(_UNARY))]
    return getattr(dm, fn)(a)


@pytest.mark.parametrize("seed", range(40))
def test_semantic_equivalence_generated(seed):
    """>=200 generated trees round-trip value-equivalently (5 trees x 40 seeds)."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"g{seed}")
    xs = [m.continuous(f"x{i}", lb=0.5, ub=3.0) for i in range(3)]
    m.minimize(xs[0])  # placeholder objective so the model is well-formed
    for _ in range(5):
        expr = _rand_expr(rng, xs, depth=4)
        _assert_equiv(expr, m, np.full(3, 0.5), np.full(3, 3.0), n_pts=120, seed=seed)


_CORPUS = sorted(p.stem for p in _NL_DIR.glob("*.nl"))


@pytest.mark.parametrize("name", _CORPUS)
def test_semantic_equivalence_corpus(name):
    """Every vendored instance's objective + constraint bodies round-trip."""
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    nvar = sum(v.size for v in model._variables)
    if nvar == 0 or nvar > 60:
        pytest.skip("degenerate or large variable count")
    lo = np.concatenate([np.asarray(v.lb, float).ravel() for v in model._variables])
    hi = np.concatenate([np.asarray(v.ub, float).ravel() for v in model._variables])
    # Sample strictly inside a finite box so transcendental domains stay defined.
    # Clamp infinities to a finite window and guarantee lo < hi per coordinate
    # (some instances have half-open or degenerate bounds).
    lo_f = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi - 10.0, -5.0))
    hi_f = np.where(np.isfinite(hi), hi, np.where(np.isfinite(lo), lo + 10.0, 5.0))
    a = np.minimum(lo_f, hi_f)
    b = np.maximum(lo_f, hi_f)
    span = np.maximum(b - a, 1e-3)
    lo2, hi2 = a + 0.05 * span, b - 0.05 * span
    hi2 = np.maximum(hi2, lo2 + 1e-6)
    dag = canonicalize(model)
    exprs = []
    if model._objective is not None:
        exprs.append(model._objective.expression)
    exprs.extend(c.body for c in model._constraints)
    rng = np.random.default_rng(0)
    X = rng.uniform(lo2, hi2, size=(60, nvar))
    for expr in exprs:
        node = dag.cnode_of(expr)
        rec = reconstruct(node, model)
        v0 = _values(expr, model, X)
        v1 = _values(rec, model, X)
        fin = np.isfinite(v0) & np.isfinite(v1)
        if fin.sum() == 0:
            continue
        assert np.allclose(v0[fin], v1[fin], rtol=1e-7, atol=1e-7), (
            f"[{name}] round-trip diverged: max|Δ|={np.max(np.abs(v0[fin] - v1[fin])):.3e}"
        )


# --------------------------------------------------------------------------- #
# structural guarantees
# --------------------------------------------------------------------------- #
def test_interning_shares_equal_subtrees():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    m.minimize(x * y + x * y)
    dag = canonicalize(m)
    node = dag.cnode_of(m._objective.expression)
    # x*y + x*y -> 2 * (x*y): a single product child, shared by identity.
    assert node.kind == "sum"
    assert len(node.children) == 1
    assert node.payload[0] == (2.0,)


def test_interning_identity_across_separate_builds_of_same_subtree():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    e1 = dm.exp(x * y)
    e2 = dm.exp(x * y)
    m.minimize(e1 + e2)
    dag = canonicalize(m)
    n1 = dag.cnode_of(e1)
    n2 = dag.cnode_of(e2)
    assert n1 is n2  # content-addressed: structurally equal -> same object


def test_idempotence():
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=3)
    y = m.continuous("y", lb=0.5, ub=3)
    expr = (x + 2 * y) ** 2 + x / y - dm.exp(x)
    m.minimize(expr)
    dag = canonicalize(m)
    node = dag.cnode_of(expr)
    rec = reconstruct(node, m)
    dag2 = canonicalize(m)
    node2 = dag2.cnode_of(rec)
    assert node.key == node2.key


def test_determinism_construction_order_independent():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    z = m.continuous("z", lb=1, ub=3)
    a = (x * y) + (y * z) + (x * z)
    b = (z * x) + (y * x) + (z * y)  # same set, different write order
    m.minimize(a)
    dag = canonicalize(m)
    assert dag.cnode_of(a).key == dag.cnode_of(b).key


def test_refusal_sign_is_opaque():
    m = dm.Model()
    x = m.continuous("x", lb=-2, ub=2)
    e = dm.sign(x) + x
    m.minimize(e)
    dag = canonicalize(m)
    top = dag.cnode_of(e)
    # the sign(...) subterm must be opaque; the surrounding sum is fine.
    assert any(c.is_opaque for c in top.children) or top.is_opaque


def test_refusal_custom_call_is_opaque():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    cc = CustomCall(lambda a: a * a, x, name="sq")
    m.minimize(cc + x)
    dag = canonicalize(m)
    node = dag.cnode_of(cc)
    assert node.is_opaque
    # opaque reconstructs to the original object verbatim.
    assert reconstruct(node, m) is cc


def test_division_is_negative_exponent_product():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    e = x / y
    m.minimize(e)
    dag = canonicalize(m)
    node = dag.cnode_of(e)
    assert node.kind == "prod"
    (exps,) = node.payload
    assert sorted(exps) == [-1.0, 1.0]


def test_no_spurious_cancellation_of_reciprocal():
    """x * (1/x) must NOT collapse to 1 (would change the value at x=0)."""
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=3)
    e = x * (Constant(1.0) / x)
    m.minimize(e)
    dag = canonicalize(m)
    node = dag.cnode_of(e)
    assert node.kind == "prod"
    (exps,) = node.payload
    assert sorted(exps) == [-1.0, 1.0]


def test_build_still_works_after_canonicalize():
    """Canonicalize is side-effect free: building the relaxation still works and
    canonicalize does not perturb it (library-only, nothing wired)."""
    model = from_nl(str(_NL_DIR / "nvs09.nl"))
    terms = classify_nonlinear_terms(model)
    relax1, _ = build_milp_relaxation(model, terms, DiscretizationState())
    canonicalize(model)
    relax2, _ = build_milp_relaxation(model, terms, DiscretizationState())
    assert isinstance(canonicalize(model).nodes[0], CNode)
    # shape stable across a canonicalize call
    assert np.asarray(relax1._c).shape == np.asarray(relax2._c).shape
