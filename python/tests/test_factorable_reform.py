"""Tests for the factorable reformulation pass (issue #130).

The pass rewrites two families of terms the relaxation pipeline cannot relax
natively into terms it can:

* sign-definite denominators ``N / D`` (``D`` bounded away from zero) are
  cleared by multiplying the constraint through by ``D``;
* mixed repeated-factor products such as ``x*x*y`` are lifted to bilinear form
  via a monomial aux variable ``w == x**2``.

Both rewrites are value-preserving, so the headline guard is that the raw
``nvs01`` instance — exp/quadratic objective, a trilinear equality, and a
division constraint — now certifies to its MINLPLib optimum with a *sound* dual
bound where before the constraint was silently dropped from the relaxation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.factorable_reform import (
    _should_lift_call_arg,
    canonicalize_entropy,
    factorable_reformulate,
    has_factorable_work,
)
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import FunctionCall, Variable

_DATA = Path(__file__).parent / "data" / "minlplib"
_NL_DATA = Path(__file__).parent / "data" / "minlplib_nl"


def _aux_names(model):
    return [v.name for v in model._variables if v.name.startswith("_fr_aux")]


def test_noop_when_nothing_applies():
    """A model with only bilinear/monomial terms is returned unchanged."""
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x * y + x**2)
    m.subject_to(x + y <= 5)
    out = factorable_reformulate(m)
    assert out is m  # same object, no rewrite performed


def test_mixed_repeated_product_is_lifted_to_bilinear():
    """``x*x*y`` is unrepresentable natively; the pass lifts ``x**2`` to an aux
    so the product becomes a bilinear term the classifier accepts."""
    m = dm.Model("mixed")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(y)
    m.subject_to(x * x * y >= 8)

    # Before: the mixed product lands in general_nl (dropped by the relaxer).
    pre = classify_nonlinear_terms(m)
    assert pre.general_nl, "expected x*x*y to be unhandled before reformulation"

    out = factorable_reformulate(m)
    assert out is not m
    assert len(_aux_names(out)) == 1

    post = classify_nonlinear_terms(out)
    # The lifted product is now a genuine bilinear term, and the aux defining
    # equality contributes the x**2 monomial — nothing left in general_nl.
    assert post.bilinear, "lifted product should be classified as bilinear"
    assert (0, 2) in post.monomial  # w == x**2
    assert not post.general_nl


def test_fractional_power_of_product_base_triggers_reform():
    """``(x*y)**0.5`` — a fractional power over a *composite* (product) base — must
    be detected by ``has_factorable_work`` so the reform fires and lifts it.

    The lift capability (``_lift_objective_atoms``: ``t == x*y`` then ``d == t**0.5``)
    is applied to constraint bodies, but the gate scanner only recognized mixed
    repeated-factor products and sqrt/exp *calls* — a fractional power of a product
    is neither, so ``has_factorable_work`` returned False, the pass never ran, the
    term dropped from the relaxation, and the dual bound froze (feasible but never
    proved; the ex1226 failure mode for a composite-base power). After detection
    the instance certifies optimality in a handful of nodes.
    """
    m = dm.Model("fracpow_prod")
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    m.minimize(-x - y)
    m.subject_to((x * y) ** 0.5 <= 2.0)

    assert has_factorable_work(m), "fractional power of a product base must be liftable work"

    out = factorable_reformulate(m)
    assert out is not m
    assert _aux_names(out), "expected aux variables from the composite-base power lift"

    res = m.solve(time_limit=30, gap_tolerance=1e-4, max_nodes=20_000)
    assert res.status == "optimal", (
        f"(x*y)**0.5 did not certify optimality (status={res.status}); the "
        "composite-base fractional power likely dropped from the relaxation"
    )
    assert res.node_count <= 200, (
        f"(x*y)**0.5 took {res.node_count} nodes — far above the ~3 expected; the "
        "dropped-term / frozen-bound regression may have returned"
    )


def test_fractional_power_simple_base_not_lifted_but_call_form_untouched():
    """The gate detector matches only composite-base **power-form** fractional
    powers. A single-variable base (``x**0.5``) is relaxed natively via
    ``fractional_power_var_map``, so it must NOT be promoted to factorable work.
    The ``sqrt(...)`` **call** form is a FunctionCall reached by the existing
    composite-univariate path, so it must likewise be left to that path (the
    fractional-power scanner only walks ``**`` nodes)."""
    m1 = dm.Model("fracpow_simple")
    x = m1.continuous("x", lb=1, ub=3)
    m1.minimize(-x)
    m1.subject_to(x**0.5 <= 1.5)
    assert not has_factorable_work(m1), "simple-base fractional power needs no reform"

    # sqrt(affine) call form: handled by the composite-univariate envelope, not the
    # power-form scanner — must remain undetected so its tight path is preserved.
    m2 = dm.Model("sqrt_call_affine")
    a = m2.continuous("a", lb=1, ub=3)
    b = m2.continuous("b", lb=1, ub=3)
    m2.minimize(-a - b)
    m2.subject_to(dm.sqrt(a + b) <= 2.0)
    assert not has_factorable_work(m2), "sqrt(affine) call form must stay on its own path"


def test_fractional_power_of_affine_base_in_power_form_closes():
    """``(x+y)**0.5`` (affine base, power form, 0<p<1) is concave and *kept* by the
    relaxation, but its native power-form envelope is too loose to close — it
    churned 3503 nodes while the equivalent call ``sqrt(x+y)`` closed in ~251.
    Detecting the composite-base power triggers the lift (``t == x+y`` then
    ``d == t**0.5``), routing it through the same single-variable fractional-power
    envelope the call form uses, so it now certifies optimality in a few nodes."""
    m = dm.Model("fracpow_affine_power")
    a = m.continuous("a", lb=1, ub=3)
    b = m.continuous("b", lb=1, ub=3)
    m.minimize(-a - b)
    m.subject_to((a + b) ** 0.5 <= 2.0)

    assert has_factorable_work(m), "affine-base power-form fractional power must be liftable"

    res = m.solve(time_limit=30, gap_tolerance=1e-4, max_nodes=20_000)
    assert res.status == "optimal", f"(x+y)**0.5 did not certify optimality (status={res.status})"
    assert res.node_count <= 200, (
        f"(x+y)**0.5 took {res.node_count} nodes — the loose power-form envelope "
        "regression may have returned"
    )


def test_multi_repeated_factor_product_all_monomials_lifted():
    """``x*x*y*y`` (= ``x**2 * y**2``) must lift BOTH monomials. Left-associated
    parsing ``((x*x)*y)*y`` never presents ``y*y`` as a standalone subproduct, so
    the monomial collector previously registered only ``x**2`` and the constraint
    dropped. Both ``(x,2)`` and ``(y,2)`` must now be lifted so the product becomes
    the bilinear ``aux(x**2)*aux(y**2)`` the relaxer keeps."""
    m = dm.Model("multi_repeat")
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    m.minimize(-x - y)
    m.subject_to(x * x * y * y <= 16.0)

    res = m.solve(time_limit=30, gap_tolerance=1e-4, max_nodes=20_000)
    assert res.status == "optimal", (
        f"x*x*y*y did not certify optimality (status={res.status}); a repeated-factor "
        "monomial may have been missed, dropping the constraint"
    )
    assert res.node_count <= 200, f"x*x*y*y took {res.node_count} nodes — regression"


def _inequality_sense(model):
    """The (single) non-equality constraint sense, as stored after discopt's
    internal canonicalisation."""
    senses = [c.sense for c in model._constraints if c.sense != "=="]
    assert len(senses) == 1
    return senses[0]


def test_positive_denominator_cleared_sense_preserved():
    """``N / (x**2 + c)`` with a strictly positive denominator is cleared and the
    constraint sense is preserved (relative to discopt's stored form)."""
    m = dm.Model("posdiv")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    m.subject_to((y + 1) / (x**2 + 1) - 2 >= 0)
    orig_sense = _inequality_sense(m)

    out = factorable_reformulate(m)
    assert out is not m
    # The division by a variable expression is gone; only the aux-defining
    # equality and a polynomial body remain.
    bodies = [repr(c.body) for c in out._constraints]
    assert all("/ (" not in b and "/(" not in b for b in bodies)
    # Positive denominator → sense unchanged from the stored original.
    assert _inequality_sense(out) == orig_sense


def test_negative_denominator_flips_sense():
    """A strictly negative denominator flips the inequality sense on clearing."""
    m = dm.Model("negdiv")
    x = m.continuous("x", lb=1, ub=5)  # so -(x**2)-1 < 0 strictly
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    # denominator -(x**2) - 1 is in [-26, -2] < 0 over the box
    m.subject_to((y + 1) / (-(x**2) - 1) <= 3)
    orig_sense = _inequality_sense(m)

    out = factorable_reformulate(m)
    assert out is not m
    flip = {"<=": ">=", ">=": "<="}
    assert _inequality_sense(out) == flip[orig_sense]


def test_grazing_denominator_not_cleared():
    """A denominator whose interval includes zero is NOT sign-definite and must
    be left untouched (no unsound multiply-through)."""
    m = dm.Model("graze")
    x = m.continuous("x", lb=-2, ub=2)  # x**2 - 1 spans [-1, 3], crosses zero
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    m.subject_to((y + 1) / (x**2 - 1) >= 1)
    out = factorable_reformulate(m)
    # No clearable denominator and no mixed product → unchanged.
    assert out is m


def test_transcendental_product_not_lifted():
    """``sqrt(x) * y`` is not a pure polynomial product and must be left for the
    native composite/general handling, not lifted."""
    m = dm.Model("trans")
    x = m.continuous("x", lb=1, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(dm.sqrt(x) * y)
    m.subject_to(x + y >= 3)
    out = factorable_reformulate(m)
    assert out is m


# ---------------------------------------------------------------------------
# Auxiliary-variable factorization of ``outer(g(x))`` (transcendental over a
# multivariate factorable argument) — the general lift for nodes such as
# ``sqrt(x4**2 + 2*x4*x5*x7 + x5**2)`` that the univariate envelope cannot reach.
# ---------------------------------------------------------------------------


def _outer_sqrt_node(model):
    """Return the (single) sqrt FunctionCall in *model*'s objective, or None."""
    found: list[FunctionCall] = []

    def walk(e):
        if isinstance(e, FunctionCall):
            found.append(e)
        for attr in ("left", "right", "operand"):
            if hasattr(e, attr):
                walk(getattr(e, attr))
        if hasattr(e, "args"):
            for a in e.args:
                walk(a)

    walk(model._objective.expression)
    return found[0] if found else None


def test_multivar_sqrt_arg_is_lifted():
    """``sqrt(x*x + 2*x*y*z + y*y)`` — a non-affine, multivariate argument the
    univariate envelope cannot reach — is lifted to ``sqrt(t)`` with ``t == g``.
    The gate fires and the rewritten node is a sqrt over a single aux Variable."""
    m = dm.Model("multivar_sqrt")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    z = m.continuous("z", lb=0.5, ub=3.0)
    m.minimize(dm.sqrt(x * x + 2.0 * x * y * z + y * y))

    node = _outer_sqrt_node(m)
    assert _should_lift_call_arg(node, m) is not None, "gate should select the multivar arg"
    assert has_factorable_work(m)

    out = factorable_reformulate(m)
    assert out is not m
    assert len(_aux_names(out)) >= 1, "expected an aux variable for t == g(x)"
    lifted = _outer_sqrt_node(out)
    assert lifted is not None and lifted.func_name == "sqrt"
    assert len(lifted.args) == 1 and isinstance(lifted.args[0], Variable), (
        "sqrt argument should now be a single aux variable"
    )


def test_affine_norm_sqrt_not_lifted():
    """An affine 2-norm ``sqrt(0.25*x**2 + (0.5*y+0.5*z)**2)`` is proven convex by
    the detector and relaxed exactly by its own envelope — the lift must NOT
    downgrade it to the looser concave-sqrt form."""
    m = dm.Model("affine_norm")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=-2.0, ub=2.0)
    m.minimize(dm.sqrt(0.25 * x**2 + (0.5 * y + 0.5 * z) ** 2))

    node = _outer_sqrt_node(m)
    assert _should_lift_call_arg(node, m) is None, "convex affine norm must not be lifted"


def test_univariate_sqrt_arg_not_lifted():
    """``sqrt(x*x + 3)`` has a single-variable argument reached by the existing
    composite-univariate path; the multivariate lift must abstain."""
    m = dm.Model("univar_sqrt")
    x = m.continuous("x", lb=0.5, ub=3.0)
    m.minimize(dm.sqrt(x * x + 3.0))

    node = _outer_sqrt_node(m)
    assert _should_lift_call_arg(node, m) is None, "single-variable arg must not be lifted"


def test_exp_multivar_arg_is_lifted():
    """The lift generalizes beyond sqrt: ``exp(x*y)`` (a transcendental over a
    bilinear, UNKNOWN-curvature argument) is lifted to ``exp(t)`` with ``t == x*y``."""
    m = dm.Model("exp_bilinear")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.exp(x * y))

    node = _outer_sqrt_node(m)
    assert node is not None and node.func_name == "exp"
    assert _should_lift_call_arg(node, m) is not None
    out = factorable_reformulate(m)
    assert out is not m
    lifted = _outer_sqrt_node(out)
    assert lifted.func_name == "exp"
    assert len(lifted.args) == 1 and isinstance(lifted.args[0], Variable)


@pytest.mark.correctness
def test_lifted_multivar_sqrt_is_sound_and_optimal():
    """End-to-end: a well-conditioned multivariate cross-term sqrt objective
    certifies to its true optimum with a sound dual bound after the lift.

    ``min sqrt(x*x + 2*x*y*z + y*y)`` over ``x,y,z in [1,3]`` is minimized at the
    box corner ``x=y=z=1``: ``sqrt(1 + 2 + 1) = 2``."""
    m = dm.Model("sound_multivar_sqrt")
    m.continuous("x", lb=1.0, ub=3.0)
    m.continuous("y", lb=1.0, ub=3.0)
    m.continuous("z", lb=1.0, ub=3.0)
    xv, yv, zv = (v for v in m._variables)
    m.minimize(dm.sqrt(xv * xv + 2.0 * xv * yv * zv + yv * yv))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.objective == pytest.approx(2.0, abs=1e-3)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"


@pytest.mark.correctness
def test_lifted_model_preserves_optimum():
    """The reformulated model is equivalent: solving it reaches the same
    optimum as the hand-checked value of the original."""
    m = dm.Model("equiv")
    x = m.integer("x", lb=1, ub=5)
    y = m.integer("y", lb=1, ub=5)
    m.minimize(y)
    # x*x*y >= 20 with x,y integer in [1,5]: minimum y is 1 (x=5 -> 25>=20).
    m.subject_to(x * x * y >= 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4


def test_lifter_expression_dedups_by_structure_not_identity():
    """Regression for the ex7_2_3 false-"optimal" SUSPECT.

    ``_Lifter.expression`` deduplicates lifted sub-expressions by a *structural*
    key (``repr``), never ``id()``. CPython recycles the ``id`` of a
    garbage-collected node, so an ``id()``-keyed cache could hand a later,
    structurally *different* ratio the aux of a freed one — silently dropping a
    denominator. In MINLPLib ``ex7_2_3`` that made the constraint
    ``1.25e6/(x3*x8) + x5/x8 - 2500*x5/x3/x8 <= 1`` satisfiable at the infeasible
    box corner ``x_i = lb_i``, so spatial B&B certified the corner (objvar = sum
    of lower bounds = 2100) as the global optimum — a false "optimal" (the true
    optimum is 7049.2479).

    The structural contract: two distinct objects with identical structure share
    one aux; structurally different expressions never do. The first assertion
    deterministically fails under ``id()`` keying — two distinct objects have
    distinct ids and would each allocate their own aux instead of deduping.
    """
    from discopt._jax.factorable_reform import _Lifter
    from discopt._jax.term_classifier import distribute_products
    from discopt.modeling.core import BinaryOp, Constant

    m = dm.Model("dedup")
    x5 = m.continuous("x5", lb=10, ub=1000)
    x3 = m.continuous("x3", lb=1000, ub=10000)
    x8 = m.continuous("x8", lb=10, ub=1000)
    lifter = _Lifter(m)

    def ratio_2500_x5_over_x3():
        return distribute_products(BinaryOp("/", BinaryOp("*", Constant(2500.0), x5), x3))

    e1, e2 = ratio_2500_x5_over_x3(), ratio_2500_x5_over_x3()
    assert e1 is not e2 and repr(e1) == repr(e2)
    a1 = lifter.expression(e1)
    a2 = lifter.expression(e2)
    # Structurally identical -> one shared aux. Fails under id(): the two distinct
    # objects miss each other's cache entry and allocate separate auxes.
    assert a1 is a2

    # A structurally different ratio gets its own aux, never the stale one.
    a3 = lifter.expression(distribute_products(BinaryOp("/", x5, x8)))
    assert a3 is not a1


@pytest.mark.correctness
def test_nested_ratio_solve_is_sound():
    """End-to-end soundness for nested ratios (ex7_2_3's e4 shape).

    A certified-global solution MUST satisfy the *original* constraint. Before
    the structural-key fix the lifted reform dropped a denominator and the solver
    certified an infeasible point; here we solve the constraint's small sibling
    and assert the returned, certified optimum honors the original e4.
    """
    m = dm.Model("nested_ratio_solve")
    x3 = m.continuous("x3", lb=1000, ub=10000)
    x5 = m.continuous("x5", lb=10, ub=1000)
    x8 = m.continuous("x8", lb=10, ub=1000)
    m.subject_to(1.25e6 / (x3 * x8) + x5 / x8 - 2500 * x5 / x3 / x8 <= 1)
    m.minimize(x3 + x5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.gap_certified, "small nested-ratio model should certify global"
    x3v, x5v, x8v = (float(r.x[n]) for n in ("x3", "x5", "x8"))
    e4 = 1.25e6 / (x3v * x8v) + x5v / x8v - 2500 * x5v / x3v / x8v
    assert e4 <= 1 + 1e-4, (
        f"certified solution violates the original e4 (={e4}); a dropped "
        "denominator would certify an infeasible point"
    )
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4  # sound dual bound


# ---------------------------------------------------------------------------
# Entropy canonicalization: x*log(x) -> entropy(x)  (issue #207)
# ---------------------------------------------------------------------------


def _has_entropy(expr) -> bool:
    """True if ``entropy(...)`` appears anywhere in the expression DAG."""
    if isinstance(expr, FunctionCall):
        return expr.func_name == "entropy" or any(_has_entropy(a) for a in expr.args)
    for attr in ("left", "right", "operand"):
        child = getattr(expr, attr, None)
        if child is not None and _has_entropy(child):
            return True
    return False


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * dm.log(x),  # canonical product
        lambda x, y: dm.log(x) * x,  # reversed factor order
        lambda x, y: 2.5 * x * dm.log(x),  # with a constant coefficient
        lambda x, y: -(x * dm.log(x)),  # negated
        lambda x, y: x * dm.log(x) + y * dm.log(y),  # separable sum
    ],
)
def test_entropy_product_is_canonicalized(build):
    """``c·x·log(x)`` in the objective is rewritten to ``entropy(x)``."""
    m = dm.Model("ent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.001, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert out is not m
    assert _has_entropy(out._objective.expression)


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * x * dm.log(x),  # x**2 * log(x) is not entropy
        lambda x, y: x * dm.log(y),  # log of a different variable
        lambda x, y: x * dm.log(2 * x),  # log of an affine argument
        lambda x, y: x * y,  # no log factor at all
        lambda x, y: x * dm.exp(x),  # transcendental that is not log
    ],
)
def test_non_entropy_products_left_untouched(build):
    """Structurally-different products must not be (mis)matched as entropy."""
    m = dm.Model("noent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.001, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert out is m  # unchanged: same object
    assert not _has_entropy(out._objective.expression)


def test_entropy_domain_guard_rejects_negative_lb():
    """``x·log(x)`` is only canonicalized when ``x``'s box is nonnegative —
    consistent with ``relax_entropy``'s ``lb >= 0`` requirement."""
    m = dm.Model("negdomain")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x * dm.log(x))
    out = canonicalize_entropy(m)
    assert out is m
    assert not _has_entropy(out._objective.expression)


# --------------------------------------------------------------------------- #
# R4 — zero-spanning product-factor lift (DISCOPT_LIFT_ZERO_SPANNING_FACTORS)  #
# --------------------------------------------------------------------------- #
#
# A product ``f(x)·g(x)`` whose non-atomic factor ``f`` has an interval spanning
# 0 already gets ``f`` lifted to a bounded aux ``w == f`` (via the blow-up
# prelift). The default spatial-branching policy deprioritizes every lifted aux
# (a product aux ``w = x_i·x_j`` cannot shrink its own envelope). For a
# zero-spanning FACTOR that reasoning is inverted: branching ``w`` at 0 splits
# the factor's sign and tightens the ``w·g`` McCormick envelope — the only move
# that un-pins the bound. The flag tags those auxes so the solver keeps them
# branchable. Default OFF => no tagging => byte-identical branching set.


def _st_e36_shaped(name="zsf"):
    """A 2-var st_e36-shaped model: a product of a zero-spanning quadratic factor
    ``f = x^2 - 6x - 11 + 0.8y`` (∋ 0 on the box) and a strictly-positive
    sum-of-squares product ``g``, constrained ``f·g == 0``. Since ``g > 0`` the
    feasible set is exactly ``f == 0``; the objective's box-min off the manifold
    pins the McCormick product bound until ``f``'s lifted aux is split at 0.
    NOT the named instance — built here so the test is a class probe, not an
    instance hack."""
    m = dm.Model(name)
    x = m.continuous("x", lb=3.0, ub=5.5)
    y = m.continuous("y", lb=15.0, ub=25.0)
    m.minimize(2 * x**2 + 0.008 * y**3 - 3.2 * x * y - 2 * y)
    f = x**2 - 6 * x - 11 + 0.8 * y
    g = (
        ((-0.62 * y + 3.25 * x) ** 2 + (-6.35 + 0.2 * y + x) ** 2)
        * ((-0.66 * y + 3.55 * x) ** 2 + (-6.85 + 0.2 * y + x) ** 2)
        * ((-0.7 * y + 3.6 * x) ** 2 + (-7.1 + 0.2 * y + x) ** 2)
        * ((-0.82 * y + 3.8 * x) ** 2 + (-7.9 + 0.2 * y + x) ** 2)
    )
    m.subject_to(f * g == 0)
    m.subject_to(-0.2 * x * y + 0.6 * y + dm.exp(x - 3) - 1 <= 0)
    return m


def test_r4_default_off_no_tagging(monkeypatch):
    """Flag OFF (default): no aux is tagged, so the branching set is unchanged."""
    monkeypatch.delenv("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", raising=False)
    m = _st_e36_shaped()
    assert has_factorable_work(m)
    m2 = factorable_reformulate(m)
    # The lift still happens (the aux exists) — only the *tagging* is gated.
    assert _aux_names(m2), "the zero-spanning factor should still be lifted"
    assert getattr(m2, "_zero_spanning_factor_auxes", set()) == set()


def test_r4_flag_on_tags_zero_spanning_factor(monkeypatch):
    """Flag ON: the zero-spanning product-factor aux is tagged for branching."""
    monkeypatch.setenv("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", "1")
    m = _st_e36_shaped()
    m2 = factorable_reformulate(m)
    tagged = getattr(m2, "_zero_spanning_factor_auxes", set())
    assert tagged, "a zero-spanning product factor must be tagged when the flag is on"
    # Every tagged aux really spans 0 over its lifted box.
    for v in m2._variables:
        if v.name in tagged:
            import numpy as np

            lo, hi = float(np.min(v.lb)), float(np.max(v.ub))
            assert lo < 0.0 < hi, f"tagged aux {v.name} box [{lo},{hi}] must span 0"


def test_r4_positive_factor_not_tagged(monkeypatch):
    """A product whose lifted factors are all strictly one-signed (never span 0)
    tags nothing — the flag must not indiscriminately mark every product aux."""
    monkeypatch.setenv("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", "1")
    m = dm.Model("pos_factors")
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    # Both factors are sums of squares plus a positive constant => strictly > 0.
    f = (x + y) ** 2 + (x - 0.5 * y) ** 2 + 1.0
    g = (
        ((0.5 * x + y) ** 2 + (x + 0.3 * y) ** 2 + 1.0)
        * ((0.4 * x + y) ** 2 + (x + 0.2 * y) ** 2 + 1.0)
        * ((0.3 * x + y) ** 2 + (x + 0.1 * y) ** 2 + 1.0)
    )
    m.minimize(x + y)
    m.subject_to(f * g <= 5000.0)
    if has_factorable_work(m):
        m2 = factorable_reformulate(m)
        for v in m2._variables:
            if v.name in getattr(m2, "_zero_spanning_factor_auxes", set()):
                import numpy as np

                lo, hi = float(np.min(v.lb)), float(np.max(v.ub))
                assert lo < 0.0 < hi  # any tagged aux must genuinely span 0


@pytest.mark.correctness
@pytest.mark.slow
def test_r4_flag_on_unpins_pinned_product(monkeypatch):
    """The st_e36-shaped pinned-product model: with the flag OFF the product's
    McCormick bound is pinned at a box-min constant far below the optimum; with
    it ON the zero-spanning factor aux becomes branchable and the bound climbs to
    (essentially) the optimum. Asserts the *lever* (bound un-pinning) and
    soundness — not a wall-clock certification race, which is machine-dependent.

    The pin (OFF) is ≈ -304.5; the optimum is ≈ -246.0. A ≥ 25 % relative gap
    reduction (the R4 acceptance threshold) is the falsifiable claim.
    """
    monkeypatch.setenv("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_on = _st_e36_shaped().solve(time_limit=60, gap_tolerance=1e-4)
    monkeypatch.setenv("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_off = _st_e36_shaped().solve(time_limit=20, gap_tolerance=1e-4)

    assert r_on.bound is not None and r_off.bound is not None
    assert r_on.objective is not None and r_off.objective is not None
    opt = r_on.objective  # both reach the same incumbent (the true optimum)
    # Soundness (both paths): the dual bound never crosses the optimum (min).
    assert r_on.bound <= opt + 1e-4, "ON bound must not exceed the optimum (false cert)"
    assert r_off.bound <= opt + 1e-4, "OFF bound must not exceed the optimum"
    # Non-regression: ON is at least as tight as OFF.
    assert r_on.bound >= r_off.bound - 1e-4, "ON bound must not regress vs OFF"
    # The lever: ON reduces the root/OFF gap to the optimum by >= 25 %.
    gap_off = abs(opt - r_off.bound)
    gap_on = abs(opt - r_on.bound)
    assert gap_off > 1e-3, "probe must actually be pinned with the flag OFF"
    reduction = (gap_off - gap_on) / gap_off
    assert reduction >= 0.25, (
        f"flag ON must un-pin the bound >= 25 % "
        f"(OFF gap {gap_off:.3f} -> ON gap {gap_on:.3f}, reduction {reduction:.1%})"
    )


def test_entropy_canonicalized_in_constraint_body():
    """The rewrite fires in constraint bodies too, not just the objective."""
    m = dm.Model("entcon")
    x = m.continuous("x", lb=0.001, ub=1.0)
    m.minimize(x)
    m.subject_to(x * dm.log(x) <= -0.1)
    out = canonicalize_entropy(m)
    assert out is not m
    assert any(_has_entropy(c.body) for c in out._constraints)


@pytest.mark.correctness
def test_separable_entropy_objective_certifies():
    """End-to-end: a separable ``Σ xᵢ·log(xᵢ)`` (entropy/Gibbs) objective is
    solved to its optimum *and certified*, where the raw product previously left
    the dual bound stuck at a constant separable floor (issue #207)."""
    import math

    m = dm.Model("gibbs")
    xs = [m.continuous(f"x{i}", lb=0.001, ub=1.0) for i in range(3)]
    m.subject_to(sum(xs) == 1.0)
    m.minimize(sum(x * dm.log(x) for x in xs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    # Optimum of Σ xᵢ log xᵢ on the simplex is uniform xᵢ = 1/3 -> -log(3).
    assert r.status == "optimal"
    assert r.gap_certified, "separable entropy objective should certify"
    assert r.objective == pytest.approx(-math.log(3), abs=1e-3)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4  # sound dual bound


# ---------------------------------------------------------------------------
# distributed entropy: x*(affine + log(x)) -> x*affine + entropy(x)  (ex6_1_4)
# ---------------------------------------------------------------------------


def _obj_values_match(m_before, m_after, lo=0.01, hi=2.0, n=200, seed=0):
    """The two objectives must evaluate identically over a random positive box."""
    import numpy as np
    from discopt._jax.dag_compiler import compile_expression

    f0 = compile_expression(m_before._objective.expression, m_before)
    f1 = compile_expression(m_after._objective.expression, m_after)
    nvar = sum(max(1, int(np.prod(v.shape))) for v in m_before._variables)
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n):
        x = rng.uniform(lo, hi, size=nvar)
        err = max(err, abs(float(f0(x)) - float(f1(x))))
    return err


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * (0.5 + dm.log(x)),  # constant + log(x)  (ex6_1_4 shape)
        lambda x, y: x * (dm.log(x) + 0.5),  # log(x) + constant  (order swap)
        lambda x, y: x * (dm.log(x) - 2.0),  # log(x) - constant
        lambda x, y: 2.5 * x * (0.5 + dm.log(x)),  # outer constant coefficient
        lambda x, y: -(x * (0.5 + dm.log(x))),  # negated
        lambda x, y: x * (3.0 * y + dm.log(x)),  # affine remainder with another var (bilinear)
    ],
)
def test_distributed_entropy_is_canonicalized(build):
    """``c·x·(affine + log(x))`` recovers ``entropy(x)`` while staying exact."""
    m = dm.Model("dent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.001, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert out is not m
    assert _has_entropy(out._objective.expression)
    # The rewrite is algebraically exact, not just structural.
    assert _obj_values_match(m, out) < 1e-9


def test_distributed_entropy_preserves_affine_remainder():
    """The affine wrapper term must survive: ``x·(c + log(x)) = c·x + entropy(x)``,
    so the objective value (incl. the linear part) is unchanged."""
    m = dm.Model("drem")
    x = m.continuous("x", lb=0.001, ub=5.0)
    y = m.continuous("y", lb=0.001, ub=5.0)
    m.minimize(x * (0.28809 + dm.log(x)) + 1.5 * x * y)
    out = canonicalize_entropy(m)
    assert _has_entropy(out._objective.expression)
    assert _obj_values_match(m, out) < 1e-9


def test_distributed_entropy_domain_guard():
    """No rewrite when ``x``'s box admits negatives (entropy domain violated)."""
    m = dm.Model("dneg")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x * (0.5 + dm.log(x)))
    out = canonicalize_entropy(m)
    assert out is m
    assert not _has_entropy(out._objective.expression)


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * (0.5 + dm.log(y)),  # log of a different variable
        lambda x, y: x * (0.5 + dm.log(2 * x)),  # log of an affine argument
        lambda x, y: x * (dm.log(x) + dm.log(x)),  # two entropy-log terms -> ambiguous
        lambda x, y: x * x * (0.5 + dm.log(x)),  # x**2 wrapper -> not entropy
    ],
)
def test_distributed_entropy_non_matches_left_untouched(build):
    """Lookalikes of the distributed form must not be (mis)matched."""
    m = dm.Model("dnoent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.001, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert not _has_entropy(out._objective.expression)


@pytest.mark.correctness
def test_distributed_entropy_with_bilinear_certifies():
    """End-to-end ex6_1_4-shaped repro: a diagonal entropy in the distributed
    ``x·(c + log(x))`` form plus an off-diagonal bilinear coupling must solve and
    *certify*. Before the distributed-form recovery the relaxer hit an
    un-decomposable ``x·log(x)`` product and left the bound at a constant floor
    (issue #207)."""
    m = dm.Model("ex6_like")
    x0 = m.continuous("x0", lb=0.01, ub=1.0)
    x1 = m.continuous("x1", lb=0.01, ub=1.0)
    m.subject_to(x0 + x1 == 1.0)
    # diagonal entropy wrapped in an affine offset + a small bilinear coupling
    m.minimize(x0 * (0.2 + dm.log(x0)) + x1 * (0.3 + dm.log(x1)) + 0.1 * x0 * x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=120, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.gap_certified, "distributed entropy + bilinear should certify"
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4  # sound dual bound


# ---------------------------------------------------------------------------
# centropy canonicalization: x*log(x/y) -> centropy(x, y)  (issue #207)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * dm.log(x / y),  # variable denominator
        lambda x, y: dm.log(x / y) * x,  # reversed factor order
        lambda x, y: 2.5 * x * dm.log(x / y),  # with a constant coefficient
        lambda x, y: -(x * dm.log(x / y)),  # negated
        lambda x, y: x * dm.log(x / 0.5),  # constant denominator
        lambda x, y: x * dm.log(x / (x + y)),  # affine-sum denominator (Gibbs form)
    ],
)
def test_centropy_product_is_canonicalized(build):
    """``c·x·log(x/y)`` is rewritten to ``centropy(x, y)``."""
    m = dm.Model("cent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert out is not m

    def has_centropy(e):
        if isinstance(e, FunctionCall):
            return e.func_name == "centropy" or any(has_centropy(a) for a in e.args)
        for attr in ("left", "right", "operand"):
            child = getattr(e, attr, None)
            if child is not None and has_centropy(child):
                return True
        return False

    assert has_centropy(out._objective.expression)


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * dm.log(y / x),  # numerator is the *wrong* variable
        lambda x, y: y * dm.log(x / y),  # bare factor != log numerator
        lambda x, y: x * x * dm.log(x / y),  # x²·log(x/y) is not centropy
    ],
)
def test_non_centropy_products_left_untouched(build):
    """Structurally-different ratios-in-log must not be matched as centropy."""
    m = dm.Model("nocent")
    x = m.continuous("x", lb=0.001, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.minimize(build(x, y))
    out = canonicalize_entropy(m)
    assert out is m


def test_centropy_domain_guard_rejects_nonpositive_denominator():
    """``x·log(x/y)`` is only canonicalized when ``y`` is provably positive."""
    m = dm.Model("centdom")
    x = m.continuous("x", lb=0.001, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)  # y can be <= 0
    m.minimize(x * dm.log(x / y))
    out = canonicalize_entropy(m)
    assert out is m


@pytest.mark.parametrize(
    "name,x_lb,y_lb,expect_convex",
    [
        ("convex", 0.0, 0.1, True),  # x >= 0, y > 0  -> jointly convex
        ("x_neg", -1.0, 0.1, False),  # x may be negative -> abstain
        ("y_nonpos", 0.0, -1.0, False),  # y not provably > 0 -> abstain
    ],
)
def test_centropy_curvature_rule(name, x_lb, y_lb, expect_convex):
    """``centropy`` is detected jointly CONVEX only on a provable ``x≥0, y>0``
    box with affine arguments; otherwise the rule soundly abstains (UNKNOWN)."""
    from discopt._jax.convexity import classify_expr
    from discopt._jax.convexity.lattice import Curvature

    m = dm.Model(name)
    x = m.continuous("x", lb=x_lb, ub=1.0)
    y = m.continuous("y", lb=y_lb, ub=2.0)
    curv = classify_expr(FunctionCall("centropy", x, y), m)
    if expect_convex:
        assert curv == Curvature.CONVEX
    else:
        assert curv == Curvature.UNKNOWN


@pytest.mark.correctness
def test_relative_entropy_objective_certifies():
    """End-to-end: a separable relative-entropy / Gibbs objective
    ``Σ xᵢ·log(xᵢ/yᵢ)`` with a *variable* denominator is solved and *certified*
    on the convex fast path (jointly-convex centropy), with a sound bound."""
    m = dm.Model("kl")
    x = [m.continuous(f"x{i}", lb=0.01, ub=1.0) for i in range(3)]
    y = [m.continuous(f"y{i}", lb=0.2, ub=0.8) for i in range(3)]
    m.subject_to(sum(x) == 1.0)
    m.subject_to(sum(y) == 1.5)
    m.minimize(sum(xi * dm.log(xi / yi) for xi, yi in zip(x, y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.gap_certified, "relative-entropy objective should certify (jointly convex)"
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4  # sound dual bound


@pytest.mark.correctness
@pytest.mark.slow
def test_nvs01_certifies_with_sound_bound():
    """Raw nvs01 (division + trilinear + composite objective) certifies to the
    MINLPLib optimum 12.4697 with a valid dual bound (bound <= objective)."""
    nl = _DATA / "nvs01.nl"
    if not nl.exists():
        pytest.skip("nvs01.nl not vendored")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = dm.from_nl(str(nl)).solve(
            time_limit=120, gap_tolerance=1e-4, max_nodes=2_000_000, subnlp_frequency=1
        )
    assert r.status == "optimal"
    assert r.objective == pytest.approx(12.4697, abs=1e-2)
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-4  # sound dual bound


# ---------------------------------------------------------------------------
# TD-A: lift an integer power of a univariate call ``g(x)**n`` to ``t**n``
# (t == g(x)), flag ``DISCOPT_LIFT_LOOSE_PRODUCTS`` (default OFF).
# ---------------------------------------------------------------------------


def _squared_log_model():
    """An nvs09-shaped probe (not the named instance): a separable sum of squared
    logs whose ``distribute_products`` form is the ``log·log`` product the MILP
    linearizer drops. Two integer vars keep it small and certifiable."""
    import math

    m = dm.Model("sqlog")
    xs = [m.integer(f"x{i}", lb=3, ub=9) for i in range(2)]
    obj = 0
    for x in xs:
        obj = obj + dm.log(x - 2) ** 2 + dm.log(10 - x) ** 2
    m.minimize(obj)
    return m, xs, math


def test_tda_flag_off_leaves_call_power_untouched(monkeypatch):
    """Flag OFF: ``g(x)**n`` is NOT lifted (reform byte-identical to baseline)."""
    monkeypatch.delenv("DISCOPT_LIFT_LOOSE_PRODUCTS", raising=False)
    m, _xs, _ = _squared_log_model()
    # The squared-log alone is not factorable work with the flag off.
    assert not has_factorable_work(m)


def test_tda_flag_on_lifts_call_power(monkeypatch):
    """Flag ON: each ``log(·)**2`` is lifted to a ``t**2`` monomial with an aux
    ``t == log(·)`` (bounded by the log's FBBT interval)."""
    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "1")
    m, _xs, _ = _squared_log_model()
    assert has_factorable_work(m)
    m2 = factorable_reformulate(m)
    assert m2 is not m
    auxes = _aux_names(m2)
    # 2 vars * 2 logs each -> 4 lifted univariate-call auxes.
    assert len(auxes) == 4, f"expected 4 call auxes, got {auxes}"

    # Every aux is a bounded continuous var (t == log(arg)) with a finite box.
    import numpy as np

    for v in m2._variables:
        if v.name in auxes:
            lo, hi = float(np.min(v.lb)), float(np.max(v.ub))
            assert np.isfinite(lo) and np.isfinite(hi)
            assert lo <= hi


def test_tda_lift_is_exact_identity_feasible_sampling(monkeypatch):
    """Feasible-point sampling: the lift ``t == g(x)`` cuts no feasible point.

    For random x in the box, the lifted objective evaluated at (x, t = g(x))
    equals the original objective at x, and every aux-defining equality holds —
    i.e. the reformulation is an exact identity substitution, never a relaxation
    that could exclude a feasible point.
    """
    import numpy as np
    from discopt._jax.dag_compiler import compile_expression

    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "1")
    m, _xs, math = _squared_log_model()
    m2 = factorable_reformulate(m)

    f_orig = compile_expression(m._objective.expression, m)
    f_lift = compile_expression(m2._objective.expression, m2)
    # aux-defining equalities: body == 0 for each ``t - g(x)``.
    aux_bodies = [
        (c.body, compile_expression(c.body, m2))
        for c in m2._constraints
        if any(vn in _aux_names(m2) for vn in _referenced_names(c.body))
    ]

    orig_vars = [v.name for v in m._variables]
    lift_vars = [v.name for v in m2._variables]
    aux_names = set(_aux_names(m2))

    rng = np.random.default_rng(0)
    max_obj_err = 0.0
    max_aux_resid = 0.0
    for _ in range(2000):
        # sample the original (non-aux) variables over their integer box
        xvals = {vn: float(rng.integers(3, 10)) for vn in orig_vars}
        # derive the aux values t = log(arg) so the point is feasible in m2
        full = dict(xvals)
        # solve aux defs in order (they are t - log(arg) = 0, arg over base vars)
        for v in m2._variables:
            if v.name in aux_names:
                full.setdefault(v.name, 0.0)
        # fixpoint once: evaluate each aux body to recover t (t appears linearly)
        for body, _fn in aux_bodies:
            # body = t - g(x); set t = g(x)
            t_name, g_val = _solve_aux_body(body, full, m2)
            if t_name is not None:
                full[t_name] = g_val
        x_orig = np.array([xvals[vn] for vn in orig_vars], dtype=float)
        x_lift = np.array([full[vn] for vn in lift_vars], dtype=float)
        o0 = float(f_orig(x_orig))
        o1 = float(f_lift(x_lift))
        max_obj_err = max(max_obj_err, abs(o0 - o1))
        for body, fn in aux_bodies:
            max_aux_resid = max(max_aux_resid, abs(float(fn(x_lift))))
    assert max_obj_err < 1e-6, f"lifted objective diverged from original: {max_obj_err}"
    assert max_aux_resid < 1e-6, f"aux equality violated at t=g(x): {max_aux_resid}"


@pytest.mark.correctness
@pytest.mark.slow
def test_tda_nvs09_root_bound_tightens(monkeypatch):
    """nvs09: the flag ON tightens the root bound >= 25 % relative vs OFF, and the
    ON bound never crosses the oracle (-43.134, min sense) — the TD-A acceptance.
    """
    nl = _NL_DATA / "nvs09.nl"
    if not nl.exists():
        pytest.skip("nvs09.nl not vendored")
    oracle = -43.1343369200

    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_on = dm.from_nl(str(nl)).solve(time_limit=8)
    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_off = dm.from_nl(str(nl)).solve(time_limit=8)

    assert r_on.root_bound is not None and r_off.root_bound is not None
    # Soundness: neither root bound crosses the oracle (min: bound <= optimum).
    assert r_off.root_bound <= oracle + 1e-4
    assert r_on.root_bound <= oracle + 1e-4
    # ON is at least as tight (higher) as OFF.
    assert r_on.root_bound >= r_off.root_bound - 1e-6
    gap_off = abs(oracle - r_off.root_bound)
    gap_on = abs(oracle - r_on.root_bound)
    reduction = (gap_off - gap_on) / gap_off
    assert reduction >= 0.25, (
        f"flag ON must tighten nvs09 root gap >= 25 % "
        f"(OFF {r_off.root_bound:.2f} -> ON {r_on.root_bound:.2f}, reduction {reduction:.1%})"
    )


def test_tda_generalizes_to_sin_power(monkeypatch):
    """The lift keys on the *structure* (call ** int), not a named function: a
    ``sin(x)**2`` term is lifted the same way ``log(x-2)**2`` is (mathopt witness
    family). Demonstrates the rule is not log-specific."""
    monkeypatch.setenv("DISCOPT_LIFT_LOOSE_PRODUCTS", "1")
    m = dm.Model("sinpow")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(dm.sin(x) ** 2 + 0.1 * x)
    assert has_factorable_work(m)
    m2 = factorable_reformulate(m)
    assert m2 is not m
    assert len(_aux_names(m2)) == 1, "sin(x)**2 must lift one aux t == sin(x)"


def _referenced_names(expr):
    names = set()

    def walk(e):
        if isinstance(e, Variable):
            names.add(e.name)
        for a in ("left", "right", "operand"):
            if hasattr(e, a) and getattr(e, a) is not None:
                walk(getattr(e, a))
        if hasattr(e, "args") and getattr(e, "args", None):
            try:
                for k in e.args:
                    walk(k)
            except TypeError:
                pass

    walk(expr)
    return names


def _solve_aux_body(body, values, model):
    """Given an aux-defining body ``t - g(x) == 0`` and a dict of base-var values,
    return (t_name, g_value) by evaluating g at the base values."""
    import numpy as np
    from discopt._jax.dag_compiler import compile_expression
    from discopt.modeling.core import BinaryOp

    # body is ``t - g(x)`` (a subtraction with t the aux on the left leaf).
    if not (isinstance(body, BinaryOp) and body.op == "-"):
        return None, None
    left = body.left
    if not isinstance(left, Variable):
        return None, None
    t_name = left.name
    g_expr = body.right
    fn = compile_expression(g_expr, model)
    lift_vars = [v.name for v in model._variables]
    xv = np.array([values.get(vn, 0.0) for vn in lift_vars], dtype=float)
    return t_name, float(fn(xv))
