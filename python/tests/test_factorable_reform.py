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
from discopt._jax.factorable_reform import canonicalize_entropy, factorable_reformulate
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import FunctionCall

_DATA = Path(__file__).parent / "data" / "minlplib"


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
