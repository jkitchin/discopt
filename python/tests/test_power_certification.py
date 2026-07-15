"""Regression locks for power-of-expression relaxations (issue #141, bucket 4).

A nonconvex power term ``g(x)**p`` with a non-integer exponent is lifted to an
auxiliary column carrying the convex/concave secant–tangent envelope for the
single-variable composite (curvature chosen from ``p`` and the sign of the base
over its box). This already works on ``main`` for bare-variable bases, affine
composite bases (e.g. ``(2x+1)**0.75``), and even product bases
(``(x*y)**0.75``), so the listed bucket-4 instances all certify with a sound dual
bound. These tests pin that behavior.

Scope note (issue #141): ``ex1221`` / ``st_e15`` / ``ex1225`` are guarded by
``test_monomial_lp_bound.py`` and ``st_e04`` by ``test_exp_certification.py``;
``st_e11`` and ``ex1226`` were unguarded and are added here. The
``(7936.51*x)**0.75`` drop the issue cited lives in ``4stufen`` / ``beuster``
(not in the bucket-4 list) — those instances are dominated by bucket-1
log-of-ratio / non-constant-division terms and would not certify from a power
fix alone; the isolated power term there is structurally identical to a
synthetic case that certifies here.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import math
from pathlib import Path

import discopt.modeling as dm
import pytest

pytestmark = [pytest.mark.claim_boundary]

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
def test_affine_base_fractional_power_certifies():
    """``(2x+1)**0.75`` (affine composite base, concave) certifies soundly."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize((2 * x + 1) ** 0.75)  # min at x=0 -> 1**0.75 = 1
    r = m.solve(time_limit=15, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, 1.0, abs_tol=1e-3)
    assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"
    assert r.bound <= 1.0 + 1e-3, "dual bound must not exceed the true optimum"


# NOTE (#632 cutover): test_shifted_integer_power_lift_guards was a unit test of the
# deleted federation predicate `_should_claim_composite` (which decided whether a
# shifted integer power was "claimed" as a composite). The uniform engine atomizes
# and relaxes every power by construction (no claim predicate); the end-to-end
# soundness that mattered — (x-1)**4 certifies with a sound bound — is asserted by
# test_shifted_even_integer_power_certifies below, so the predicate unit test was
# deleted rather than rewritten against a removed API. The `_affine_base_power_curvature`
# curvature helper it sat next to is KEPT and still unit-tested here.


def test_affine_power_curvature_guards_crossing_zero_base():
    from discopt._jax.milp_relaxation import _affine_base_power_curvature

    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    box = {}

    assert _affine_base_power_curvature((x - 1.0) ** 4, m, box) == "convex"
    assert _affine_base_power_curvature((x - 1.0) ** 3, m, box) is None
    assert _affine_base_power_curvature((x - 1.0) ** 1.5, m, box) is None
    assert _affine_base_power_curvature((x - 1.0) ** (-1.0), m, box) is None


@pytest.mark.correctness
def test_shifted_even_integer_power_certifies():
    """``(x - 1)**4`` uses the composite power lift and certifies soundly."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize((x - 1.0) ** 4)
    r = m.solve(time_limit=15, gap_tolerance=1e-4)

    assert r.status == "optimal"
    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, 0.0, abs_tol=1e-4)
    assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"
    assert r.gap_certified


def test_nonaffine_even_integer_power_abstains_from_affine_curvature_shortcut():
    from discopt._jax.milp_relaxation import _affine_base_power_curvature

    m = dm.Model()
    x = m.continuous("x", lb=math.pi / 2.0 - 1.0, ub=math.pi / 2.0 + 1.0)

    assert _affine_base_power_curvature(dm.sin(x) ** 4, m, {}) is None


def test_nonaffine_even_integer_power_bound_remains_sound():
    m = dm.Model()
    x = m.continuous("x", lb=math.pi / 2.0 - 1.0, ub=math.pi / 2.0 + 1.0)
    m.minimize(dm.sin(x) ** 4)
    r = m.solve(time_limit=15, gap_tolerance=1e-4)

    true_min = math.cos(1.0) ** 4
    assert r.status == "optimal"
    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, true_min, abs_tol=1e-3)
    assert r.bound <= true_min + 1e-3, "dual bound must not exceed the true optimum"


@pytest.mark.correctness
def test_product_base_fractional_power_certifies():
    """``(x*y)**0.75`` (product base) certifies soundly with a valid bound."""
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize((x * y) ** 0.75)  # min at x=y=0.5 -> 0.25**0.75
    r = m.solve(time_limit=15, gap_tolerance=1e-4)

    opt = 0.25**0.75
    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, opt, rel_tol=1e-3)
    assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"
    assert r.bound <= opt + 1e-3, "dual bound must not exceed the true optimum"


@pytest.mark.correctness
@pytest.mark.parametrize(
    "p, expected",
    [
        (2, (0.0, 4.0)),  # already handled on main; pin it
        (4, (0.0, 16.0)),  # C-34/FR-1: was (16.0, 16.0) pre-fix
        (6, (0.0, 64.0)),  # C-34/FR-1: was (64.0, 64.0) pre-fix
        (8, (0.0, 256.0)),
    ],
)
def test_even_power_straddle_bound_includes_interior_min(p, expected):
    """C-34/FR-1: for an even power over a base straddling 0, the interval
    bound must include the interior minimum at 0. Endpoint-only bounds omit
    it, under-approximating the range → invalid aux box → false optimum."""
    from discopt._jax.gdp_reformulate import _bound_expression

    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    assert _bound_expression(x**p, m) == expected


@pytest.mark.correctness
@pytest.mark.parametrize(
    "lb, ub, p, expected",
    [
        (1.0, 3.0, 4, (1.0, 81.0)),  # positive base: monotone endpoints
        (-3.0, -1.0, 4, (1.0, 81.0)),  # negative base: monotone endpoints
        (-2.0, 2.0, 3, (-8.0, 8.0)),  # odd power: monotone endpoints (unchanged)
    ],
)
def test_power_bound_non_straddling_and_odd_unchanged(lb, ub, p, expected):
    """The fix must not perturb one-signed even powers or odd powers, which
    are monotone on the interval and correctly bounded by their endpoints."""
    from discopt._jax.gdp_reformulate import _bound_expression

    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    assert _bound_expression(x**p, m) == expected


@pytest.mark.correctness
def test_c34_even_power_straddle_no_false_optimum():
    """C-34/FR-1 end-to-end: ``x**4 * y`` on a box where x straddles 0. The
    buggy [16,16] aux box for x**4 forced x=+-2, cutting off the true optimum
    at (0.5, 1) and returning a false certified obj ~= 3.13. The true optimum
    is 0."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize((x - 0.5) ** 2 + (y - 1.0) ** 2)
    m.subject_to(x**4 * y <= 1.0)
    r = m.solve(time_limit=60, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    # (0.5, 1) is feasible: 0.5**4 * 1 = 0.0625 <= 1, obj = 0.
    assert r.objective <= 1e-3, f"false optimum: obj={r.objective} (true optimum is 0)"
    assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"


@pytest.mark.correctness
@pytest.mark.parametrize(
    "instance, optimum",
    [
        # st_e11 (x**0.6 fractional-power terms): oracle optimum from
        # minlplib.solu = 189.3116297. discopt certifies bound==objective==189.3116
        # (gap_certified), which matches the oracle. The prior 189.3292 reference
        # was stale (Δ0.018) — adjudicated 2026-07-14, xfail removed.
        ("st_e11", 189.3116297),
        ("ex1226", -17.0),  # power terms in objective/constraints
    ],
)
def test_unguarded_power_instances_certify(instance, optimum):
    """Previously-unguarded bucket-4 instances certify with a sound dual bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert abs(r.objective - optimum) <= 1e-2, f"[{instance}] obj={r.objective} != {optimum}"
    # Soundness: a valid dual bound never exceeds the optimum.
    assert r.bound <= optimum + 1e-2, f"[{instance}] unsound dual bound {r.bound} > {optimum}"
    assert r.gap_certified, f"[{instance}] expected certified optimality"
