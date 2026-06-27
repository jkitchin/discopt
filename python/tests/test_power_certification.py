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
    "instance, optimum",
    [
        ("st_e11", 189.3292),  # x**0.6 fractional-power terms
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
