"""GDP-1 / GDP-2 regressions: big-M validity in the GDP reformulation (#413).

GDP-1 — *too-small* big-M cuts feasible points of the active disjunct.
    ``_compute_big_m`` treated any variable bound ≥ 1e15 as "effectively
    infinite" and substituted ``_DEFAULT_BIG_M = 1e4``. For a variable left at
    the large-but-*finite* default bounds (±~1e20), that shrank a *valid* big-M
    (the true finite bound) down to 1e4, which is NOT a valid over-estimate of
    the body: the inactive disjunct's relaxed constraint stayed binding and cut
    off the whole active disjunct → a false-**infeasible** certificate on the
    default ``gdp_method="big-m"`` path. The sound behavior: use the true finite
    bound (a large valid M), and for a *truly* infinite bound refuse loudly
    (no valid finite M exists). The same defect lived in the SOS linking big-M.

GDP-2 — mbigm crashes on any disjunction that adds a selector.
    ``_compute_big_m_lp`` built the LP ``bounds`` list from the *mutated* model
    (which already carries the reformulation's selector binaries), so ``bounds``
    was longer than the ``c``/``A`` columns precomputed from the original model.
    The exact LP backend then panicked on the dimension mismatch, breaking every
    mbigm/auto disjunction. Fix: align ``bounds`` to the precomputed ``n_vars``.

Before the fixes: GDP-1 solve returns ``status="infeasible"`` on a feasible
model; GDP-2 reformulation raises a Rust ``PanicException``. After: GDP-1 finds
the true optimum (or refuses loudly on unbounded vars); GDP-2 reformulates and
solves soundly.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.gdp_reformulate import reformulate_gdp

pytestmark = pytest.mark.smoke


def _optx(result) -> float:
    x = (result.x or {}).get("x")
    return float(np.ravel(x)[0]) if x is not None else float("nan")


def test_gdp1_large_finite_bounds_no_false_infeasible():
    """Default (large-finite) bounds must NOT yield a false-infeasible cert."""
    m = dm.Model("gdp1")
    x = m.continuous("x")  # default ±~1e20 (finite)
    m.minimize((x - 25000.0) * (x - 25000.0))
    # Feasible region: x<=2 OR x>=20000. True optimum: x=25000, obj=0.
    m.either_or([[x <= 2], [x >= 20000]])
    r = m.solve(gdp_method="big-m")
    assert r.status == "optimal", f"expected optimal, got {r.status}"
    assert r.objective == pytest.approx(0.0, abs=1e-3)
    assert _optx(r) == pytest.approx(25000.0, abs=1e-2)


def test_gdp1_truly_infinite_bound_refuses_loudly():
    """A genuinely unbounded disjunct body has no valid finite big-M."""
    m = dm.Model("gdp1inf")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 8]])
    with pytest.raises(ValueError, match="unbounded|finite"):
        reformulate_gdp(m, method="big-m")


def test_gdp1_sos_infinite_bound_refuses_loudly():
    """SOS linking big-M must refuse an unbounded variable, not clamp it."""
    m = dm.Model("sosinf")
    y = m.continuous("y", lb=0.0, ub=np.inf)
    z = m.continuous("z", lb=0.0, ub=5.0)
    m.minimize(y + z)
    m.sos1([y, z])
    with pytest.raises(ValueError, match="infinite|finite"):
        reformulate_gdp(m, method="big-m")


def test_gdp2_mbigm_disjunction_does_not_crash():
    """mbigm on a selector-adding disjunction must reformulate + solve."""
    m = dm.Model("gdp2")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 8]])
    # Reformulation alone previously panicked in the exact LP backend.
    rm = reformulate_gdp(m, method="mbigm")
    assert len(rm._constraints) >= 3
    r = m.solve(gdp_method="mbigm")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-4)


def test_gdp2_mbigm_matches_bigm_optimum():
    """mbigm must produce the same optimum as big-m (tighter M, same region)."""

    def build():
        m = dm.Model("cmp")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y = m.continuous("y", lb=0.0, ub=10.0)
        m.minimize(-(x + y))
        m.either_or([[x <= 2, y <= 2], [x >= 7, y >= 7]])
        return m

    r_big = build().solve(gdp_method="big-m")
    r_mbm = build().solve(gdp_method="mbigm")
    assert r_big.status == r_mbm.status == "optimal"
    assert r_mbm.objective == pytest.approx(r_big.objective, abs=1e-3)
