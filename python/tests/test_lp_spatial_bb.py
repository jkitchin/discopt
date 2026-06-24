"""LP-node spatial branch-and-bound engine (discopt#280: SCIP-grade integer
products). Pins correctness (matches brute force / valid dual bound) and the
scope gate. See docs/dev/scip-gap-nvs-diagnosis.md."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.lp_spatial_bb import solve_lp_spatial_bb  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


def _tiny(ux, uy, rhs, coef):
    m = dm.Model("t")
    a = m.integer("a", lb=0, ub=ux)
    b = m.integer("b", lb=0, ub=uy)
    m.minimize(a + coef * b)
    m.subject_to(a * b >= rhs)
    return m


def _brute(ux, uy, rhs, coef):
    return min(
        (a + coef * b for a in range(ux + 1) for b in range(uy + 1) if a * b >= rhs),
        default=None,
    )


# --------------------------------------------------------------------------- #
# scope gate (pure logic, fast)
# --------------------------------------------------------------------------- #


def test_out_of_scope_continuous_returns_none():
    """A model with a continuous variable is out of scope -> None (caller falls
    back). The collapsed-box exactness argument needs every var integer."""
    m = dm.Model("c")
    x = m.continuous("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x + y)
    m.subject_to(x * y >= 4)
    assert solve_lp_spatial_bb(m, time_limit=5) is None


def test_out_of_scope_maximize_returns_none():
    """The McCormick relaxation bound is only a valid *lower* bound for minimize."""
    m = dm.Model("mx")
    a = m.integer("a", lb=0, ub=5)
    b = m.integer("b", lb=0, ub=5)
    m.maximize(a + b)
    m.subject_to(a * b <= 6)
    assert solve_lp_spatial_bb(m, time_limit=5) is None


# --------------------------------------------------------------------------- #
# correctness: matches brute force exactly, and proves optimality
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.requires_pounce
@pytest.mark.parametrize(
    "ux,uy,rhs,coef",
    [(5, 4, 7, 1), (6, 6, 20, 2), (8, 4, 15, 1), (7, 3, 11, 3)],
)
def test_matches_brute_force(ux, uy, rhs, coef):
    r = solve_lp_spatial_bb(_tiny(ux, uy, rhs, coef), time_limit=20, gap_tolerance=1e-6)
    assert r is not None and r.status == "optimal"
    assert r.objective == pytest.approx(_brute(ux, uy, rhs, coef), abs=1e-6)


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_nvs17_dual_bound_is_valid_and_tight():
    """On nvs17 the LP-node engine must (a) never report a dual bound *above* the
    true optimum (soundness) and (b) get far closer than the default path's frozen
    root value (-65842). Full closure needs cuts (a later step)."""
    path = os.path.join(_DATA, "nvs17.nl")
    if not os.path.exists(path):
        pytest.skip("nvs17 unavailable")
    r = solve_lp_spatial_bb(dm.from_nl(path), time_limit=45, gap_tolerance=1e-4)
    assert r is not None
    assert r.bound is not None and r.bound <= -1100.4 + 1e-4  # valid lower bound
    assert r.bound >= -2000.0  # vastly tighter than the -65842 frozen root
    if r.objective is not None:
        assert r.objective >= -1100.4 - 1e-4  # incumbent can't beat the true optimum
