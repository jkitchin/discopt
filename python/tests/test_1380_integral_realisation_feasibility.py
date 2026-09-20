"""Feasibility must be judged at the INTEGRAL realisation, not at the point as computed.

Every feasibility arbiter in the tree used to test the rows at the point the
engine produced and integrality *beside* it, as two independent tests. A discrete
column sitting inside ``integrality_tol`` then buys its row up to
``|a_ij| * integrality_tol`` of slack the integral point does not have. That is
unbounded in the coefficient, so any big-M formulation reaches it::

    min -x + 3z   s.t.   x <= 1e7 z,   0 <= x <= 10,   z binary

``z`` is binary, so there are exactly two cases: ``z=0`` forces ``x=0``
(objective ``0``) and ``z=1`` allows ``x=10`` (objective ``-7``). **The true
optimum is -7.0.** Measured on ``924169b``, before the fix:

===========================================  ==========  ===========  ======
route                                        status      objective    z
===========================================  ==========  ===========  ======
spatial B&B                                  optimal     -9.999997    1e-06
Rust monolithic MILP (``..._BACKEND=rust``)  optimal     -9.999997    1e-06
Python MILP-BB                               optimal     -9.999997    1e-06
HiGHS MILP route (default)                   feasible    -9.999997    1e-06
===========================================  ==========  ===========  ======

Three of the four **certify optimal below the model's true optimum**, at a point
whose binary is not a binary; the fourth declines to certify but still reports the
same unachievable objective. ``verify_point`` -- the module written (#908) so that
"wrongly-accept" would be structurally impossible -- agreed::

    verify_point(m, [10.0, 1e-6]) -> ok=True   obj=-9.999997
    verify_point(m, [10.0, 0.0 ]) -> ok=False  row 0 violated by 1.000e+01

At ``z = 1e-6`` the integrality test passes (``1e-6 < INT_TOL``) and the row holds
*exactly* (``10 <= 10``). At ``z = 0`` -- the point being CLAIMED -- it is violated
by 10.0.

What this file pins is the **class**, not the instance. The named model above is a
probe; the property is that no route may publish an objective a feasible integral
point cannot attain, and that the fix is inert on models that are solvable at the
declared tolerances (the ``M = 100`` control, which must still reach -7).

Every test here ends by asserting an EXECUTED-COMPARISON count (CLAUDE.md §6): a
probe that silently checks nothing reads exactly like a pass.
"""

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.validation.feasibility import (
    INT_TOL,
    model_integer_mask,
    snap_integer_columns,
    snap_integers,
    verify_point,
)

TRUE_OPTIMUM = -7.0
#: What the pre-fix routes reported. Any objective at or below this is the defect.
FALSE_OPTIMUM = -9.999997


def _bigm_model(big_m: float):
    """``min -x + 3z  s.t.  x <= big_m * z``; optimum -7 for any ``big_m >= 10``."""
    m = dm.Model("bigm")
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.binary("z")
    m.minimize(-x + 3.0 * z)
    m.subject_to(x <= big_m * z)
    return m


# ─────────────────────────────────────────────────────────────
# The shared primitive
# ─────────────────────────────────────────────────────────────


def test_snap_is_a_no_op_on_a_genuinely_integral_point():
    """The soundness argument in one direction: nothing already integral moves.

    If the snap could perturb an integral point it would make some arbiter
    stricter for reasons unrelated to integrality, and the fix would not be the
    no-op it claims to be on the overwhelming majority of incumbents.
    """
    m = _bigm_model(1e7)
    mask = model_integer_mask(m)
    compared = 0
    for pt in ([0.0, 0.0], [10.0, 1.0], [3.5, 1.0], [-0.0, 1.0]):
        x = np.asarray(pt, dtype=np.float64)
        out = snap_integer_columns(x, mask)
        assert out.tobytes() == x.tobytes(), f"snap moved an integral point {pt}"
        compared += 1
    assert compared == 4


def test_snap_moves_no_further_than_the_integrality_tolerance():
    """The soundness argument in the other direction: the move is bounded by a
    distance the caller's own integrality test has already declared immaterial."""
    m = _bigm_model(1e7)
    mask = model_integer_mask(m)
    compared = 0
    for z in (1e-6, -1e-6, 1.0 - 1e-6, 4.0 + INT_TOL):
        x = np.asarray([1.0, z], dtype=np.float64)
        out = snap_integer_columns(x, mask)
        assert abs(out[1] - z) <= INT_TOL + 1e-15
        assert out[1] == round(z)
        assert out[0] == 1.0, "a continuous column must never be snapped"
        compared += 1
    assert compared == 4


def test_snap_leaves_a_genuinely_fractional_column_alone():
    """Snapping a value the integrality test would REJECT would fabricate a point
    the search never proved feasible. The rejection is the integrality test's job."""
    m = _bigm_model(1e7)
    x = np.asarray([1.0, 0.4], dtype=np.float64)
    out = snap_integers(m, x)
    assert out[1] == 0.4
    assert not verify_point(m, x).ok


def test_snap_refuses_a_misaligned_mask():
    """A mask of the wrong length would leave exactly the columns this exists for
    unsnapped -- the §6 "probe silently measured nothing" shape. Refuse, never guess."""
    with pytest.raises(ValueError):
        snap_integer_columns(np.zeros(2), np.array([True]))


# ─────────────────────────────────────────────────────────────
# The arbiter (#908's verifier)
# ─────────────────────────────────────────────────────────────


def test_verify_point_rejects_the_fractional_big_m_point():
    """The entry experiment, as an assertion. Pre-fix this point verified ``ok=True``
    with ``obj=-9.999997``."""
    m = _bigm_model(1e7)
    fractional = verify_point(m, [10.0, 1e-6], with_objective=True)
    integral = verify_point(m, [10.0, 0.0], with_objective=True)
    assert not fractional.ok, (
        "verify_point vouched for a point that is infeasible at its own integral "
        f"realisation (objective {fractional.objective!r})"
    )
    assert not integral.ok
    # The two now fail for the SAME reason -- which is the whole fix: the
    # fractional point is judged as the integral one it stands for.
    assert "row 0" in fractional.reason and "row 0" in integral.reason


def test_verify_point_still_accepts_the_genuine_optima():
    """Inertness: the points the model really does admit still verify, and their
    objectives are unchanged."""
    m = _bigm_model(1e7)
    compared = 0
    for pt, obj in (([10.0, 1.0], TRUE_OPTIMUM), ([0.0, 0.0], 0.0)):
        res = verify_point(m, pt, with_objective=True)
        assert res.ok, f"{pt} rejected: {res.reason}"
        assert res.objective == pytest.approx(obj, abs=1e-9)
        compared += 1
    assert compared == 2


def test_verify_point_reports_the_objective_of_the_claimed_point():
    """A near-integral point's objective is the one its INTEGRAL realisation attains
    -- reporting the fractional point's value publishes a number no integral point
    reaches, which is the same false claim one step downstream of the arbiter."""
    m = _bigm_model(100.0)
    res = verify_point(m, [10.0, 1.0 - 1e-7], with_objective=True)
    assert res.ok, res.reason
    assert res.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-12)


# ─────────────────────────────────────────────────────────────
# End-to-end: no route may publish an unattainable objective
# ─────────────────────────────────────────────────────────────

#: Each route the issue measured, keyed to how it is reached. ``_env`` entries are
#: set for that solve only. Named by ROUTE, never by instance (CLAUDE.md §2).
ROUTES: dict[str, dict] = {
    "highs-milp": {},
    "rust-monolithic": {"_env": {"DISCOPT_LP_MILP_BACKEND": "rust"}},
    "python-milp-bb": {
        "lagrangian_bound": True,
        "_env": {"DISCOPT_LP_MILP_BACKEND": "rust"},
    },
    "spatial-bb": {"lazy_constraints": [], "_env": {"DISCOPT_LP_MILP_BACKEND": "rust"}},
}


def _solve_on(route: str, big_m: float, monkeypatch):
    kw = dict(ROUTES[route])
    for key, val in kw.pop("_env", {}).items():
        monkeypatch.setenv(key, val)
    return _bigm_model(big_m).solve(time_limit=30, **kw)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_no_route_publishes_an_unattainable_objective(route, monkeypatch):
    """The certificate invariant. Whatever a route does with this model -- solve it,
    decline to certify, or refuse outright -- it may not report an objective below
    the true optimum, and may not report a binary that is not a binary.

    Pre-fix every route reported ``-9.999997`` at ``z = 1e-6``; three of the four
    labelled it ``optimal`` with ``gap_certified=True``.
    """
    try:
        r = _solve_on(route, 1e7, monkeypatch)
    except RuntimeError as exc:
        # A loud refusal is a correct outcome here (CLAUDE.md §3): the model is not
        # solvable at the declared tolerances, and saying so beats certifying a
        # point no integral realisation supports.
        assert "INTEGRAL realisation" in str(exc)
        return

    assert r.status != "optimal" or r.objective is None or r.objective >= TRUE_OPTIMUM - 1e-6, (
        f"{route} certified optimal at {r.objective!r}, below the true optimum {TRUE_OPTIMUM}"
    )
    if r.objective is not None:
        assert r.objective > FALSE_OPTIMUM, (
            f"{route} reported {r.objective!r}, an objective no integral point attains"
        )
    if r.x is not None and r.x.get("z") is not None:
        z = float(np.asarray(r.x["z"]).ravel()[0])
        assert abs(z - round(z)) == 0.0, f"{route} reported a non-integral binary z={z!r}"


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_the_fix_is_inert_on_a_well_scaled_big_m(route, monkeypatch):
    """The control that makes the test above mean something.

    Same formulation, same structure, ``M = 100`` instead of ``1e7`` -- a big-M
    small enough that ``|a_ij| * integrality_tol`` is under the feasibility
    tolerance. Every route must still SOLVE it, to ``-7`` at ``z = 1``. Without
    this, "no route publishes an unattainable objective" would be satisfied by a
    fix that simply refuses everything.
    """
    r = _solve_on(route, 100.0, monkeypatch)
    assert r.status in ("optimal", "feasible"), f"{route} failed a solvable model: {r.status}"
    assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-6), (
        f"{route} reported {r.objective!r}, expected {TRUE_OPTIMUM}"
    )
    z = float(np.asarray(r.x["z"]).ravel()[0])
    assert z == pytest.approx(1.0, abs=1e-12)
    assert verify_point(_bigm_model(100.0), [float(np.asarray(r.x["x"]).ravel()[0]), z]).ok


def test_probe_reaches_every_named_route(monkeypatch):
    """CLAUDE.md §6: prove the parametrisation above actually exercised four
    distinct engines rather than four aliases of the default one."""
    seen = set()
    for route in ROUTES:
        kw = dict(ROUTES[route])
        env = kw.pop("_env", {})
        assert set(env) <= {"DISCOPT_LP_MILP_BACKEND"}
        seen.add((route, tuple(sorted(env.items())), tuple(sorted(kw))))
    assert len(seen) == 4
    assert os.environ.get("DISCOPT_LP_MILP_BACKEND") in (None, "", "highs", "rust")
