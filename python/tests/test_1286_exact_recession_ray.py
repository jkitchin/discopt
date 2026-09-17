"""An ``unbounded`` certificate needs an exact recession ray (#1286).

``x1 + x2 = 0``, ``x1 + x2 + eps*x3 = 0``, ``max x3`` has optimum 0 (the rows
force ``x3 = 0``), but both LP routes certified ``unbounded``: the float ray
tests (relative residual 1e-9, and the interior-point ray LP) cannot see the
row whose only coupling to ``d = (1, -1, 1)`` is ``eps``.
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers.lp_milp_highs import StdForm, primal_ray_verified


def _model(eps):
    m = dm.Model("A")
    x1, x2, x3 = (m.continuous(n, lb=-1e20, ub=1e20) for n in ("x1", "x2", "x3"))
    m.subject_to(x1 + x2 == 0)
    m.subject_to(x1 + x2 + eps * x3 == 0)
    m.maximize(x3)
    return m


@pytest.mark.parametrize("backend", [None, "rust"])
@pytest.mark.parametrize("eps", [1e-12, 1e-13, 5e-13])
def test_finite_lp_is_never_certified_unbounded(monkeypatch, backend, eps):
    if backend is None:
        monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    r = _model(eps).solve()
    assert r.status in ("optimal", "error"), r.status
    if r.status == "optimal":
        assert r.objective == pytest.approx(0.0, abs=1e-9)


def test_tiny_coupling_ray_is_refused():
    A = sp.csc_matrix([[1.0, 1.0, 0.0], [1.0, 1.0, 1e-12]])
    sf = StdForm.from_arrays([0.0, 0.0, -1.0], A, [0.0, 0.0], [-1e20] * 3, [1e20] * 3)
    assert not primal_ray_verified(np.array([1.0, -1.0, 1.0]), sf)


def test_genuine_and_approximate_rays_are_accepted():
    # min -x  s.t.  x - y = 0,  x, y >= 0: the ray (1, 1) is exact.
    sf = StdForm.from_arrays([-1.0, 0.0], sp.csc_matrix([[1.0, -1.0]]), [0.0], [0, 0], [1e20] * 2)
    assert primal_ray_verified(np.array([1.0, 1.0]), sf)
    # A float ray off by 1e-12 still proves a ray exists on its support.
    assert primal_ray_verified(np.array([1.0, 1.0 + 1e-12]), sf)
    # Wrong sign pattern and zero-cost directions are refused.
    assert not primal_ray_verified(np.array([1.0, -1.0]), sf)
    assert not primal_ray_verified(np.array([-1.0, -1.0]), sf)


@pytest.mark.parametrize("backend", [None, "rust"])
def test_genuinely_unbounded_lp_still_certified(monkeypatch, backend):
    if backend is None:
        monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    m = dm.Model("U")
    x = m.continuous("x", lb=0.0, ub=1e20)
    y = m.continuous("y", lb=0.0, ub=1e20)
    m.subject_to(x - 2.0 * y <= 1.0)
    m.subject_to(x + y >= 3.0)
    m.maximize(x + y)
    assert m.solve().status == "unbounded"


# --- the ray LP's own direction is an interior-point iterate ------------------
#
# ``_certify_unbounded_ray`` hands ``primal_ray_verified`` the direction its ray
# LP returned, and that LP runs at ``constr_viol_tol = 1e-8`` without
# ``bound_relax_factor = 0``. So a column the recession cone pins to 0 comes back
# at ~1e-8 — outside the direction box on a half-open column, inside it on a free
# one — and the exact check refused the ray for round-off, turning the convex QP
# ``min ½x0² - x1`` on ``x >= 0`` from ``unbounded`` into ``error``.

_INF = 1e20

# ``min ½x0² - x1  s.t.  x0 <= 10``: the recession rows are the ``x0 <= 10`` row
# and ``Q d = 0``, which pin d0 and leave the ray ``(0, 1)``.
_QP_RAY_SYSTEM = dict(
    c=np.array([0.0, -1.0]),
    A_ray=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
    cl_ray=np.array([-_INF, 0.0, 0.0]),
    cu_ray=np.array([0.0, 0.0, 0.0]),
)


@pytest.mark.parametrize(
    "d, d_lo, d_hi",
    [
        # Exact: the ray the cone actually contains.
        (np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        # Measured on a half-open column: Ipopt relaxes d0 >= 0 by its default
        # ``bound_relax_factor``, so d0 lands 1e-8 below its own lower bound.
        (np.array([-1.0e-8, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        # Measured on a free column: there is no bound to fall outside, and the
        # equality row ``Q d = 0`` is met only to ``constr_viol_tol``.
        (np.array([5.0e-9, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    ],
)
def test_interior_point_dirt_on_a_pinned_column_keeps_the_ray(d, d_lo, d_hi):
    from discopt.solvers.lp_pounce import _ray_verified_exactly

    assert _ray_verified_exactly(d, d_lo=d_lo, d_hi=d_hi, **_QP_RAY_SYSTEM)


@pytest.mark.parametrize("d0", [1.0e-6, 1.0e-3, 0.5])
def test_a_pinned_column_above_the_dirt_floor_is_still_refused(d0):
    """The cleaning is a floor, not a blanket: a genuine nonzero on a column the
    cone pins to 0 is not a ray, however the rest of the direction looks."""
    from discopt.solvers.lp_pounce import _ray_verified_exactly

    assert not _ray_verified_exactly(
        np.array([d0, 1.0]),
        d_lo=np.array([-1.0, -1.0]),
        d_hi=np.array([1.0, 1.0]),
        **_QP_RAY_SYSTEM,
    )


def test_cleaning_does_not_undo_the_tiny_coupling_refusal():
    """The #1286 instance, put through the POUNCE wrapper: every entry of
    ``d = (1, -1, 1)`` is at the direction's max-norm, so nothing is cleaned and
    the 1e-12 coupling still decides the ray exactly."""
    from discopt.solvers.lp_pounce import _ray_verified_exactly

    assert not _ray_verified_exactly(
        np.array([1.0, -1.0, 1.0]),
        c=np.array([0.0, 0.0, -1.0]),
        A_ray=np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1e-12]]),
        cl_ray=np.zeros(2),
        cu_ray=np.zeros(2),
        d_lo=np.full(3, -1.0),
        d_hi=np.full(3, 1.0),
    )
