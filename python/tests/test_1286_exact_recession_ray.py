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
