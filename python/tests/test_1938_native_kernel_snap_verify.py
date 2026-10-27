"""The native kernel must not adopt a point that is integral only within ``int_tol`` (#1938).

``min -y*v - x + c*z + w`` s.t. ``x - M*z <= 0``, ``x - w <= eps``, ``y + v == 2``
(``z`` binary). The LP pays ``c*x/M`` for ``x`` through ``z``, so the node LPs return
``x = eps``, ``z = eps/M = 1e-6``, which passes the kernel's 1e-5 integrality test.
Snapped to ``z = 0`` (what ``verify_point`` judges) the point violates the big-M row
by ``eps``. Before the fix the kernel adopted it at -1.00994 (true optimum -1.005),
exited ``optimal``, and the #789 check after the search discarded the whole kernel
result; the Python engine then ended ``feasible`` at -1.0 with the bound -1.01.
"""

from __future__ import annotations

import discopt
import discopt._rust as native
import pytest

pytestmark = pytest.mark.smoke


def _bigm_snap(eps: float, big_m: float, c: float) -> discopt.Model:
    m = discopt.Model("bigm_snap")
    x = m.continuous("x", lb=0, ub=big_m)
    w = m.continuous("w", lb=0, ub=big_m)
    z = m.binary("z")
    y = m.continuous("y", lb=0, ub=2)
    v = m.continuous("v", lb=0, ub=2)
    m.subject_to(x - big_m * z <= 0)
    m.subject_to(x - w <= eps)
    m.subject_to(y + v == 2)  # loose McCormick relaxation of y*v
    m.minimize(-y * v - x + c * z + w)
    return m


@pytest.mark.parametrize("eps,big_m,c", [(1e-2, 1e4, 5e-3), (1e-3, 1e5, 2e-4)])
def test_kernel_certifies_the_integral_optimum(monkeypatch, eps, big_m, c):
    kernel = []
    tree = native.solve_spatial_tree_py

    def trace(**kwargs):
        res = tree(**kwargs)
        # #1983's root probe is a one-node call with no hooks; only the search
        # itself prices incumbents through ``incumbent_value``.
        if "incumbent_value" in kwargs:
            kernel.append(res)
        return res

    monkeypatch.setattr(native, "solve_spatial_tree_py", trace)
    r = _bigm_snap(eps, big_m, c).solve(time_limit=30, threads=1)

    opt = -1.0 - eps + c
    assert len(kernel) == 1, "the native kernel did not run"
    k = kernel[0]
    assert k["status"] == "optimal"
    # Never below the optimum: a 1e-6 binary priced at -1.00994 is not adopted.
    assert k["incumbent"] >= opt - 1e-6, (k["incumbent"], opt)
    # ...and the result was kept, not discarded by #789.
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=1e-6)
    assert r.bound <= opt + 1e-6
    assert r.objective == pytest.approx(k["incumbent"], abs=1e-9)
    assert float(r.x["z"]) == pytest.approx(1.0)
