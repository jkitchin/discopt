"""#1749: ``Model.sensitivity`` raised ``LinAlgError: Singular matrix`` at a degenerate
LP vertex.

At ``min c x0 + 2 x1  s.t. x0 + x1 = 1, x in [0, 1]^2`` with ``c = 1`` the optimum
``x = (1, 0)`` has three active constraints in two variables. Both variables are
frozen at their bounds, so the equality row and its multiplier column of the
active-set KKT matrix vanish and the matrix is singular -- the ordinary case at a
degenerate vertex. ``dx*/dp`` is still unique (zero), and ``dobj/dc = x0 = 1``. Only
the multiplier derivative is undetermined.

#1615 replaced the JAX ``custom_root`` solve (which does not raise on a singular
matrix) by ``numpy.linalg.solve`` (which does). The fix solves a singular system by
SVD: unique ``dx`` is returned exactly, undetermined ``dλ`` rows are ``nan``, and a
point with no derivative raises :class:`DegenerateSensitivityError`.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers.sipopt import DegenerateSensitivityError


def _degenerate_lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    c = m.parameter("c", value=1.0)
    m.subject_to(x[0] + x[1] == 1)
    m.minimize(c * x[0] + 2 * x[1])
    m.solve()
    return m, c


@pytest.mark.parametrize("method", ["exact", "fd"])
def test_degenerate_vertex_returns_unique_primal_derivative(method):
    m, c = _degenerate_lp()
    s = m.sensitivity(wrt=[c], method=method)
    assert s.status == "optimal"
    np.testing.assert_allclose(s.x, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(s.dx_dp.ravel(), [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(s.dobj_dp, [1.0], atol=1e-6)
    # The equality's multiplier is not unique at this vertex, so neither is its
    # derivative: reported as nan, not as an arbitrary choice.
    assert s.dlambda_dp.shape == (1, 1)
    assert np.isnan(s.dlambda_dp[0, 0])


def test_degenerate_vertex_order2_refuses():
    m, c = _degenerate_lp()
    with pytest.raises(DegenerateSensitivityError, match="order=2"):
        m.sensitivity(wrt=[c], order=2)


def test_kink_without_derivative_raises_clear_error():
    # min x s.t. x >= p, x in [0, 5] at p = 0: the bound and the row are both active,
    # and dx/dp is 0 from the left and 1 from the right -- no derivative exists.
    m = dm.Model("kink")
    x = m.continuous("x", lb=0, ub=5)
    p = m.parameter("p", value=0.0)
    m.subject_to(x >= p)
    m.minimize(x)
    m.solve()
    with pytest.raises(DegenerateSensitivityError, match="not differentiable"):
        m.sensitivity(wrt=[p])
    # Still a LinAlgError, which is what callers caught before.
    with pytest.raises(np.linalg.LinAlgError):
        m.sensitivity(wrt=[p])
    # Off the kink the ordinary path is untouched.
    p.value = 1.0
    m.solve()
    s = m.sensitivity(wrt=[p])
    np.testing.assert_allclose(s.dx_dp.ravel(), [1.0], atol=1e-8)
    np.testing.assert_allclose(s.dobj_dp, [1.0], atol=1e-8)
