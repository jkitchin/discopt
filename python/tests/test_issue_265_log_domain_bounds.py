"""Issue #265: a free variable fed to ``log`` must not yield a false ``infeasible``.

MINLPLib ``ex8_5_4`` is feasible (optimum ``-0.0004251471``) but discopt reported
``infeasible``: all five variables are declared *free* and the objective contains
``log(x0)*x0``. With no bound forcing ``x0 > 0`` the local NLP wandered into the
undefined ``log`` domain, every node solve failed, and the search exhausted with no
incumbent — which the spatial B&B reports as ``infeasible``.

``FunctionDomainBoundRule`` derives the implied domain bound (``log`` argument
``> 0``, ``sqrt`` argument ``>= 0``, …) wherever the function appears — objective
or constraint, nested in any expression — so the NLP stays in-domain and the
feasible optimum is found. Only single-variable affine arguments are bounded; a
multi-variable argument such as ``log(x1 - x2)`` is conservatively left alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.nonlinear_bound_tightening import (  # noqa: E402
    FunctionDomainBoundRule,
    build_flat_variable_metadata,
)

pytestmark = pytest.mark.unit


def _tighten(model):
    meta = build_flat_variable_metadata(model)
    n = len(meta.flat_var_types)
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    return FunctionDomainBoundRule().tighten(model, lb, ub, meta)


def test_log_argument_gets_nonnegative_lower_bound():
    m = dm.Model("logobj")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x * dm.log(x))  # log nested inside a product, in the objective
    lb, ub = _tighten(m)
    assert lb[0] == 0.0
    assert ub[0] == np.inf


def test_sqrt_argument_gets_nonnegative_lower_bound():
    m = dm.Model("sqrtobj")
    y = m.continuous("y", lb=-np.inf, ub=np.inf)
    m.minimize(dm.sqrt(y) + y)
    lb, _ = _tighten(m)
    assert lb[0] == 0.0


def test_affine_log_argument_inverts_bound():
    m = dm.Model("afflog")
    z = m.continuous("z", lb=-np.inf, ub=np.inf)
    m.minimize(dm.log(2.0 * z + 1.0))  # 2z + 1 > 0  ->  z >= -0.5
    lb, _ = _tighten(m)
    assert lb[0] == pytest.approx(-0.5)


def test_multivariable_log_argument_is_not_bounded():
    """log(x0 - x1) > 0 implies nothing about x0 or x1 individually — abstain."""
    m = dm.Model("multivar")
    x = m.continuous("x", shape=(2,), lb=-np.inf, ub=np.inf)
    m.minimize(dm.log(x[0] - x[1]) + x[0])
    lb, ub = _tighten(m)
    assert np.all(lb == -np.inf)
    assert np.all(ub == np.inf)


def test_log_in_constraint_body_is_bounded():
    m = dm.Model("logcon")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(y)
    m.subject_to(dm.log(x) + y >= 1.0)  # x must be > 0 for log to be defined
    lb, _ = _tighten(m)
    assert lb[0] == 0.0


def test_free_variable_xlogx_is_feasible_not_infeasible():
    """The end-to-end #265 symptom: free log argument → feasible, never infeasible."""
    m = dm.Model("xlogx")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x * dm.log(x))  # min at x = 1/e, value -1/e
    m.subject_to(x <= 5.0)
    r = m.solve(time_limit=5, gap_tolerance=1e-4)
    assert r.status != "infeasible"
    assert r.status in ("optimal", "feasible")
    assert r.objective is not None
    assert r.objective == pytest.approx(-1.0 / np.e, abs=1e-3)
