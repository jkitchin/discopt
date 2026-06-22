"""Stage 0 of the native-NLP integration (discopt#281): prove POUNCE's native
AD on the ``.nl`` problem matches the JAX ``NLPEvaluator`` *before* the node
solve is wired to use it.

Two layers of evidence:

* **Derivative equivalence** — at random points the native objective and
  gradient (and, for the identity case, the Hessian) agree with the evaluator
  to a tight tolerance. The gradient pins variable ordering and sign; the
  Hessian pins second-order curvature. The ``.nl`` constraint *representation*
  differs from the evaluator's (canonicalized bodies / row order), so
  constraint values are intentionally *not* compared element-wise.
* **Solve equivalence** — solving the relaxation natively and via the JAX
  callback path reaches the same optimum. This is the meaningful end-to-end
  check that the differing constraint representation describes the same
  feasible region.

Also covers the two non-trivial cases the production path relies on: a
``to_nl``-emitted model whose canonical reorder makes the permutation
*non-identity*, and a ``maximize`` model whose objective sense must be flipped.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.solvers import nlp_native as N  # noqa: E402

# `slow`, not `unit`: these build a POUNCE native base directly, which creates a
# pounce PyNlProblem. That object is unsendable (pyo3); under the fast suite's
# pytest-xdist parallelism it can be GC'd on a different worker thread than it was
# created on, raising "unsendable ... dropped on another thread" and tipping
# co-scheduled CPU-bound batch tests over the timeout. Native-AD is opt-in
# (default off), so this opt-in path belongs in the serial slow suite.
pytestmark = [pytest.mark.slow, pytest.mark.requires_pounce]

if not N._POUNCE_OK:  # pragma: no cover
    pytest.skip("pounce native API unavailable", allow_module_level=True)

import pounce  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# A basket spanning MIQP, nonconvex, and larger models. All minimize.
_INSTANCES = ["st_miqp1", "nvs01", "ex1221", "alan", "gbd", "fac2"]


def _instance_path(name: str) -> str:
    p = os.path.join(_DATA, f"{name}.nl")
    if not os.path.exists(p):
        pytest.skip(f"missing instance {name}")
    return p


def _densify_jac(base, x):
    r, c = base.jacobian_structure()
    v = base.jacobian(x)
    J = np.zeros((base.m, base.n))
    for rr, cc, vv in zip(r, c, v):
        J[rr, cc] += vv
    return J


def _densify_hess(base, x, lam, of):
    r, c = base.hessian_structure()
    v = base.hessian(x, lam, of)
    H = np.zeros((base.n, base.n))
    for rr, cc, vv in zip(r, c, v):
        H[rr, cc] += vv
        if rr != cc:
            H[cc, rr] += vv
    return H


@pytest.mark.parametrize("name", _INSTANCES)
def test_native_derivatives_match_evaluator(name):
    """Objective, gradient, and *objective* Hessian agree with the evaluator.

    POUNCE reports everything in minimization sense (matching the evaluator), so
    no sign correction is applied. Only the objective Hessian (``lam=0``) is
    compared — the *Lagrangian* Hessian mixes in the ``.nl``'s canonicalized
    constraints, which intentionally differ from the evaluator's representation.
    """
    path = _instance_path(name)
    ev = NLPEvaluator(dm.from_nl(path))
    base = pounce.read_nl(path)
    n, m = ev.n_variables, ev.n_constraints
    assert base.n == n and base.m == m

    lo, hi = ev.variable_bounds
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    zero_lam = np.zeros(m)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = lo + rng.uniform(size=n) * (hi - lo)
        assert abs(ev.evaluate_objective(x) - base.objective(x)) <= 1e-7 * (
            1 + abs(ev.evaluate_objective(x))
        )
        np.testing.assert_allclose(
            ev.evaluate_gradient(x), np.asarray(base.gradient(x)), rtol=1e-7, atol=1e-7
        )
        # Objective Hessian only (constraint Hessian representation differs).
        np.testing.assert_allclose(
            ev.evaluate_lagrangian_hessian(x, 1.0, zero_lam),
            _densify_hess(base, x, zero_lam, 1.0),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("name", _INSTANCES)
def test_native_solve_matches_jax_solve(name):
    """A native relaxation solve reaches the same optimum as the JAX path."""
    from discopt.solvers.nlp_pounce import solve_nlp as jax_solve_nlp

    path = _instance_path(name)
    ev = NLPEvaluator(dm.from_nl(path))
    nb = N.build_native_base(ev)
    assert nb is not None, "native base should build for an .nl-originated model"
    assert nb.perm is None, "from_nl model must align identically to its .nl"

    lo, hi = ev.variable_bounds
    x0 = 0.5 * (np.clip(lo, -1e6, 1e6) + np.clip(hi, -1e6, 1e6))
    rj = jax_solve_nlp(ev, np.asarray(x0, float), None, {"max_iter": 400})
    rn = N.solve_node_native(nb, x0, lo, hi, {"max_iter": 400})
    assert rn.status == rj.status
    assert rn.objective == pytest.approx(rj.objective, rel=1e-4, abs=1e-4)


def test_to_nl_permutation_is_nonidentity_and_correct():
    """A model whose .nl canonical reorder is non-trivial round-trips correctly.

    ``a`` appears only linearly and is declared first; ``b`` appears nonlinearly.
    The .nl writer pushes the linear variable to the tail, so the recovered
    permutation must be non-identity for the native solve to be correct.
    """
    from discopt.solvers.nlp_pounce import solve_nlp as jax_solve_nlp

    m = dm.Model("perm")
    a = m.continuous("a", lb=0.0, ub=10.0)  # linear-only, declared first
    b = m.continuous("b", lb=-3.0, ub=3.0)  # nonlinear
    m.minimize(2.0 * a + (b - 1.0) ** 2)
    m.subject_to(a + b >= 1.0)
    ev = NLPEvaluator(m)
    nb = N.build_native_base(ev)
    assert nb is not None and nb.source == "to_nl"
    assert nb.perm is not None and not np.array_equal(nb.perm, np.arange(nb.n)), (
        f"expected a non-identity permutation, got {None if nb.perm is None else nb.perm.tolist()}"
    )

    lo, hi = ev.variable_bounds
    x0 = 0.5 * (lo + hi)
    rj = jax_solve_nlp(ev, np.asarray(x0, float), None, {"max_iter": 300})
    rn = N.solve_node_native(nb, x0, lo, hi, {"max_iter": 300})
    assert rn.objective == pytest.approx(rj.objective, rel=1e-5, abs=1e-6)
    np.testing.assert_allclose(rn.x, rj.x, rtol=1e-4, atol=1e-4)
    nb.cleanup()


def test_maximize_sense_is_normalized():
    """A maximize model's native objective is returned in minimization sense."""
    from discopt.solvers.nlp_pounce import solve_nlp as jax_solve_nlp

    m = dm.Model("maxsense")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.maximize(-((x - 1.0) ** 2) - (y - 1.5) ** 2 + 5.0)
    m.subject_to(x + y <= 3.0)
    ev = NLPEvaluator(m)
    nb = N.build_native_base(ev)
    # base.minimize is False (a maximize .nl), but the native path still builds:
    # POUNCE minimizes the negated objective internally, matching the evaluator.
    assert nb is not None and nb.base.minimize is False

    lo, hi = ev.variable_bounds
    x0 = 0.5 * (lo + hi)
    rj = jax_solve_nlp(ev, np.asarray(x0, float), None, {"max_iter": 300})
    rn = N.solve_node_native(nb, x0, lo, hi, {"max_iter": 300})
    # Both report minimization sense (the evaluator negates the maximize obj).
    assert rn.objective == pytest.approx(rj.objective, rel=1e-5, abs=1e-6)
    nb.cleanup()


def test_node_bound_tightening_is_honored():
    """Tightened node bounds change the native optimum the same way as JAX."""
    from discopt.solvers.nlp_pounce import solve_nlp as jax_solve_nlp

    path = _instance_path("st_miqp1")
    ev = NLPEvaluator(dm.from_nl(path))
    nb = N.build_native_base(ev)
    lo, hi = ev.variable_bounds
    # Tighten the box around an interior point.
    mid = 0.5 * (np.clip(lo, -10, 10) + np.clip(hi, -10, 10))
    nlb = np.maximum(lo, mid - 0.25 * np.abs(mid) - 0.1)
    nub = np.minimum(hi, mid + 0.25 * np.abs(mid) + 0.1)
    x0 = 0.5 * (nlb + nub)

    # JAX path with the same tightened bounds via the bound-override proxy.
    from discopt.solver import _BoundOverrideEvaluator

    proxy = _BoundOverrideEvaluator(ev, nlb, nub)
    rj = jax_solve_nlp(proxy, np.asarray(x0, float), None, {"max_iter": 400})
    rn = N.solve_node_native(nb, x0, nlb, nub, {"max_iter": 400})
    if rj.status.name == "OPTIMAL" and rn.status.name == "OPTIMAL":
        assert rn.objective == pytest.approx(rj.objective, rel=1e-4, abs=1e-4)
