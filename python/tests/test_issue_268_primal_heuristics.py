"""Issue #268: primal-heuristic improvements for combinatorial MINLPs.

Two changes are exercised here:

* the heuristic *sub-NLP* solves are iteration-capped (they only need an
  approximately feasible point, not a tight optimum) so a pathological projection
  cannot burn the whole branch-and-bound budget, and
* ``fractional_diving`` is wired into both the convex NLP-BB and the spatial
  root-heuristic ladders as a fallback, so models whose relaxation rounds to a
  constraint-infeasible assignment (the feasibility pump returns nothing) still
  get a feasible incumbent instead of exhausting with no solution.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt._jax.primal_heuristics import (  # noqa: E402
    _HEURISTIC_NLP_MAX_ITER,
    feasibility_pump,
)
from discopt.solvers import NLPResult, SolveStatus  # noqa: E402

pytestmark = pytest.mark.unit


def _small_minlp() -> dm.Model:
    m = dm.Model("c")
    y = m.binary("y", shape=(3,))
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x + y[0] + 2 * y[1] + 3 * y[2])
    m.subject_to(x + y[0] + y[1] + y[2] >= 1.0)
    return m


def test_heuristic_subnlp_solves_are_iteration_capped():
    """The feasibility pump must forward a bounded ``max_iter`` to its sub-NLP."""
    m = _small_minlp()
    ev = NLPEvaluator(m)
    captured: list[dict] = []

    def mock_backend(evaluator, x0, options=None):
        captured.append(dict(options or {}))
        return NLPResult(
            status=SolveStatus.OPTIMAL,
            x=np.asarray(x0, dtype=float),
            objective=0.0,
            multipliers=None,
            bound_multipliers_lower=None,
            bound_multipliers_upper=None,
            iterations=1,
            wall_time=0.0,
        )

    x_relax = np.array([0.4, 0.6, 0.5, 1.0], dtype=float)
    feasibility_pump(m, x_relax, backend=mock_backend, evaluator=ev)

    assert captured, "pump never invoked the NLP backend"
    assert all(o.get("max_iter") == _HEURISTIC_NLP_MAX_ITER for o in captured)


def test_caller_can_override_the_iteration_cap():
    """An explicit ``ipopt_options['max_iter']`` wins over the default cap."""
    m = _small_minlp()
    ev = NLPEvaluator(m)
    captured: list[dict] = []

    def mock_backend(evaluator, x0, options=None):
        captured.append(dict(options or {}))
        return NLPResult(
            status=SolveStatus.INFEASIBLE,
            x=None,
            objective=None,
            multipliers=None,
            bound_multipliers_lower=None,
            bound_multipliers_upper=None,
            iterations=1,
            wall_time=0.0,
        )

    feasibility_pump(
        m,
        np.array([0.4, 0.6, 0.5, 1.0]),
        backend=mock_backend,
        evaluator=ev,
        ipopt_options={"max_iter": 1234},
    )
    assert captured and all(o.get("max_iter") == 1234 for o in captured)


def test_cap_is_a_sane_positive_bound():
    assert isinstance(_HEURISTIC_NLP_MAX_ITER, int)
    assert 50 <= _HEURISTIC_NLP_MAX_ITER <= 5000


def test_combinatorial_minlp_still_solves_with_capped_heuristics():
    """End-to-end sanity: a small combinatorial MINLP returns a sound result and a
    feasible incumbent (the capped heuristics + diving fallback must not regress
    the easy case)."""
    m = _small_minlp()
    r = m.solve(time_limit=10, gap_tolerance=1e-4)
    assert r.status in ("optimal", "feasible")
    assert r.objective is not None
    # x=1, all binaries 0 satisfies x >= 1 with objective 1.0 — the optimum.
    assert r.objective == pytest.approx(1.0, abs=1e-4)
