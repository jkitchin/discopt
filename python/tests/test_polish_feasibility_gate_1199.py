"""#1199 — the terminal KKT polish must never adopt a point it moved off the rows.

``solve_model``'s terminal polish re-solves the incumbent's continuous completion
with the integers fixed and adopts the result when the objective is *unchanged*
(the arm its own comment called "the always-safe purification") or, on convex
models, *improved*. Only the improving arm ever looked at the constraints. An
objective that did not move says nothing about whether the point still satisfies
the rows -- the NLP backend reports its own convergence status, not the model's.

Measured on ``nvs05`` before this fix (``max_nodes=5000, time_limit=300``):

    tree incumbent      max row violation 9.0e-6   objective 5.470934109
    polish output       max row violation 1.3e+2   objective 5.4709340754

The polish output was adopted -- its objective had moved 3.3e-8, deep inside the
1e-4 relative window ``_unchanged`` accepts -- and the solve reported
``status="optimal"``, ``gap_certified=True`` at an objective BELOW the
``minlplib.solu`` oracle (5.470934108), on a point the repo's own acceptance
arbiter (``_check_constraint_feasibility``) rejects. The tree incumbent it
replaced matched that oracle to nine digits.
"""

from __future__ import annotations

import pathlib

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt._relax.nlp_evaluator import cached_evaluator
from discopt.modeling.core import from_nl
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds


class _LinearRows:
    """Minimal evaluator over equality rows ``A x == 0``.

    Enough surface for the arbiter and the ratio: ``n_constraints``,
    ``evaluate_constraints`` and ``evaluate_jacobian``.
    """

    def __init__(self, A):
        self.A = np.asarray(A, dtype=float)

    @property
    def n_constraints(self):
        return int(self.A.shape[0])

    def evaluate_constraints(self, x):
        return self.A @ np.asarray(x, dtype=float)

    def evaluate_jacobian(self, x):
        return self.A


# One row ``x0 - x1 == 0`` evaluated near 1e5: the large-magnitude regime the
# scale-aware test exists for. scale = |1|*1e5 + |-1|*(1e5) ~ 2e5, so the
# combined tolerance is 1e-6 + 1e-9*2e5 ~ 2.01e-4.
_BIG = 1.0e5
_ROWS = _LinearRows([[1.0, -1.0]])
_CL = np.zeros(1)
_CU = np.zeros(1)


def _pt(resid):
    return np.array([_BIG + resid, _BIG])


def test_ratio_and_arbiter_agree_at_the_boundary():
    """``scaled_violation_ratio(...) <= 1`` exactly when the arbiter accepts.

    The gate ranks two points by this number, so it may not be a second, drifting
    definition of the same test.
    """
    compared = 0
    for resid in (0.0, 1e-9, 1e-4, 2.0e-4, 2.1e-4, 1e-3, 1.0):
        x = _pt(resid)
        accepts = ph._check_constraint_feasibility(x=x, evaluator=_ROWS, cl=_CL, cu=_CU)
        ratio = ph.scaled_violation_ratio(_ROWS, x, _CL, _CU)
        assert accepts == (ratio <= 1.0), f"resid={resid:g}: accepts={accepts} ratio={ratio:g}"
        compared += 1
    assert compared == 7, f"only {compared} comparisons executed"


def test_ratio_is_scale_aware_not_absolute():
    """A residual accepted at scale 1e5 is refused on the same row at scale 1."""
    big = ph.scaled_violation_ratio(_ROWS, _pt(1.0e-4), _CL, _CU)
    small = ph.scaled_violation_ratio(_ROWS, np.array([1.0e-4, 0.0]), _CL, _CU)
    assert big <= 1.0 < small, f"big={big:g} small={small:g}"


def test_gate_rejects_a_polish_that_moves_the_point_off_the_rows():
    """The nvs05 signature, at unit scale: feasible incumbent, degraded candidate."""
    x_old = _pt(0.0)
    x_new = _pt(1.0e-2)  # ~50x the combined tolerance
    assert not solver_mod._polish_preserves_feasibility(_ROWS, x_new, x_old, _CL, _CU)


def test_gate_accepts_a_feasible_polish():
    x_old = _pt(1.0e-4)
    x_new = _pt(1.0e-9)
    assert solver_mod._polish_preserves_feasibility(_ROWS, x_new, x_old, _CL, _CU)


def test_gate_accepts_a_polish_that_improves_an_already_infeasible_incumbent():
    """Never-degrade, not "feasible": a purification of a marginal point must not
    be blocked by a bar its own input could not clear."""
    x_old = _pt(1.0)
    x_new = _pt(1.0e-2)  # still outside the arbiter, but 100x closer
    assert solver_mod._polish_preserves_feasibility(_ROWS, x_new, x_old, _CL, _CU)
    assert not solver_mod._polish_preserves_feasibility(_ROWS, x_old, x_new, _CL, _CU)


def test_gate_is_a_no_op_without_row_bounds():
    assert solver_mod._polish_preserves_feasibility(_ROWS, _pt(1.0), _pt(0.0), [], [])


def _pinned_model():
    """``min n  s.t.  x**3 - 1000 == 0``, x continuous, n integer.

    The row pins ``x = 10`` and involves no objective variable, so perturbing
    ``x`` moves the point off the row while leaving the objective EXACTLY
    unchanged — the purification arm's own condition, met perfectly.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=100.0)
    n = m.integer("n", lb=1, ub=3)
    m.subject_to(x * x * x - 1000.0 == 0.0)
    m.minimize(n)
    return m


# Row scale at the solution: |d/dx (x**3)| * |x| = 3*10**2 * 10 = 3000, so the
# acceptance tolerance there is 1e-6 + 1e-9*3000 = 4e-6 and the deliberately loose
# false-primal screen is 1e-3 + 1e-6*3000 = 4e-3. The injected 1e-4 violation sits
# between them: only the polish gate can refuse this point.
_INJECTED_VIOLATION = 1.0e-4


def test_a_degrading_polish_is_not_reported(monkeypatch):
    """End to end: an off-manifold polish output with an IDENTICAL objective.

    Fails before the fix — the purification arm adopts any point whose objective
    did not move — and passes after. Route-agnostic on purpose: both terminal
    re-solves (spatial-B&B polish, NLP-BB refine) go through
    ``_solve_node_nlp_kkt``, and the invariant asserted is the one the issue is
    about: the point that leaves satisfies the rows it is reported against.
    """
    calls = {"n": 0}
    real = solver_mod._solve_node_nlp_kkt

    def _degrading(evaluator, x0, node_lb, node_ub, constraint_bounds, options, *a, **k):
        res = real(evaluator, x0, node_lb, node_ub, constraint_bounds, options, *a, **k)
        if res.x is not None and len(res.x) >= 2:
            calls["n"] += 1
            bad = np.asarray(res.x, dtype=float).copy()
            bad[0] = float((1000.0 + _INJECTED_VIOLATION) ** (1.0 / 3.0))
            res.x = bad
        return res

    monkeypatch.setattr(solver_mod, "_solve_node_nlp_kkt", _degrading)

    r = _pinned_model().solve(time_limit=30)
    assert calls["n"] > 0, "no terminal re-solve ran — this test asserted nothing"
    assert r.x is not None, f"no incumbent (status={r.status})"
    x_rep = float(np.asarray(r.x["x"]).ravel()[0])
    resid = abs(x_rep**3 - 1000.0)
    threshold = 1.0e-6 + 1.0e-9 * (3.0 * x_rep**2 * abs(x_rep))
    assert resid <= threshold, (
        f"the reported point sits {resid:.3e} off the row it must satisfy "
        f"(tolerance {threshold:.3e}, x={x_rep!r}): a re-solve that moved the "
        f"incumbent off the manifold was adopted because its objective had not moved"
    )


@pytest.mark.slow
def test_nvs05_incumbent_passes_the_acceptance_arbiter():
    """The issue's own instance, verified against a freshly parsed ORIGINAL model."""
    path = pathlib.Path(__file__).parent / "data" / "minlplib_nl" / "nvs05.nl"
    r = from_nl(str(path)).solve(max_nodes=5000, time_limit=180.0)
    assert r.x is not None, f"no incumbent (status={r.status})"

    fresh = from_nl(str(path))
    ev = cached_evaluator(fresh)
    assert ev.n_constraints > 0, "fixture has no rows — the assertion below is vacuous"
    x = np.concatenate(
        [np.atleast_1d(np.asarray(r.x[v.name], dtype=float)).ravel() for v in fresh._variables]
    )
    cl, cu = (np.asarray(b, dtype=float) for b in _infer_constraint_bounds(ev))
    ratio = ph.scaled_violation_ratio(ev, x, cl, cu)
    assert ph._check_constraint_feasibility(ev, x), (
        f"the reported nvs05 incumbent (objective={r.objective!r}) is "
        f"{ratio:.3g}x the acceptance tolerance outside the rows — the arbiter that "
        f"decides whether a primal heuristic may offer a point rejects the point the "
        f"solve returns"
    )
    # The oracle: 5.470934108 (minlplib.solu). Below it means off the manifold.
    assert r.objective is not None and r.objective >= 5.470934108 - 1e-6
