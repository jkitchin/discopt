"""#2221: the HiGHS LP engine's ``kUnknown`` mapping, in both directions.

highspy 1.16 can label a primal/dual pair ``kUnknown`` that 1.15 labelled ``kOptimal``.
``HighsEngine.solve`` maps ``kUnknown`` to ``"optimal"`` only when HiGHS holds a primal
AND a dual solution it calls feasible, and to ``"error"`` otherwise. Every consumer of an
``"optimal"`` answer re-verifies it, so the mapping only decides whether the checks run.
No released highspy at the CI pins produces ``kUnknown`` on a small LP, so the label is
injected at ``_status_name`` and the solution statuses at ``_both_solutions_feasible``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

H = pytest.importorskip("discopt.solvers.lp_milp_highs")
pytest.importorskip("highspy")

from discopt.solvers import lp_std_engine as E  # noqa: E402


def _lp() -> "H.StdForm":
    # min -x0 - 2 x1  s.t.  x0 + x1 + s = 4,  0 <= x <= 3,  s >= 0.  Optimum -7 at (1, 3).
    a = sp.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
    return H.StdForm.from_arrays(
        np.array([-1.0, -2.0, 0.0]),
        a,
        np.array([4.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 3.0, np.inf]),
        0.0,
    )


def _as_unknown(monkeypatch, feasible: bool) -> list:
    seen: list = []
    real = H._status_name

    def unknown(h):
        seen.append(real(h))
        return "kUnknown"

    monkeypatch.setattr(H, "_status_name", unknown)
    monkeypatch.setattr(E, "_both_solutions_feasible", lambda info, highspy: feasible)
    return seen


def test_kunknown_with_both_solutions_feasible_is_handed_on_as_optimal(monkeypatch):
    seen = _as_unknown(monkeypatch, feasible=True)
    raw = E.HighsEngine().solve(_lp(), time_limit=None, presolve=None, tight_tolerances=True)
    assert seen, "MEASURED NOTHING: the engine never read the HiGHS status"
    assert raw.status == "optimal"
    assert raw.x is not None and raw.y is not None and raw.rc is not None
    assert float(np.dot([-1.0, -2.0, 0.0], raw.x)) == pytest.approx(-7.0)


def test_kunknown_without_both_solutions_is_an_error(monkeypatch):
    seen = _as_unknown(monkeypatch, feasible=False)
    raw = E.HighsEngine().solve(_lp(), time_limit=None, presolve=None, tight_tolerances=True)
    assert seen, "MEASURED NOTHING: the engine never read the HiGHS status"
    assert raw.status == "error"
    assert raw.x is None


def test_both_solutions_feasible_reads_both_statuses():
    import highspy

    feas = highspy.SolutionStatus.kSolutionStatusFeasible
    infeas = highspy.SolutionStatus.kSolutionStatusInfeasible

    class Info:
        def __init__(self, p, d):
            self.primal_solution_status, self.dual_solution_status = p, d

    assert E._both_solutions_feasible(Info(feas, feas), highspy)
    assert not E._both_solutions_feasible(Info(feas, infeas), highspy)
    assert not E._both_solutions_feasible(Info(infeas, feas), highspy)
