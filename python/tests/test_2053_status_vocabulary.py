"""#2053: the status vocabulary, and three places a status said the wrong thing.

1. ``"unknown"`` and ``"no_feasible_point"`` were returned but not defined in
   :mod:`discopt.status`, so a caller dispatching on the documented vocabulary had
   no branch for them. Both are now members of ``discopt.status.STATUSES``, and a
   scan of the package's status literals fails on any value outside it.
2. A HiGHS ``kIterationLimit`` read as ``time_limit`` (LP) or ``node_limit`` (MILP).
3. ``solver="pounce"`` on an NLP whose iterates diverge returned ``status="error"``
   beside ``x`` and an objective of about -1.2e22.
"""

from __future__ import annotations

import pathlib
import re

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import status as st

PKG = pathlib.Path(st.__file__).resolve().parent

# ``status=`` literals in these files belong to other result types (a node LP
# relaxation, a module's registry entry), not to ``SolveResult.status``.
_OTHER_VOCABULARIES = ("_relax/", "solver_tuning.py")
_STATUS_LITERAL = re.compile(r"""\bstatus\s*=\s*"([a-z_]+)\"""")


def test_both_undocumented_statuses_are_defined():
    assert st.UNKNOWN == "unknown"
    assert st.NO_FEASIBLE_POINT == "no_feasible_point"
    assert {st.UNKNOWN, st.NO_FEASIBLE_POINT} <= st.STATUSES
    # Neither claims anything.
    for s in (st.UNKNOWN, st.NO_FEASIBLE_POINT):
        assert not st.is_certified_status(s)
        assert not st.is_local_status(s)


def test_every_status_literal_in_the_package_is_in_the_vocabulary():
    checked = 0
    stray: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if any(part in rel for part in _OTHER_VOCABULARIES):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in _STATUS_LITERAL.finditer(line):
                checked += 1
                if m.group(1) not in st.STATUSES:
                    stray.append(f"{rel}:{lineno}: {m.group(1)!r}")
    assert checked > 100, f"the scan only saw {checked} status literals"
    assert not stray, "status values outside discopt.status.STATUSES:\n" + "\n".join(stray)


def test_the_solve_result_docstring_names_both():
    from discopt.modeling.core import SolveResult

    doc = SolveResult.__doc__ or ""
    assert '``"unknown"``' in doc and '``"no_feasible_point"``' in doc


def test_the_diverging_iterates_reason_is_a_termination_reason():
    assert st.TERMINATION_DIVERGING in st.TERMINATION_REASONS


# ── 2. HiGHS iteration limit ──


def _cap_iterations(monkeypatch):
    """Every HiGHS instance the LP/MILP route builds stops after one iteration.

    ``simplex_iteration_limit`` is deliberately not a user-passable
    ``highs_options`` key (#1620), so the cap is applied at the factory.
    """
    import discopt.solvers.lp_milp_highs as H

    real = H._new_highs

    def capped(highspy, opts):
        return real(highspy, list(opts) + [("simplex_iteration_limit", 1)])

    monkeypatch.setattr(H, "_new_highs", capped)


def _lp_model() -> dm.Model:
    rng = np.random.default_rng(0)
    A = rng.uniform(0.0, 1.0, (15, 20))
    m = dm.Model("lp_iter_cap")
    v = m.continuous("v", shape=(20,), lb=0, ub=10)
    for i in range(15):
        m.subject_to(sum(float(A[i, j]) * v[j] for j in range(20)) >= 5)
    m.minimize(sum((1 + j % 3) * v[j] for j in range(20)))
    return m


def test_highs_lp_iteration_limit_is_iteration_limit(monkeypatch):
    from discopt.solver import _highs_std_form
    from discopt.solvers.lp_milp_highs import solve_lp_std

    _cap_iterations(monkeypatch)
    _, _, sf = _highs_std_form(_lp_model())
    out = solve_lp_std(sf, time_limit=30.0)
    assert out.highs_status == "kIterationLimit"
    assert out.status == "iteration_limit"


def test_model_solve_reports_iteration_limit_on_the_lp_route(monkeypatch):
    _cap_iterations(monkeypatch)
    r = _lp_model().solve(time_limit=30.0)
    assert r.status == "iteration_limit"
    assert r.status in st.STATUSES
    assert r.x is None and not r.gap_certified


# ── 3. Diverging_Iterates on solver="pounce" ──


@pytest.mark.requires_pounce
def test_pounce_diverging_iterates_is_local_limit_with_the_iterate_withheld():
    pytest.importorskip("pounce")
    m = dm.Model("diverge")
    x = m.continuous("x", lb=-1e20, ub=1e20)
    y = m.continuous("y", lb=-1e20, ub=1e20)
    m.subject_to(x + y >= 1)
    m.minimize(-(x**2) - y**2)  # unbounded below
    r = m.solve(solver="pounce", initial_solution={x: 2.0, y: 2.0})
    assert r.status == st.LOCAL_LIMIT
    assert r.termination == st.TERMINATION_DIVERGING
    # The diverged iterate is never reported as a solution.
    assert r.x is None and r.objective is None
    assert r.bound is None and not r.gap_certified
    assert r.last_iterate is not None
    assert abs(float(r.last_iterate["x"])) > 1e6
    assert r.last_iterate_violation is not None
