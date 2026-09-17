"""#1298: the root GMI separator must not call HiGHS basis accessors that segfault.

``Model.solve`` died with SIGSEGV (exit -11) on a tiny convex MINLP with a
``max(v, c) - v <= 0`` row. ``_RootLP`` classifies a constraint whose Jacobian is
constant over its sampled points as *linear*, and that row's coefficients are all
zero, so the root LP went to HiGHS with rows but **no matrix entries**. HiGHS then
reports ``kOptimal`` after zero simplex iterations without factorizing a basis, and
``getBasicVariables`` reads a basis that was never built.

A native crash is not catchable, so the end-to-end check runs in a subprocess and
asserts on the exit code.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from discopt.solvers._root_cuts import _basis_accessors_are_safe, separate_gmi

_REPO_PYTHON = str(Path(__file__).resolve().parents[1])

# The #1298 instance as reported. It segfaults on 108bcf3b; since #1297 this model
# routes to the spatial B&B and no longer reaches the separator at all, so this
# subprocess check locks the reported symptom rather than exercising the guard --
# the three tests above are what discriminate on the guard itself.
_REPRO = """
import discopt.modeling as dm

m = dm.Model("D")
v = m.continuous("v", lb=0, ub=4)
s = m.integer("s", lb=0, ub=3)
m.minimize(v - 1.5 * s)
m.subject_to(dm.maximum(v, 1.0) - v <= 0)
r = m.solve(time_limit=30)
print(r.status, r.objective)
"""


def _highs_lp(num_row, nnz_rows=()):
    """A HiGHS model with ``num_row`` rows, entries only in ``nnz_rows``."""
    import highspy

    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 2, num_row
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = np.array([1.0, -1.5])
    lp.col_lower_ = np.array([0.0, 0.0])
    lp.col_upper_ = np.array([4.0, 3.0])
    lp.row_lower_ = np.full(num_row, -inf)
    lp.row_upper_ = np.full(num_row, 2.0)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    starts, idx, vals = [0], [], []
    for r in range(num_row):
        if r in nnz_rows:
            idx.append(0)
            vals.append(1.0)
        starts.append(len(idx))
    lp.a_matrix_.start_ = np.array(starts, np.int32)
    lp.a_matrix_.index_ = np.array(idx, np.int32)
    lp.a_matrix_.value_ = np.array(vals, float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    return h


def test_accessors_are_refused_on_a_matrix_with_no_entries():
    # Exactly the shape that crashes: rows present, zero matrix entries.
    for num_row in (1, 2, 3):
        assert not _basis_accessors_are_safe(_highs_lp(num_row))


def test_accessors_are_allowed_when_the_basis_exists():
    assert _basis_accessors_are_safe(_highs_lp(0))
    assert _basis_accessors_are_safe(_highs_lp(1, nnz_rows=(0,)))
    # One empty row beside an ordinary one still factorizes a basis.
    assert _basis_accessors_are_safe(_highs_lp(2, nnz_rows=(1,)))


def test_separate_gmi_declines_instead_of_crashing(caplog):
    """The guard is what stands between this call and a SIGSEGV."""
    import logging

    h = _highs_lp(1)
    root = type("Root", (), {"A_eq": np.zeros((0, 2)), "n": 2, "is_int": np.array([False, True])})()
    a_all = np.zeros((1, 2))
    b_all = np.array([2.0])
    with caplog.at_level(logging.DEBUG, logger="discopt.solvers._root_cuts"):
        cuts = separate_gmi(root, h, np.array([1.0, 3.0]), a_all, b_all)
    assert cuts == []
    assert any("GMI declined" in r.getMessage() for r in caplog.records)


@pytest.mark.slow
def test_solve_does_not_segfault():
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(_REPRO)],
        capture_output=True,
        text=True,
        timeout=300,
        env={"PYTHONPATH": _REPO_PYTHON, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    # Before the fix: returncode -11 (SIGSEGV) with "Fatal Python error" on stderr.
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-2000:])
    assert "Fatal Python error" not in proc.stderr
    status, objective = proc.stdout.split()[:2]
    assert status == "optimal", proc.stdout
    # v = 1, s = 3 is the optimum; the separator declining costs no correctness.
    assert float(objective) == pytest.approx(-3.5, abs=1e-6)
