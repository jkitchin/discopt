"""IIS portfolio members stay within the caller's thread budget.

HiGHS keeps one task scheduler per process. The first ``run()`` sizes it from that
run's ``threads`` option, and ``0`` (discopt's default) means "automatic", about
half the cores. Each portfolio member is its own process, so each one sized its own
automatic scheduler: three members under ``MIPCC_THREADS=3`` used 975% CPU.
:func:`discopt._iis_backend.pin_highs_threads` sizes the scheduler before a
member's first solve, and ``threads=`` shares the caller's budget out among the
members.
"""

import subprocess
import sys
import textwrap

import discopt.modeling as dm
import pytest
from discopt._iis_milp import milp_iis_arrays

pytestmark = pytest.mark.requires_highs


def _run(code: str) -> subprocess.CompletedProcess:
    # A fresh interpreter: the scheduler is process-wide, so this test process
    # (which may already have run HiGHS) cannot show the first-run sizing.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


_LP = """
import highspy, numpy as np

def solve(threads):
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("threads", threads)
    lp = highspy.HighsLp()
    lp.num_col_ = 1
    lp.num_row_ = 0
    lp.col_cost_ = np.ones(1)
    lp.col_lower_ = np.zeros(1)
    lp.col_upper_ = np.ones(1)
    h.passModel(lp)
    return h.run() == highspy.HighsStatus.kOk
"""


def test_pin_sizes_the_scheduler_for_later_runs():
    p = _run(
        _LP
        + """
from discopt._iis_backend import pin_highs_threads
assert pin_highs_threads(1) is True
# A later run with the default threads=0 reuses the pinned scheduler ...
assert solve(0), "threads=0 after the pin should run on the pinned scheduler"
# ... and one asking for another size is refused, which shows the size is 1.
assert not solve(2), "threads=2 after pinning to 1 should be refused"
print("OK")
"""
    )
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout + p.stderr


def test_pin_after_a_different_first_run_raises():
    p = _run(
        _LP
        + """
from discopt._iis_backend import pin_highs_threads
assert solve(2)
try:
    pin_highs_threads(1)
except RuntimeError as e:
    print("RAISED", e)
"""
    )
    assert p.returncode == 0 and "RAISED" in p.stdout, p.stdout + p.stderr


def test_pin_rejects_nonpositive():
    from discopt._iis_backend import pin_highs_threads

    with pytest.raises(ValueError):
        pin_highs_threads(0)


def _arrays():
    import numpy as np
    import scipy.sparse as sp

    # x <= 0 and x >= 1, x integer in [-10, 10].
    A = sp.csr_matrix(np.array([[1.0], [1.0]]))
    return (
        A,
        np.array([-1e20, 1.0]),
        np.array([0.0, 1e20]),
        {0: np.array([0]), 1: np.array([1])},
        np.array([-10.0]),
        np.array([10.0]),
        np.array([True]),
    )


def test_portfolio_accepts_a_thread_budget():
    res = milp_iis_arrays(
        _arrays(), iis_engine="highs", portfolio=("ICB", "ICB-r"), workers=2, threads=2
    )
    assert sorted(res.constraints) == [0, 1]


def test_thread_budget_validation():
    with pytest.raises(ValueError, match="threads= needs portfolio="):
        milp_iis_arrays(_arrays(), iis_engine="highs", threads=2)
    with pytest.raises(ValueError, match="threads must be >= 1"):
        milp_iis_arrays(_arrays(), iis_engine="highs", portfolio=("ICB",), threads=0)


def test_model_api_unchanged():
    # compute_iis passes no threads=; the default budget is every core.
    from discopt.infeasibility import compute_iis

    m = dm.Model("t")
    x = m.integer("x", lb=-10, ub=10)
    m.subject_to(x <= 0, name="c0")
    m.subject_to(x >= 1, name="c1")
    m.minimize(x)
    res = compute_iis(m, include_integrality=True, include_bounds=False, portfolio=("ICB",))
    assert sorted(c.name for c in res.constraints) == ["c0", "c1"]
