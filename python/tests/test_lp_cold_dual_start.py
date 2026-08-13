"""Cold dual start basis for deadline-free pure LPs (``DISCOPT_LP_COLD_DUAL_START``).

``solve_lp_warm_std`` builds a sign-matched, dual-feasible slack start basis
(:func:`_dual_start_slack_basis`, #928) but engaged it only when the caller
passed a finite ``time_limit`` — #928 wanted a bankable anytime floor, not speed.
Since the deadline itself is default-OFF, every cold node LP on the default path
took the Rust cold PRIMAL loop, which stalls on the equality-rich lifted
relaxations this solver produces: an equality reaches the LP layer as two
opposing ``<=`` rows, both tight at every feasible point, so such an LP is
massively primal-degenerate. Measured on the RLT-on QPLIB_1157 root LP the cold
primal exhausted ``max_iter`` after >150 s; the dual start returned the same
optimum in 6.2 s.

These tests lock what the flag may and may not change:

1. it flips the start basis on a cold, deadline-free solve, and only then;
2. the LP optimum is unchanged — the flag is a start-basis choice, not a
   relaxation change, so ON and OFF must agree with each other and with SciPy;
3. an LP whose selected side is open stays on the primal path (the basis would
   not be dual-feasible), flag or not;
4. a solve that carries a finite deadline is unaffected — that path already used
   the dual start and must keep doing so with the flag OFF.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import (
    _cold_dual_start_enabled,
    _dual_start_slack_basis,
    solve_lp_warm_std,
)
from scipy.optimize import linprog

pytestmark = [pytest.mark.correctness]


def _degenerate_lp(n: int = 8, seed: int = 0):
    """A small stand-in for the pathology: a lifted LP carrying linear equalities
    encoded as opposing ``<=`` pairs, so every one of them is tight everywhere.

    Generic construction (random equalities over a box), not a named instance.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1.0, 1.0, size=n)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    x0 = rng.uniform(0.2, 0.8, size=n)  # a strictly interior feasible point
    for _ in range(3):
        a = rng.uniform(-1.0, 1.0, size=n)
        r = float(a @ x0)
        rows.append(a)
        rhs.append(r)  # a'x <= r
        rows.append(-a)
        rhs.append(-r)  # -a'x <= -r  =>  together a'x == r
    for _ in range(4):  # a few ordinary inequalities on top
        a = rng.uniform(-1.0, 1.0, size=n)
        rows.append(a)
        rhs.append(float(a @ x0) + 0.5)
    A = sp.csr_matrix(np.array(rows))
    b = np.array(rhs, dtype=float)
    bounds = [(0.0, 1.0)] * n
    return c, A, b, bounds


def _reference(c, A, b, bounds):
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def test_flag_controls_the_cold_deadline_free_start(monkeypatch):
    """The flag is what decides whether a cold, deadline-free LP gets the basis."""
    monkeypatch.delenv("DISCOPT_LP_COLD_DUAL_START", raising=False)
    assert _cold_dual_start_enabled() is False
    monkeypatch.setenv("DISCOPT_LP_COLD_DUAL_START", "1")
    assert _cold_dual_start_enabled() is True


def _record_start_basis(monkeypatch) -> list:
    """Capture the ``(col_status, basic_vars)`` actually handed to Rust.

    Without this the behavioural tests below cannot see the gate at all: both
    arms converge to the same optimum by construction, so a gate that never
    fires reads as a pass. ``solve_lp_warm_std`` imports the binding inside the
    function body, so patching the module attribute is what intercepts it.
    """
    import discopt._rust as _rust

    seen: list = []
    real = _rust.solve_lp_warm_csc_py

    def spy(*args, **kwargs):
        seen.append(args[9])  # cs0, the start column-status vector (None = cold primal)
        return real(*args, **kwargs)

    monkeypatch.setattr(_rust, "solve_lp_warm_csc_py", spy)
    return seen


@pytest.mark.parametrize(
    ("env", "deadline", "expect_basis"),
    [
        (None, None, False),  # today's default: cold primal
        ("1", None, True),  # what the flag exists to change
        (None, 30.0, True),  # #928's deadline path, unchanged
        ("1", 30.0, True),
    ],
)
def test_the_gate_is_wired(monkeypatch, env, deadline, expect_basis):
    """The flag/deadline combination decides what reaches the Rust binding.

    Reverting the gate to its pre-flag form (deadline only) fails the second row.
    """
    if env is None:
        monkeypatch.delenv("DISCOPT_LP_COLD_DUAL_START", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_LP_COLD_DUAL_START", env)
    seen = _record_start_basis(monkeypatch)

    c, A, b, bounds = _degenerate_lp(seed=5)
    res, _basis = solve_lp_warm_std(c, A, b, bounds, in_basis=None, time_limit=deadline)
    assert res is not None and res.status == SolveStatus.OPTIMAL

    assert seen, "the Rust binding was never called -- the spy measured nothing"
    got_basis = seen[0] is not None
    assert got_basis is expect_basis, (
        f"env={env} deadline={deadline}: start basis passed={got_basis}, want {expect_basis}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimum_is_identical_with_and_without_the_flag(monkeypatch, seed):
    """A start basis cannot move the LP optimum. Both arms are run with no
    deadline, so both must converge, and both must match SciPy."""
    c, A, b, bounds = _degenerate_lp(seed=seed)
    ref = _reference(c, A, b, bounds)

    got = {}
    for arm in ("off", "on"):
        if arm == "on":
            monkeypatch.setenv("DISCOPT_LP_COLD_DUAL_START", "1")
        else:
            monkeypatch.delenv("DISCOPT_LP_COLD_DUAL_START", raising=False)
        res, _basis = solve_lp_warm_std(c, A, b, bounds, in_basis=None, time_limit=None)
        assert res is not None, f"arm {arm}: no result"
        assert res.status == SolveStatus.OPTIMAL, f"arm {arm}: {res.status}"
        got[arm] = float(res.objective)

    assert got["on"] == pytest.approx(ref, abs=1e-9, rel=1e-9)
    assert got["off"] == pytest.approx(ref, abs=1e-9, rel=1e-9)
    assert got["on"] == pytest.approx(got["off"], abs=1e-9, rel=1e-9)


def test_open_selected_side_keeps_the_primal_path():
    """``_dual_start_slack_basis`` refuses when the side the objective sign picks
    is unbounded — that basis would not be dual-feasible. The flag must not be
    able to force it: this is the eligibility guard, not a preference."""
    n = 3
    m = 4
    c = np.array([1.0, -1.0, 1.0])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1e20, 1.0])  # c[1] < 0 selects the OPEN upper bound
    assert _dual_start_slack_basis(c, lb, ub, m) is None

    ub_ok = np.array([1.0, 1.0, 1.0])
    start = _dual_start_slack_basis(c, lb, ub_ok, m)
    assert start is not None
    col_status, basic_vars = start
    # AT_LOWER=0, BASIC=1, AT_UPPER=2; slacks basic, structurals at the sign-picked side.
    assert list(col_status[:n]) == [0, 2, 0]
    assert list(col_status[n:]) == [1] * m
    assert list(basic_vars) == list(range(n, n + m))


def test_deadline_path_is_untouched_by_the_flag_being_off(monkeypatch):
    """A finite deadline already engaged the dual start (#928) and still must,
    with the flag OFF. Guards against the gate rewrite dropping that arm."""
    monkeypatch.delenv("DISCOPT_LP_COLD_DUAL_START", raising=False)
    c, A, b, bounds = _degenerate_lp(seed=3)
    ref = _reference(c, A, b, bounds)
    res, _basis = solve_lp_warm_std(c, A, b, bounds, in_basis=None, time_limit=30.0)
    assert res is not None and res.status == SolveStatus.OPTIMAL
    assert float(res.objective) == pytest.approx(ref, abs=1e-9, rel=1e-9)
