"""The warm pure-LP node path honours its caller's ``time_limit`` (#917 follow-up).

``MilpRelaxationModel.solve`` accepts a ``time_limit``, but its DEFAULT
``backend="simplex"`` pure-LP fast path dropped it on the floor: ``_solve_lp_warm``
and ``_solve_lp_warm_equilibrated`` took no deadline, ``solve_lp_warm_std`` took
none, and ``lp_bindings.rs`` hardcoded ``SimplexOptions { deadline: None }`` — while
the MILP route (``solve_milp_csc_py(time_limit_s=…)``) wired it up and the dual and
primal pivot loops already poll it every 256 pivots.

Measured on nvs24 (``scratchpad/nvs24_arm.py``,
``scratchpad/nvs24_profile_evidence.txt``): ``solve(time_limit=0.202)`` reached
``_solve_lp_warm`` and ran **47.03 s** — one ``DualPivotLoop`` 59 494 degenerate dual
pivots deep, Bland's rule never activated — turning a 3.9 s solve budget into 53 s
(13.5x). ~13 call sites in ``_relax/mccormick_lp.py`` plus ``lp_spatial_bb.py`` and
``integer_ratio.py`` compute a per-LP budget and pass it here, so the drop was
general rather than an nvs24 quirk.

Cutting an LP short changes the bound it returns, so the fix is bound-CHANGING and
ships behind ``DISCOPT_LP_WARM_DEADLINE`` (CLAUDE.md §5).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._relax.milp_relaxation import (  # noqa: E402
    MilpRelaxationModel,
    _lp_warm_deadline_enabled,
)
from discopt.solvers.milp_simplex import solve_lp_warm_std  # noqa: E402


@pytest.fixture
def warm_dl(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_LP_WARM_DEADLINE", value)

    return _set


def _small_lp():
    """A tiny bounded LP: min -x0 - x1 s.t. x0 + x1 <= 1, x in [0,1]^2. Optimum -1."""
    c = np.array([-1.0, -1.0])
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    b = np.array([1.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    return c, A, b, bounds


# ── the flag ────────────────────────────────────────────────────────────────


def test_flag_defaults_on_after_graduation(warm_dl):
    """GRADUATED default-ON 2026-08-11 (#928, performance-plan §14f); ``=0`` opts out.

    Four-arm panel on the merged tree (19 binding instances, 3 reps at each of two
    budgets, arms interleaved per instance): with the round budget, total overrun vs
    base is -24.2/-43.8/+4.4 s at 20 s and -6.2/-6.2/+1.1 s at 15 s; the bound ledger
    is positive (4-7 tighter against 1-4 looser per rep, no bound lost, hda
    -2.07e13 -> -119286.3 in 6/6 reps); no incumbent lost in 6/6 and a better one every
    rep; zero unsound cells and zero ceiling violations over 168 counted comparisons.
    The lost incumbent that failed the two earlier panels was attributed to
    ``DISCOPT_HESS_COMPILE_GATE``, which stays OFF.

    The flag-graduation convention keeps ``=0`` as the escape hatch back to the legacy
    path that dropped the caller's ``time_limit`` — asserted by ``test_flag_parses``."""
    warm_dl(None)
    assert _lp_warm_deadline_enabled() is True


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("off", False)])
def test_flag_parses(warm_dl, value, expected):
    warm_dl(value)
    assert _lp_warm_deadline_enabled() is expected


# ── the seam that dropped the budget ────────────────────────────────────────


def test_solve_lp_warm_std_accepts_a_time_limit():
    """The parameter's absence WAS the bug: every caller computed a per-LP budget and
    this signature silently ignored it."""
    c, A, b, bounds = _small_lp()
    result, _basis = solve_lp_warm_std(c, A, b, bounds, time_limit=None)
    assert result is not None
    assert result.objective == pytest.approx(-1.0, abs=1e-6)


def test_zero_time_limit_yields_instead_of_solving():
    """``time_limit=0.0`` is an already-elapsed deadline — the state of a caller whose
    outer budget is spent. The simplex must yield a limit exit (``result is None``,
    the same shape as a pivot-cap exit) rather than run the LP anyway.

    Deterministic by construction: the deadline is in the past before the first pivot,
    so this asserts the plumbing, not a race."""
    c, A, b, bounds = _small_lp()
    result, basis = solve_lp_warm_std(c, A, b, bounds, time_limit=0.0)
    assert result is None, "a spent budget must not be silently ignored"
    assert basis is None


def test_time_limit_reaches_the_rust_binding():
    """Guards the whole chain, not just the Python signature: a ``time_limit`` that
    stops at the marshalling layer would leave the simplex unbounded exactly as
    before, and the only observable difference is this argument."""
    import discopt._rust as _rust

    seen = {}
    orig = _rust.solve_lp_warm_csc_py

    def spy(*a, **kw):
        seen["time_limit_s"] = kw.get("time_limit_s", "ABSENT")
        return orig(*a, **kw)

    _rust.solve_lp_warm_csc_py = spy
    try:
        c, A, b, bounds = _small_lp()
        solve_lp_warm_std(c, A, b, bounds, time_limit=1.25)
    finally:
        _rust.solve_lp_warm_csc_py = orig
    assert seen.get("time_limit_s") == pytest.approx(1.25)


def test_rust_binding_rejects_a_nan_budget():
    """A NaN duration would become a nonsense ``Instant`` and silently disable the
    deadline; it must raise instead."""
    import discopt._rust as _rust

    c, A, b, bounds = _small_lp()
    a_std = sp.hstack([sp.csc_matrix(A), sp.identity(1, format="csc")], format="csc").tocsc()
    with pytest.raises(ValueError):
        _rust.solve_lp_warm_csc_py(
            np.ascontiguousarray(np.concatenate([c, [0.0]])),
            1,
            3,
            np.ascontiguousarray(a_std.indptr, dtype=np.int64),
            np.ascontiguousarray(a_std.indices, dtype=np.int64),
            np.ascontiguousarray(a_std.data, dtype=np.float64),
            np.ascontiguousarray(b),
            np.ascontiguousarray(np.array([lo for lo, _ in bounds] + [0.0])),
            np.ascontiguousarray(np.array([hi for _, hi in bounds] + [1e20])),
            None,
            None,
            time_limit_s=float("nan"),
        )


def test_rust_binding_treats_infinity_as_no_limit():
    """``+inf`` is the natural spelling of "uncapped"; rejecting it would push a
    ValueError into a defensive ``except`` and silently disable the fast path."""
    c, A, b, bounds = _small_lp()
    result, _ = solve_lp_warm_std(c, A, b, bounds, time_limit=float("inf"))
    assert result is not None
    assert result.objective == pytest.approx(-1.0, abs=1e-6)


# ── one budget shared across the attempts ───────────────────────────────────


def _relaxation():
    c, A, b, bounds = _small_lp()
    return MilpRelaxationModel(c=c, A_ub=A, b_ub=b, bounds=bounds)


def test_budget_is_shared_across_attempts_when_enabled(warm_dl, monkeypatch):
    """``solve`` may try warm, equilibrated and cold in turn. Each must draw on ONE
    budget: handing all three a fresh copy of the caller's duration would silently
    triple the limit the caller set."""
    warm_dl("1")
    seen = []

    import discopt.solvers.milp_simplex as MS

    orig = MS.solve_lp_warm_std

    def spy(*a, **kw):
        seen.append(kw.get("time_limit", "ABSENT"))
        return orig(*a, **kw)

    monkeypatch.setattr(MS, "solve_lp_warm_std", spy)
    _relaxation().solve(time_limit=5.0, backend="simplex")
    assert seen, "the warm path never ran — this test would assert nothing"
    assert all(v != "ABSENT" and v is not None for v in seen)
    assert all(v <= 5.0 + 1e-9 for v in seen)


def test_warm_path_is_unbounded_when_disabled(warm_dl, monkeypatch):
    """Flag OFF must reproduce the historical call exactly: no deadline on the warm
    path, whatever the caller passed."""
    warm_dl("0")
    seen = []

    import discopt.solvers.milp_simplex as MS

    orig = MS.solve_lp_warm_std

    def spy(*a, **kw):
        seen.append(kw.get("time_limit", "ABSENT"))
        return orig(*a, **kw)

    monkeypatch.setattr(MS, "solve_lp_warm_std", spy)
    _relaxation().solve(time_limit=5.0, backend="simplex")
    assert seen, "the warm path never ran — this test would assert nothing"
    assert all(v is None for v in seen)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_result_is_unchanged_under_either_flag(warm_dl, flag):
    """The deadline can only ever shorten an LP, never alter what an LP that finishes
    returns. On a solve that completes far inside its budget both arms must agree."""
    warm_dl(flag)
    res = _relaxation().solve(time_limit=30.0, backend="simplex")
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-1.0, abs=1e-6)


# ── a yielded LP must not silently drop to "nothing known" ──────────────────


def test_timed_out_solve_surfaces_a_recovered_floor(warm_dl, monkeypatch):
    """Honouring the deadline must not cost a bound outright.

    When the warm LP yields, the simplex's own dual still gives a
    Neumaier-Shcherbina floor, and ``g(y)`` is a valid lower bound for ANY
    multiplier vector by weak duality — stopping early only loosens it. So a
    timed-out ``solve`` reports that floor rather than ``bound=None``.

    Regression: at a 15 s budget the first cut of this change turned bchoco08's
    ``bound=1.0`` into ``None`` and contvar's 171244.81 into ``None`` — trading a
    budget overrun for a lost certificate, which is the wrong trade (CLAUDE.md §1).
    """
    warm_dl("1")
    from discopt.solvers.milp_simplex import LpWarmCert

    rel = _relaxation()
    import discopt.solvers.milp_simplex as MS

    def _yield_with_a_floor(*a, **kw):
        # What the Rust side returns on a deadline exit: no result, but a dual that
        # still implies a sound floor.
        return None, None, LpWarmCert(safe_bound=-7.5, farkas_certified=False)

    monkeypatch.setattr(MS, "solve_lp_warm_std", _yield_with_a_floor)
    res = rel.solve(time_limit=0.0, backend="simplex")
    assert res.status == "time_limit"
    assert res.objective is None, "a yielded LP proves no incumbent"
    assert res.bound == pytest.approx(-7.5), "the recovered floor must not be discarded"
    assert res.bound <= -1.0 + 1e-9, "UNSOUND: floor above the true optimum -1.0"


def test_timed_out_solve_without_a_floor_reports_no_bound(warm_dl, monkeypatch):
    """The converse: with no recoverable dual there is nothing sound to report, and
    the result must say so rather than invent one."""
    warm_dl("1")
    import discopt.solvers.milp_simplex as MS
    from discopt.solvers.milp_simplex import LpWarmCert

    monkeypatch.setattr(
        MS,
        "solve_lp_warm_std",
        lambda *a, **kw: (None, None, LpWarmCert(safe_bound=None, farkas_certified=False)),
    )
    res = _relaxation().solve(time_limit=0.0, backend="simplex")
    assert res.status == "time_limit"
    assert res.bound is None


def test_deadline_exit_banks_the_warm_basis_floor():
    """#928: a deadline exit must return the best bound the solve already had.

    Warm-start the LP from its own OPTIMAL basis and cut it with an already-spent
    budget. The dual loop's deadline exit banks the current (here: optimal) basis's
    row duals, so the recovered Neumaier-Shcherbina floor must sit at the LP
    optimum (within its rigor margin) — not at the trivial ``g(y=0)`` box bound.

    Before the fix the dual loop discarded its basis on a deadline (``return
    None`` → cold fallback → ``IterLimit`` from the cold INITIAL basis), so the
    banked floor collapsed to ``g(0)``: measured on the hda separated-relaxation
    node LP, floor -141697 at 15%%, 40%% and 75%% deadlines alike, with the
    optimum -64473 fractions of a second away.

    Deterministic: the deadline is in the past before the first pivot, so the
    banked basis is exactly the one passed in.
    """
    c, A, b, bounds = _small_lp()
    res_full, out_basis, _cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert res_full is not None and res_full.objective == pytest.approx(-1.0, abs=1e-6)
    assert out_basis is not None

    res_cut, _, cert = solve_lp_warm_std(
        c, A, b, bounds, tuple(out_basis), return_cert=True, time_limit=0.0
    )
    assert res_cut is None, "a spent budget must still yield, never solve on"
    assert cert.safe_bound is not None, "the deadline exit must bank a floor"
    assert cert.safe_bound <= -1.0 + 1e-9, "UNSOUND: floor above the true optimum"
    assert cert.safe_bound == pytest.approx(-1.0, abs=1e-6), (
        "the optimal-basis floor was discarded: a deadline exit must bank the "
        f"bound it already had (got {cert.safe_bound}, optimum -1.0)"
    )


def test_cold_deadline_solve_starts_the_dual_simplex():
    """#928: a COLD solve carrying a finite deadline threads the sign-matched
    slack basis to the engine, so the (anytime, monotone-bound) dual simplex runs
    instead of the primal — the primal proves no usable floor mid-run. Without a
    deadline the historical no-basis call is preserved bit-for-bit."""
    import discopt._rust as _rust

    calls = []
    orig = _rust.solve_lp_warm_csc_py

    def spy(*a, **kw):
        calls.append(a[9] is not None)  # start_col_status positional slot
        return orig(*a, **kw)

    _rust.solve_lp_warm_csc_py = spy
    try:
        c, A, b, bounds = _small_lp()
        solve_lp_warm_std(c, A, b, bounds, time_limit=None)
        solve_lp_warm_std(c, A, b, bounds, time_limit=30.0)
        solve_lp_warm_std(c, A, b, bounds, time_limit=float("inf"))
    finally:
        _rust.solve_lp_warm_csc_py = orig
    assert calls == [False, True, False], (
        "slack-dual start must engage exactly on finite-deadline cold solves; "
        f"got {calls} for [no limit, 30s, inf]"
    )


def test_dual_start_slack_basis_eligibility():
    """The sign-matched slack basis is proposed only when it is dual-feasible:
    every objective-selected bound side must be finite. ``PreparedDual::prepare``
    re-verifies the same precondition exactly, so this gate is about not wasting a
    prepare, never about soundness."""
    import numpy as np
    from discopt.solvers.milp_simplex import _dual_start_slack_basis

    lb = np.array([0.0, -1e20])
    ub = np.array([1.0, 1e20])
    # c[1] > 0 selects the lower bound of a free-below column: dual-infeasible.
    assert _dual_start_slack_basis(np.array([1.0, 1.0]), lb, ub, m=2) is None
    # c[1] < 0 selects the (open) upper bound: dual-infeasible.
    assert _dual_start_slack_basis(np.array([1.0, -1.0]), lb, ub, m=2) is None
    # c[1] = 0 constrains nothing: eligible; slacks basic, structurals at bounds.
    out = _dual_start_slack_basis(np.array([-1.0, 0.0]), lb, ub, m=2)
    assert out is not None
    col_status, basic_vars = out
    assert col_status.tolist() == [2, 0, 1, 1]  # AT_UPPER, AT_LOWER, BASIC, BASIC
    assert basic_vars.tolist() == [2, 3]
    # No rows -> no slack basis to build.
    assert _dual_start_slack_basis(np.array([-1.0, 0.0]), lb, ub, m=0) is None


def test_iter_limit_exports_a_dual_candidate():
    """The Rust blocker: a deadline exit must still hand back a dual candidate.

    ``primal.rs`` exported ``y = B⁻ᵀc_B`` on a ``Numerical`` exit but kept an EMPTY
    dual on ``IterLimit``, so an LP cut short by its caller's budget returned nothing
    at all — no optimum AND no floor — and every Python-side attempt to bank a bound
    from it was banking an empty vector. Measured before the fix:
    ``solve_lp_warm_csc_py(..., time_limit_s=0.0)`` returned ``dual=[]``.

    A Neumaier-Shcherbina bound is valid for ANY multiplier vector by weak duality, so
    exporting a mid-solve candidate can only produce a looser floor, never an unsound
    one.
    """
    import discopt._rust as _rust

    c, A, b, bounds = _small_lp()
    a_std = sp.hstack([sp.csc_matrix(A), sp.identity(1, format="csc")], format="csc").tocsc()
    args = (
        np.ascontiguousarray(np.concatenate([c, [0.0]])),
        1,
        3,
        np.ascontiguousarray(a_std.indptr, dtype=np.int64),
        np.ascontiguousarray(a_std.indices, dtype=np.int64),
        np.ascontiguousarray(a_std.data, dtype=np.float64),
        np.ascontiguousarray(b),
        np.ascontiguousarray(np.array([lo for lo, _ in bounds] + [0.0])),
        np.ascontiguousarray(np.array([hi for _, hi in bounds] + [1e20])),
        None,
        None,
    )
    status, _x, _obj, _iters, _cs, _bv, dual, _ray = _rust.solve_lp_warm_csc_py(
        *args, time_limit_s=0.0
    )
    assert status == "iter_limit", f"expected a limit exit, got {status}"
    assert np.size(dual) == 1, "a yielded LP must still export its dual candidate"
