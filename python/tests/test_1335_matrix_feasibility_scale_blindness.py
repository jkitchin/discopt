"""#1335: the matrix-form feasibility gate was scale-blind in the accept direction.

`_matrix_solution_feasible` is *the arbiter* -- the check that decides whether an
LP/QP engine's point may be returned as the answer. Its per-row test was
``|viol| <= tol + rtol*row_scale`` with ``rtol = 1e-9`` and ``row_scale =
sum_j |A_ij| |x_j|``, and nothing else. On a row whose terms total ``2e15`` that
licenses a violation of **two million**.

The consequence, from the issue::

    x, y in [1e15, 3e15];  x - y <= 1;  x - y >= 10;  min x + y

which is infeasible by inspection -- no ``x - y`` is both -- and whose minimum
total violation is 9. With ``DISCOPT_LP_MILP_BACKEND=rust`` the engine returned
``x - y = 5.5``, violating BOTH rows by 4.5; the gate passed it, and the solve
came back ``status=optimal, gap_certified=True``. A certified wrong answer on a
default-reachable route, which CLAUDE.md §1 admits no slack on.

The fix applies #1254's rule -- which the NLP-side verifier has had since then and
this one never received -- capping the residual allowance by POSITION: a violation
is only forgivable if a small move in variable space fixes it, and for a linear
row that distance is ``viol / ||A_i||_inf``. At 4.5 on a unit-coefficient row the
point is 4.5 away from feasible, which no tolerance makes near-feasible.

The companion floor (#937) is kept, but stated as the row's own finest step
``min_j |A_ij| ulp(x_j)`` rather than ``1e-9 * sum_j |A_ij x_j|``: the MINIMUM
answers "can any column absorb this?", which is the question that decides the
case, and the sum does not.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke

_NEVER = ("optimal", "feasible")


def _conflicting_lp(big: float):
    """``x - y <= 1`` and ``x - y >= 10`` over ``[big, 3*big]``: infeasible at any
    scale, by inspection, with minimum total violation 9."""
    m = dm.Model("i1335")
    x = m.continuous("x", lb=big, ub=3 * big)
    y = m.continuous("y", lb=big, ub=3 * big)
    m.subject_to(x - y <= 1.0)
    m.subject_to(x - y >= 10.0)
    m.minimize(x + y)
    return m


@pytest.mark.parametrize("big", [1.0e8, 1.0e11, 1.0e14, 1.0e15, 1.0e16])
def test_conflicting_rows_are_never_certified_feasible(big, monkeypatch):
    """The §1 line across the magnitude band, on the opt-out route the issue names.

    ``infeasible`` (the right answer, and what the small-magnitude cases give) or
    ``error`` (honest: no engine could decide it) are both acceptable; ``optimal``
    and ``feasible`` are not, at any scale. Pre-fix this returned certified
    ``optimal`` at 1e15 and 1e16.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _conflicting_lp(big).solve(time_limit=60)
    assert r.status not in _NEVER, (
        f"big={big:e}: an infeasible LP came back {r.status!r} "
        f"(certified={r.gap_certified}) at {dict(r.x) if r.x else None}"
    )


def test_the_issues_own_repro_is_not_a_certified_optimum(monkeypatch):
    """Verbatim from #1335, including the certification flag it carried."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _conflicting_lp(1.0e15).solve(time_limit=60)
    assert r.status != "optimal"
    assert not (r.status in _NEVER and r.gap_certified)


def test_a_feasible_large_scale_lp_still_certifies(monkeypatch):
    """The other direction: tightening the gate must not cost a correct answer.
    Same box and rows made satisfiable (``1 <= x - y <= 10``)."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    big = 1.0e15
    m = dm.Model("i1335_ok")
    x = m.continuous("x", lb=big, ub=3 * big)
    y = m.continuous("y", lb=big, ub=3 * big)
    m.subject_to(x - y <= 10.0)
    m.subject_to(x - y >= 1.0)
    m.minimize(x + y)
    r = m.solve(time_limit=60)
    assert r.status == "optimal", f"got {r.status}"
    d = float(np.asarray(r.x["x"])) - float(np.asarray(r.x["y"]))
    assert 1.0 - 1e-6 <= d <= 10.0 + 1e-6, f"certified point violates its own row: x-y={d}"


# ── unit: the gate itself ────────────────────────────────────────────────


def test_gate_rejects_the_violating_point_directly():
    """The arbiter, called on the exact point the engine returned. Pre-fix the
    threshold was ``1e-6 + 1e-9*2e15 = 2e6`` against a violation of 4.5."""
    x = np.array([1.0000000000000054e15, 9.9999999999999888e14])
    A_ub = np.array([[1.0, -1.0], [-1.0, 1.0]])
    b_ub = np.array([1.0, -10.0])
    viol = A_ub @ x - b_ub
    assert viol.max() > 4.0, f"probe must present a real violation, got {viol}"
    assert not S._matrix_solution_feasible(x, A_ub, b_ub, None, None, [(1e15, 3e15), (1e15, 3e15)])


@pytest.mark.parametrize("big", [1.0e12, 1.0e15, 1.0e16, 1.0e18])
def test_a_unit_coefficient_row_never_forgives_a_resolvable_violation(big):
    """Scale-independence of the position cap: on a row whose coefficients are ±1,
    a violation the row's own columns CAN resolve means the point is that far from
    feasible in variable space, however large the columns are.

    The offset is taken above the column's ulp on purpose. Asking for 4.5 at
    ``big = 1e18`` asks for nothing: ``ulp(1e18) = 256``, so ``1e18 + 5.5`` IS
    ``1e18`` and the constructed point satisfies the row -- a probe that tests
    nothing while reading as a pass (CLAUDE.md §6). The assertions below prove the
    violation is both real and above the representability floor before requiring
    it to be rejected."""
    off = max(4.5, 64.0 * float(np.spacing(big)))
    x = np.array([big + off, big])
    A_ub = np.array([[1.0, -1.0]])
    b_ub = np.array([1.0])

    viol = float((A_ub @ x - b_ub)[0])
    floor = S._row_representability_floor(A_ub[0], x)
    assert viol > 1.0, f"big={big:e}: probe built no violation (viol={viol})"
    assert viol > floor, (
        f"big={big:e}: probe's violation {viol} is inside the row's own "
        f"resolution {floor} -- it tests nothing"
    )
    assert not S._matrix_solution_feasible(x, A_ub, b_ub, None, None, None), (
        f"big={big:e}: forgave a resolvable violation of {viol} on a unit-coefficient row"
    )


def test_representability_floor_is_the_minimum_column_not_the_sum():
    """The #937 floor, restated. A row of two 1e15 columns can only step as finely
    as one of them; the SUM of their magnitudes is not a resolution limit and using
    it is what forgave 4.5."""
    checked = 0

    a = np.array([1.0, -1.0])
    x = np.array([1e15, 1e15])
    floor = S._row_representability_floor(a, x)
    # min_j |A_ij| ulp(x_j) ~ 2*eps*1e15, NOT 2*eps*2e15 and certainly not 1e-9*2e15
    assert floor == pytest.approx(2 * np.finfo(np.float64).eps * 1e15, rel=1e-9)
    checked += 1
    assert floor < 4.5, "the floor must not reach the violation it has to catch"
    checked += 1

    # A column at 0 can be nudged arbitrarily finely -> no floor at all.
    assert (
        S._row_representability_floor(np.array([1.0, -1.0, 1.0]), np.array([1e15, 1e15, 0.0]))
        == 0.0
    )
    checked += 1

    # A structurally empty row cannot be moved, so a residual on it is real.
    assert S._row_representability_floor(np.array([0.0, 0.0]), np.array([1e15, 1e15])) == 0.0
    checked += 1

    # Zero coefficients do not participate: the 1e15 column is masked out by A=0.
    assert S._row_representability_floor(
        np.array([0.0, 1.0]), np.array([1e15, 1e3])
    ) == pytest.approx(2 * np.finfo(np.float64).eps * 1e3, rel=1e-9)
    checked += 1

    assert checked == 5, f"probe ran {checked} assertions, expected 5"


def test_the_850_cancellation_noise_boundary_is_preserved():
    """The guard this fix must not weaken: a point carrying only noise proportional
    to a large-term row is still accepted (``test_matrix_guard_accepts_exact_and_noise_points``)."""
    A_ub = np.array([[1.0]])
    b_ub = np.array([1e4])
    assert S._matrix_solution_feasible(np.array([1e4]), A_ub, b_ub, None, None, [(0.0, 1e6)])
    assert S._matrix_solution_feasible(np.array([1e4 + 5e-6]), A_ub, b_ub, None, None, [(0.0, 1e6)])
    # ...and the point 1e-4 on the infeasible side is still rejected.
    assert not S._matrix_solution_feasible(
        np.array([10000.0000999]), A_ub, b_ub, None, None, [(0.0, 1e6)]
    )
