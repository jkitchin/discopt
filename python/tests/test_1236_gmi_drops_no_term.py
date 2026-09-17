"""#1236: the GMI separator must not delete a term from a cut's left-hand side.

``separate_gomory_cols`` builds ``sum psi_j xtilde_j >= 1`` with ``psi_j >= 0``
and ``xtilde_j >= 0``. It used to remove terms two ways -- snapping a tableau
coefficient to the nearest integer (which zeroes a continuous column's ``psi``)
and skipping ``|psi| <= tol`` -- without charging anything to the right-hand
side. Removing a nonnegative term from the LHS of a ``>=`` is a STRENGTHENING,
so the cut can exclude feasible integer points.

Harmless while ``xtilde`` is O(1), which is every binary. On a big-M master the
dropped terms sit on continuous slacks with ``u = 1e20``: measured on the ``fac2``
OA master captured here, two coefficients of ~1e-10 multiplied ``xtilde`` values
of 7e7 and 2.3e8, so they were worth 0.47 and 0.025 against a right-hand side of
1. The cut cut off the optimum and the in-house driver returned a **certified
false** ``optimal`` 7839 above it -- which OA then republished as the MINLP's
answer.

The fix charges a dropped term's maximum over its box to the rhs (a relaxation)
and keeps the term exactly when that range is unbounded. The Rust unit test
``gmi_tiny_coefficient_on_wide_continuous_column_is_not_dropped`` pins the
mechanism on a three-column LP; this pins the real instance end to end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.modeling.core import from_nl
from discopt.solvers import milp_simplex
from discopt.solvers.lp_backend import get_milp_solver

DATA = os.path.join(os.path.dirname(__file__), "data")
MASTER = os.path.join(DATA, "oa_masters", "fac2_master0.npz")
NL = os.path.join(DATA, "minlplib_nl", "fac2.nl")

#: MINLPLib reference optimum for fac2.
FAC2_REFERENCE = 331837498.2

#: Absolute slack for "not above the witness", in objective units.
#:
#: ABSOLUTE, never scale-relative -- the fixture README says so, and this test was
#: shipped violating it (review finding 9). ``1e-6 * |attained|`` reads as tight
#: and is **331.8** on a 3.3e8 objective: it catches the original 7839 defect with
#: only a 24x margin and would miss any regression an order of magnitude smaller.
#: A scale-relative gate is exactly how this defect was missed once already.
#:
#: The numbers are measured on this fixture, not chosen. Against the witness at
#: 331837498.17693394 the four backends land at ``auto`` -1.67e-4, ``simplex``
#: -1.67e-4, ``highs`` -1.64e-4 (all BELOW it, i.e. no excess at all) and
#: ``pounce`` +0.32 -- a 1e-9 relative excess, tolerance noise at this scale.
#:
#: So: 1.0 for the in-house driver, whose measured margin is 1.7e-4, and 10.0 for
#: the cross-backend comparison, which must clear POUNCE's +0.32 with headroom.
#: Both catch the 7839 defect with a margin of ~800x or better, against 24x before.
ABS_SLACK_DRIVER = 1.0
ABS_SLACK_BACKENDS = 10.0


def _load_master():
    d = np.load(MASTER)
    kw = dict(
        c=d["c"],
        bounds=list(zip(d["lb"].tolist(), d["ub"].tolist())),
        integrality=d["integrality"],
    )
    for mat, rhs in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        kw[mat] = sp.csr_matrix(
            (d[f"{mat}_data"], d[f"{mat}_indices"], d[f"{mat}_indptr"]),
            shape=tuple(d[f"{mat}_shape"]),
        )
        kw[rhs] = d[rhs]
    return kw, d["z_star"], float(d["z_star_objective"])


@pytest.mark.smoke
def test_captured_master_is_not_certified_above_a_feasible_point():
    problem, z_star, attained = _load_master()

    # The witness must really be feasible, or the assertion below proves nothing
    # (CLAUDE.md §6): re-verify it here rather than trusting the fixture.
    rows = problem["A_ub"] @ z_star - problem["b_ub"]
    assert np.all(rows <= 1e-6 * np.maximum(1.0, np.abs(problem["b_ub"])))
    eq = np.abs(problem["A_eq"] @ z_star - problem["b_eq"])
    assert np.all(eq <= 1e-6 * np.maximum(1.0, np.abs(problem["b_eq"])))
    lo = np.array([b[0] for b in problem["bounds"]])
    hi = np.array([b[1] for b in problem["bounds"]])
    assert np.all(z_star >= lo - 1e-6) and np.all(z_star <= hi + 1e-6)
    ints = np.nonzero(problem["integrality"])[0]
    assert np.all(np.abs(z_star[ints] - np.round(z_star[ints])) <= 1e-6)
    assert abs(float(problem["c"] @ z_star) - attained) <= 1e-6

    res = milp_simplex.solve_milp(max_nodes=500_000, time_limit=60.0, **problem)

    # Tolerances are ABSOLUTE against the attained value. A scale-relative gate
    # is what hid this defect during the investigation: |z*|max is 3.3e8 here, so
    # a 1e-9-relative threshold is 0.86 -- larger than the 0.017 cut violation and
    # far larger than any honest rounding allowance on a 3.3e8 objective.
    slack = ABS_SLACK_DRIVER
    assert res.objective is not None
    assert res.objective <= attained + slack, (
        f"driver reports objective {res.objective!r}, above a point of "
        f"{attained!r} that is feasible for this very master"
    )
    assert res.bound is not None and res.bound <= attained + slack, (
        f"driver reports dual bound {res.bound!r} above a feasible point "
        f"of {attained!r} -- a false certificate"
    )


@pytest.mark.smoke
def test_all_milp_backends_agree_on_the_captured_master():
    """The in-house engine must not be the odd one out.

    POUNCE and HiGHS solve this master correctly; before the fix the in-house
    simplex was alone in certifying a wrong answer, which is the signature that
    made the defect attributable at all.
    """
    problem, _z, attained = _load_master()
    slack = ABS_SLACK_BACKENDS
    checked = 0
    for backend in ("auto", "simplex", "pounce", "highs"):
        solve = get_milp_solver(backend=backend)
        res = solve(time_limit=60.0, gap_tolerance=1e-4, **problem)
        checked += 1
        assert res.objective is not None and res.objective <= attained + slack, (
            f"backend {backend!r} returned {res.objective!r} above {attained!r}"
        )
    assert checked == 4, "a backend was skipped; this comparison proves nothing"


@pytest.mark.smoke
def test_fac2_end_to_end_matches_the_reference_optimum():
    """The defect reached the user as a wrong certified MINLP answer; pin that."""
    result = from_nl(NL).solve(time_limit=60.0)
    assert result.status == "optimal", result.status
    assert result.objective is not None
    rel = abs(result.objective - FAC2_REFERENCE) / abs(FAC2_REFERENCE)
    assert rel <= 1e-6, (
        f"fac2 objective {result.objective!r} differs from the MINLPLib "
        f"reference {FAC2_REFERENCE} by {rel:.3e}"
    )
    assert result.bound is not None
    # Absolute, for the same reason as above: 1e-6 * |objective| is 331.8 here, so
    # the certificate invariant `bound <= incumbent` would be checked with 331.8 of
    # slack. Measured margin on this instance is -1.12e-4 (the bound sits BELOW the
    # incumbent, as it must), so 1.0 is ~9000x the observed noise and still catches
    # a violation four orders of magnitude smaller than the one this test exists for.
    assert result.bound <= result.objective + ABS_SLACK_DRIVER, (
        f"bound {result.bound!r} above incumbent {result.objective!r}"
    )
