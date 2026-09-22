"""Big-M indicator rows must reach the true optimum on EVERY route (#1414, #1380).

#1380's repro is the minimal witness for the class:

    min -x + 3z   s.t.  x <= M*z,  x in [0,10],  z binary

``z`` is binary, so ``z=0`` forces ``x=0`` (objective 0) and ``z=1`` allows
``x=10`` (objective -7). The true optimum is **-7.0** for every ``M >= 10``.

The defect is not unsoundness — the #1380/#1383 guards correctly refuse to
certify the bad point — but *weakness*, and it is a function of ``M``:

  * the LP relaxation admits ``z = 10/M`` with ``x = 10``, objective ~= -10;
  * once ``10/M`` drops below the integrality tolerance, that point is declared
    integer-feasible, and rounding ``z`` to 0 makes the row violated by 10;
  * discopt's rust MILP route (``INT_TOL = 1e-5``) is defeated at ``M = 1e6``;
    HiGHS's own MIP integrality tolerance (1e-6) is defeated at ``M = 1e7``.

Measured on ``29ef5423`` (flag OFF), against a true optimum of -7.0:

    | route     | M    | status     | objective  |
    |-----------|------|------------|------------|
    | highs     | 1e7  | error      | None       |
    | highs     | 1e6  | optimal    | -7.0       |
    | rust      | 1e7  | feasible   | -4.75e-09  |
    | rust      | 1e6  | feasible   | -1.60e-08  |

No fixed *absolute* integrality tolerance resolves this class, because the
offending ``z = 10/M`` shrinks without bound as ``M`` grows. Big-M coefficient
tightening removes the mechanism instead of chasing it: ``u_rest = 10`` and
``rhs = 0`` give ``a_z' = rhs - u_rest = -10``, so the row becomes ``x <= 10z``,
the relaxation becomes exact, and every route returns -7.0 with nothing to
branch on.

Two things are pinned here, and the tests are written over a *parametrized*
class (several ``M`` and several upper bounds), never a single instance:

  1. the optimum is reached on **both** backends across the ``M`` range;
  2. the tightening actually reaches the **HiGHS pure-LP/MILP route**. That
     route sets ``presolve = False`` (``solver.py``, the ``_highs_takes_pure_lp_milp``
     skip) on the premise that "presolve only tightens bounds, and HiGHS
     presolves the model itself". Coefficient tightening rewrites *coefficients*,
     and a big-M coefficient is precisely what HiGHS cannot recover on its own,
     so it is gated on ``presolve_requested`` instead. Test 2 fails before that
     change and passes after.

The flag-OFF soundness control is kept deliberately: it must pass on BOTH arms.
A route that answers "-7.0" because the feature is on is worth nothing if the
route with it off is free to certify a wrong number.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from discopt import Model

SOLVER_SRC = Path(__file__).parents[1] / "discopt" / "solver.py"

# (M, x_upper_bound) — the optimum is always ``3 - ub``, never a hardcoded -7.
BIGM_CASES = [
    (1e6, 10.0),
    (1e7, 10.0),
    (1e8, 10.0),
    (1e7, 7.5),
    (1e9, 4.0),
]


def _bigm_model(M: float, ub: float) -> Model:
    m = Model("bigm")
    x = m.continuous("x", lb=0.0, ub=ub)
    z = m.binary("z")
    m.minimize(-x + 3.0 * z)
    m.subject_to(x <= M * z)
    return m


def _true_optimum(ub: float) -> float:
    # z=1 -> x=ub -> -ub+3 ; z=0 -> x=0 -> 0.  Optimal iff ub > 3.
    assert ub > 3.0, "case must have a nontrivial optimum"
    return 3.0 - ub


def test_marker_present_the_highs_skip_no_longer_suppresses_coef_tightening():
    """§8 load gate: pin the marker this module's fix is identified by.

    Asserted on the SOURCE, so a test run against a tree without the fix fails
    here rather than silently exercising the old gating.
    """
    src = SOLVER_SRC.read_text()
    assert "presolve_requested = presolve" in src, (
        "#1414 marker absent: coefficient tightening is still gated on the "
        "post-HiGHS-skip `presolve`, so it cannot reach the pure-LP/MILP route."
    )
    assert "if presolve_requested and not _deadline_exhausted():" in src, (
        "#1414 marker absent: the coefficient-tightening block is not gated on "
        "`presolve_requested`."
    )


@pytest.mark.parametrize("M,ub", BIGM_CASES)
@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_bigm_indicator_row_reaches_true_optimum_on_every_route(
    monkeypatch, backend: str, M: float, ub: float
):
    """Every route reaches the true optimum, for every M in the class."""
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)

    result = _bigm_model(M, ub).solve(time_limit=30)
    want = _true_optimum(ub)

    assert result.objective is not None, (
        f"{backend} route at M={M:g} produced no objective (status={result.status}); "
        f"the true optimum {want} is reachable by inspection (z=1, x={ub})."
    )
    assert result.objective == pytest.approx(want, abs=1e-6), (
        f"{backend} route at M={M:g}: got {result.objective}, want {want}."
    )


@pytest.mark.parametrize("M,ub", BIGM_CASES)
@pytest.mark.parametrize("flag", ["0", "1"])
def test_bigm_route_never_certifies_better_than_the_true_optimum(
    monkeypatch, flag: str, M: float, ub: float
):
    """Soundness control — must pass with the feature OFF *and* ON.

    A test that only passes when the feature is on cannot tell "the feature
    works" from "the feature hid the guard". The bad point in this class has
    objective ~= -ub (below the true optimum ``3-ub``), so a *certified* value
    below the true optimum is exactly the #1380 false certificate.
    """
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", flag)
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)

    result = _bigm_model(M, ub).solve(time_limit=30)
    want = _true_optimum(ub)

    if result.objective is None:
        return  # refusing is weak, not unsound — covered by the test above
    assert result.objective >= want - 1e-6, (
        f"super-optimal result at M={M:g} with flag={flag}: {result.objective} "
        f"is better than the true optimum {want}."
    )


def test_coefficient_tightening_reaches_the_highs_pure_milp_route(monkeypatch, caplog):
    """The class pin for this change: the HiGHS route must SEE the tightening.

    This model classifies as a pure MILP and is routed to HiGHS, which sets
    ``presolve = False``. Before the fix that also suppressed coefficient
    tightening, so the row reached HiGHS with its original ``M`` and the solve
    ended in ``error`` at M=1e7. Asserting on the emitted log makes "it fired"
    observable rather than inferred from the answer (CLAUDE.md §6).
    """
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")

    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        result = _bigm_model(1e7, 10.0).solve(time_limit=30)

    tightened = [r for r in caplog.records if "Coefficient tightening: strengthened" in r.message]
    assert tightened, (
        "coefficient tightening never ran on the HiGHS pure-LP/MILP route — the "
        "`presolve = False` skip is still suppressing it."
    )
    assert result.objective == pytest.approx(-7.0, abs=1e-6)


def test_tightening_is_inert_when_the_flag_is_off(monkeypatch, caplog):
    """Capability control: the change must do nothing with the flag off.

    Without this, the fix could be 'working' because it enabled a pass
    unconditionally — which would make the graduation panel compare a config
    against itself.
    """
    monkeypatch.delenv("DISCOPT_COEF_TIGHTEN", raising=False)
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")

    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        _bigm_model(1e7, 10.0).solve(time_limit=30)

    assert not [r for r in caplog.records if "Coefficient tightening: strengthened" in r.message], (
        "coefficient tightening ran with DISCOPT_COEF_TIGHTEN unset"
    )


# --------------------------------------------------------------------------
# The flag-OFF half of #1414: the default configuration must reach the true
# optimum on its own, and must never report a non-integral integer variable.
#
# Three defects were found and fixed for this (all on the default path, all
# measured on the class above with DISCOPT_COEF_TIGHTEN unset):
#
#   1. ``_root_dive`` accepted a coordinate within 1e-6 of an integer as
#      "integral", returned it UNROUNDED with the RELAXATION objective, and the
#      caller injected it as an incumbent. At M=1e7 that was a cutoff of
#      -9.999997 -- a value nothing attains -- which closed the gap at the root
#      and pruned the true optimum (nodes=1).
#   2. The tree FATHOMED a node whose point was integral only within 1e-5 and
#      promoted it. Sentinelling the bound routes it to branching instead, but
#      that alone changed nothing: most-fractional selection finds no candidate
#      inside the same tolerance, so the node was dropped with no children and
#      the search still ended at nodes=1. It takes an explicit branch hint on
#      the offending column, honored by `process_evaluated` for any value that
#      is not exactly integral.
#   3. ``_pounce_snap_incumbent`` trusted the engine's value for a column it had
#      itself PINNED. At M=1e12, with ``z`` pinned to [0,0], the re-solve
#      returned ``optimal`` at ``z = 1.336e-11``; ``1e12 * 1.336e-11 = 13.4`` is
#      enough row slack to carry ``x = 10``, so the "purified" point passed every
#      downstream row check and was reported ``optimal -9.99999999994`` against a
#      true optimum of -7.0. This was a FALSE CERTIFICATE, and it is the reason
#      the M range below runs past 1e9.
#
# Measured after the fixes, flag OFF, rust route: optimal -7.0 at
# M = 1e3, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12, 1e13, 1e14, 1e16, 1e18.
# --------------------------------------------------------------------------

# Deliberately reaching past 1e9: defect 3 above only appears once M is large
# enough that a pinned-bound rounding error buys more row slack than the box.
FLAG_OFF_CASES = [
    (1e6, 10.0),
    (1e7, 10.0),
    (1e9, 10.0),
    (1e12, 10.0),
    (1e14, 10.0),
    (1e12, 7.5),
]


def test_markers_present_for_the_three_flag_off_fixes():
    """§8 load gate for the flag-OFF fixes, asserted on the SOURCE.

    Without this, a run against a tree missing any of the three would exercise
    the old code and the end-to-end assertions below would be the only signal.
    """
    src = SOLVER_SRC.read_text()
    tree_src = (
        Path(__file__).parents[2] / "crates" / "discopt-core" / "src" / "bnb" / "tree_manager.rs"
    ).read_text()
    assert "_off = [j for j in int_idx if float(x[j]) != float(round(x[j]))]" in src, (
        "#1414 marker absent: the root dive still returns a tolerance-integral "
        "point unrounded, with the relaxation objective."
    )
    assert "_integral_claim_branch_col" in src, (
        "#1414 marker absent: the node-level unpromotable-claim guard."
    )
    assert "tree.set_branch_hints(" in src and "_unpromotable_hints" in src, (
        "#1414 marker absent: the branch hint on the offending column."
    )
    assert "x[idx] = snapped" in src, (
        "#1414 marker absent: `_pounce_snap_incumbent` still trusts the engine's "
        "value for a column it pinned."
    )
    assert "let actionable = if frac > 1e-5 && frac < 1.0 - 1e-5 {" in tree_src, (
        "#1414 marker absent: process_evaluated still discards an explicit branch "
        "hint whose value sits inside the integrality tolerance."
    )
    assert "if _dev == 0.0:" in src, (
        "#1414 marker absent: `_pounce_snap_incumbent`'s repair is no longer gated "
        "on the engine having missed a pin. Ungated, it recomputes the objective on "
        "EVERY snap, which perturbs the reported value in its last digits on the "
        "default path (measured: -6.9999999994785931 vs the engine's "
        "-6.9999999994786322) -- a bound-neutrality violation under CLAUDE.md §5."
    )


@pytest.mark.parametrize("M,ub", FLAG_OFF_CASES)
def test_rust_route_certifies_the_true_optimum_with_the_flag_off(monkeypatch, M, ub):
    """The DEFAULT configuration certifies the optimum across the M range."""
    monkeypatch.delenv("DISCOPT_COEF_TIGHTEN", raising=False)
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")

    result = _bigm_model(M, ub).solve(time_limit=30)
    want = _true_optimum(ub)

    assert result.objective == pytest.approx(want, rel=1e-6), (
        f"rust route at M={M:g} with the flag OFF: got {result.objective}, want {want}."
    )
    assert result.status == "optimal", (
        f"rust route at M={M:g} with the flag OFF: status={result.status}; the "
        f"optimum {want} is reachable by inspection (z=1, x={ub})."
    )


@pytest.mark.parametrize("M,ub", FLAG_OFF_CASES)
@pytest.mark.parametrize("backend", ["highs", "rust"])
@pytest.mark.parametrize("flag", ["0", "1"])
def test_reported_binary_is_exactly_integral(monkeypatch, flag, backend, M, ub):
    """A reported binary must be exactly 0 or 1 — on every route and both arms.

    This is the invariant defect 3 violated: ``z = 1.336e-11`` was reported as
    the answer's value for a *binary* variable, and it was that non-integrality
    which bought the row slack that made the false certificate pass its checks.
    A near-integral reported value is never acceptable, independent of whether
    the objective happens to be right.
    """
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", flag)
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)

    result = _bigm_model(M, ub).solve(time_limit=30)
    if result.objective is None:
        return  # refusing is weak, not unsound
    z = float(result.x["z"])
    assert z in (0.0, 1.0), (
        f"{backend} route at M={M:g} (flag={flag}) reported a BINARY variable at "
        f"{z!r} — not an integer, so the point is not an answer to the model."
    )
