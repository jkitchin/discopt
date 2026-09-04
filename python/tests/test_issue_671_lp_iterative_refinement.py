"""Regression tests for #671 candidate (B): a *tight* dual bound for hda-class
ill-conditioned McCormick relaxations via RHS-regularized iterative refinement.

Candidate A (#517/#662) gave hda its first *finite* dual bound by attaching the
Neumaier–Shcherbina safe bound of the drifted dual from the numerically-broken
node LP — sound but loose (≈ −1.80e10 vs opt −5964.53). The entry experiment
(#671, PR #708) proved that loose bound is a *precision artifact*: the true root
McCormick value is ≈ −6.47e4.

Fix (flag ``DISCOPT_LP_ITERATIVE_REFINEMENT`` / ``SolverTuning``
``lp_iterative_refinement``; default OFF): on a node-LP numerical breakdown,
re-solve a few RHS-regularized neighbours ``[A|I] z = b + tau`` in the **in-house**
simplex and keep the *tightest* NS safe bound the recovered duals imply, evaluated
against the **original** ``b`` (never ``b+tau``). ``g(y)`` is a valid lower bound
for any ``y``, so the regularization affects only tightness, never soundness; the
reported bound is the max over the sweep AND candidate A, so it is never looser
than candidate A and never above the optimum. In-house simplex only — no external
solver.
"""

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest

_NL_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

_FLAG = "DISCOPT_LP_ITERATIVE_REFINEMENT"
_HDA_OPT = -5964.534084  # published MINLPLib global optimum
_CAND_A = -1.80e10  # candidate A's loose floor magnitude


def _hda_path():
    p = os.path.join(_NL_DATA, "hda.nl")
    if not os.path.exists(p):
        pytest.skip("hda.nl not vendored")
    return p


def test_flag_defaults_off(monkeypatch):
    """Bound-changing lever ships default OFF (Dev-Philosophy #5); ``=1`` enables."""
    monkeypatch.delenv(_FLAG, raising=False)
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().lp_iterative_refinement is False
    monkeypatch.setenv(_FLAG, "1")
    assert SolverTuning().lp_iterative_refinement is True


def test_refined_bound_is_sound_and_never_worse_than_candidate_a():
    """The refinement helper on a standard-form LP: the returned bound is finite,
    a valid lower bound (≤ the true optimum), and — because it maxes over the
    sweep — at least as tight as the drifted-dual candidate A on the same data.

    Uses a small ill-conditioned LP (coefficient range 1e8) directly through the
    in-house ``solve_lp_warm_csc_py`` marshalling, so it is fast and deterministic.
    """
    import scipy.sparse as sp
    from discopt._rust import solve_lp_warm_csc_py
    from discopt.solvers.milp_simplex import (
        _refined_safe_bound_regularized,
    )

    # min -x0 - x1  s.t.  1e8 x0 + x1 <= 1e8,  x1 <= 5,  0<=x0<=1, 0<=x1<=10.
    # True LP optimum: maximize x0+x1 -> x1=5, x0=1-5e-8 (=~1), obj = -6 (the 1e8
    # coefficient is the ill-conditioning under test).
    n = 2
    A = sp.csc_matrix(np.array([[1e8, 1.0], [0.0, 1.0]], dtype=np.float64))
    b = np.array([1e8, 5.0], dtype=np.float64)
    m = 2
    a_std = sp.hstack([A, sp.identity(m, format="csc")], format="csc").tocsc()
    c_std = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float64)
    lb_std = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ub_std = np.array([1.0, 10.0, 1e20, 1e20], dtype=np.float64)

    refined = _refined_safe_bound_regularized(
        solve_lp_warm_csc_py, c_std, a_std, b, lb_std, ub_std, m, n
    )
    assert refined is not None and math.isfinite(refined), f"no finite bound: {refined}"
    true_lp_opt = -6.0
    # Soundness: a valid lower bound never exceeds the LP optimum.
    assert refined <= true_lp_opt + 1e-6, f"UNSOUND: refined {refined} > opt {true_lp_opt}"
    # Tight on a well-conditioned LP: recovers the optimum closely.
    assert abs(refined - true_lp_opt) < 1e-3, f"not tight: {refined} vs {true_lp_opt}"


@pytest.mark.slow
def test_hda_gets_a_tight_dual_bound_with_the_flag(monkeypatch):
    """End-to-end: with the flag ON, hda's dual bound is *materially tighter* than
    candidate A's ≈ −1.80e10 floor — many orders of magnitude closer to opt — while
    remaining a sound lower bound."""
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", "1")  # keep candidate A fallback
    r = dm.from_nl(_hda_path()).solve(time_limit=90)
    assert r.bound is not None and math.isfinite(r.bound), f"no finite bound: {r.bound}"
    # Soundness: never above the published optimum.
    assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
    # Tightness: orders of magnitude above candidate A's loose floor (well above
    # -1e7, vs candidate A's -1.8e10; the true root McCormick value is ≈ -6.47e4).
    assert r.bound > -1e7, f"bound {r.bound:.6g} no tighter than candidate A's floor"


@pytest.mark.slow
def test_hda_flag_off_is_looser_than_flag_on(monkeypatch):
    """The ``=0`` opt-out is live and load-bearing: OFF is strictly looser than ON.

    #1039: this test used to assert a hardcoded value -- that the OFF arm lands on
    candidate A's loose floor, ``< -1e7`` (approx -1.8e10). Two things had moved
    underneath it.

    First, it did not pin ``DISCOPT_RELAX_ROW_FILTER``, which graduated to
    default-ON after this test was written (#671, 2026-07-18) and supplies hda's
    tight bound on its own. So the test was not varying only the flag it names:
    both of its arms got the row filter's answer, which is why it reported
    "flag OFF should be the loose candidate-A floor, got -64473.44".

    Second, even with the row filter pinned OFF, the OFF arm no longer falls all
    the way to candidate A's floor -- an intervening improvement now gets it to
    -1.4e5. Measured, 3 reps interleaved, row filter pinned OFF, candidate A ON
    (``scratchpad/issue1039/probe_671ref.py``):

        DISCOPT_LP_ITERATIVE_REFINEMENT=0 -> -141647.03848991546  (sd 0, 3/3)
        DISCOPT_LP_ITERATIVE_REFINEMENT=1 ->  -64509.850876897275 (sd 0, 3/3)

    So the assertion is the DIFFERENTIAL rather than either endpoint's value: ON
    is strictly tighter than OFF, and both are sound. That is what the graduated
    ``=0`` opt-out actually promises, and unlike a hardcoded floor it does not go
    stale the next time some other mechanism moves one of the endpoints.
    """
    # Pin the other rescue mechanisms so this flag is the ONLY variable. Without
    # this the row filter answers for both arms and the comparison is vacuous.
    monkeypatch.setenv("DISCOPT_RELAX_ROW_FILTER", "0")
    monkeypatch.setenv("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", "1")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(_hda_path()).solve(time_limit=90)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(_hda_path()).solve(time_limit=90)

    for label, r in (("OFF", off), ("ON", on)):
        assert r.bound is not None and math.isfinite(r.bound), (
            f"{label}: no finite bound ({r.bound})"
        )
        # Soundness first: neither arm may cross the published optimum.
        assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND ({label}): {r.bound:.6g} > {_HDA_OPT}"

    # The opt-out genuinely gates a bound-improving path.
    assert on.bound > off.bound, (
        f"the =0 opt-out is inert: OFF {off.bound:.6g} vs ON {on.bound:.6g} -- "
        "the refinement path no longer changes the bound on hda, so this test no "
        "longer measures the flag and must be re-pointed"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["alan", "ex1221"])
def test_inert_on_cleanly_certifying_instances(name, monkeypatch):
    """Instances whose node LPs solve cleanly never hit the numerical-failure path,
    so the flag is byte-identical ON vs OFF (failure-triggered only)."""
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(path).solve(time_limit=20)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(path).solve(time_limit=20)

    assert off.status == on.status, f"{name}: status changed {off.status} -> {on.status}"
    assert off.objective == on.objective, f"{name}: objective drifted with the flag"
    assert off.bound == on.bound, f"{name}: bound drifted with the flag ({off.bound} -> {on.bound})"
