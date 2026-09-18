"""#1337: an unbounded root LP defeated the root cross-check on an INTEGRAL conflict.

``2x == 1`` with ``x`` integer has no solution, but the infeasibility is purely
INTEGRAL -- the root LP relaxation is perfectly happy at ``x = 0.5``. Add an
objective that is unbounded on the relaxation (``min -y`` over ``y in [0, 1e20]``)
and the NS-safe root cross-check comes back ``unbounded``, which the route read as
"settled nothing" and used to decertify HiGHS's ``kInfeasible`` into ``error``.

But the only question that branch asks of a ``kInfeasible`` claim is *is the
relaxation feasible?*, and an unbounded objective says nothing about that. The
re-check drops the objective -- same LP, same box, zero ``c`` -- and:

* a Farkas-infeasible relaxation proves the MILP empty (stronger than needed);
* a feasible relaxation leaves the ``kInfeasible`` claim standing on exactly the
  footing the ordinary ``lp.status in ("optimal", "feasible")`` path already gives
  it, which is the pre-existing accepted rule of this route, not a new one;
* anything else still decertifies, unchanged.

``main`` reports ``infeasible`` here too, but by trusting the tree label with no
cross-check at all -- which is what #1295/#1320 exist to stop. This gets the same
answer with the check actually run.
"""

import discopt.modeling as dm
import numpy as np
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import StdForm, solve_milp_std

# No marker: fast regression fences on a certification defect, like #1320's.


def _integral_conflict_unbounded_objective() -> dm.Model:
    """The issue's model, verbatim."""
    m = dm.Model("i1337")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 1)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    return m


def test_the_issues_repro_is_certified_infeasible():
    """Pre-fix: ``error``, route label "unverified ... kInfeasible"."""
    r = _integral_conflict_unbounded_objective().solve(time_limit=60)
    assert r.status == "infeasible", f"got {r.status!r}"
    assert r.gap_certified
    # The §1 line is kept alongside: never a wrong certified status.
    assert r.status not in ("optimal", "unbounded")


def test_a_relaxation_infeasible_sibling_gets_a_farkas_proof():
    """When the conflict is NOT integral the objective-free re-check finds the
    relaxation empty, which is a stronger proof than the branch needs."""
    m = dm.Model("i1337_lp")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(x >= 3)
    m.subject_to(x <= 1)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status == "infeasible"
    assert r.gap_certified


def test_an_integrally_feasible_model_is_never_called_infeasible():
    """The direction that would be a false certificate: the same shape with a
    conflict that DOES have an integer solution (``2x == 2``)."""
    m = dm.Model("i1337_feas")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 2)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status != "infeasible", "certified 'infeasible' on a model with x = 1"
    assert r.status != "optimal", "the objective really is unbounded here"


def test_a_bounded_objective_sibling_is_unchanged():
    """Regression fence: with a bounded objective the original root check concludes
    and the re-check never runs, exactly as before."""
    m = dm.Model("i1337_bounded")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 1)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status == "infeasible"
    assert r.gap_certified


# ── unit: the re-check fires where it should and nowhere else ────────────


def _integral_conflict_sf(c, xu) -> StdForm:
    """``2x == 1`` (x integer) plus a free continuous column carrying ``c``."""
    return StdForm.from_arrays(
        c=np.asarray(c, dtype=np.float64),
        A=np.array([[2.0, 0.0]]),
        b=np.array([1.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.asarray(xu, dtype=np.float64),
        int_idx=np.array([0]),
    )


def test_recheck_runs_on_an_unbounded_root_and_settles_it():
    sf = _integral_conflict_sf([0.0, -1.0], [5.0, 1e20])
    out = solve_milp_std(sf, time_limit=60.0, gap_tolerance=1e-4, max_nodes=10_000)
    assert out.stats.get("milp/root_check_ran") == 1.0
    assert out.stats.get("milp/root_check_feasibility_only") == 1.0
    assert out.stats.get("milp/root_check_inconclusive") is None
    assert out.status == "infeasible"
    assert out.gap_certified
    assert out.labels.get("milp/infeasible_provenance") == "highs-root-feasible"


def test_recheck_does_not_run_when_the_root_check_already_concluded():
    """A bounded objective settles at the root, so the extra solve is never paid."""
    sf = _integral_conflict_sf([0.0, -1.0], [5.0, 10.0])
    out = solve_milp_std(sf, time_limit=60.0, gap_tolerance=1e-4, max_nodes=10_000)
    assert out.status == "infeasible"
    assert out.gap_certified
    assert out.stats.get("milp/root_check_feasibility_only") is None


def test_an_inconclusive_root_check_on_a_kOPTIMAL_result_still_decertifies(monkeypatch):
    """The re-check is scoped to a ``kInfeasible`` claim. A kOptimal result whose
    root check settles nothing is still downgraded -- the #1320 rule is untouched.
    """
    calls = []
    real = lp_milp_highs.solve_lp_std

    def _inconclusive(sf, **kw):
        calls.append(sf)
        from discopt.solvers.lp_milp_highs import HighsOutcome

        return HighsOutcome("error", message="kUnknown", highs_status="kUnknown")

    monkeypatch.setattr(lp_milp_highs, "solve_lp_std", _inconclusive)
    sf = StdForm.from_arrays(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )
    out = solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert calls, "the root cross-check never ran, so this asserts nothing"
    assert real is not None
    assert out.stats.get("milp/root_check_inconclusive") == 1.0
    assert not out.gap_certified
    assert out.status == "feasible"


def test_a_still_inconclusive_recheck_decertifies(monkeypatch):
    """If the objective-free re-check ALSO settles nothing, the claim is decertified
    exactly as before -- the re-check adds an answer, it never assumes one."""
    from discopt.solvers.lp_milp_highs import HighsOutcome

    def _always_inconclusive(sf, **kw):
        return HighsOutcome("error", message="kUnknown", highs_status="kUnknown")

    monkeypatch.setattr(lp_milp_highs, "solve_lp_std", _always_inconclusive)
    sf = _integral_conflict_sf([0.0, -1.0], [5.0, 1e20])
    out = solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_feasibility_only") == 1.0
    assert out.stats.get("milp/root_check_inconclusive") == 1.0
    assert not out.gap_certified
    assert out.status == "error"


def test_the_recheck_asks_a_zero_objective_question(monkeypatch):
    """The re-solve must differ from the first one in the objective ALONE: same
    rows, same box, same integrality, ``c = 0`` and no constant."""
    seen = []
    real = lp_milp_highs.solve_lp_std

    def _record(sf, **kw):
        seen.append(sf)
        return real(sf, **kw)

    monkeypatch.setattr(lp_milp_highs, "solve_lp_std", _record)
    sf = _integral_conflict_sf([0.0, -1.0], [5.0, 1e20])
    solve_milp_std(sf, time_limit=60.0, gap_tolerance=1e-4, max_nodes=10_000)

    assert len(seen) == 2, f"expected the root solve then the re-check, got {len(seen)}"
    first, second = seen
    assert np.any(first.c != 0.0), "the first solve must carry the real objective"
    assert np.all(second.c == 0.0), "the re-check must drop the objective"
    assert second.obj_const == 0.0
    assert np.array_equal(second.b, first.b)
    assert np.array_equal(second.xl, first.xl)
    assert np.array_equal(second.xu, first.xu)
    assert np.array_equal(second.A.toarray(), first.A.toarray())
    assert np.array_equal(second.int_idx, first.int_idx)
