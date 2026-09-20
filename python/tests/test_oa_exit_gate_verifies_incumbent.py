"""The OA route must not certify a point its own verifier rejects.

``solve_oa`` built its ``SolveResult`` with nothing checking the vector it
returned -- the hole #952 closed on the matrix paths and #954 closed on NLP-BB
("the last of the five solve exit paths with no verification of the point it
returns"; OA was a sixth). Measured on ``portfol_roundlot`` before the fix::

    STATUS       optimal
    OBJECTIVE    0.028292461076062838
    BOUND        0.028290211674170968
    CERTIFIED    True
    VERIFY_POINT ok=False reason='row 5 violated by 5.632e-06 (allowed 1.000e-06)'

``verify_point`` is this repository's own shipped arbiter and it was confirmed to
have RUN on that solve (zero "false-primal guard is disabled" warnings), so this
is a solver certifying a point its own verifier refuses -- a soundness defect
regardless of the magnitude.

The cause is a scale mismatch, not a bug in any one instance: OA admits a
fixed-NLP point through ``_is_primal_feasible`` at ``1e-4`` (100x the declared
``abs=1e-6``) and the NLP backend's convergence test is on *its* error, not the
model's rows. POUNCE reported ``nlp_err=7.22e-07`` while leaving four columns
``7.22e-11`` above ``lb=0``; a row with a ``78000`` coefficient turns that into
``5.6e-06``.

The fix is bound-neutral here (CLAUDE.md §5 regime (a)) and these tests pin that:
after the gate the status, objective and bound are bit-identical and only the
verification flips::

    STATUS optimal  OBJECTIVE 0.028292461076062838  BOUND 0.028290211674170968
    CERTIFIED True  VERIFY_POINT ok=True

The end-to-end case runs in ~2.6 s and its answer is stable across
``time_limit`` 5/10/20, so the budget below is not load-bearing.
"""

import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solvers.oa import _exit_verified_incumbent
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke

_NL = "python/tests/data/minlplib/portfol_roundlot.nl"

# Measured before AND after the gate -- identical, which is the point.
_OPT = 0.028292461076062838
_BOUND = 0.028290211674170968


def _flat(model, x_dict):
    return np.concatenate(
        [np.asarray(x_dict[v.name], dtype=np.float64).ravel() for v in model._variables]
    )


def test_oa_returned_incumbent_passes_the_shipped_verifier():
    """The regression: OA certified a point ``verify_point`` rejects."""
    m = from_nl(_NL)
    r = m.solve(time_limit=10, solver="mip-nlp", mip_nlp_method="oa")

    assert r.x is not None, "no incumbent -- the test measured nothing"
    verdict = verify_point(m, _flat(m, r.x))
    assert verdict.ok, (
        "OA returned a point the shipped verifier rejects while reporting "
        f"status={r.status} gap_certified={r.gap_certified}: {verdict.reason}"
    )


def test_the_gate_is_bound_neutral_on_this_instance():
    """Regime (a): the repair moves the point, never the answer.

    The four snapped columns move ``7.22e-11`` each and the objective is
    bit-identical, so a drift here -- in EITHER direction -- means the gate is
    doing something other than clearing round-off.
    """
    m = from_nl(_NL)
    r = m.solve(time_limit=10, solver="mip-nlp", mip_nlp_method="oa")

    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(_OPT, rel=1e-9)
    assert r.bound == pytest.approx(_BOUND, rel=1e-9)
    # The certificate must not be inverted by the repair.
    assert r.bound <= r.objective + 1e-9


def _tiny_model():
    """A round-lot linking row: ``c*x - n == 0`` with a large ``c``.

    The shape the defect takes, in three variables, so the gate's branches can be
    exercised without depending on what any NLP backend happens to return.
    """
    import discopt.modeling as dm

    m = dm.Model("roundlot")
    x = m.continuous("x", lb=0.0, ub=1.0)
    n = m.integer("n", lb=0, ub=10)
    m.subject_to(78000.0 * x - n == 0.0)
    m.minimize(x * x + n)
    return m, x, n


def test_gate_passes_a_clean_point_through_unchanged():
    """A verifiable point is returned untouched, objective included."""
    m, _, _ = _tiny_model()
    x = np.array([0.0, 0.0], dtype=np.float64)
    assert verify_point(m, x).ok, "probe setup is wrong: the control point is infeasible"

    out, obj, refusal = _exit_verified_incumbent(m, x, 0.0, 1.0)

    assert refusal is None
    assert obj == 0.0
    np.testing.assert_array_equal(out, x)


def test_gate_repairs_a_near_bound_point_and_reports_the_repaired_objective():
    """The measured failure shape: a column left just off the bound its row forces."""
    m, _, _ = _tiny_model()
    off = np.array([7.220331e-11, 0.0], dtype=np.float64)
    # Precondition: this is exactly the defect -- feasible-looking, verifier-rejected.
    assert not verify_point(m, off).ok, "the probe's bad point is not actually rejected"

    out, obj, refusal = _exit_verified_incumbent(m, off, float(off[0] ** 2), 1.0)

    assert refusal is None, f"the gate failed to repair a near-bound point: {refusal}"
    assert verify_point(m, out).ok
    assert out[0] == 0.0
    assert obj == pytest.approx(0.0, abs=1e-12)


def test_gate_refuses_to_certify_a_point_it_cannot_repair():
    """Unrepairable: a gross violation is reported, never silently accepted.

    This branch is what stops the gate from being a repair tool that quietly
    launders anything handed to it. ``x=0.5`` is nowhere near a bound, so no snap
    applies and the point stays rejected.
    """
    m, _, _ = _tiny_model()
    bad = np.array([0.5, 0.0], dtype=np.float64)
    assert not verify_point(m, bad).ok

    out, obj, refusal = _exit_verified_incumbent(m, bad, 0.25, 1.0)

    assert refusal is not None, "an unrepairable point was accepted by the gate"
    assert "violated" in refusal or "bound" in refusal.lower()
    # The point and its objective are handed back unchanged; it is the
    # CERTIFICATE that is withheld, not the information.
    np.testing.assert_array_equal(out, bad)
    assert obj == 0.25


def test_a_refusal_downgrades_the_solve_result(monkeypatch):
    """The refusal must reach the CALLER, not just the helper's return value.

    The corpus never exercises this branch (measured: 1 rejection in 62 OA
    incumbents over ``data/minlplib``, and it repairs), so without this test the
    wiring from "the gate refused" to "the SolveResult says so" would ship
    unexercised -- the kind of path that is discovered to be inert years later.
    Forcing the refusal is the only way to reach it deterministically.
    """
    import discopt.solvers.oa as oa

    def _always_refuse(model, x_flat, obj, obj_sign):
        return np.asarray(x_flat, dtype=np.float64), obj, "row 99 violated by 1.000e-03"

    monkeypatch.setattr(oa, "_exit_verified_incumbent", _always_refuse)

    m = from_nl(_NL)
    r = m.solve(time_limit=10, solver="mip-nlp", mip_nlp_method="oa")

    assert r.x is not None, "the forced-refusal run lost its incumbent entirely"
    # The point and its objective still reach the caller; the CERTIFICATE does not.
    assert r.objective == pytest.approx(_OPT, rel=1e-9)
    assert r.status == "feasible"
    assert r.gap_certified is False
    assert r.gap is None
    # The dual bound comes from the master relaxation, which never saw this
    # point, so a primal defect must not discard it.
    assert r.bound == pytest.approx(_BOUND, rel=1e-9)


def test_maximize_objective_units_survive_the_repair():
    """The repaired objective is converted back to OA's internal minimize units.

    ``verify_point`` reports in MODEL units (it un-negates a MAXIMIZE model) while
    ``incumbent_obj`` is internal; a missing conversion would flip the sign of
    every repaired maximize result, which no feasibility assertion would catch.
    """
    import discopt.modeling as dm

    m = dm.Model("roundlot_max")
    x = m.continuous("x", lb=0.0, ub=1.0)
    n = m.integer("n", lb=0, ub=10)
    m.subject_to(78000.0 * x - n == 0.0)
    m.maximize(x + n)

    off = np.array([7.220331e-11, 0.0], dtype=np.float64)
    assert not verify_point(m, off).ok

    # obj_sign=-1 is what solve_oa passes for a MAXIMIZE model: the internal
    # objective is the negated model objective.
    out, obj, refusal = _exit_verified_incumbent(m, off, -float(off[0]), -1.0)

    assert refusal is None
    assert verify_point(m, out).ok
    # Model objective at the repaired point is 0.0, so the internal one is too --
    # and critically NOT +0.0 via a double negation of a nonzero value.
    assert obj == pytest.approx(0.0, abs=1e-12)
