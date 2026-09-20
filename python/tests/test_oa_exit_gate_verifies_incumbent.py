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

The fix is bound-neutral (CLAUDE.md §5 regime (a)). That is asserted here as a
DIFFERENTIAL -- the same instance solved with the gate and with a pass-through
stub, in one process -- not as a pinned float. An earlier revision of this file
did pin one machine's floats and asserted ``status == "optimal"``; CI reached a
different incumbent and reported ``feasible``, and the equalities failed while
the gate was working exactly as intended. A snapshot measures how far the search
got, which is hardware-dependent and is not the property under test.

The end-to-end cases share one module-scoped solve where they can, and assert
against the MINLPLib oracle (``best=0.0282906349``) rather than against this
machine, so they hold at any search depth.
"""

import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solvers.oa import _exit_verified_incumbent
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke

_NL = "python/tests/data/minlplib/portfol_roundlot.nl"

# MINLPLib reference (``minlplib.solu``) -- the ORACLE, not a snapshot of this
# machine. An earlier revision of this file pinned the objective to the exact
# float one laptop produced (0.028292461076062838) and asserted
# ``status == "optimal"``. Both describe how far the search happened to get, not
# whether the answer is sound: CI reached 0.0284247108 and reported "feasible",
# and the equality failed while nothing was wrong. The invariants below hold on
# any machine at any search depth, and they are STRICTLY STRONGER as correctness
# statements -- a snapshot cannot catch a false primal, whereas
# ``objective >= _BEST`` can.
_BEST = 0.0282906349
_BESTDUAL = 0.0282902203
_ATOL = 1e-6  # conftest's declared abs tolerance
_RTOL = 1e-4  # conftest's declared rel tolerance


def _flat(model, x_dict):
    return np.concatenate(
        [np.asarray(x_dict[v.name], dtype=np.float64).ravel() for v in model._variables]
    )


def _solve(**kw):
    m = from_nl(_NL)
    return m, m.solve(time_limit=10, solver="mip-nlp", mip_nlp_method="oa", **kw)


@pytest.fixture(scope="module")
def oa_run():
    """One real OA solve, shared -- the end-to-end cases assert on the same run."""
    return _solve()


def test_oa_returned_incumbent_passes_the_shipped_verifier(oa_run):
    """The regression: OA certified a point ``verify_point`` rejects."""
    m, r = oa_run

    assert r.x is not None, "no incumbent -- the test measured nothing"
    verdict = verify_point(m, _flat(m, r.x))
    assert verdict.ok, (
        "OA returned a point the shipped verifier rejects while reporting "
        f"status={r.status} gap_certified={r.gap_certified}: {verdict.reason}"
    )


def test_the_certificate_survives_the_gate(oa_run):
    """The soundness invariants CLAUDE.md §1 names, against the MINLPLib oracle.

    These are what the gate must not break, and unlike a pinned float they say
    something true on every machine: the dual bound never crosses the reference
    optimum, the incumbent never sits below it (that would BE a false primal),
    and the certificate is not inverted.
    """
    _m, r = oa_run

    assert r.objective is not None and r.bound is not None
    # No false primal: a point claiming to beat the known optimum is unsound.
    assert r.objective >= _BEST - max(_ATOL, _RTOL * abs(_BEST)), (
        f"incumbent {r.objective!r} is below the reference optimum {_BEST!r}"
    )
    # The dual bound must never cross the oracle.
    assert r.bound <= _BEST + max(_ATOL, _RTOL * abs(_BEST)), (
        f"dual bound {r.bound!r} exceeds the reference optimum {_BEST!r}"
    )
    # The certificate must not be inverted by the repair.
    assert r.bound <= r.objective + _ATOL
    # A certified run must actually be within the gap it claims.
    if r.gap_certified:
        assert r.status == "optimal"


def test_the_gate_is_bound_neutral(monkeypatch):
    """Regime (a), as a DIFFERENTIAL rather than a snapshot.

    "The repair moves the point, never the answer" is a claim about the gate, so
    it is tested by running the same instance with the gate replaced by a
    pass-through and comparing the two runs in one process. That comparison is
    machine-independent; the hardcoded float it replaces was not, and it is the
    assertion that failed on CI while the gate was working correctly.
    """
    import discopt.solvers.oa as oa

    real = oa._exit_verified_incumbent

    def _identity(model, x_flat, obj, obj_sign):
        return np.asarray(x_flat, dtype=np.float64), obj, None

    monkeypatch.setattr(oa, "_exit_verified_incumbent", _identity)
    _m0, ungated = _solve()

    monkeypatch.setattr(oa, "_exit_verified_incumbent", real)
    _m1, gated = _solve()

    assert ungated.objective is not None and gated.objective is not None
    # EXACTLY unchanged -- any drift, in either direction, means the gate is
    # doing something other than clearing round-off.
    assert gated.objective == ungated.objective, (
        f"gate moved the objective: {ungated.objective!r} -> {gated.objective!r}"
    )
    assert gated.bound == ungated.bound
    assert gated.status == ungated.status
    assert gated.gap_certified == ungated.gap_certified


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

    seen: dict = {}

    def _always_refuse(model, x_flat, obj, obj_sign):
        # Record what the gate was handed, so the assertions below compare the
        # result against THIS run rather than against a float from one machine.
        seen["obj"] = obj
        seen["x"] = np.asarray(x_flat, dtype=np.float64).copy()
        return np.asarray(x_flat, dtype=np.float64), obj, "row 99 violated by 1.000e-03"

    monkeypatch.setattr(oa, "_exit_verified_incumbent", _always_refuse)

    m, r = _solve()

    assert seen, "the gate was never called -- this test measured nothing"
    assert r.x is not None, "the forced-refusal run lost its incumbent entirely"
    # The point and its objective still reach the caller; the CERTIFICATE does not.
    assert r.objective == seen["obj"]
    np.testing.assert_array_equal(_flat(m, r.x)[: seen["x"].size], seen["x"])
    assert r.status == "feasible"
    assert r.gap_certified is False
    assert r.gap is None
    # The dual bound comes from the master relaxation, which never saw this
    # point, so a primal defect must not discard it.
    assert r.bound is not None, "a primal refusal discarded a valid dual bound"
    assert r.bound <= r.objective + _ATOL


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
