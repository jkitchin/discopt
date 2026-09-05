"""Regression tests for #517 candidate (A): in-house numerical-dual safe bound.

Root cause (``docs/dev/hda-no-bound-simplex-robustness-2026-07-16.md``): on the
hda-class ill-conditioned flowsheet relaxations the in-house simplex's phase 1
finds a feasible basis but phase 2 drifts / breaks down (``Numerical``), so the
node certifies no dual bound and the tree never fathoms — hda has *no* dual bound
at all.

Fix (flag ``DISCOPT_NODE_NUMERICAL_DUAL_BOUND`` /
``SolverTuning.node_numerical_dual_bound``; shipped default OFF under the
bound-changing regime, graduated to default ON with #362 —
``DISCOPT_NODE_NUMERICAL_DUAL_BOUND=0`` restores the legacy no-rescue behavior):
export the Optimal-style dual candidate ``y = B⁻ᵀc_B`` from the broken basis and
attach the in-repo Neumaier–Shcherbina safe lower bound it yields. The NS bound is
valid for ANY multiplier vector, so a drifted-basis dual only *loosens* it — never
lifts it above the optimum — and it is reported as a bound-only node (no fabricated
incumbent), so it never fathoms on its own. No external solver is used.
"""

import math
import os

import discopt.modeling as dm
import pytest

_NL_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

_FLAG = "DISCOPT_NODE_NUMERICAL_DUAL_BOUND"
_HDA_OPT = -5964.534084  # published MINLPLib global optimum


def _hda_path():
    p = os.path.join(_NL_DATA, "hda.nl")
    if not os.path.exists(p):
        pytest.skip("hda.nl not vendored")
    return p


@pytest.mark.slow
def test_hda_gets_first_finite_dual_bound(monkeypatch):
    """With the flag ON, hda reports a *finite* dual bound (its first) that is a
    sound lower bound (never above the published optimum)."""
    monkeypatch.setenv(_FLAG, "1")
    r = dm.from_nl(_hda_path()).solve(time_limit=25)
    assert r.bound is not None, "hda should get its first finite dual bound with the flag ON"
    assert math.isfinite(r.bound), f"bound must be finite, got {r.bound}"
    # Soundness: a valid dual (lower) bound never crosses the true optimum.
    assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
    # Bound-only: the floor must not fabricate an incumbent or a false optimality.
    assert r.status != "optimal", "a loose dual floor must not claim optimality"


@pytest.mark.slow
def test_hda_flag_disabled_is_sound_but_no_longer_discriminating(monkeypatch):
    """The ``=0`` opt-out stays sound -- but hda no longer discriminates this flag.

    #1039. This test used to assert ``r.bound is None``: with the candidate-A
    floor disabled, hda was supposed to fall back to the legacy no-rescue
    baseline of no dual bound at all. That baseline is unreachable now. Two
    mechanisms moved underneath it:

      * ``DISCOPT_RELAX_ROW_FILTER`` graduated to default-ON (#671, 2026-07-18)
        and supplies hda's tight bound regardless of this flag, so the test was
        never varying only the flag it names -- hence its reported
        "flag disabled must be the no-bound baseline, got -64473.44".
      * Even with the row filter pinned OFF, something else now produces a finite
        bound in BOTH arms. Measured, 3 reps interleaved, row filter pinned OFF
        (``scratchpad/issue1039/probe_517.py``):

            DISCOPT_NODE_NUMERICAL_DUAL_BOUND=0 -> -141697.43348991615 (sd 0, 3/3)
            DISCOPT_NODE_NUMERICAL_DUAL_BOUND=1 -> -141697.43348991545 (sd 0, 3/3)

        The arms agree to 12 significant digits: the flag is still consulted, but
        its floor is no longer what hda's reported bound comes from.

    This follows the C-42/C-43/C-44 precedent the repo has already established
    for exactly this shape -- when the condition a runtime mechanism recovers
    from stops occurring, the honest assertion is inertness WITH the measurement
    recorded, not a demand that it fire (which would be asserting a premise known
    to be false). The flag's ON path keeps its own coverage in
    ``test_hda_gets_first_finite_dual_bound`` above, so the mechanism is not left
    untested; what is lost is hda's ability to discriminate ON from OFF.

    KNOWN GAP, deliberately not papered over: re-pointing this flag at an
    instance where its floor is still load-bearing needs an invocation counter
    for the #517 path (the #1039 ``row_filter/invocations`` counter is what made
    the equivalent search tractable for the row filter). That is the one piece of
    #1039 left open rather than closed.

    #1165 -- the differential assertion is GONE, not weakened. This test used to
    close with ``abs(on.bound - off.bound) <= 1e-6 * max(1, abs(off.bound))``
    between two 25 s solves. **A wall-clock-truncated bound is not a comparable
    quantity**: each arm reports wherever the search happened to be when the
    clock ran out, so that assertion measured machine load, not the flag. Three
    measurements, all in ``scratchpad/issue1165/``:

      * The number the arms agree on is a property of the BOX. On a quiet
        machine both arms are bit-stable at ``-100217.95933512867`` (6 solves,
        3 reps interleaved, sd 0) -- not the ``-141697.43348991615`` recorded
        above from a different machine, in the same arm at the same budget.
      * The failure that opened #1165 had both arms at ``status='time_limit'``,
        ``nodes=3``, ``objective=None`` and reported OFF ``-141697.43348991615``
        vs ON ``-13992288065.862448`` -- five orders of magnitude, from load.
      * Making the comparison VALID is not affordable on hda. The #1116 recipe
        (``deterministic=True`` + a node budget, so both arms terminate on WORK)
        gives the OFF arm ``-141697.43348991615`` in 305 s at ``max_nodes=1``
        -- exactly the value recorded above, i.e. the work-terminated bound is
        the reproducible one -- while the ON arm did not return in 2095 s at the
        same budget. hda is in the #1039 ``_NOT_AFFORDABLY_COMPARABLE`` class.

    So the ON/OFF differential moved to where it can be made valid:
    ``test_inert_on_cleanly_certifying_instances`` below now runs both arms to a
    deterministic node budget and compares them bit-for-bit, which is the same
    claim tested on instances where the comparison has a well-defined subject.
    What is left here is exactly what survives truncation and is asserted
    unconditionally: a sound bound is sound wherever the search was cut off, and
    the OFF arm's bound being finite is the whole content of "the legacy
    no-bound baseline is unreachable".
    """
    # Pin the later-graduated row filter OFF so this flag is the only variable.
    monkeypatch.setenv("DISCOPT_RELAX_ROW_FILTER", "0")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(_hda_path()).solve(time_limit=25)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(_hda_path()).solve(time_limit=25)

    # Soundness is the non-negotiable part and is asserted unconditionally.
    for label, r in (("OFF", off), ("ON", on)):
        assert r.bound is not None and math.isfinite(r.bound), (
            f"{label}: no finite bound ({r.bound})"
        )
        assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND ({label}): {r.bound:.6g} > {_HDA_OPT}"
        assert r.status != "optimal", f"{label}: a loose dual floor must not claim optimality"

    # NO differential assertion here, deliberately -- see the #1165 section of
    # this test's docstring. Everything above is truncation-proof; a comparison
    # between two ``time_limit`` runs is not, and must not be added back.


#: Both arms terminate on WORK, not on the wall clock -- the #1116 recipe already
#: used by ``test_relax_row_filter.py``. A solve cut off by ``time_limit`` stops at
#: a machine-speed-dependent point, so two such solves are two different amounts
#: of search and comparing them bit-for-bit measures load (#1165). Measured on the
#: two instances below, 2 reps x 2 arms each
#: (``scratchpad/issue1165/probe_affordable_comparisons.py``): identical bounds in
#: every run under BOTH budgets, and the work budget is the cheaper one
#: (alan 2.4 s -> 0.1 s per arm pair; ex1221 unchanged at ~4 s).
_WORK_TERMINATED_KW = {"deterministic": True, "max_nodes": 25, "time_limit": 120}


@pytest.mark.slow
@pytest.mark.parametrize("name", ["alan", "ex1221"])
def test_inert_on_cleanly_certifying_instances(name, monkeypatch):
    """Instances whose node LPs solve cleanly (no numerical breakdown) are
    byte-identical with the flag ON: the floor fires only on a failed node LP, so
    a well-conditioned certifying instance never triggers it.

    This carries the ON/OFF differential that ``hda`` can no longer host (#1165):
    hda cannot terminate on work affordably, these two can, so the comparison is
    made here where it has a well-defined subject. Both arms run under
    ``_WORK_TERMINATED_KW`` and the precondition is ASSERTED rather than assumed
    (CLAUDE.md §6) -- an arm that ends on ``time_limit`` fails loudly instead of
    silently degrading into a comparison of two differently-truncated searches.
    """
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(path).solve(**_WORK_TERMINATED_KW)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(path).solve(**_WORK_TERMINATED_KW)

    for arm, r in (("OFF", off), ("ON", on)):
        assert r.status != "time_limit", (
            f"{name}: the {arm} arm terminated on time_limit, so this ON/OFF "
            "comparison is not valid -- raise time_limit or lower max_nodes in "
            "_WORK_TERMINATED_KW rather than comparing two truncated searches"
        )

    assert off.status == on.status, f"{name}: status changed {off.status} -> {on.status}"
    assert off.objective == on.objective, f"{name}: objective drifted with the flag"
    assert off.bound == on.bound, f"{name}: bound drifted with the flag ({off.bound} -> {on.bound})"
    assert off.node_count == on.node_count, f"{name}: node count drifted with the flag"


def test_flag_defaults_on(monkeypatch):
    """The tuning flag is default-ON (graduated with #362; ``=0`` restores the
    legacy no-rescue behavior).

    Check the *code* default in the absence of the env override (a CI shell that
    exports the flag must not distort the default), and the escape hatch."""
    monkeypatch.delenv(_FLAG, raising=False)
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().node_numerical_dual_bound is True
    monkeypatch.setenv(_FLAG, "0")
    assert SolverTuning().node_numerical_dual_bound is False
