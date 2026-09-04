"""Regression tests for the #671 float64-intractable-row filter.

hda-class relaxations emit a small set of rows whose coefficients span more
orders than float64 can resolve at the LP feasibility tolerance (hda: 130 rows,
raw spread 2.837e26). Those rows made every float64 LP engine false-fail while
contributing zero root tightness (measured:
``docs/dev/hda-certification-rowfilter-entry-2026-07-18.md``).

Fix (flag ``DISCOPT_RELAX_ROW_FILTER`` / ``SolverTuning.relax_row_filter``,
default OFF): **failure-triggered** — only when a node LP breaks down without a
certified verdict (``numerical``, or a spurious ``infeasible`` with no Farkas
proof) does ``mccormick_lp._solve_at_node_impl`` drop such rows and re-solve
once. Sound by construction — removing relaxation rows yields a superset
feasible region (a valid, weaker outer approximation), so the bound can only
loosen, never falsify. Firing only on failure keeps every already-solving node
byte-identical (its LP is optimal/Farkas-infeasible, so the filter never runs).
"""

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

_NL_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
_FLAG = "DISCOPT_RELAX_ROW_FILTER"
_HDA_OPT = -5964.534084  # published MINLPLib global optimum


def test_flag_defaults_on(monkeypatch):
    """Graduated default ON (#671, 2026-07-18); ``=0`` opts out."""
    monkeypatch.delenv(_FLAG, raising=False)
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().relax_row_filter is True
    monkeypatch.setenv(_FLAG, "0")
    assert SolverTuning().relax_row_filter is False
    monkeypatch.setenv(_FLAG, "1")
    assert SolverTuning().relax_row_filter is True


class _FakeMilp:
    def __init__(self, a_ub, b_ub):
        self._A_ub = a_ub
        self._b_ub = b_ub


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_filter_drops_wide_rows_and_keeps_normal_ones(sparse):
    """The helper drops exactly the float64-intractable rows (ratio > 1e6 or a
    coefficient outside [1e-8, 1e8]), preserves normal and empty rows, and keeps
    the container kind."""
    from discopt._relax.milp_relaxation import _filter_unresolvable_rows

    a = np.array(
        [
            [1.0, 2.0, 0.0],  # normal — keep
            [6.3e10, -1.0, 0.0],  # |a| > 1e8 (hda Arrhenius row) — drop
            [1.0, -6.2e-10, 0.0],  # |a| < 1e-8 — drop
            [1e5, 1e-3, 0.0],  # ratio 1e8 > 1e6 — drop
            [0.0, 0.0, 0.0],  # empty — keep (may encode infeasibility)
            [-3.0, 0.5, 7.0],  # normal — keep
        ]
    )
    b = np.array([1.0, 0.0, -6.6e-11, 2.0, -1.0, 3.0])
    milp = _FakeMilp(sp.csr_matrix(a) if sparse else a, b)
    dropped = _filter_unresolvable_rows(milp)
    assert dropped == 3
    kept = sp.csr_matrix(milp._A_ub).toarray()
    assert kept.shape == (3, 3)
    np.testing.assert_array_equal(kept[0], a[0])
    np.testing.assert_array_equal(kept[1], a[4])
    np.testing.assert_array_equal(kept[2], a[5])
    np.testing.assert_array_equal(milp._b_ub, [1.0, -1.0, 3.0])
    assert sp.issparse(milp._A_ub) == sparse, "container kind must be preserved"


def test_filter_noop_on_well_conditioned_matrix():
    """A relaxation with no wide rows is untouched (byte-identical object data)."""
    from discopt._relax.milp_relaxation import _filter_unresolvable_rows

    a = np.array([[1.0, -2.0], [3.5, 0.25]])
    b = np.array([1.0, 2.0])
    milp = _FakeMilp(a.copy(), b.copy())
    assert _filter_unresolvable_rows(milp) == 0
    np.testing.assert_array_equal(np.asarray(milp._A_ub), a)
    np.testing.assert_array_equal(milp._b_ub, b)


@pytest.mark.slow
def test_hda_certifies_a_tight_bound_at_default(monkeypatch):
    """End-to-end at DEFAULT settings (#671 graduated the filter ON): hda's node
    LPs solve cleanly and the reported dual bound is the tight root-relaxation
    value (≈ −6.47e4 or better via branching) instead of candidate A's −1.80e10,
    while staying sound."""
    monkeypatch.delenv(_FLAG, raising=False)  # rely on the graduated default (ON)
    r = dm.from_nl(os.path.join(_NL_DATA, "hda.nl")).solve(time_limit=60)
    assert r.bound is not None and math.isfinite(r.bound), f"no finite bound: {r.bound}"
    # Sound: never above the published optimum.
    assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
    # Tight: at or above the true root McCormick value (−64675.25 − slack); far
    # above candidate A's −1.80e10 floor.
    assert r.bound >= -7e4, f"bound {r.bound:.6g} not the tight root value"


@pytest.mark.slow
def test_hda_optout_restores_loose_candidate_a_floor(monkeypatch):
    """The ``=0`` opt-out restores the legacy no-filter path: hda's ill-conditioned
    root LP false-fails and the reported bound falls back to candidate A's loose
    floor (≪ −1e7), never the tight value. Proves the legacy path is intact and
    the graduated behavior is genuinely gated (not hardcoded)."""
    monkeypatch.setenv(_FLAG, "0")
    r = dm.from_nl(os.path.join(_NL_DATA, "hda.nl")).solve(time_limit=60)
    # Sound either way; the point is the bound is the LOOSE floor without the filter.
    if r.bound is not None and math.isfinite(r.bound):
        assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
        assert r.bound < -1e7, (
            f"opt-out bound {r.bound:.6g} is unexpectedly tight — the legacy "
            "no-filter path should give the loose candidate-A floor"
        )


# Instances whose *whole solve* is a valid ON/OFF comparison: under
# ``deterministic=True`` they terminate on WORK with real slack against the wall
# limit, which is the precondition #1116 attaches to reproducibility
# (``solver_tuning.SolverTuning.deterministic``). Measured 3 reps x 2 arms each
# (``scratchpad/issue1039/probe_deterministic.py``): byte-identical within and
# across arms, 3/3.
_WORK_TERMINATING = ["alan", "ex1221", "nvs09"]

# Instances on which the whole-solve comparison is NOT affordably valid, so only
# the *mechanism* claim is tested — see
# ``test_failure_triggered_never_fires_on_solving_instances``. Two distinct
# reasons, both measured:
#   * bchoco07, beuster: cannot terminate on work at all within a budget this
#     suite can spend (bchoco07 reached 3 nodes in 120 s, beuster 5-9), so both
#     arms end on ``time_limit`` and are truncated at machine-speed-dependent
#     points.
#   * casctanks: DOES terminate on work and is byte-identical 6/6 under
#     ``deterministic=True`` -- but each arm costs ~321 s (against the 120 s
#     ``time_limit`` it was given; that 2.7x role-1 overrun is #1039 bucket B,
#     tracked separately), so a two-arm comparison exceeds the suite's 300 s
#     per-test timeout.
_NOT_AFFORDABLY_COMPARABLE = ["bchoco07", "beuster", "casctanks"]

# The instances that actually PROVOKE the failure-triggered branch. Measured by
# sweeping all 66 vendored ``.nl`` instances with the #1039
# ``row_filter/invocations`` counter (``scratchpad/issue1039/probe_bucketA.py``);
# exactly three open it, and every other instance reports zero:
#
#     bchoco07  2 invocations / 158 rows dropped
#     bchoco08  2 invocations / 144 rows dropped
#     hda       2 invocations / 356 rows dropped
#
# This is the re-pointing target the issue's own bucket-A retraction comment
# asked for. That comment measured ``filter_invocations=0`` in BOTH arms on hda
# and concluded the mechanism was dormant; with the counter surfaced through
# ``solver_stats`` the count on hda is 2, not 0, and the two arms are far apart
# (FLAG=0 -> -13992288065.86, FLAG=1 -> -64509.85, 2 reps interleaved,
# ``probe_hda_arms.py``). The opt-out is live and load-bearing here.
_FILTER_FIRES = ["bchoco07", "bchoco08", "hda"]

# The complement: vendored instances whose node LPs all certify, so the branch
# never opens. Derived from the same sweep rather than hand-listed.
_FILTER_DORMANT = [
    n for n in _WORK_TERMINATING + _NOT_AFFORDABLY_COMPARABLE if n not in _FILTER_FIRES
]

_DETERMINISTIC_KW = {"deterministic": True, "max_nodes": 25, "time_limit": 120}


@pytest.mark.slow
@pytest.mark.parametrize("name", _WORK_TERMINATING)
def test_failure_triggered_is_byte_identical_on_solving_instances(name, monkeypatch):
    """The failure-triggered filter is byte-identical ON vs OFF on an already-solving
    instance: the un-filtered node LP is optimal/Farkas-infeasible, so the filter
    never fires (it only re-solves a numerically-failed node).

    #1039: this test used to run both arms under a bare ``time_limit=20`` and
    compare the results. That is not a comparison. A solve cut off by the wall
    clock is truncated at a point that depends on machine speed, so the two arms
    are two *different amounts of search* and any difference measures the
    stopwatch. Measured (``probe_deterministic.py``, 3 reps x 2 arms):

        beuster (time_limit=120, truncated)   OFF -> {9 nodes / 8362.516450208394,
                                                     9 nodes / 8362.516450208394,
                                                     5 nodes / 6395.348953445055}
                                              ON  -> {5 nodes / 6395.348953445055,
                                                     9 nodes / 8362.516450208394,
                                                     5 nodes / 6395.348953445055}

    Both outcomes occur in BOTH arms -- the variation is within an arm, so the
    old assertion was failing on truncation, not on the flag. bchoco07's
    reported "bound drifted 1.0000000000002582 -> 1.0000000000002498" is the same
    artifact: under ``deterministic=True`` all six of its runs return the
    identical 0.9999818893334098.

    So both arms now run with ``deterministic=True`` (which neutralizes the
    role-2 wall sub-budgets that cause the drift, per #1116) on the instances
    that terminate on work, and the precondition is ASSERTED rather than assumed
    -- a run that ends on ``time_limit`` fails loudly here instead of silently
    degrading back into the invalid comparison (CLAUDE.md §6).
    """
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(path).solve(**_DETERMINISTIC_KW)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(path).solve(**_DETERMINISTIC_KW)

    # §6: the comparison is only meaningful if neither arm was truncated by the
    # wall clock. Without this the test silently reverts to comparing two
    # differently-truncated searches.
    for arm, r in (("OFF", off), ("ON", on)):
        assert r.status != "time_limit", (
            f"{name}: the {arm} arm terminated on time_limit, so this ON/OFF "
            "comparison is not valid -- raise the budget or move the instance to "
            "_NOT_AFFORDABLY_COMPARABLE"
        )

    assert off.status == on.status, f"{name}: status changed {off.status} -> {on.status}"
    assert off.objective == on.objective, f"{name}: objective drifted with the flag"
    assert off.bound == on.bound, f"{name}: bound drifted ({off.bound} -> {on.bound})"
    assert off.node_count == on.node_count, f"{name}: node count drifted with the flag"


@pytest.mark.slow
@pytest.mark.parametrize("name", _FILTER_DORMANT)
def test_failure_triggered_never_fires_on_solving_instances(name, monkeypatch):
    """The mechanism claim, asserted directly: the filter never fires on these.

    #1039. The byte-identical comparison above is an *indirect* test of "the
    filter never fires" -- and it is unavailable on the three instances that
    cannot finish on work (beuster, casctanks), because a truncated search is not
    comparable to another truncated search. Counting invocations tests the same
    claim directly and truncation cannot invalidate it: whether or not the tree
    finished, the filter either opened its branch or it did not.

    So coverage of those instances is retained here rather than dropped with the
    invalid comparison. Note bchoco07 is deliberately NOT in this list: the
    corpus sweep shows it fires twice, so it was never an "already-solving
    instance" and asserting zero on it was asserting a false premise. It is
    covered by ``test_row_filter_fires_where_the_corpus_says_it_does`` instead.

    ``row_filter/invocations`` is surfaced by
    ``MccormickLPRelaxer._row_filter_stats`` (#1039) and is only present when
    non-zero; the two positive controls below mean a zero here reads as "never
    fired" rather than "never wired".
    """
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "1")
    r = dm.from_nl(path).solve(time_limit=20)
    stats = r.solver_stats or {}

    # "The filter stayed dormant" has to be a POSITIVE observation. Asserting
    # only `stats.get("row_filter/invocations", 0) == 0` passes identically when
    # the counter was never wired up AND when the relaxer never ran at all --
    # the §6 "probe never fired" shape. There are two honest ways to be dormant
    # and each is asserted on its own terms:
    #
    #   (a) the McCormick-LP path ran and the failure branch never opened, so
    #       the counter is present and zero; or
    #   (b) the solve never reached that path, because it was routed to the
    #       convex mip-nlp/oa algorithm, which has no McCormick LP relaxer to
    #       filter rows for. `alan` is this case -- it emits NO solver_stats at
    #       all, so before this assertion it was passing vacuously.
    #
    # `algorithm_route is None` means the DEFAULT path, which is case (a); only a
    # named non-default route can excuse a missing counter.
    route = r.algorithm_route
    if "row_filter/invocations" in stats:
        assert stats["row_filter/invocations"] == 0, (
            f"{name}: the failure-triggered filter fired "
            f"{stats['row_filter/invocations']} time(s) on an already-solving "
            "instance; its node LPs are supposed to certify without it"
        )
    else:
        assert route is not None and "mip-nlp/oa" in route, (
            f"{name}: row_filter/invocations is absent but the solve was not "
            f"routed away from the McCormick-LP path (algorithm_route={route!r}). "
            "That is an unwired counter, not a dormant mechanism."
        )
        assert not stats, (
            f"{name}: routed to mip-nlp/oa yet solver_stats is non-empty "
            f"({sorted(stats)[:5]}); the row-filter counter should have been "
            "emitted alongside them"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", _FILTER_FIRES)
def test_row_filter_fires_where_the_corpus_says_it_does(name, monkeypatch):
    """The other half of the mechanism claim: on the instances that DO provoke an
    uncertified node LP, the filter opens its branch and drops rows.

    #1039. Without this, ``test_failure_triggered_never_fires_on_solving_instances``
    is a one-sided test -- a mechanism deleted outright would pass every one of
    its cases. It is also the direct rebuttal of the issue's bucket-A retraction
    comment, which measured zero invocations on hda and concluded the branch
    never opens: it opens twice.

    This is a *real-instance* control, so unlike the monkeypatched one below it
    also proves the provoking condition still exists in the corpus. If a future
    numerics improvement makes all three of these certify cleanly, this test
    fails and says so, instead of the mechanism quietly losing its last live
    exercise the way #517's did.

    The budget is pinned at the sweep's ``time_limit=15`` deliberately, because
    whether the branch opens is itself budget-dependent -- a #1116-shaped role-2
    effect, measured in ``probe_bchoco_tl.py``:

        bchoco07  tl=15 -> 2 inv   tl=30 -> 2 inv   tl=60 -> 0 inv (status flips
                  to ``feasible``, and the bound goes from 1.0000000000002498 to
                  the LOOSER 0.9999909424984251)
        bchoco08  tl=15 -> 2 inv   tl=30 -> 2 inv   tl=60 -> 2 inv
        hda       tl=15 -> 2 inv   tl=30 -> 2 inv   tl=60 -> 2 inv

    ``deterministic=True`` does not remove the dependence, it moves it: under it
    bchoco07 reaches 0 nodes at tl<=60 and only opens the branch at tl=120. So a
    larger budget is not a safer choice here, and 15s sits inside a plateau all
    three instances share (15 and 30 agree for every one of them).
    """
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "1")
    r = dm.from_nl(path).solve(time_limit=15)
    stats = r.solver_stats or {}
    assert stats.get("row_filter/invocations", 0) >= 1, (
        f"{name}: the failure-triggered filter did not fire at time_limit=15, but "
        "the corpus sweep recorded 2 invocations there -- either the branch "
        "condition changed, the instance now certifies cleanly, or this machine "
        "sits at a different point on the budget curve in the docstring; re-point "
        "this test rather than deleting it"
    )
    assert stats.get("row_filter/rows_dropped", 0) >= 1, (
        f"{name}: the filter was invoked but dropped no rows"
    )
    # Soundness is never conditional on which path produced the bound.
    if r.bound is not None:
        assert math.isfinite(r.bound), f"{name}: non-finite bound {r.bound}"


def test_row_filter_counter_moves_when_the_branch_opens(monkeypatch):
    """Positive control for the #1039 counter (CLAUDE.md §6).

    ``test_failure_triggered_never_fires_on_solving_instances`` reads a zero as
    "the mechanism did not fire". That inference is only valid if the counter can
    move at all -- a miswired counter, or one whose stats never reach
    ``solver_stats``, reports the same zero and reads as a pass. Force the branch
    open by making the node LP report a non-certified verdict, and require the
    count to appear.
    """
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    _real_solve = MilpRelaxationModel.solve
    state = {"doctored": 0}

    def _uncertified_once(self, *a, **kw):
        res = _real_solve(self, *a, **kw)
        # Doctor only the first solve: "numerical" is neither ``optimal`` nor a
        # Farkas-certified ``infeasible``, which is exactly the condition the
        # failure-triggered filter exists to handle.
        if state["doctored"] == 0:
            state["doctored"] += 1
            res.status = "numerical"
            res.farkas_certified = False
        return res

    monkeypatch.setattr(MilpRelaxationModel, "solve", _uncertified_once)
    monkeypatch.setenv(_FLAG, "1")

    m = dm.Model("ctl")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[1] - x[0])
    r = m.solve(time_limit=30)

    assert state["doctored"] == 1, "the control never doctored a node LP verdict"
    stats = r.solver_stats or {}
    assert stats.get("row_filter/invocations", 0) >= 1, (
        "the filter branch opened but row_filter/invocations did not move -- the "
        "counter or its solver_stats plumbing is broken, which would make the "
        "zero asserted above meaningless"
    )
