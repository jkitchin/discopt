"""#1236 item E: what node-count determinism actually guarantees.

Issue #1236 reported its node counts as "deterministic and match the earlier
panel exactly", and used that to justify comparing `tls2`'s node count across
arms. The claim is true for a **certifying** exit and false for a **budgeted**
one, and the difference is not a nuance: measured over six solves at a 60 s
limit, `tls2` returned 197 / 227 / 357 / 359 nodes with status `feasible`. Its
node count is a wall-clock measurement, so any A/B that treats it as a fixed
number is comparing noise.

The contract this pins:

* A solve that CERTIFIES (`status == "optimal"`) under ``deterministic=True``
  explores the same tree every time. (Not in the default mode: clock-bounded steps
  inside a route can steer a certifying solve to a different tree, #2240. There a
  certifying solve promises the certificate -- the same optimum, soundly bounded --
  and not the node count.) That is what makes a bound-neutrality check
  (CLAUDE.md §5 regime 1) meaningful at all, and why such a check must run with
  ``deterministic=True``.
* A solve that exits on its BUDGET (`time_limit` / `feasible` at the limit)
  carries no such guarantee, and the suite must not assert one.

Both halves are asserted, because pinning only the first would let the second
silently acquire a guarantee it never had -- which is how #1236's framing went
wrong.
"""

from __future__ import annotations

import os

import pytest
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

#: Instances that certify quickly, one per engine family the issue exercised:
#: the MILP driver via the integer-bilinear lift, and the native spatial kernel.
CERTIFYING = ("nvs02", "nvs14", "nvs13")

REPEATS = 3


def _solve(name, time_limit=60.0, deterministic=False):
    return from_nl(os.path.join(DATA, f"{name}.nl")).solve(
        time_limit=time_limit, deterministic=deterministic
    )


def _solve_profiled(name, time_limit):
    """One solve with the Rust profile counters bracketed around it.

    Returns the result and the counters it moved (name -> count, nonzero only).
    The counters are process-wide run totals; an xdist worker runs one test at a
    time, so nothing else writes them between the reset and the snapshot.
    """
    from discopt._rust import profile_counters_py, profile_reset_py

    profile_reset_py()
    r = _solve(name, time_limit=time_limit)
    moved = {k: int(v) for k, v in profile_counters_py().items() if v}
    return r, moved


def _divergence_report(runs, counters):
    """What differs between two certifying runs that disagree.

    Used when two default-mode certificates of `tls2` disagree on the optimum (a
    soundness failure). It was added for differing node counts, which #2240 showed
    the default mode does not promise.

    #2234's CI run certified `tls2` twice in one worker with 104 then 70 nodes, and
    nothing in the failure said why; 70+ local runs (contention, clock scaling,
    hash seeds, heap churn, Python 3.12, git pounce) all explored 82. This puts the
    evidence a root cause needs into the failure message itself: each run's
    route, timing split and solver stats, and every profile counter that differs.
    """
    lines = []
    for i, r in enumerate(runs):
        lines.append(
            f"run {i}: status={r.status} nodes={r.node_count} mip_count={r.mip_count} "
            f"wall={r.wall_time:.3f}s rust={r.rust_time:.3f}s python={r.python_time:.3f}s "
            f"pounce={r.pounce_time:.3f}s route={r.algorithm_route!r}"
        )
        lines.append(f"run {i}: solver_stats={r.solver_stats!r}")
    a, b = counters
    names = sorted(set(a) | set(b))
    diff = {k: (a.get(k, 0), b.get(k, 0)) for k in names if a.get(k, 0) != b.get(k, 0)}
    lines.append(f"profile counters that differ (run0, run1): {diff!r}")
    lines.append(f"profile counters equal in both runs: {len(names) - len(diff)}")
    return "\n".join(lines)


@pytest.mark.smoke
@pytest.mark.parametrize("name", CERTIFYING)
def test_a_certifying_solve_explores_the_same_tree_every_time(name):
    """Under ``deterministic=True``, the mode that promises it.

    #2240: "nothing in a certifying solve was cut short by a clock" is false in the
    default mode. Clock-bounded steps inside the route (the native MILP's
    ``root_cut_time_s`` sub-budget and capped sub-MIPs, the #1940 lift race's
    slices) can end early on a loaded machine and steer a different tree to the
    same certificate. Measured on nvs02, one core shared with 3 busy loops: default
    mode 2 / 258 nodes across runs on both ``main`` 340a4f43 and PR #2238, and CI
    fast shard 3 saw ``[38, 6, 38]`` and ``[38, 11, 38]``; ``deterministic=True``
    gives 38 on every run, loaded or not. A bound-neutrality A/B that compares node
    counts must therefore run deterministic, which is the contract pinned here.
    """
    runs = [_solve(name, deterministic=True) for _ in range(REPEATS)]

    # The probe is only meaningful on a certifying exit (CLAUDE.md §6): a
    # budgeted run would make the assertion below a statement about the clock.
    for r in runs:
        assert r.status == "optimal", (
            f"{name} exited {r.status!r}, not `optimal` -- this test asserts the "
            "CERTIFYING contract and cannot run against a budgeted exit"
        )

    nodes = {r.node_count for r in runs}
    assert len(nodes) == 1, (
        f"{name} node counts differ across {REPEATS} certifying runs: "
        f"{[r.node_count for r in runs]}"
    )
    objectives = {round(r.objective, 9) for r in runs}
    assert len(objectives) == 1, (
        f"{name} certified objectives differ across runs: {[r.objective for r in runs]}"
    )
    # The certificate invariant, sense-aware, on every repetition.
    for r in runs:
        assert r.bound is not None
        assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective))


@pytest.mark.smoke
def test_the_suite_does_not_claim_determinism_for_a_budgeted_exit(monkeypatch):
    """`tls2` at a short budget: whatever it returns, it is not a fixed number.

    This does not assert that the node count *varies* -- that would be a flaky
    test asserting flakiness. It asserts that the contract for whichever regime
    the solve lands in actually holds, and that the result is SOUND either way.

    It deliberately does NOT assert ``status != "optimal"`` (review finding 10).
    That shipped here and is a test that breaks the moment the solver gets FAST
    ENOUGH to certify `tls2` in 5 s -- a self-inflicted future break in the
    per-PR smoke gate, punishing exactly the improvement this repository is for.
    Nor does it skip when that happens: a skip is the §6 no-op that reads as a
    pass. It asserts the OTHER regime's contract instead, and counts which arm
    ran so neither can pass vacuously.

    The two solves run with the Rust profile counters armed, so a certifying pair
    that disagrees reports which counters moved differently (#2234): the failure
    is then diagnosable from the CI log alone.
    """
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    profiled = [_solve_profiled("tls2", time_limit=5.0) for _ in range(2)]
    runs = [r for r, _ in profiled]
    counters = [c for _, c in profiled]
    # Prove the instrument fired (CLAUDE.md §6): a solve that moved no counter
    # means the profile never armed, and the divergence report would be empty.
    for i, c in enumerate(counters):
        assert c, f"run {i}: DISCOPT_PROFILE=1 moved no profile counter"
    budgeted = 0
    certified = 0
    for r in runs:
        # Sound on every run, in every regime -- minimize model, so the dual
        # bound is a LOWER bound and may never sit above the incumbent.
        if r.objective is not None and r.bound is not None:
            assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (
                f"{r.status} exit reports bound {r.bound!r} above incumbent {r.objective!r}"
            )
        if r.status == "optimal":
            # The solver got fast enough (it does today, ~1 s unloaded). Hold it
            # to the certifying contract of the default mode instead of the
            # budgeted one -- a certified exit must actually be certified.
            assert r.gap_certified is True, (
                "an `optimal` exit that is not gap_certified is the #1262 defect"
            )
            certified += 1
        else:
            # Budgeted exit: #1236's determinism claim does not extend to it.
            assert r.gap_certified is False, f"a {r.status!r} exit claims gap_certified"
            budgeted += 1
    assert budgeted + certified == 2, "both repetitions must be inspected"
    if certified == 2:
        # Both runs certified. In the DEFAULT mode that pins the certificate, not
        # the tree (#2240): role-2 wall budgets steer which tree reaches it, so
        # equal node counts are promised only under ``deterministic=True`` (pinned
        # by test_a_certifying_solve_explores_the_same_tree_every_time). This test
        # cannot run tls2 in that mode: it certifies there in 51 s / 209 nodes,
        # against ~1 s / 82 nodes in the default mode. Measured on tls2 at 5 s, one
        # core shared with 2 busy loops, 8 runs: one certified (82 nodes, 3.0 s),
        # seven took a slower path and hit the budget at 47-71 nodes; #2234's CI
        # worker certified it twice at 104 then 70 nodes. Two certificates of one
        # instance must still agree, and when they do not the report says why.
        tol = 1e-6 * max(1.0, abs(runs[0].objective))
        assert abs(runs[0].objective - runs[1].objective) <= tol, (
            f"tls2 certified two different optima, {runs[0].objective!r} then "
            f"{runs[1].objective!r}\n" + _divergence_report(runs, counters)
        )


@pytest.mark.smoke
def test_the_divergence_report_names_the_counters_that_differ():
    """The #2234 diagnostic must say something when it is finally needed.

    It renders only on a failure that has been seen once in CI and never locally,
    so its content is pinned here rather than discovered broken on the next one.
    """
    from types import SimpleNamespace

    def run(nodes):
        return SimpleNamespace(
            status="optimal",
            node_count=nodes,
            mip_count=1,
            wall_time=2.0,
            rust_time=1.0,
            python_time=1.0,
            pounce_time=0.5,
            algorithm_route="mip-nlp/lp_nlp_bb",
            solver_stats={"oa/support_cuts": 2.0},
        )

    report = _divergence_report(
        [run(104), run(70)],
        [{"DiveRuns": 3, "SubMip": 1, "NodeLpSolve": 104}, {"DiveRuns": 3, "NodeLpSolve": 70}],
    )
    assert "run 0: status=optimal nodes=104" in report
    assert "run 1: status=optimal nodes=70" in report
    assert "route='mip-nlp/lp_nlp_bb'" in report
    assert "{'NodeLpSolve': (104, 70), 'SubMip': (1, 0)}" in report
    assert "profile counters equal in both runs: 1" in report
