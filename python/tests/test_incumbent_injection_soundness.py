"""Certification must be robust to which incumbent is injected, and when.

A B&B certificate ("optimal") claims the incumbent is the global optimum. That
claim must not depend on the *order/value* of incumbents fed to the search — a
*valid suboptimal* incumbent (from a warm start or a primal heuristic) must never
flip a sound search into a false certificate.

This guards the nvs19 regression: in nonconvex mode the Rust tree never promotes a
node's relaxation bound to the incumbent, and the per-node NLP that normally
injects feasible points is strided, so a node whose relaxation solution was already
an integer- and constraint-feasible point (the true optimum at a fully-branched
leaf) could be fathomed without its objective ever being recorded. Seeded with the
*suboptimal* feasible point [1,7,2,3,6,7,7,1], the solver then exhausted its tree
and certified -1098.0 as optimal while -1098.4 is feasible — a false certificate,
exposed once the convex-objective bound made the dual bound tight enough to
"exhaust". The fix injects every verified integer/constraint-feasible node as an
incumbent candidate, so a feasible point can never be fathomed unrecorded.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.modeling.core import from_nl

_NVS19 = "python/tests/data/minlplib/nvs19.nl"
_NVS19_OPT = -1098.4  # verified-feasible global optimum at [2,6,3,2,8,5,7,1]
# Valid, FEASIBLE, but SUBOPTIMAL warm starts (each was confirmed feasible).
_SUBOPTIMAL_SEEDS = [
    [1, 7, 2, 3, 6, 7, 7, 1],  # obj -1097.6 — the seed that triggered the false cert
    [2, 6, 3, 2, 7, 6, 7, 1],  # obj -1098.0
]


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_NVS19), reason="nvs19.nl not vendored")
@pytest.mark.parametrize("seed", _SUBOPTIMAL_SEEDS)
def test_suboptimal_warm_start_never_false_certifies(seed):
    """Seeding a valid suboptimal incumbent must not yield a false certificate."""
    m = from_nl(_NVS19)
    init = {v: float(seed[i]) for i, v in enumerate(m._variables)}
    r = m.solve(
        time_limit=50,
        gap_tolerance=1e-4,
        initial_solution=init,
        use_start_as_incumbent=True,
    )
    if r.gap_certified:
        # A certificate must be the TRUE optimum — never a value above it.
        assert r.objective is not None
        assert r.objective <= _NVS19_OPT + 1e-3, (
            f"FALSE CERTIFICATE: certified {r.objective} but {_NVS19_OPT} is feasible"
        )
        # The reported bound must be a valid lower bound on the optimum.
        assert r.bound is not None and r.bound <= _NVS19_OPT + 1e-3


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_NVS19), reason="nvs19.nl not vendored")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1039 bucket F: nvs19 does not reach its optimum at ANY budget up to "
        "480s, and MORE budget yields a WORSE incumbent. A completeness miss with "
        "no false certificate; the soundness sibling above still passes."
    ),
)
def test_feasible_integer_node_objective_is_recorded():
    """The reported incumbent on a clean solve is at least as good as the best
    feasible point the search encounters — i.e. feasible integer leaves are not
    silently fathomed without being recorded.

    This is a *completeness* check (the search must reach the known optimum), not a
    soundness one — the false-certificate invariant is guarded by
    ``test_suboptimal_warm_start_never_false_certifies`` above, which only asserts
    on certification and is timing-robust. The nvs19 solve reaches the optimum in
    ~11s locally, but it is a CPU-bound B&B that starves under the PR-fast job's
    xdist oversubscription (returning a suboptimal incumbent / hitting the
    pytest-timeout), so it is marked ``slow`` to run outside the parallel gate.

    #1039: the "~11s locally" premise is measurably false now. Swept on a quiet
    machine (load 2.63, ``scratchpad/issue1039/probe_nvs19.py``); the optimum
    -1098.4 is never reached, at any budget:

        tl   wall     status      nodes    objective   bound
        30    30.2s   time_limit   38403   -1098.2     -2076.33
        60    80.7s   time_limit    7619   -1001.2     -7401.66
       120   110.7s   feasible    100009   -1097.6     -1104.24
       240   188.4s   feasible    100001   -1097.6     -1104.21
       480   344.1s   feasible    100013   -1097.6     -1103.83

    Two things in that table matter more than the missed threshold:

    * **More budget buys a worse answer.** tl=30 finds -1098.2 (0.2 short of the
      optimum); tl=60 finds -1001.2, and explores 5x FEWER nodes (38403 -> 7619)
      despite twice the wall clock. That is the #1116 role-2 signature -- a larger
      ``time_limit`` inflates the sub-budgets carved from it, so the search spends
      far more per node and covers far less tree. It is a performance pathology,
      not a budget shortfall, and raising the budget cannot fix it.
    * At tl>=120 the search stops at ~100,000 nodes with ``status=feasible`` and
      wall under the limit, i.e. it is terminating on a NODE cap, not the clock.

    Everything here is sound: every bound is a valid lower bound on -1098.4, and
    every incumbent is >= the optimum (feasible, merely suboptimal). No false
    certificate, which is what the soundness sibling above guards and why that one
    still passes.

    Pinned STRICT with the assertion untouched: the fix is to stop the
    non-monotonicity, and when the search reaches the optimum again this xfail
    turns into a suite failure that says so."""
    r = from_nl(_NVS19).solve(time_limit=60, gap_tolerance=1e-4)
    assert r.objective is not None
    # Must reach the true optimum (the whole point of recording feasible leaves).
    assert r.objective <= _NVS19_OPT + 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
