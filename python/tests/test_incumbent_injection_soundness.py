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


@pytest.mark.skipif(not os.path.exists(_NVS19), reason="nvs19.nl not vendored")
def test_feasible_integer_node_objective_is_recorded():
    """The reported incumbent on a clean solve is at least as good as the best
    feasible point the search encounters — i.e. feasible integer leaves are not
    silently fathomed without being recorded.

    The budget is generous (the solve reaches the optimum in ~10s locally) so the
    assertion is about *completeness*, not speed: under heavily-oversubscribed
    parallel CI a 50s wall-clock cap was too tight and the search returned a
    suboptimal incumbent (-1097.6) — a flaky timeout, not a soundness failure."""
    r = from_nl(_NVS19).solve(time_limit=120, gap_tolerance=1e-4)
    assert r.objective is not None
    # Must reach the true optimum (the whole point of recording feasible leaves).
    assert r.objective <= _NVS19_OPT + 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
