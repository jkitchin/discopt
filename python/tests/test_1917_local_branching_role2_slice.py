"""#1917: a certifying ``deterministic`` solve must not depend on machine speed.

GDPlib ``positioning`` under ``gdp_method="hull"`` certifies in about 75 s,
well inside its ``time_limit``, so it is squarely in the ``deterministic``
contract. Measured on ``6fb6b7a``, it still explored 147 nodes in most runs and
199 in one of three under real CPU load. With Python's clocks scaled 3x it
explored 179 / 193 / 307, which made the defect reproducible.

Bisecting the patched modules traced it to NLP-BB's local-branching heuristic.
Its sub-MIP ``time_limit`` was ``min(2.0, max(0.5, remaining - 0.2))`` seconds,
a role-2 slice handed to a nested ``solve_model``. Inside that solve, the
convexity proof's deadline is the nested role-1 deadline, about 2 s away. So
whether the sub-MIP proved its model convex, and which incumbent it returned,
depended on the clock. This is the same class as #1187's RENS budget.

Both local-branching slices (spatial and NLP-BB) now go through
``solver._role2_slice``. After the fix, 5/5 runs at clock scales 1, 3, 3, 3 and
0.3 are bit-identical: 147 nodes, the same bound and the same objective. Wall
time is unchanged.

This test runs the scaled-clock experiment. The structural guard that runs in
CI is ``test_912_wall_budget_inventory.test_local_branching_slices_stay_routed``.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import pytest

pytest.importorskip("pounce", reason="the default NLP route is pounce")
pytest.importorskip("pyomo.environ")
pytest.importorskip("gdplib")

_BENCH = Path(__file__).resolve().parents[2] / "discopt_benchmarks"


def _positioning_model():
    sys.path.insert(0, str(_BENCH))
    try:
        from benchmarks import gdplib_runner as gr
    finally:
        sys.path.remove(str(_BENCH))
    from discopt.pyomo.gdp import convert_pyomo_gdp

    specs = gr.discover_models(include=["positioning"])
    if not specs:
        pytest.skip("this gdplib has no `positioning` (install gdplib from source)")
    return convert_pyomo_gdp(specs[0].builder()).model


def _solve(scale: float, monkeypatch) -> tuple:
    model = _positioning_model()
    reads = [0]
    if scale != 1.0:
        real, t0 = time.perf_counter, time.perf_counter()

        def scaled() -> float:
            reads[0] += 1
            return t0 + (real() - t0) * scale

        monkeypatch.setattr(time, "perf_counter", scaled)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = model.solve(gdp_method="hull", time_limit=3000.0, deterministic=True)
    finally:
        monkeypatch.undo()
    return r, reads[0]


@pytest.mark.slow
@pytest.mark.correctness
def test_certifying_deterministic_hull_solve_is_clock_independent(monkeypatch):
    base, _ = _solve(1.0, monkeypatch)
    fast, reads = _solve(3.0, monkeypatch)

    # CLAUDE.md §6: prove the scaled clock was actually read during the solve,
    # and that both runs are in the regime the contract covers.
    assert reads > 0, "the scaled clock was never read; the probe measured nothing"
    for r in (base, fast):
        assert r.status == "optimal", f"not a certifying exit: {r.status!r}"
        assert math.isclose(r.objective, -8.064136166293226, rel_tol=1e-4, abs_tol=1e-6)

    assert fast.node_count == base.node_count, (
        f"the tree depends on clock speed: {base.node_count} nodes at 1x, "
        f"{fast.node_count} at 3x (#1917)"
    )
    assert (fast.bound, fast.objective) == (base.bound, base.objective)
