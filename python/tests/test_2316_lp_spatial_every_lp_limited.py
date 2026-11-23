"""#2316: every LP ``solve_lp_spatial_bb`` solves carries a time limit.

The engine passed its remaining time only to the first LP of each node (#1009).
The LPs of its dive and feasibility pump, and the LP after each cut round, went
through ``IncrementalMcCormickLP.solve``/``solve_assembled`` with no limit, and the
engine's own deadline checks run only BETWEEN LPs -- so one slow LP ran far past
the limit (1017 s for a 600 s ``Model.solve`` on MINLPLib sonet24v2).
"""

from __future__ import annotations

import traceback

import discopt.modeling as dm
import pytest
from discopt._relax.lp_spatial_bb import solve_lp_spatial_bb
from discopt.solvers import milp_simplex

PARTS = ("node_relax", "dive", "feasibility_pump")


def _small_model():
    m = dm.Model("small")
    x = m.integer("x", lb=0, ub=6)
    y = m.integer("y", lb=0, ub=5)
    z = m.integer("z", lb=0, ub=4)
    m.minimize(x + y + z)
    m.subject_to(x * y + z >= 10)
    return m


@pytest.mark.unit
def test_dive_pump_and_cut_round_lps_all_carry_a_limit(monkeypatch):
    real = milp_simplex.solve_lp_warm_std
    seen: list[tuple[str, object]] = []

    def spy(*args, **kwargs):
        stack = [f.name for f in traceback.extract_stack()]
        part = [name for name in stack if name in PARTS]
        seen.append((part[-1] if part else "other", kwargs.get("time_limit")))
        return real(*args, **kwargs)

    monkeypatch.setattr(milp_simplex, "solve_lp_warm_std", spy)
    limit = 60.0
    r = solve_lp_spatial_bb(_small_model(), time_limit=limit, use_obbt=False, root_cut_rounds=2)
    assert r is not None and r.status == "optimal"
    assert r.objective == pytest.approx(7.0)

    parts = {p for p, _ in seen}
    # Prove the probe reached every LP kind the issue names (CLAUDE.md §6).
    assert {"node_relax", "dive", "feasibility_pump"} <= parts, parts
    checked = 0
    for part, tl in seen:
        assert tl is not None, f"an LP in {part!r} ran without a time limit"
        assert 0.0 < float(tl) <= limit
        checked += 1
    assert checked == len(seen) > 0
