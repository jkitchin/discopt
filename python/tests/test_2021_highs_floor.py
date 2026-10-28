"""The ``lp_nlp_bb`` HiGHS master runs on discopt's highspy floor.

The floor was ``highspy>=1.10`` while the master already needed 1.11 APIs: on
1.10.0 ``milp_highs`` raised ``HiGHS rejected option mip_heuristic_run_rins=False``
(``DISCOPT_HIGHS_MASTER_NO_SUBMIP``, #2021), and with that switched off it failed
on ``Highs.clearCallbacks`` and on the callback's ``mip_solution`` (a pointer
wrapper there, an ndarray from 1.11). The default convex route caught the
exception and fell back, so only the explicit route crashed; the CI lane pinned
to the floor never ran it. That lane runs this file. Measured 2026-10-28.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

pytest.importorskip("highspy")


def _build():
    m = dm.Model("floor2021")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=10)
    m.minimize((x - 2.5) ** 2 + (y - 3.4) ** 2)
    m.subject_to(x**2 + y**2 <= 30)
    return m


def test_lp_nlp_bb_highs_master_certifies():
    r = _build().solve(
        solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="highs", time_limit=30
    )
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    # x = 2.5, y = 3 is feasible (6.25 + 9 <= 30): objective 0.16.
    assert r.objective == pytest.approx(0.16, abs=1e-4)
    # The sub-MIP switches reached HiGHS (the option the floor lacked).
    assert (r.solver_stats or {}).get("oa/highs_no_submip") == 1.0
