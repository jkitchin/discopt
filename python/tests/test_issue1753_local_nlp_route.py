"""#1753: a deliberate local solve names its route and is not blamed on the relaxation.

``skip_convex_check=True`` runs one local NLP and builds no relaxation, so the
missing dual bound is by design; it used to report ``algorithm_route=None`` and log
"the relaxation layer could not bound this objective" with epigraph advice.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke


def test_local_solve_route_and_no_relaxation_warning(caplog):
    m = dm.Model("rb")
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    m.minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = m.solve(skip_convex_check=True)
    assert r.status == "feasible"
    assert r.bound is None and not r.gap_certified
    assert r.algorithm_route.startswith("local-nlp:")
    assert "skip_convex_check=True" in r.algorithm_route
    assert not [rec for rec in caplog.records if "No valid dual bound" in rec.getMessage()]
