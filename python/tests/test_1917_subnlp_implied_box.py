"""#1917: regression fixture left by the retired ``DISCOPT_SUBNLP_IMPLIED_BOX``.

Under ``gdp_method="hull"``, a heuristic that fixes a disjunct's indicator to 0 leaves
the disjunct's disaggregated copy ``v`` on its whole box. POUNCE's bound push then
starts ``v`` at 0.01, and the hull row ``yhat * exp(a v / yhat)`` overflows at
``yhat = eps``: ``Invalid_Number_Detected`` before the first iteration. The model is
the Trespalacios ex. 6.3.3 GDP (``pyomo.ex_633_trespalacios`` in the parity panel).

The flag solved heuristic sub-NLPs on the node-FBBT box instead (164 -> 6
``Invalid_Number_Detected`` here). Its pre-registered §5 panel
(``docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.prereg.md``) retired it
on 2026-11-03: net-positive failed (2 cells gained against a bar of 3, heatexch_gen3
lost its incumbent in both reps, primal worse on 2 cells and better on 0). The
flag and its ``_solve_sub_nlp`` branch were deleted; ``_implied_box``, which
``subnlp``'s default-on refusal uses, stays. This test keeps the hull solve of
the model that motivated it correct.
"""

from __future__ import annotations

import pytest

ORACLE = 4.46037  # baron+scip (gdplib_runner.panel_oracle)


def _ex633():
    """The parity-panel model, converted from Pyomo.

    Not rebuilt natively: the converted model orders the indicators first, and a
    native build of the same GDP (indicators after ``x1, x2``) sends the heuristics
    down other dives and hit no ``Invalid_Number_Detected`` at all (measured: 0 of
    184, 235 nodes), so it would not exercise the class.
    """
    gr = pytest.importorskip("benchmarks.gdplib_runner")
    pytest.importorskip("pyomo")
    from discopt.pyomo.gdp import convert_pyomo_gdp

    spec = gr.discover_models(include=["pyomo.ex_633_trespalacios"])
    if not spec:
        pytest.skip("pyomo.ex_633_trespalacios not available")
    return convert_pyomo_gdp(spec[0].builder()).model


@pytest.mark.slow
def test_hull_solve_is_correct():
    r = _ex633().solve(gdp_method="hull", time_limit=120)
    assert str(r.status).endswith("optimal")
    assert r.objective == pytest.approx(ORACLE, abs=1e-4)
    assert r.bound <= r.objective + 1e-6
