"""#1059: the convex-MINLP auto-route must certify or fall back, never strand.

The §5 graduation panel refused ``DISCOPT_CONVEX_MINLP_ROUTE`` for a specific,
reproducible reason: on three in-repo instances the routed OA path terminated
*without a dual bound*, and on ``alan`` — a 9-variable convex MIQP — it burned
the entire 180 s budget and returned an uncertified 3.000 where the default
spatial path certifies 2.925 in 0.18 s (3/3 reps). A route is a decision, not a
commitment; these tests pin the decision being revisited.

Two mechanisms, tested separately:

* the auto-route is **budgeted** to a fraction of the caller's limit, so there
  is a remainder left to recover with (an explicit ``solver="mip-nlp"`` is a
  caller's choice and keeps the whole budget);
* an auto-routed result that did not certify **falls back** to the default path
  with the remainder, seeded by whatever the route did find, and says so on
  ``SolveResult.algorithm_route``.
"""

from pathlib import Path

import numpy as np
import pytest
from discopt.modeling.core import Model, from_nl
from discopt.solver import (
    _CONVEX_ROUTE_BUDGET_FRACTION,
    _route_incumbent_seed,
    _route_result_is_certified,
)

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"

pytest.importorskip("highspy", reason="the OA master needs a MILP backend")


def _load(name: str) -> Model:
    path = NL_DIR / f"{name}.nl"
    assert path.exists(), f"missing corpus instance {path}"
    m = from_nl(str(path))
    m._convexity_time_budget = 10.0
    return m


class _Res:
    """Minimal stand-in with just the fields the predicate reads."""

    def __init__(self, status=None, gap_certified=False, x=None):
        self.status = status
        self.gap_certified = gap_certified
        self.x = x


@pytest.mark.unit
class TestCertifiedPredicate:
    """``feasible`` with no bound is the failure the fallback exists for."""

    def test_certified_result_is_certified(self):
        assert _route_result_is_certified(_Res("optimal", gap_certified=True)) is True

    def test_feasible_without_a_certificate_is_not(self):
        assert _route_result_is_certified(_Res("feasible", gap_certified=False)) is False

    def test_optimal_without_gap_certified_is_not(self):
        """``optimal`` is a claim; ``gap_certified`` is the proof. Trust the proof."""
        assert _route_result_is_certified(_Res("optimal", gap_certified=False)) is False

    def test_infeasible_is_a_certificate(self):
        assert _route_result_is_certified(_Res("infeasible")) is True

    def test_unbounded_is_a_certificate(self):
        assert _route_result_is_certified(_Res("unbounded")) is True

    def test_none_is_not(self):
        assert _route_result_is_certified(None) is False


@pytest.mark.unit
class TestIncumbentSeed:
    """The fallback starts from the route's answer, so it can only improve."""

    def test_dict_solution_is_flattened_in_model_order(self):
        m = _load("gbd")
        n = sum(v.size for v in m._variables)
        x = {v.name: np.zeros(v.size) for v in m._variables}
        seed = _route_incumbent_seed(m, _Res("feasible", x=x))
        assert seed is not None
        assert seed.shape == (n,)

    def test_no_solution_gives_no_seed(self):
        assert _route_incumbent_seed(_load("gbd"), _Res("time_limit")) is None

    def test_a_nonfinite_point_is_refused(self):
        m = _load("gbd")
        x = {v.name: np.full(v.size, np.nan) for v in m._variables}
        assert _route_incumbent_seed(m, _Res("feasible", x=x)) is None


@pytest.mark.unit
class TestRouteBudget:
    """An auto-route is budgeted; an explicit choice is not."""

    @staticmethod
    def _capture(monkeypatch):
        seen = {}
        import discopt.solvers.mip_nlp as mn

        real = mn.solve_mip_nlp

        def spy(model, **kw):
            seen["time_limit"] = kw.get("time_limit")
            return real(model, **kw)

        monkeypatch.setattr(mn, "solve_mip_nlp", spy)
        return seen

    def test_auto_route_gets_a_fraction_of_the_limit(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        seen = self._capture(monkeypatch)
        _load("gbd").solve(time_limit=20.0)
        assert seen, "solve_mip_nlp was never called -- the router did not fire"
        assert seen["time_limit"] == pytest.approx(20.0 * _CONVEX_ROUTE_BUDGET_FRACTION)

    def test_explicit_mip_nlp_keeps_the_whole_limit(self, monkeypatch):
        """The caller chose the algorithm; there is no fallback to reserve for."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        seen = self._capture(monkeypatch)
        _load("gbd").solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=20.0)
        assert seen, "solve_mip_nlp was never called"
        assert seen["time_limit"] == pytest.approx(20.0)


@pytest.mark.slow
class TestFallbackEndToEnd:
    """``alan``: the panel's sharpest regression, and the reason for this change.

    Reference optimum 2.925 (``minlplib.solu``). Before the fallback the routed
    arm returned an uncertified 3.000 after spending the whole budget.
    """

    def test_alan_recovers_its_certificate_through_the_fallback(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        result = _load("alan").solve(time_limit=60.0)
        assert result.gap_certified is True
        assert result.objective == pytest.approx(2.925, abs=1e-3)

    def test_the_fallback_is_visible_on_the_result(self, monkeypatch):
        """A fallback solve must not read as one that was never routed."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        result = _load("alan").solve(time_limit=60.0)
        assert result.algorithm_route is not None
        assert "mip-nlp/oa" in result.algorithm_route
        assert "fell back" in result.algorithm_route

    def test_route_off_is_untouched(self, monkeypatch):
        """The default path keeps its own answer and its empty route field."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        result = _load("alan").solve(time_limit=60.0)
        assert result.algorithm_route is None
        assert result.gap_certified is True
        assert result.objective == pytest.approx(2.925, abs=1e-3)
