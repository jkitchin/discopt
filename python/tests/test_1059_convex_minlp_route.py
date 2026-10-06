"""#1059: auto-routing a convexity-certified MINLP to the MIP-NLP family.

discopt certifies the MINLPLib ``syn``/``rsyn`` family fully convex at the root in
hundredths of a second and then discards the certificate: ``solver="mip-nlp"``
was reachable only through an explicit kwarg, so a default ``Model.solve()`` ran
the *spatial global* algorithm on a convex MINLP. These tests pin the router's
gates and the end-to-end effect.

The route is bound-changing under CLAUDE.md §5 regime 2, so it ships default-OFF
behind ``DISCOPT_CONVEX_MINLP_ROUTE``; every test here sets the variable
explicitly rather than relying on the ambient default, so the file keeps testing
what it says after any future graduation.
"""

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Model, from_nl
from discopt.solver import _convex_minlp_auto_route, _convex_minlp_route_enabled

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"
MASTER_ENV = "DISCOPT_CONVEX_ROUTE_OA_MASTER"


def _load(name: str) -> Model:
    path = NL_DIR / f"{name}.nl"
    assert path.exists(), f"missing corpus instance {path}"
    m = from_nl(str(path))
    # Keep classification bounded so a slow box cannot turn a gate test into a
    # timeout; every instance here classifies in well under 0.05 s.
    m._convexity_time_budget = 10.0
    return m


@pytest.mark.unit
class TestRouteFlag:
    def test_default_is_on_after_graduation(self, monkeypatch):
        """§5 regime 2: default-OFF until the panel clears BOTH bars -- it did.

        75 instances at 60 s (in-repo corpus + a syn/rsyn supplement, since the
        in-repo corpus holds one ``syn`` and no ``rsyn``): 0 unsound bounds, 0
        certification regressions, bound tighter/looser 8/3, nodes fewer/more
        27/1, incumbent gained/lost 2/0, objective better/worse 6/2, total wall
        1561.6 s -> 1469.6 s. Cert-clean and net-positive.
        """
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        assert _convex_minlp_route_enabled() is True

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        assert _convex_minlp_route_enabled() is True

    def test_zero_is_the_opt_out(self, monkeypatch):
        """``=0`` must stay a hard opt-out, so a graduation cannot strand users."""
        monkeypatch.setenv(ROUTE_ENV, "0")
        assert _convex_minlp_route_enabled() is False

    def test_disabled_flag_declines_a_routable_model(self, monkeypatch):
        """With the flag off the router declines a model it would otherwise take."""
        monkeypatch.setenv(ROUTE_ENV, "0")
        method, reason, _opts = _convex_minlp_auto_route(_load("gbd"))
        assert method is None
        assert "disabled" in reason


@pytest.mark.unit
class TestRouteGates:
    """Every gate is a *refusal*: the route never fires on unproven convexity."""

    def test_fires_on_convex_minlp(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.delenv(MASTER_ENV, raising=False)
        monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_METHOD", raising=False)
        method, reason, opts = _convex_minlp_auto_route(_load("gbd"))
        # `"lp_nlp_bb"` since #1658 (performance-plan §75; `"oa"` from #1141 to then).
        assert method == "lp_nlp_bb"
        assert reason.startswith("mip-nlp/lp_nlp_bb:")
        assert "certified convex" in reason
        # HiGHS master by default since performance-plan §25.13; the reason says so.
        assert opts == {"milp_solver": "highs"}
        assert "master=highs" in reason

    def test_the_retired_in_house_master_raises(self, monkeypatch):
        """``DISCOPT_CONVEX_ROUTE_OA_MASTER=auto`` (the in-house simplex master)
        was retired: HiGHS is the MILP solver. Asking for it must raise, not
        silently run HiGHS while the caller believes otherwise."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(MASTER_ENV, "auto")
        with pytest.raises(ValueError, match="retired"):
            _convex_minlp_auto_route(_load("gbd"))

    def test_explicit_highs_is_accepted(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(MASTER_ENV, "highs")
        monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_METHOD", raising=False)
        method, reason, opts = _convex_minlp_auto_route(_load("gbd"))
        assert method == "lp_nlp_bb" and opts == {"milp_solver": "highs"}, reason

    def test_route_method_opt_out_restores_oa(self, monkeypatch):
        """``DISCOPT_CONVEX_ROUTE_METHOD=oa`` restores the pre-#1658 target."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_METHOD", "oa")
        method, reason, opts = _convex_minlp_auto_route(_load("gbd"))
        assert method == "oa" and reason.startswith("mip-nlp/oa:"), reason
        assert opts == {"milp_solver": "highs"}

    def test_an_unknown_route_method_raises(self, monkeypatch):
        """A typo must not silently pick an algorithm."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_METHOD", "ecp")
        with pytest.raises(ValueError, match="DISCOPT_CONVEX_ROUTE_METHOD"):
            _convex_minlp_auto_route(_load("gbd"))

    def test_an_unknown_master_raises(self, monkeypatch):
        """A typo must not silently pick an engine."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(MASTER_ENV, "gurobi")
        with pytest.raises(ValueError, match="DISCOPT_CONVEX_ROUTE_OA_MASTER"):
            _convex_minlp_auto_route(_load("gbd"))

    def test_missing_highspy_is_a_loud_broken_install(self, monkeypatch):
        """highspy is a core dependency since #1229, so its absence is a broken
        install -- the route raises, as the pure LP/MILP route does, instead of
        silently choosing another engine (the #1141 defect: the default
        algorithm must never change with what happens to be importable). An
        environment without highspy turns the route off instead."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        import builtins

        real_import = builtins.__import__

        def _no_highs(name, *a, **kw):
            if name == "highspy":
                raise ImportError("highspy is not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_highs)
        monkeypatch.delenv(MASTER_ENV, raising=False)
        with pytest.raises(ImportError, match="highspy"):
            _convex_minlp_auto_route(_load("gbd"))
        monkeypatch.setenv(ROUTE_ENV, "0")
        method, _reason, _opts = _convex_minlp_auto_route(_load("gbd"))
        assert method is None

    def test_declines_pure_continuous(self, monkeypatch):
        """A convex NLP is already served by the continuous convex fast path."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = Model("convex_nlp")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(dm.exp(x) + x**2)
        m.subject_to(x >= -2)
        method, reason, _opts = _convex_minlp_auto_route(m)
        assert method is None
        assert "pure continuous" in reason

    def test_declines_milp(self, monkeypatch):
        """A MILP has no nonlinearity for a MIP-NLP decomposition to decompose."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = Model("milp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 2 * y)
        m.subject_to(x + y >= 1)
        method, reason, _opts = _convex_minlp_auto_route(m)
        assert method is None
        assert "not a discrete NLP" in reason

    def test_declines_nonconvex_minlp(self, monkeypatch):
        """A nonconvex MINLP must stay on the sound spatial path."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = Model("nonconvex_minlp")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        z = m.binary("z")
        m.minimize(x * y + z)  # bilinear: nonconvex
        m.subject_to(x + y + z >= 1)
        method, reason, _opts = _convex_minlp_auto_route(m)
        assert method is None
        assert reason.startswith("not routed")

    def test_declines_opaque_custom_body(self, monkeypatch):
        """``dm.custom`` is AD-only; a MILP master cannot linearize it."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = Model("custom")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        opaque = dm.custom(lambda v: v**2)
        m.minimize(opaque(x) + y)
        m.subject_to(x + y >= 1)
        method, reason, _opts = _convex_minlp_auto_route(m)
        assert method is None
        assert "dm.custom" in reason


@pytest.mark.smoke
class TestRouteDispatch:
    """The router must never override an explicit ``solver=``."""

    def test_explicit_bb_is_honoured(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = _load("gbd")
        result = m.solve(solver="bb", time_limit=30)
        # #1614: every route now names itself; the auto-route must not be it.
        assert (result.algorithm_route or "").startswith("miqp-bb:"), result.algorithm_route

    def test_auto_route_is_recorded_on_the_result(self, monkeypatch):
        """The routing decision must be visible, not silent."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = _load("gbd")
        result = m.solve(time_limit=30)
        assert result.algorithm_route is not None
        assert result.algorithm_route.startswith("mip-nlp/lp_nlp_bb:"), result.algorithm_route

    def test_route_off_leaves_the_field_unset(self, monkeypatch):
        """The opt-out must restore the pre-graduation behaviour exactly."""
        monkeypatch.setenv(ROUTE_ENV, "0")
        m = _load("gbd")
        result = m.solve(time_limit=30)
        # #1614: the default path names its own engine instead of leaving None;
        # the opt-out must still keep the solve off the auto-route.
        assert (result.algorithm_route or "").startswith("miqp-bb:"), result.algorithm_route


@pytest.mark.unit
class TestRoutedMasterEngine:
    """End to end: the engine the route names is the one the OA master runs on."""

    @staticmethod
    def _count_highs_calls(monkeypatch):
        import discopt.solvers.milp_highs as milp_highs

        calls = []
        # Both entry points: OA's master is ``solve_milp``, LP/NLP-BB's (the route
        # target since #1658) is ``solve_milp_with_lazy_cuts``.
        for name in ("solve_milp", "solve_milp_with_lazy_cuts"):
            real = getattr(milp_highs, name)

            def _counting(*a, _real=real, **kw):
                calls.append(1)
                return _real(*a, **kw)

            monkeypatch.setattr(milp_highs, name, _counting)
        return calls

    def test_default_solve_runs_the_master_on_highs(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.delenv(MASTER_ENV, raising=False)
        calls = self._count_highs_calls(monkeypatch)
        m = _load("gbd")
        r = m.solve(time_limit=60)
        assert r.algorithm_route is not None
        assert r.algorithm_route.startswith("mip-nlp/lp_nlp_bb:"), r.algorithm_route
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(2.2, abs=1e-6)
        assert len(calls) > 0
