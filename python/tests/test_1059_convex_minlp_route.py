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
    def test_default_is_off(self, monkeypatch):
        """§5 regime 2: a bound-changing route ships default-OFF."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        assert _convex_minlp_route_enabled() is False

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        assert _convex_minlp_route_enabled() is True

    def test_zero_is_the_opt_out(self, monkeypatch):
        """``=0`` must stay a hard opt-out, so a graduation cannot strand users."""
        monkeypatch.setenv(ROUTE_ENV, "0")
        assert _convex_minlp_route_enabled() is False

    def test_disabled_flag_declines_a_routable_model(self, monkeypatch):
        """With the flag off the router declines a model it would otherwise take."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        method, reason = _convex_minlp_auto_route(_load("gbd"))
        assert method is None
        assert "disabled" in reason


@pytest.mark.unit
class TestRouteGates:
    """Every gate is a *refusal*: the route never fires on unproven convexity."""

    def test_fires_on_convex_minlp(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        method, reason = _convex_minlp_auto_route(_load("gbd"))
        assert method == "oa"
        assert "certified convex" in reason

    def test_declines_pure_continuous(self, monkeypatch):
        """A convex NLP is already served by the continuous convex fast path."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = Model("convex_nlp")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(dm.exp(x) + x**2)
        m.subject_to(x >= -2)
        method, reason = _convex_minlp_auto_route(m)
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
        method, reason = _convex_minlp_auto_route(m)
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
        method, reason = _convex_minlp_auto_route(m)
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
        method, reason = _convex_minlp_auto_route(m)
        assert method is None
        assert "dm.custom" in reason


@pytest.mark.smoke
class TestRouteDispatch:
    """The router must never override an explicit ``solver=``."""

    def test_explicit_bb_is_honoured(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = _load("gbd")
        result = m.solve(solver="bb", time_limit=30)
        assert result.algorithm_route is None

    def test_auto_route_is_recorded_on_the_result(self, monkeypatch):
        """The routing decision must be visible, not silent."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        m = _load("gbd")
        result = m.solve(time_limit=30)
        assert result.algorithm_route is not None
        assert "mip-nlp/oa" in result.algorithm_route

    def test_route_off_leaves_the_field_unset(self, monkeypatch):
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        m = _load("gbd")
        result = m.solve(time_limit=30)
        assert result.algorithm_route is None
