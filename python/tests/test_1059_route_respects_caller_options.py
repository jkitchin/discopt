"""#1059: the auto-route must not answer a question the caller did not ask.

The convex-MINLP auto-route graduated default-ON. It fires only when the caller
"expressed no preference" -- but the first version of that gate read only
``solver=``, and ``solver`` is not the only way a caller names an engine.
``m.solve(nlp_bb=True, node_callback=cb)`` names the NLP branch-and-bound loop
*and* asks to observe its nodes; the MIP-NLP family honours neither. Routing
such a call silently discards both.

This is not hypothetical. Flipping the default ON turned nine tests red at once,
all of them for the same reason -- they passed ``nlp_bb=True`` or a callback and
were routed away from the loop they were written to exercise, so their callbacks
never fired. The gate now consults ``_mip_nlp_ignored_options``, the same table
the ``solver="mip-nlp"`` block warns from, so the two can never drift: an option
this family drops is either a warning (the caller chose the family) or a refusal
to route (the caller did not).
"""

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Model
from discopt.solver import _mip_nlp_ignored_options

ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"


def _convex_minlp() -> Model:
    """A small convex MINLP the route certifies and takes by default."""
    m = dm.Model("route_pref")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    z = m.integer("z", lb=0, ub=4)
    m.subject_to(x + y + z >= 2.0)
    m.minimize(x * x + y * y + 0.3 * z)
    return m


@pytest.mark.unit
class TestIgnoredOptionTable:
    """The table is shared, so its predicates are worth pinning directly."""

    def test_an_untouched_default_is_not_caller_intent(self):
        """A default solve must stay routable.

        If this regressed the route would silently switch itself off for
        everyone and the graduation panel would be undone.

        The defaults are read from ``solve_model``'s own signature rather than
        transcribed. A transcribed copy goes stale the moment a default moves and
        then tests a value nobody passes -- which is what happened when #1116
        moved ``deterministic`` from ``True`` to ``False``.
        """
        import inspect

        from discopt.solver import solve_model

        sig = inspect.signature(solve_model).parameters
        names = [
            "threads",
            "deterministic",
            "batch_size",
            "strategy",
            "nlp_bb",
            "node_callback",
            "incumbent_callback",
            "presolve",
            "cuts",
            "subnlp_frequency",
            "in_tree_presolve_stride",
        ]
        defaults = {}
        for n in names:
            assert n in sig, f"{n} is no longer a solve_model parameter"
            assert sig[n].default is not inspect.Parameter.empty, f"{n} has no default"
            defaults[n] = sig[n].default
        assert len(defaults) == len(names), "a name was dropped -- the probe shrank"
        assert _mip_nlp_ignored_options(defaults) == []

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("nlp_bb", True),
            ("node_callback", lambda ctx, model: None),
            ("incumbent_callback", lambda *a: None),
            ("threads", 4),
            ("strategy", "depth_first"),
            ("subnlp_frequency", 5),
            ("presolve", False),
            ("cuts", "off"),
        ],
    )
    def test_a_set_option_is_reported(self, name, value):
        assert _mip_nlp_ignored_options({name: value}) == [name]

    def test_names_absent_from_the_mapping_are_not_consulted(self):
        # The route passes a snapshot; a caller asking about a subset must not
        # get a verdict about options it did not supply.
        assert _mip_nlp_ignored_options({}) == []

    def test_every_reported_name_is_a_real_solve_model_parameter(self):
        """A stale entry here would silently stop gating (CLAUDE.md §6)."""
        import inspect

        from discopt.solver import _MIP_NLP_IGNORED_OPTIONS, solve_model

        params = set(inspect.signature(solve_model).parameters)
        checked = 0
        for name, _pred in _MIP_NLP_IGNORED_OPTIONS:
            assert name in params, f"{name} is not a solve_model parameter"
            checked += 1
        assert checked >= 30, f"table shrank unexpectedly: {checked} entries"


@pytest.mark.smoke
class TestTheRouteDeclines:
    def test_a_node_callback_still_fires_on_a_routable_model(self, monkeypatch):
        """The regression that caught this: the callback simply never ran."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)  # ambient (graduated) default
        seen: list[int] = []

        res = _convex_minlp().solve(
            time_limit=30.0,
            nlp_bb=True,
            node_callback=lambda ctx, model: seen.append(ctx.node_count),
        )
        assert res.status in ("optimal", "feasible")
        assert seen, "node callback must fire; the route swallowed it"

    def test_the_reason_names_the_option_that_declined_it(self, monkeypatch):
        """The refusal must be legible, not silent (CLAUDE.md §3)."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        res = _convex_minlp().solve(time_limit=30.0, nlp_bb=True)
        route = getattr(res, "algorithm_route", None)
        assert route is None or "nlp_bb" in route, route

    def test_a_plain_solve_is_still_routed(self, monkeypatch):
        """Guard the other direction: the gate must not disable the route."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        res = _convex_minlp().solve(time_limit=30.0)
        route = getattr(res, "algorithm_route", None)
        assert route is not None and "mip-nlp" in route, route
        assert res.status == "optimal"
