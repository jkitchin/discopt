"""A bound on one element of an array must not cap the whole array (hull GDP).

`_extract_disjunct_bounds` keys the per-disjunct bounds it extracts by variable
**name**, so whatever it returns applies to every component of an array
variable. Its body patterns used to dig `.base` out of an `IndexExpression`,
which meant a constraint on `x[0]` produced a bound on all of `x`. The hull
reformulation caps every disaggregated component with that bound, so the
components nobody constrained were capped too -- cutting feasible points and
returning `status=optimal` on a value that is not the optimum.

That is the worst failure this codebase has (CLAUDE.md §1: a wrong certificate),
so the tests below are written to fail loudly on the exact reproduction rather
than on an internal detail that a refactor could move.

The asymmetry in the old code is worth recording: the `const - var` branch
already required a bare `Variable` and so declined an `IndexExpression`,
returning `{}` and no tightening. Only the `var - const` branch leaked. One
direction of the same pattern was sound and the other was not, which is why the
bug survived -- half the obvious test cases pass either way.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import _extract_disjunct_bounds, reformulate_gdp
from discopt.modeling import Model

N = 3
METHODS = ("big-m", "hull", "mbigm")


def _leaky_model():
    """Disjunct 0 constrains ONLY `x[0]`; `x[1]` and `x[2]` stay free in [0, 10]."""
    m = Model("leak")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=10.0)
    m.either_or([[x[0] <= 1.0], [x[0] >= 9.0]], name="d")
    m.subject_to(x[0] <= 1.0, name="pin")  # force the first disjunct
    m.maximize(x[1] + x[2])
    return m


def _reference_objective():
    """The same model with the disjunction removed: 20.0."""
    m = Model("ref")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=10.0)
    m.subject_to(x[0] <= 1.0, name="pin")
    m.maximize(x[1] + x[2])
    return m.solve().objective


@pytest.mark.parametrize(
    "body,expected",
    [
        ("element", {}),
        ("partial slice", {}),
        ("whole variable", {"x": (0.0, 1.0)}),
        ("full slice", {"x": (0.0, 1.0)}),
    ],
)
def test_only_a_whole_variable_yields_a_whole_variable_bound(body, expected):
    m = Model("t")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=10.0)
    con = {
        "element": x[0] <= 1.0,
        "partial slice": x[0:2] <= 1.0,
        "whole variable": x <= 1.0,
        "full slice": x[0:N] <= 1.0,
    }[body]
    assert _extract_disjunct_bounds([con], m) == expected


def test_a_ge_element_bound_also_does_not_leak():
    """The direction that was already sound must stay sound."""
    m = Model("t")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=10.0)
    assert _extract_disjunct_bounds([x[0] >= 9.0], m) == {}
    assert _extract_disjunct_bounds([x >= 9.0], m) == {"x": (9.0, 10.0)}


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_no_reformulation_returns_a_false_optimum(method):
    """Before the fix `hull` returned optimal/2.0 against a true optimum of 20.0."""
    truth = _reference_objective()
    assert truth == pytest.approx(20.0, abs=1e-5), f"reference model changed: {truth}"

    result = reformulate_gdp(_leaky_model(), method=method).solve()
    assert result.status == "optimal", f"{method}: {result.status}"
    assert result.objective == pytest.approx(truth, abs=1e-5), (
        f"{method} reports {result.objective} against a true optimum of {truth} -- "
        "a feasible point was cut and the status still says optimal"
    )


@pytest.mark.slow
def test_all_three_methods_agree_on_the_leaky_model():
    """Agreement across methods is the cheap cross-check that caught this."""
    values = [reformulate_gdp(_leaky_model(), method=m).solve().objective for m in METHODS]
    assert values[0] == pytest.approx(values[1], abs=1e-5)
    assert values[0] == pytest.approx(values[2], abs=1e-5)


def test_a_whole_array_disjunct_still_tightens():
    """The fix must not throw away the sound tightening it is guarding."""
    m = Model("tight")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=10.0)
    m.either_or([[x <= 3.0], [x >= 7.0]], name="d")
    m.subject_to(dm.sum(x) >= 5.0, name="tot")
    m.minimize(dm.sum(x))
    flat = reformulate_gdp(m, "hull")

    from discopt.export import to_lp

    rows = [ln.strip() for ln in to_lp(flat).split("\n") if ln.strip().startswith("_hull_lb_d_x_1")]
    assert rows, "no disaggregated lower-bound rows for disjunct 1"
    # Disjunct 1 is `x >= 7` for the WHOLE array, so `v_1 >= 7*y_1` is valid and
    # is the tightening worth keeping; a plain `v_1 >= 0` would mean the fix
    # over-corrected.
    assert all("7 " in r for r in rows), rows
