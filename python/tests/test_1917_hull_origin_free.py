"""#1917: hull of a disjunct row whose ``g(0)`` is not finite.

The Furman-Sawaya-Grossmann eps-perspective needs ``g(0)`` to cancel its residual on
a de-selected disjunct (#1043), so ``log(x)`` or ``1/x`` in a disjunct raised
``HullPerspectiveOriginError`` -- 4 ``gdp_parity`` instances (``gdp_col``,
``kaibel``, ``hda``, ``stranded_gas``), all of which Pyomo's hull solves.
``_origin_free_hull_row`` lowers such a row by the eps-free lifted perspective
(``v - y t = 0``, ``t`` in the disjunct's own box, row ``y g(t) <sense> y rhs``), which
needs no ``g(0)``, when ``g`` is bounded on that box; otherwise the refusal stands.

Pinned here: which rows take the new form and which keep today's (a model the
eps-perspective can lower is untouched), the refusal that remains, and solves that
agree with big-M. The differential battery (``test_gdp_differential_battery``,
shapes ``log_off_origin`` / ``recip_off_origin``) checks soundness point by point
under every method.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import HullPerspectiveOriginError, reformulate_gdp
from test_gdp_differential_battery import SHAPES, _log_off_origin, _recip_off_origin


def _ratio0(low):
    return (
        [v for v in low._variables if "_t0_" in v.name],
        [c for c in low._constraints if c.name and c.name.startswith("_hull_ratio0_")],
    )


@pytest.mark.parametrize("build", [_log_off_origin, _recip_off_origin])
def test_a_row_undefined_at_the_origin_now_lowers_under_hull(build):
    low = reformulate_gdp(build(), "hull")
    ts, links = _ratio0(low)
    # x and z of disjunct 0 (the only nonlinear row is there)
    assert sorted(t.name.rsplit("_", 1)[-1] for t in ts) == ["x", "z"], [t.name for t in ts]
    assert len(links) == 2 and all(c.sense == "==" for c in links)
    # t lives on the disjunct's OWN box: x >= 1 (log) / 0.5 (recip), not [0, 10],
    # so g is never evaluated where it is undefined.
    tx = next(t for t in ts if t.name.endswith("_x"))
    assert float(np.min(tx.lb)) in (1.0, 0.5) and float(np.max(tx.ub)) == 10.0


def test_a_row_unbounded_on_its_own_box_still_refuses():
    """``1/x`` with no bound away from 0 in the disjunct: the lifted row's relaxation
    would be unbounded too, so the refusal stands -- and now says why."""
    m = dm.Model("recip_reaches_zero")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=-5, ub=5)
    m.either_or([[1.0 / x <= z + 1], [x <= 0.2, z <= -1]], name="d")
    m.minimize((x - 4) ** 2 - z)
    with pytest.raises(HullPerspectiveOriginError, match="#1917"):
        reformulate_gdp(m, "hull")


@pytest.mark.parametrize(
    "shape",
    [s for s in SHAPES if not s.name.endswith("_off_origin")],
    ids=lambda s: s.name,
)
def test_rows_the_eps_perspective_can_lower_are_untouched(shape):
    """Bound-neutral outside the refusal: no shape whose ``g(0)`` is finite gets a
    lifted row (the eps-perspective is kept, so its convexity is kept too)."""
    try:
        low = reformulate_gdp(shape.build(), "hull")
    except Exception:  # noqa: BLE001 - shapes that declare a hull refusal
        assert "hull" in shape.refuses, shape.name
        return
    assert _ratio0(low) == ([], [])


@pytest.mark.correctness
@pytest.mark.parametrize("build", [_log_off_origin, _recip_off_origin])
def test_hull_and_big_m_certify_the_same_optimum(build):
    big = build().solve(gdp_method="big-m", time_limit=60)
    hull = build().solve(gdp_method="hull", time_limit=60)
    assert big.status == hull.status == "optimal", (big.status, hull.status)
    assert hull.objective == pytest.approx(big.objective, abs=1e-6, rel=1e-4)
    for r in (big, hull):
        assert r.bound <= r.objective + 1e-9
    # Neither bound may pass the other's incumbent (a false bound would).
    assert hull.bound <= big.objective + 1e-6
    assert big.bound <= hull.objective + 1e-6


def _two_owned_columns_in_one_row():
    """One origin-free row over two variables whose disjunct boxes are disjoint.

    Optimum in disjunct 0 at ``x = 2, z = 6`` (objective ``-x - z/10 = -2.6``);
    disjunct 1 only allows ``x <= 0.5``. Before the fix both ``t``s of the row got
    the same column, so ``v_z - y t_z = 0`` read ``t_x``'s box ``[1, 2]`` against
    ``z in [5, 6]`` and disjunct 0 became (falsely) infeasible.
    """
    m = dm.Model("two_t_one_row")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.either_or(
        [[dm.log(x) - dm.log(z) <= 0.0, x >= 1, x <= 2, z >= 5, z <= 6], [x <= 0.5, z <= 1]],
        name="d",
    )
    m.minimize(-x - z / 10)
    return m


def test_every_generated_column_is_its_own():
    """A variable's ``_index`` is its position: two ``t``s of one row used to share
    one, so every expression over the second read the first's value and box (GDPlib
    ``hda``: hull returned a false ``infeasible``)."""
    low = reformulate_gdp(_two_owned_columns_in_one_row(), "hull")
    ts, _ = _ratio0(low)
    assert len(ts) == 2, [t.name for t in ts]
    n = 0
    for pos, v in enumerate(low._variables):
        assert v._index == pos, (v.name, v._index, pos)
        n += 1
    assert n == len(low._variables) > 0


def test_registering_a_stale_column_is_refused():
    from discopt.modeling.core import Variable, VarType

    m = dm.Model("stale")
    m.continuous("x", lb=0, ub=1)
    a = Variable("a", VarType.CONTINUOUS, (), 0.0, 1.0, m)
    b = Variable("b", VarType.CONTINUOUS, (), 0.0, 1.0, m)
    m._register_generated_variable(a)
    with pytest.raises(ValueError, match="column"):
        m._register_generated_variable(b)


@pytest.mark.correctness
def test_two_lifted_columns_in_one_row_solve_to_the_big_m_optimum():
    big = _two_owned_columns_in_one_row().solve(gdp_method="big-m", time_limit=60)
    hull = _two_owned_columns_in_one_row().solve(gdp_method="hull", time_limit=60)
    assert big.status == hull.status == "optimal", (big.status, hull.status)
    assert hull.objective == pytest.approx(-2.6, abs=1e-6)
    assert big.objective == pytest.approx(-2.6, abs=1e-6)
    assert hull.bound <= big.objective + 1e-6
