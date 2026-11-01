"""#1917: hull of a row whose operand is bounded away from 0 only by a sum row.

GDPlib ``stranded_gas`` has 18 disjunct rows ``... log(S) ...`` with ``S`` a sum of
360 columns, each on ``[0, 10800]``. The disjunct's own row ``S >= 1`` keeps ``log``
in its domain wherever the disjunct holds, but that is a multivariate row: neither
the disjunct's variable bounds nor the lifted perspective's box can see it, so the
hull refused the row with ``HullPerspectiveOriginError``.

``_hull_split_rows`` now hands such a row to the #1999 argument split: ``S`` becomes
a fresh column ``w`` boxed by an LP over the disjunct's rows (``[1, 20]`` here), with
``S - w == 0`` in the disjunct, and the lifted perspective takes the split row.
Pinned: the synthetic model lowers and solves to the big-M optimum, the refusal
stands where the split cannot bound the row, and a row the hull lowered before is
not split.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    GDP_AUX_PREFIX,
    HullPerspectiveOriginError,
    reformulate_gdp,
)


def _log_of_sum(floor: float = 1.0):
    """``z == log(x1 + x2)`` in a disjunct that also says ``x1 + x2 >= floor``."""
    m = dm.Model("log_of_sum")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    m.either_or(
        [
            [x1 + x2 >= floor, z - dm.log(x1 + x2) == 0],
            [x1 + x2 <= 0.5, z == -2.0],
        ],
        name="d",
    )
    m.minimize((x1 - 0.2) ** 2 + (x2 - 0.3) ** 2 + 0.5 * z)
    return m


def _args(low):
    return [v for v in low._variables if v.name.startswith(f"{GDP_AUX_PREFIX}arg_")]


def test_a_log_of_a_sum_bounded_only_by_a_sum_row_lowers_under_hull():
    low = reformulate_gdp(_log_of_sum(), "hull")
    args = _args(low)
    assert len(args) == 1, [v.name for v in low._variables]
    # W = [1, 20]: the disjunct's own LP, so log is finite on it.
    assert float(np.min(args[0].lb)) == pytest.approx(1.0, abs=1e-6)
    assert float(np.max(args[0].ub)) == pytest.approx(20.0, abs=1e-6)
    # The split row takes the lifted perspective, over w's own box.
    t_w = [v for v in low._variables if v.name.endswith(f"_t0_{args[0].name}")]
    assert len(t_w) == 1 and float(np.min(t_w[0].lb)) > 0.99


def test_the_split_cannot_bound_a_sum_with_no_floor_so_the_refusal_stands():
    with pytest.raises(HullPerspectiveOriginError, match="#1917"):
        reformulate_gdp(_log_of_sum(floor=0.0), "hull")


def test_a_row_the_hull_lowered_before_is_not_split():
    m = dm.Model("log_of_var")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    m.either_or([[x >= 1, z - dm.log(x) == 0], [x <= 0.5, z == -2.0]], name="d")
    m.minimize((x - 0.2) ** 2 + 0.5 * z)
    low = reformulate_gdp(m, "hull")
    assert _args(low) == []


@pytest.mark.correctness
def test_hull_and_big_m_certify_the_same_optimum():
    big = _log_of_sum().solve(gdp_method="big-m", time_limit=60)
    hull = _log_of_sum().solve(gdp_method="hull", time_limit=60)
    assert big.status == hull.status == "optimal", (big.status, hull.status)
    assert hull.objective == pytest.approx(big.objective, abs=1e-5)


# --- PR #2076 review -------------------------------------------------------------


def _nested_log_of_sum():
    """The ``log(x1 + x2)`` row one level down: ``outer: [u >= 1, inner] v [...]``.

    The optimum (0.225, at ``x1 + x2 = 1``, ``z = 0``, ``u = 1``) lies in the nested
    disjunct whose row is split, so the split is load-bearing for the answer.
    """
    m = dm.Model("nested_log_of_sum")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    u = m.continuous("u", lb=0, ub=5)
    inner = m.disjunction(
        [[x1 + x2 >= 1, z - dm.log(x1 + x2) == 0], [x1 + x2 >= 2, z == 1.0]], name="inner"
    )
    m.either_or([[u >= 1, inner], [x1 + x2 <= 0.5, z == 2.0, u <= 0.5]], name="outer")
    m.minimize((x1 - 0.2) ** 2 + (x2 - 0.3) ** 2 + 0.5 * z + 0.1 * u)
    return m


def test_a_nested_split_column_is_disaggregated_by_the_parent():
    from discopt._relax.gdp_reformulate import _collect_variables

    low = reformulate_gdp(_nested_log_of_sum(), "hull")
    args = _args(low)
    assert len(args) == 1, [v.name for v in low._variables]
    w = args[0].name
    # w is shared at the parent, not local: the parent made a copy of it per disjunct
    # and its aggregation row is the only row left that reads w itself (every child
    # row was rewritten over the parent's copies).
    copies = sorted(v.name for v in low._variables if v.name.startswith(f"_hull_outer_v_{w}_"))
    assert copies == [f"_hull_outer_v_{w}_0", f"_hull_outer_v_{w}_1"]
    readers = [c.name for c in low._constraints if w in _collect_variables(c.body)]
    assert readers == [f"_hull_agg_outer_{w}"]


@pytest.mark.correctness
def test_nested_split_hull_and_big_m_certify_the_same_optimum():
    big = _nested_log_of_sum().solve(gdp_method="big-m", time_limit=60)
    hull = _nested_log_of_sum().solve(gdp_method="hull", time_limit=60)
    assert big.status == hull.status == "optimal", (big.status, hull.status)
    assert big.objective == pytest.approx(0.225, abs=1e-5)
    assert hull.objective == pytest.approx(big.objective, abs=1e-5)


def _nested_log_of_var():
    """A nested row lowered by the eps-free lifted perspective, with no split."""
    m = dm.Model("nested_log_of_var")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    u = m.continuous("u", lb=0, ub=5)
    inner = m.disjunction([[x >= 1, z - dm.log(x) == 0], [x >= 2, z == 1.0]], name="inner")
    m.either_or([[u >= 1, inner], [x <= 0.5, z == 2.0, u <= 0.5]], name="outer")
    m.minimize((x - 0.2) ** 2 + 0.5 * z + 0.1 * u)
    return m


@pytest.mark.correctness
def test_a_nested_lifted_perspective_row_is_kept_by_the_parent():
    # Before: the child's ratio column t in [1, 10] was taken for a local of the
    # parent, which requires 0 in [lb, ub], and the lowering died on an
    # AssertionError. t is a ratio, not a quantity: its rows are kept as is.
    low = reformulate_gdp(_nested_log_of_var(), "hull")
    assert _args(low) == []
    big = _nested_log_of_var().solve(gdp_method="big-m", time_limit=60)
    hull = _nested_log_of_var().solve(gdp_method="hull", time_limit=60)
    assert big.status == hull.status == "optimal", (big.status, hull.status)
    assert hull.objective == pytest.approx(big.objective, abs=1e-5)


def _array_log(lb, ub):
    m = dm.Model("array_log")
    x = m.continuous("x", shape=(2,), lb=np.array(lb), ub=np.array(ub))
    z = m.continuous("z", lb=-10, ub=10)
    m.either_or([[x[0] >= 0.2, z - dm.log(x[1]) == 0], [x[0] <= 0.1, z == -2.0]], name="d")
    m.minimize(x[0] + z)
    return m


def test_the_probe_judges_an_array_variable_on_the_hulls_own_box(monkeypatch):
    """x[0] in [0, 1], x[1] in [1, 2], row z == log(x[1]): never split.

    The hull's ratio column for ``x`` takes ``(min lb, max ub) = [0, 2]``, on which
    ``log`` is unbounded, so the hull refused this row before #1917 and still does;
    the probe asks the same question on the same box. The split does not narrow
    ``x[1]`` (its box already is its active range), so no column is added.
    """
    from discopt._relax import gdp_reformulate as G

    calls = []
    real = G._ArgumentDisaggregator.split

    def spy(self, con, active, accept):
        out = real(self, con, active, accept)
        calls.append(out)
        return out

    monkeypatch.setattr(G._ArgumentDisaggregator, "split", spy)
    with pytest.raises(HullPerspectiveOriginError, match="#1917"):
        reformulate_gdp(_array_log([0.0, 1.0], [1.0, 2.0]), "hull")
    assert calls and all(c is None for c in calls)
    # Bounds that keep the whole array off log's pole: lowered as before, unsplit.
    calls.clear()
    low = reformulate_gdp(_array_log([1.0, 2.0], [2.0, 3.0]), "hull")
    assert _args(low) == [] and calls == []


def test_the_hull_accepts_a_split_by_its_own_test_not_big_ms(monkeypatch):
    from discopt._relax import gdp_reformulate as G

    # Big-M's test says no to everything: the hull's split must not consult it.
    monkeypatch.setattr(G, "_big_m_unavailable", lambda con, model: True)
    assert len(_args(reformulate_gdp(_log_of_sum(), "hull"))) == 1


def test_a_rejected_split_adds_no_column():
    from discopt._relax import gdp_reformulate as G

    m = _log_of_sum()
    dc = next(c for c in m._constraints if isinstance(c, G._DisjunctiveConstraint))
    own = [c for c in dc.disjuncts[0] if isinstance(c, G.Constraint)]
    made = []

    def add(prefix, lb, ub):
        made.append(m.continuous(f"made{len(made)}", lb=lb, ub=ub))
        return made[-1]

    disagg = G._ArgumentDisaggregator(m, add)
    seen = []
    assert disagg.split(own[1], own, lambda rows, scratch: seen.append(rows) or False) is None
    assert seen and made == []  # the predicate was asked, and refused: no column
    assert disagg.split(own[1], own, lambda rows, scratch: True) is not None
    assert len(made) == 1


def test_an_uncollected_variable_is_its_own_error_class():
    from discopt._relax import gdp_reformulate as G

    m = dm.Model("c")
    x = m.continuous("x", lb=1, ub=2)
    with pytest.raises(G.HullOriginCollectionError, match="not collected"):
        G._body_at_zero(dm.log(x), {})
    assert issubclass(G.HullOriginCollectionError, HullPerspectiveOriginError)
    # The probe hands it back to the lowering (which re-raises it), never to a split.
    assert G._hull_origin_unbounded(dm.log(x) <= 1, {}) is False


def test_an_interval_evaluator_defect_propagates(monkeypatch):
    from discopt._relax.convexity import interval_eval

    def broken(*a, **k):
        raise TypeError("a node the evaluator does not know")

    monkeypatch.setattr(interval_eval, "evaluate_interval", broken)
    with pytest.raises(TypeError, match="does not know"):
        reformulate_gdp(_log_of_sum(), "hull")
