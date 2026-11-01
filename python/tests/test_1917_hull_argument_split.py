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
