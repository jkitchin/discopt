"""GDP reformulation must preserve an array-valued constraint body (#1215).

The premise this file was written to check turned out to be false, and that is
worth recording: `_relax/gdp_reformulate.py` was believed to emit one
`Constraint` per row, putting every GDP model on the slow side of the 6.9-11.1x
idiom gap measured in `docs/dev/performance-plan.md` §48. It does not. Its loops
run over *disjuncts* and over *input constraint objects*, never over rows, so an
array-valued body passes through as an array-valued body:

    if_then, 8-row disjunct constraint:   9 objects per-element ->  2 vectorised
    either_or, 8-row:                    18 objects per-element ->  4 vectorised
    equality (== inside a disjunct):     17 objects per-element ->  3 vectorised

These tests pin that, because it is a property nothing else asserts and an
innocuous-looking `for i in range(...)` added to the emitter would silently undo
it.

They also pin the stronger claim the reformulation has to satisfy: the two
idioms must produce the *same model*. Checking that is what surfaced C-43, a
certified false optimum in the hull path (see
`test_1215_hull_element_bound_leak.py`). Row ORDER may legitimately differ -- an
`==` body emits all `_le` rows then all `_ge` rows when vectorised, and
interleaves them when written per-element -- so the comparison here is over the
multiset of row bodies, not the raw text.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.export import to_lp
from discopt.modeling import Model

METHODS = ("big-m", "hull", "mbigm")
SHAPES = ("if_then", "equality", "either_or")


def _build(shape, vectorised, n):
    m = Model("g")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    y = m.binary("y")
    if shape == "if_then":
        body = [x <= 3.0] if vectorised else [x[i] <= 3.0 for i in range(n)]
        m.if_then(y, body, name="ind")
    elif shape == "equality":
        body = [x == 3.0] if vectorised else [x[i] == 3.0 for i in range(n)]
        m.if_then(y, body, name="ind")
    else:
        a = [x <= 3.0] if vectorised else [x[i] <= 3.0 for i in range(n)]
        b = [x >= 7.0] if vectorised else [x[i] >= 7.0 for i in range(n)]
        m.either_or([a, b], name="d")
    m.subject_to(dm.sum(x) >= 5.0, name="tot")
    m.minimize(dm.sum(x) - 10.0 * y)
    return m


def _row_bodies(model):
    """Multiset of written row bodies, names stripped -- order-insensitive."""
    out = []
    for line in to_lp(model).split("\n"):
        s = line.strip()
        if ":" in s and not s.startswith("\\") and not s.startswith("obj"):
            out.append(s.split(":", 1)[1].strip())
    return sorted(out)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("method", ("big-m", "mbigm"))
@pytest.mark.parametrize("n", (4, 8))
def test_bigm_reformulations_are_row_for_row_identical(shape, method, n):
    per_element = reformulate_gdp(_build(shape, False, n), method=method)
    vectorised = reformulate_gdp(_build(shape, True, n), method=method)
    rows_e, rows_v = _row_bodies(per_element), _row_bodies(vectorised)
    assert rows_e, "the per-element arm produced no rows"
    assert rows_e == rows_v, (
        f"{shape}/{method}/n={n}: the idioms disagree on the model.\n"
        f"  only per-element: {[r for r in rows_e if r not in rows_v]}\n"
        f"  only vectorised:  {[r for r in rows_v if r not in rows_e]}"
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", (4, 8))
def test_hull_rows_differ_only_by_a_valid_tightening(shape, n):
    """`hull` is the one method where the idioms give different ROWS, on purpose.

    Its disaggregated bounds come from `_extract_disjunct_bounds`, which extracts
    a whole-variable bound only from a body covering the whole variable (C-43).
    An array body `x <= 3` therefore tightens `v_0 <= 3*y_0`, while the
    per-element bodies `x[i] <= 3` fall back to the global `v_0 <= 10*y_0`. Both
    are valid; the vectorised one is a tighter relaxation of the same set.

    The invariant is the row COUNT and the optimum, not the row text. If this
    ever starts failing on the count, the reformulation has changed shape rather
    than tightness.
    """
    per_element = reformulate_gdp(_build(shape, False, n), method="hull")
    vectorised = reformulate_gdp(_build(shape, True, n), method="hull")
    assert len(_row_bodies(per_element)) == len(_row_bodies(vectorised))


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("method", METHODS)
def test_the_two_idioms_solve_to_the_same_optimum(shape, method):
    """The property that actually matters, and the one that caught C-43.

    Same model, two ways of writing it: the answer must not depend on which.
    """
    n = 4
    a = reformulate_gdp(_build(shape, False, n), method=method).solve()
    b = reformulate_gdp(_build(shape, True, n), method=method).solve()
    assert a.status == b.status, f"{shape}/{method}: {a.status} vs {b.status}"
    if a.status == "optimal":
        assert a.objective == pytest.approx(b.objective, abs=1e-5), (
            f"{shape}/{method}: per-element {a.objective} vs vectorised {b.objective}"
        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("method", METHODS)
def test_an_array_body_stays_one_object_through_reformulation(shape, method):
    """The row count is equal; the OBJECT count must be far smaller."""
    n = 8
    per_element = reformulate_gdp(_build(shape, False, n), method=method)
    vectorised = reformulate_gdp(_build(shape, True, n), method=method)
    n_e, n_v = len(per_element._constraints), len(vectorised._constraints)
    assert len(_row_bodies(per_element)) == len(_row_bodies(vectorised))
    assert n_v < n_e, (
        f"{shape}/{method}: {n_v} objects vectorised vs {n_e} per-element -- the "
        "reformulation is expanding an array body into one object per row"
    )
    # It must not grow with n either: re-run at 4 and require the same count.
    smaller = reformulate_gdp(_build(shape, True, 4), method=method)
    assert len(smaller._constraints) == n_v, (
        f"{shape}/{method}: the vectorised object count grew from "
        f"{len(smaller._constraints)} to {n_v} when the family grew from 4 to 8 rows"
    )
