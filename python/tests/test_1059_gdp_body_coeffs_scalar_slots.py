"""#1059/#941: the GDP big-M layer must use the canonical flat-slot resolver.

``_extract_body_coeffs`` open-coded the offset arithmetic that
``discopt._flat_index.resolve_scalar_slot`` exists to own, and got it wrong in
the two ways #941 documents:

* ``int(e.index)`` raises ``TypeError`` on the tuple index of a 2-D variable, so
  a single 2-D reference anywhere in a disjunct crashed the whole
  reformulation instead of declining one row; and
* a negative index was taken literally, so ``v[-1]`` resolved to
  ``base_offset - 1`` -- an in-range slot belonging to a *different* variable.
  Nothing downstream errors on a wrong-but-in-range slot, so the big-M would
  simply have been computed against the wrong column.

Both are silent-or-fatal rather than conservative, which is backwards: refusing
(``None``) is always sound here because every caller falls back to a more
general path. These tests fail before the delegation and pass after.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from discopt._flat_index import resolve_scalar_slot
from discopt._relax.gdp_reformulate import _extract_body_coeffs


def _model_with_a_2d_variable():
    m = dm.Model("two_d")
    m.continuous("a", lb=0, ub=1)
    m.continuous("X", shape=(2, 3), lb=0, ub=1)
    return m


def _n_vars(model):
    return sum(int(v.size) for v in model._variables)


def test_a_2d_reference_resolves_instead_of_raising():
    """The crash case: ``X[1, 2]`` used to hit ``int(tuple)``."""
    m = _model_with_a_2d_variable()
    X = m._variables[1]
    n = _n_vars(m)

    result = _extract_body_coeffs(X[1, 2], m, n)
    assert result is not None, "a plain 2-D scalar reference must be extractable"
    c_vec, offset = result

    expected_slot = resolve_scalar_slot(X[1, 2], m)
    assert expected_slot is not None
    assert offset == 0.0
    assert c_vec[expected_slot] == 1.0
    assert np.count_nonzero(c_vec) == 1
    # C-order: slot = 1 (for "a") + 1*3 + 2
    assert expected_slot == 6


def test_a_negative_index_does_not_land_on_another_variable():
    """``X[-1, -1]`` is the last entry of ``X``, never a slot before ``X``."""
    m = _model_with_a_2d_variable()
    X = m._variables[1]
    n = _n_vars(m)

    c_vec, offset = _extract_body_coeffs(X[-1, -1], m, n)
    assert offset == 0.0
    assert np.count_nonzero(c_vec) == 1
    assert int(np.flatnonzero(c_vec)[0]) == n - 1, "negative index escaped its own variable"


def test_a_partial_index_is_refused_rather_than_guessed():
    """``X[0]`` is a row, not a scalar; ``None`` keeps the caller conservative."""
    m = _model_with_a_2d_variable()
    X = m._variables[1]
    assert _extract_body_coeffs(X[0], m, _n_vars(m)) is None


def test_a_2d_row_survives_an_end_to_end_mbigm_reformulation():
    """The class-level statement: a disjunct mentioning a 2-D entry reformulates.

    ``mbigm`` is the method that actually reaches ``_extract_body_coeffs`` (via
    ``_precompute_lp_relaxation``). Before the fix this raised ``TypeError`` out
    of ``reformulate_gdp`` -- the whole model failed, not just this row.
    """
    from discopt._relax.gdp_reformulate import reformulate_gdp

    m = dm.Model("gdp_two_d")
    X = m.continuous("X", shape=(2, 2), lb=0, ub=4)
    m.minimize(X[0, 0] + X[1, 1])
    d1 = m.make_disjunct("d1")
    d1.subject_to(X[0, 0] + X[1, 1] >= 3)
    d2 = m.make_disjunct("d2")
    d2.subject_to(X[0, 1] >= 2)
    m.add_disjunction([d1, d2], name="dj")

    out = reformulate_gdp(m, method="mbigm")
    assert out is not None
    assert len(out._constraints) > 0
