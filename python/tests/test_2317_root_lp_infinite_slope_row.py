"""#2317: the NLP-BB root-cut LP must not take a row with an infinite sampled slope.

``_RootLP`` classes a row as linear when its Jacobian agrees at two random points
of the root box. ``np.isclose`` calls equal infinities close and a fixed column is
sampled at the same value twice, so ``sqrt(x)`` with ``x`` fixed at 0 (whose slope
is infinite since #1491) became an LP row ``[-inf, -1] <= nan`` and the Rust
engine raised ``ValueError: standard form b contains NaN`` from ``Model.solve``
(the HiGHS engine silently lost the whole stage instead).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _model(form: str = "sqrt"):
    # x is fixed at 0, where d/dx sqrt(x) is infinite. The optimum is y = 2.
    m = dm.Model("sqrt_at_fixed_zero")
    x = m.integer("x", lb=0, ub=0)
    y = m.continuous("y", lb=0, ub=4)
    if form == "sqrt":
        m.subject_to(2 - dm.sqrt(x) - y <= 0)
    elif form == "pow":
        m.subject_to(2 - x**0.5 - y <= 0)
    else:
        m.subject_to(dm.sqrt(x) + y >= 2)
    m.minimize(y)
    return m


@pytest.mark.unit
def test_root_lp_rows_stay_finite_when_a_fixed_column_has_an_infinite_slope():
    from discopt._tape_nlp_evaluator import cached_tape_evaluator
    from discopt.solvers._root_cuts import _RootLP

    m = _model()
    root = _RootLP(
        m,
        cached_tape_evaluator(m),
        np.array([0.0, 0.0]),
        np.array([0.0, 4.0]),
        np.array([True, False]),
        np.zeros(2, bool),
        False,
    )
    checked = 0
    for arr in (root.A_le, root.b_le, root.A_eq, root.b_eq):
        assert np.isfinite(np.asarray(arr, float)).all()
        checked += 1
    assert checked == 4
    assert root.nl_rows == [0]


@pytest.mark.parametrize("engine", ["rust", "highs"])
@pytest.mark.parametrize("form", ["sqrt", "pow", "ge"])
def test_solve_with_sqrt_of_a_variable_fixed_at_zero(engine, form, monkeypatch):
    if engine == "highs":
        pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", engine)
    r = _model(form).solve(threads=1, time_limit=20)
    assert r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.bound is None or r.bound <= 2.0 + 1e-6


def test_fbbt_fixed_column_variant(monkeypatch):
    """``x`` is not declared fixed; bound tightening fixes it at 0."""
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", "rust")
    m = dm.Model("sqrt_fbbt_fixed")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x <= 0.5)
    m.subject_to(2 - dm.sqrt(x) - y <= 0)
    m.minimize(y)
    r = m.solve(threads=1, time_limit=20)
    assert r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.bound is None or r.bound <= 2.0 + 1e-6


def _fixed_charge_with_sqrt_of_fixed_zero():
    """The #781 fixed-charge model (min sense, optimum 8.5 with the extra row)
    plus ``z`` integer fixed at 0 in ``0.5 - sqrt(z) - f1 <= 0``."""
    m = dm.Model("rc_min_sqrt_fixed")
    f0 = m.continuous("f0", lb=0.0, ub=10.0)
    f1 = m.continuous("f1", lb=0.0, ub=10.0)
    y0 = m.binary("y0")
    y1 = m.binary("y1")
    z = m.integer("z", lb=0, ub=0)
    m.subject_to(f0 - 8.0 * y0 <= 0.0)
    m.subject_to(f1 - 8.0 * y1 <= 0.0)
    m.subject_to(f0 + f1 >= 3.0)
    m.subject_to(f0 * f0 + f1 * f1 <= 16.0)
    m.subject_to(0.5 - dm.sqrt(z) - f1 <= 0.0)
    m.minimize(f0 + 2.0 * f1 + 2.5 * y0 + 2.5 * y1)
    return m


@pytest.mark.parametrize("engine", ["rust", "highs"])
def test_root_cut_stage_keeps_working_on_the_other_rows(engine, monkeypatch):
    if engine == "highs":
        pytest.importorskip("highspy")
    from discopt._tape_nlp_evaluator import cached_tape_evaluator
    from discopt.solvers._root_cuts import generate_root_cuts

    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", engine)
    m = _fixed_charge_with_sqrt_of_fixed_zero()
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 1.0, 1.0, 0.0])
    is_int = np.array([False, False, True, True, True])
    is_bin = np.array([False, False, True, True, False])
    res = generate_root_cuts(m, cached_tape_evaluator(m), lb, ub, is_int, is_bin)
    assert res.lp_bound is not None, "the stage produced no bound"
    assert res.lp_bound <= 8.5 + 1e-6, f"root LP bound {res.lp_bound} above the optimum 8.5"
    assert len(res.cuts) >= 1
    # Every cut keeps the optimum (f0 = 0, f1 = 3, y1 = 1, z = 0).
    x_opt = np.array([0.0, 3.0, 0.0, 1.0, 0.0])
    for alpha, rhs in res.cuts:
        assert float(np.asarray(alpha) @ x_opt) <= rhs + 1e-6
