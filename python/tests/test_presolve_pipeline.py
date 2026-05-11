"""Integration tests for the root presolve orchestrator.

Verifies that the Rust-side passes ``eliminate_variables`` (M10) and
``reformulate_polynomial`` (M4 + M5) of issue #51 are correctly
sequenced by ``discopt._jax.presolve_pipeline.run_root_presolve`` and
that tightened bounds are propagated back into the Python ``Model``
object so that ``Model.solve()`` can use them.
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import discopt
import numpy as np
import pytest
from discopt._jax.presolve_pipeline import (
    propagate_bounds_to_model,
    run_reverse_ad_tightening,
    run_root_presolve,
)


def _model_repr(model):
    from discopt._rust import model_to_repr

    return model_to_repr(model, getattr(model, "_builder", None))


def test_eliminate_pins_singleton_equality():
    """A continuous var fully determined by a singleton equality is pinned."""
    m = discopt.Model("singleton_eq")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(2.0 * x == 6.0, name="fix_x")
    m.minimize((x - y) ** 2)

    repr_in = _model_repr(m)
    repr_out, stats = run_root_presolve(repr_in, eliminate=True, polynomial=False, fbbt=False)

    assert stats["elimination"]["variables_fixed"] >= 1
    assert stats["elimination"]["constraints_removed"] >= 1
    # x's bounds should be pinned to 3.0 in the new repr.
    lb = repr_out.var_lb(0)
    ub = repr_out.var_ub(0)
    assert lb[0] == pytest.approx(3.0)
    assert ub[0] == pytest.approx(3.0)


def test_propagate_bounds_to_python_model():
    """Pinned bounds in the new ``ModelRepr`` flow back into ``Model``."""
    m = discopt.Model("propagate")
    x = m.continuous("x", lb=-100.0, ub=100.0)
    y = m.continuous("y", lb=-100.0, ub=100.0)
    m.subject_to(x == 7.5, name="fix_x")
    m.subject_to(y >= -1.0)
    m.minimize(x * y)

    repr_in = _model_repr(m)
    repr_out, _ = run_root_presolve(repr_in, eliminate=True, polynomial=False, fbbt=False)
    n_tightened = propagate_bounds_to_model(m, repr_out)

    assert n_tightened >= 2  # both lb and ub of x get tightened
    assert float(np.asarray(x.lb)) == pytest.approx(7.5)
    assert float(np.asarray(x.ub)) == pytest.approx(7.5)


def test_polynomial_reformulation_runs():
    """Opt-in polynomial reformulation completes and returns stats."""
    m = discopt.Model("poly")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    # Cubic monomial that should trigger M4 reformulation.
    m.minimize(x**3 + y**3)
    m.subject_to(x + y >= 0.5)

    repr_in = _model_repr(m)
    repr_out, stats = run_root_presolve(repr_in, eliminate=False, polynomial=True, fbbt=False)
    assert "polynomial" in stats
    # Either the rewriter rewrote some cubic terms (and added aux vars)
    # or it conservatively skipped them; both outcomes are sound. We
    # assert the stats schema is intact and aux counts are consistent.
    poly = stats["polynomial"]
    assert poly["aux_variables_introduced"] >= 0
    assert poly["aux_constraints_introduced"] >= 0
    # If aux vars were introduced, the new repr must report them.
    if poly["aux_variables_introduced"] > 0:
        assert repr_out.n_var_blocks > repr_in.n_var_blocks


def test_reverse_ad_tightening_shrinks_box():
    """Reverse-mode interval AD over a square constraint shrinks the box."""
    m = discopt.Model("rad")
    x = m.continuous("x", lb=-100.0, ub=100.0)
    y = m.continuous("y", lb=-100.0, ub=100.0)
    # x**2 + y**2 <= 4 forces |x|, |y| <= 2. Use Pow nodes (x**2) rather
    # than Mul (x*x) because the reverse-AD module's power inverse fires
    # only on Pow.
    m.subject_to(x**2 + y**2 <= 4.0)
    m.minimize(x + y)

    n_tight = run_reverse_ad_tightening(m)
    assert n_tight >= 1
    assert float(np.asarray(x.ub)) <= 2.0 + 1e-6
    assert float(np.asarray(x.lb)) >= -2.0 - 1e-6
    assert float(np.asarray(y.ub)) <= 2.0 + 1e-6
    assert float(np.asarray(y.lb)) >= -2.0 - 1e-6


def test_solve_with_presolve_does_not_change_optimum():
    """End-to-end: enabling root presolve does not change the optimum."""
    m = discopt.Model("e2e")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    z = m.continuous("z", lb=-5.0, ub=5.0)
    m.subject_to(2.0 * z == 1.0, name="pin_z")  # M10 should pin z := 0.5
    m.subject_to(x + y >= 1.0)
    m.minimize((x - 1.0) ** 2 + (y - 1.0) ** 2 + z)

    res_no = m.solve(time_limit=30.0, presolve=False)
    # Re-build a fresh model since solve() may mutate state.
    m2 = discopt.Model("e2e2")
    x2 = m2.continuous("x", lb=-5.0, ub=5.0)
    y2 = m2.continuous("y", lb=-5.0, ub=5.0)
    z2 = m2.continuous("z", lb=-5.0, ub=5.0)
    m2.subject_to(2.0 * z2 == 1.0, name="pin_z")
    m2.subject_to(x2 + y2 >= 1.0)
    m2.minimize((x2 - 1.0) ** 2 + (y2 - 1.0) ** 2 + z2)
    res_yes = m2.solve(time_limit=30.0, presolve=True)

    assert res_no.status in ("optimal", "feasible")
    assert res_yes.status in ("optimal", "feasible")
    assert res_yes.objective == pytest.approx(res_no.objective, abs=1e-4, rel=1e-4)


def _build_full_pipeline_model(name: str):
    """Model exercising elimination, polynomial reformulation, and
    reverse-AD tightening simultaneously.

    - z is uniquely determined by ``2*z == 1.0``  → M10 fires.
    - Objective contains ``x**3 + y**3`` cubic monomials → M4 + M5 fires
      when ``presolve_polynomial=True``.
    - Constraint ``x**2 + y**2 <= 4`` shrinks the [-5, 5] box on x, y to
      [-2, 2] via reverse-AD when ``presolve_reverse_ad=True`` → M9.
    """
    m = discopt.Model(name)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    z = m.continuous("z", lb=-5.0, ub=5.0)
    m.subject_to(2.0 * z == 1.0, name="pin_z")
    m.subject_to(x**2 + y**2 <= 4.0, name="disk")
    m.subject_to(x + y >= 0.5)
    m.minimize(x**3 + y**3 + 0.1 * z + (x - 0.5) ** 2 + (y - 0.5) ** 2)
    return m


def test_full_pipeline_preserves_global_optimum():
    """Enabling all three new presolve passes does not change the optimum."""
    base = _build_full_pipeline_model("baseline")
    res_base = base.solve(
        time_limit=60.0,
        presolve=False,
        presolve_polynomial=False,
        presolve_reverse_ad=False,
    )
    full = _build_full_pipeline_model("full")
    res_full = full.solve(
        time_limit=60.0,
        presolve=True,
        presolve_polynomial=True,
        presolve_reverse_ad=True,
    )
    assert res_base.status in ("optimal", "feasible")
    assert res_full.status in ("optimal", "feasible")
    assert res_full.objective == pytest.approx(res_base.objective, abs=1e-3, rel=1e-3)
