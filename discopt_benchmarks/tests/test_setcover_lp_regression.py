"""Regression guard for the sparse set-covering LP path (issues #332 / #341).

A many-row set-covering MILP is the canonical sparse instance: its warm-started
node LPs stress the `feral` sparse-LU refactor, whose cost is O(m²) per refactor
*only* with the issue-#89 `u_above` reindex fix. Without that fix the refactor is
O(m³) in the row count — invisible on few-row knapsacks but catastrophic on the
800+ row covering LPs, where it turned a ~3 s solve into a >30 s timeout (#341
moved `feral` to the crates.io 0.11.x release, which lacks the fix; reverting to
the pinned git rev restored it).

This test pins that behaviour: a 2000-col / 800-row covering instance must solve
to **proven optimality** (not merely feasible at the deadline) with the known
objective, inside a deadline generous enough to be healthy-fast but far below the
regressed timeout. It asserts a *status* + *objective*, not a wall-clock number,
so it is not timing-flaky — a feral LU regression flips optimal→feasible and trips
it; a marginally slower machine does not.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression

dm = pytest.importorskip("discopt.modeling")
_rust = pytest.importorskip("discopt._rust")
if not hasattr(_rust, "solve_milp_py"):
    pytest.skip("discopt._rust.solve_milp_py unavailable", allow_module_level=True)


def _gen_setcover(ncol: int, nrow: int, seed: int, per_col: int = 6):
    """Deterministic random set-covering model (each column covers ~per_col rows)."""
    rng = np.random.default_rng(seed)
    cols = [rng.choice(nrow, size=per_col, replace=False) for _ in range(ncol)]
    r2c: dict[int, list[int]] = {i: [] for i in range(nrow)}
    for j, c in enumerate(cols):
        for i in c:
            r2c[i].append(j)
    for i in range(nrow):
        if not r2c[i]:
            r2c[i].append(int(rng.integers(0, ncol)))
    cost = rng.integers(1, 100, ncol).astype(float)
    m = dm.Model(f"sc{ncol}x{nrow}")
    x = m.binary("x", shape=(ncol,))
    m.minimize(dm.sum(lambda j: cost[j] * x[j], over=range(ncol)))
    m.subject_to([dm.sum(lambda j: x[j], over=r2c[i]) >= 1 for i in range(nrow)], name="cov")
    return m


def _solve_engine(model, time_limit_s: float):
    """Call the Rust MILP engine directly (no Python budget cap), like the bench."""
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.solver import _extract_variable_info

    lp = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)
    _, _, _, ioff, isz = _extract_variable_info(model)
    int_idx = [j for off, s in zip(ioff, isz, strict=False) for j in range(off, off + int(s))]
    args = (
        np.ascontiguousarray(lp.c, dtype=np.float64),
        np.ascontiguousarray(lp.A_eq, dtype=np.float64),
        np.ascontiguousarray(lp.b_eq, dtype=np.float64),
        np.ascontiguousarray(lp.x_l, dtype=np.float64),
        np.ascontiguousarray(lp.x_u, dtype=np.float64),
        np.ascontiguousarray(np.asarray(int_idx, dtype=np.int64)),
        n_orig,
        float(lp.obj_const),
        5_000_000,
        1e-6,
    )
    status, _x, obj, _bound, _nodes, _ = _rust.solve_milp_py(*args, time_limit_s=time_limit_s)
    return status, obj


def test_setcover_2000x800_solves_to_optimality():
    """sc2000x800 must prove optimality (obj=2232) well inside the deadline.

    Healthy: ~3 s. A feral-LU regression (O(m³) refactor) blows this past 30 s and
    the engine returns 'feasible' at the deadline instead of 'optimal' — the
    assertion below flips and fails. The 20 s deadline is ~6× the healthy solve and
    far below the regressed timeout, so it is robust to machine-speed variation.
    """
    model = _gen_setcover(2000, 800, seed=3)
    status, obj = _solve_engine(model, time_limit_s=20.0)
    assert status == "optimal", (
        f"expected proven optimality, got {status!r} — a sparse-LP (feral LU) "
        f"regression makes the covering node solves O(m³) and the engine only "
        f"reaches a feasible point by the deadline. See Cargo.toml feral pin."
    )
    assert abs(obj - 2232.0) <= 1e-4 * (1 + 2232.0), f"wrong optimum {obj} (expected 2232)"
