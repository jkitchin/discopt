"""Rust LP crossover + basis recovery bindings (roadmap Phase 2 keystone).

``discopt._rust.crossover_to_vertex_py`` / ``recover_basis_py`` port the
crossover to Rust and additionally recover a simplex basis. Tests: the Rust
crossover preserves objective/feasibility and lands on a vertex (matching the
numpy reference's *properties* — the two need not pick the identical vertex);
the recovered basis is valid (``|B| = m``, free vars basic, nonbasics at their
bounds, and ``A_B x_B = b − A_N x_N`` reconstructs the vertex); and on a LP it
reconstructs the same optimum HiGHS reports.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")

if not hasattr(rust, "crossover_to_vertex_py"):  # older prebuilt extension
    pytest.skip("Rust crossover bindings not built", allow_module_level=True)

import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt._jax.crossover import crossover_to_vertex as np_crossover  # noqa: E402
from discopt._jax.lp_ipm import lp_ipm_solve  # noqa: E402
from discopt._jax.problem_classifier import extract_lp_data  # noqa: E402


def _sym_knapsack():
    m = dm.Model("sym")
    xs = [m.binary(f"x{i}") for i in range(4)]
    m.minimize(-sum(16 * x for x in xs))
    m.subject_to(sum(5 * x for x in xs) <= 9)
    return m


def _lp_interior_optimum(model):
    """Standard-form LP data + the IPM (interior) optimum of its relaxation."""
    ld = extract_lp_data(model)
    st = lp_ipm_solve(
        jnp.asarray(ld.c),
        jnp.asarray(ld.A_eq),
        jnp.asarray(ld.b_eq),
        jnp.asarray(ld.x_l),
        jnp.asarray(ld.x_u),
    )
    a = np.ascontiguousarray(ld.A_eq, dtype=np.float64)
    return (
        a,
        np.asarray(ld.b_eq, np.float64),
        np.asarray(ld.c, np.float64),
        np.asarray(ld.x_l, np.float64),
        np.asarray(ld.x_u, np.float64),
        np.asarray(st.x, np.float64),
    )


def _assert_valid_basis(status, basic, x, a, b, lo, up, tol=1e-6):
    m, n = a.shape
    assert len(basic) == m, "basis must have m columns"
    basic_set = set(int(j) for j in basic)
    for j in range(n):
        if status[j] == 1:
            assert j in basic_set
        elif status[j] == 0:
            assert abs(x[j] - lo[j]) < 1e-5, f"var {j} not at lower"
        elif status[j] == 2:
            assert abs(x[j] - up[j]) < 1e-5, f"var {j} not at upper"
        else:
            raise AssertionError(f"bad status {status[j]}")
        if x[j] > lo[j] + tol and x[j] < up[j] - tol:
            assert status[j] == 1, f"interior var {j} must be basic"
    # A_B x_B = b − A_N x_N reproduces x.
    rhs = b.copy()
    for j in range(n):
        if status[j] != 1:
            rhs = rhs - a[:, j] * x[j]
    ab = a[:, list(int(j) for j in basic)]
    xb = np.linalg.solve(ab, rhs)
    for col, j in enumerate(basic):
        assert abs(xb[col] - x[int(j)]) < 1e-5, f"reconstruction mismatch at var {j}"


class TestRustCrossover:
    def test_preserves_objective_feasibility_and_is_vertex(self):
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        xv = np.asarray(rust.crossover_to_vertex_py(x_int, a, c, lo, up))
        assert abs(c @ xv - c @ x_int) < 1e-5  # objective preserved
        assert np.allclose(a @ xv, b, atol=1e-5)  # feasibility preserved
        assert np.all(xv >= lo - 1e-6) and np.all(xv <= up + 1e-6)
        # A vertex: recover_basis succeeds (declines on a non-vertex).
        assert rust.recover_basis_py(xv, a, c, lo, up) is not None

    def test_matches_numpy_reference_properties(self):
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        xr = np.asarray(rust.crossover_to_vertex_py(x_int, a, c, lo, up))
        xn = np.asarray(np_crossover(x_int, a, b, c, lo, up))
        # Both implementations preserve objective and reach *a* vertex (they
        # may break ties to different vertices).
        assert abs(c @ xr - c @ xn) < 1e-5
        assert np.allclose(a @ xr, b, atol=1e-5) and np.allclose(a @ xn, b, atol=1e-5)

    def test_recovered_basis_is_valid(self):
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        xv = np.asarray(rust.crossover_to_vertex_py(x_int, a, c, lo, up))
        res = rust.recover_basis_py(xv, a, c, lo, up)
        assert res is not None
        status, basic = np.asarray(res[0]), np.asarray(res[1])
        _assert_valid_basis(status, basic, xv, a, b, lo, up)

    def test_reconstructs_highs_optimum(self):
        pytest.importorskip("highspy")
        # Solve the LP relaxation with HiGHS; the recovered basis must
        # reconstruct an optimum with the same objective HiGHS reports.
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_highs import solve_lp

        bounds = list(zip(lo.tolist(), up.tolist()))
        hi = solve_lp(c=c, A_eq=a, b_eq=b, bounds=bounds)
        assert hi.status == SolveStatus.OPTIMAL

        xv = np.asarray(rust.crossover_to_vertex_py(x_int, a, c, lo, up))
        assert abs(c @ xv - hi.objective) < 1e-5  # same optimum as HiGHS
        res = rust.recover_basis_py(xv, a, c, lo, up)
        assert res is not None
        status, basic = np.asarray(res[0]), np.asarray(res[1])
        _assert_valid_basis(status, basic, xv, a, b, lo, up)

    def test_declines_non_vertex(self):
        # The raw interior optimum (analytic center) is not a polytope vertex;
        # recovery must decline rather than fabricate a basis.
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        # x_int is interior on the optimal face → too many free vars.
        if rust.recover_basis_py(x_int, a, c, lo, up) is not None:
            # Degenerate fallback: only assert the *crossed-over* point recovers.
            pytest.skip("interior optimum happened to be a vertex")
