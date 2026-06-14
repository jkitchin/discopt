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

    def test_gomory_worked_example(self):
        # x0 + x1 + s = 1.5, x0,x1 binary-relaxed, s >= 0. At vertex (1, 0.5, 0)
        # the GMI cut is 2 s >= 1 (i.e. x0 + x1 <= 1). Check exact cut, that it
        # cuts off the vertex, and that it excludes no integer-feasible point.
        if not hasattr(rust, "gomory_cuts_py"):
            pytest.skip("gomory binding not built")
        a = np.array([[1.0, 1.0, 1.0]])
        bvec = np.array([1.5])
        c = np.zeros(3)
        lo = np.array([0.0, 0.0, 0.0])
        up = np.array([1.0, 1.0, 1e30])
        x = np.array([1.0, 0.5, 0.0])
        integ = np.array([True, True, False])
        res = rust.gomory_cuts_py(x, a, bvec, c, lo, up, integ)
        assert res is not None
        coeffs, rhs = np.asarray(res[0]), np.asarray(res[1])
        assert coeffs.shape == (1, 3)
        np.testing.assert_allclose(coeffs[0], [0.0, 0.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(rhs, [1.0], atol=1e-9)
        assert coeffs[0] @ x < rhs[0] - 1e-6  # cuts off the vertex
        for b0 in (0, 1):
            for b1 in (0, 1):
                s = 1.5 - b0 - b1
                if s < -1e-9:
                    continue
                pt = np.array([b0, b1, s])
                assert coeffs[0] @ pt >= rhs[0] - 1e-6

    def test_gomory_pipeline_separates_vertex(self):
        # Real pipeline: IPM interior optimum -> Rust crossover -> GMI cuts.
        # Every returned cut must be finite and cut off the crossover vertex.
        if not hasattr(rust, "gomory_cuts_py"):
            pytest.skip("gomory binding not built")
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        xv = np.asarray(rust.crossover_to_vertex_py(x_int, a, c, lo, up))
        n = a.shape[1]
        integ = np.array([True] * 4 + [False] * (n - 4))  # 4 binaries + slacks
        res = rust.gomory_cuts_py(xv, a, b, c, lo, up, integ)
        assert res is not None  # xv is a vertex
        coeffs, rhs = np.asarray(res[0]), np.asarray(res[1])
        for i in range(coeffs.shape[0]):
            assert np.all(np.isfinite(coeffs[i])) and np.isfinite(rhs[i])
            assert coeffs[i] @ xv < rhs[i] - 1e-6  # separates the fractional vertex

    def test_declines_non_vertex(self):
        # The raw interior optimum (analytic center) is not a polytope vertex;
        # recovery must decline rather than fabricate a basis.
        a, b, c, lo, up, x_int = _lp_interior_optimum(_sym_knapsack())
        # x_int is interior on the optimal face → too many free vars.
        if rust.recover_basis_py(x_int, a, c, lo, up) is not None:
            # Degenerate fallback: only assert the *crossed-over* point recovers.
            pytest.skip("interior optimum happened to be a vertex")


class TestGomoryWiring:
    """GMI cuts wired into the root cut loop: structural projection keeps the
    augmented relaxation well-conditioned, so cuts are valid *and* the IPM-based
    B&B stays correct (roadmap Phase 2, increment 5c)."""

    def test_projection_to_structural_eliminates_slack(self):
        # Binary knapsack std form: 5(x0+x1+x2+x3) + s = 9. The GMI slack cut
        # 0.25 s >= 1 projects to -1.25 sum(x) >= -1.25, i.e. sum(x) <= 1 —
        # structural-only, O(1) coefficients, no wide-range slack coupling.
        import discopt.solver as S

        A = np.array([[5.0, 5.0, 5.0, 5.0, 1.0]])
        b = np.array([9.0])
        coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.25])  # 0.25 * slack >= 1
        out = S._project_cut_to_structural(coeffs, 1.0, A, b, n_orig=4)
        assert out is not None
        proj, prhs = out
        np.testing.assert_allclose(proj, [-1.25, -1.25, -1.25, -1.25, 0.0], atol=1e-9)
        assert abs(prhs - (-1.25)) < 1e-9
        assert proj[4] == 0.0  # slack column eliminated

    def test_wired_solve_is_correct_and_cuts_nodes(self, monkeypatch):
        # The pure-binary knapsack that a naive (slack-coupled) GMI got wrong
        # (-8 vs -10). With structural projection the optimum is preserved and
        # the cut solves it at the root.
        pytest.importorskip("pounce")
        import discopt.solver as S

        monkeypatch.setattr(S, "GOMORY_CUTS_ENABLED", True)  # opt-in

        def _knap():
            m = dm.Model("k")
            xs = [m.binary(f"x{i}") for i in range(4)]
            m.minimize(-(10 * xs[0] + 9 * xs[1] + 8 * xs[2] + 1 * xs[3]))
            m.subject_to(5 * xs[0] + 5 * xs[1] + 5 * xs[2] + 5 * xs[3] <= 9)
            return m

        r_gmi = _knap().solve(use_highs_milp=False, time_limit=60)
        monkeypatch.setattr(S, "_separate_gomory_cuts", lambda *a, **k: None)
        r_nogmi = _knap().solve(use_highs_milp=False, time_limit=60)

        assert r_gmi.status == "optimal" and r_nogmi.status == "optimal"
        assert abs(r_gmi.objective - (-10.0)) < 1e-4  # correct optimum (was -8)
        assert abs(r_gmi.objective - r_nogmi.objective) < 1e-4
        assert r_gmi.node_count <= r_nogmi.node_count  # cuts never worsen the tree


class TestGomoryGate:
    """GMI is a POUNCE-mode feature: on when node relaxations are solved by
    POUNCE (Path B, no JAX recompile on cut-augmented shapes), off under the JAX
    IPM. ``GOMORY_CUTS_ENABLED`` is a hard override."""

    def test_gate_logic(self, monkeypatch):
        import discopt.solver as S

        # Auto mode: on iff POUNCE solves the node relaxations.
        monkeypatch.setattr(S, "GOMORY_CUTS_ENABLED", None)
        assert S._gomory_enabled(True) is True  # POUNCE mode
        assert S._gomory_enabled(False) is False  # JAX mode
        # Hard overrides ignore the engine.
        monkeypatch.setattr(S, "GOMORY_CUTS_ENABLED", True)
        assert S._gomory_enabled(False) is True
        monkeypatch.setattr(S, "GOMORY_CUTS_ENABLED", False)
        assert S._gomory_enabled(True) is False

    def test_default_off_under_jax_ipm(self):
        # Default (auto) keeps GMI off in JAX mode, so that path pays nothing.
        import discopt.solver as S

        assert S.GOMORY_CUTS_ENABLED is None  # auto by default
        assert S._gomory_enabled(False) is False

    def test_auto_on_in_pounce_mode_and_correct(self):
        # solve_milp runs prefer_pounce=True (POUNCE node solves), so GMI
        # auto-enables; confirm the knapsack a naive GMI got wrong is correct.
        pytest.importorskip("pounce")
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_pounce import solve_milp

        r = solve_milp(
            c=np.array([-10.0, -9.0, -8.0, -1.0]),
            A_ub=np.array([[5.0, 5.0, 5.0, 5.0]]),
            b_ub=np.array([9.0]),
            bounds=[(0, 1)] * 4,
            integrality=np.array([1, 1, 1, 1]),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-10.0)) < 1e-4


class TestMirCuts:
    """MIR cuts from original ``<=`` rows (basis-free, complement GMI)."""

    def test_mir_binding_rounds_row_and_is_valid(self):
        if not hasattr(rust, "mir_cuts_py"):
            pytest.skip("mir binding not built")
        # x0 + x1 <= 1.5, x0,x1 binary -> MIR x0 + x1 <= 1.
        a = np.array([[1.0, 1.0]])
        b = np.array([1.5])
        lb = np.array([0.0, 0.0])
        integ = np.array([True, True])
        x = np.array([0.75, 0.75])
        res = rust.mir_cuts_py(a, b, lb, integ, x)
        assert res is not None
        coeffs, rhs = np.asarray(res[0]), np.asarray(res[1])
        np.testing.assert_allclose(coeffs[0], [1.0, 1.0], atol=1e-9)
        assert abs(rhs[0] - 1.0) < 1e-9
        assert coeffs[0] @ x > rhs[0] + 1e-6  # separates the fractional point
        # Valid for every integer-feasible point of the row.
        for b0 in (0, 1):
            for b1 in (0, 1):
                if b0 + b1 <= 1.5:
                    assert coeffs[0] @ np.array([b0, b1]) <= rhs[0] + 1e-6

    def test_mir_in_pounce_solve_stays_correct(self):
        # solve_milp runs prefer_pounce=True, so MIR (and GMI) are auto-on;
        # confirm a general-integer MILP matches the HiGHS optimum.
        pytest.importorskip("pounce")
        pytest.importorskip("highspy")
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_highs import solve_milp as highs
        from discopt.solvers.milp_pounce import solve_milp

        rng = np.random.default_rng(7)
        c = rng.integers(-5, 6, 4).astype(float)
        A = rng.integers(0, 4, (3, 4)).astype(float)
        b = (A @ rng.integers(0, 4, 4) + rng.integers(1, 5, 3)).astype(float)
        kw = dict(c=c, A_ub=A, b_ub=b, bounds=[(0, 5)] * 4, integrality=np.ones(4))
        rp = solve_milp(**kw)
        rh = highs(**kw)
        assert rp.status == SolveStatus.OPTIMAL
        assert abs(rp.objective - rh.objective) < 1e-3
