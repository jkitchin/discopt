"""Tests for GNN branching policy: graph construction, forward pass, inference.

Tests cover:
  - Bipartite graph construction from a Model + solution
  - Variable/constraint feature shapes and values
  - Edge incidence correctness
  - GNN forward pass shape tests
  - Inference produces valid branching variable index
  - Fallback to most-fractional when no trained model
  - Strong branching data collection
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt.modeling.core import Model

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def simple_milp():
    """Simple MILP: min x + y, s.t. x + y >= 1, x <= 2; x cont, y binary."""
    m = Model("simple_milp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.binary("y")
    m.minimize(x + y)
    m.subject_to(x + y >= 1.0, name="cover")
    m.subject_to(x <= 2.0, name="x_bound")
    return m


@pytest.fixture
def two_integer_model():
    """MINLP with 2 integer + 1 continuous variable."""
    m = Model("two_int")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=5)
    z = m.binary("z")
    m.minimize(x + 2 * y + z)
    m.subject_to(x + y >= 3.0, name="c1")
    m.subject_to(y + z <= 4.0, name="c2")
    m.subject_to(x * z <= 5.0, name="c3")
    return m


# ─────────────────────────────────────────────────────────────
# Graph construction tests
# ─────────────────────────────────────────────────────────────


class TestGraphConstruction:
    def test_graph_shapes(self, simple_milp):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])  # x=0.5, y=0.5
        graph = build_graph(simple_milp, solution)

        assert graph.n_vars == 2
        assert graph.n_cons == 2
        assert graph.var_features.shape == (2, 7)
        assert graph.con_features.shape == (2, 5)
        assert graph.edge_indices.shape[0] == 2  # (2, n_edges)

    def test_variable_features(self, simple_milp):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.7])  # x=0.5, y=0.7
        graph = build_graph(simple_milp, solution)

        # Variable 0: x, continuous, value=0.5
        assert float(graph.var_features[0, 0]) == pytest.approx(0.5)  # value
        assert float(graph.var_features[0, 1]) == pytest.approx(0.0)  # lb
        assert float(graph.var_features[0, 2]) == pytest.approx(2.0)  # ub
        assert float(graph.var_features[0, 3]) == pytest.approx(0.0)  # not integer
        assert float(graph.var_features[0, 4]) == pytest.approx(0.0)  # no frac

        # Variable 1: y, binary, value=0.7
        assert float(graph.var_features[1, 0]) == pytest.approx(0.7)  # value
        assert float(graph.var_features[1, 1]) == pytest.approx(0.0)  # lb
        assert float(graph.var_features[1, 2]) == pytest.approx(1.0)  # ub
        assert float(graph.var_features[1, 3]) == pytest.approx(1.0)  # integer
        # fractionality of 0.7: 0.5 - |0.7-0.5| = 0.3
        assert float(graph.var_features[1, 4]) == pytest.approx(0.3)

    def test_constraint_features(self, simple_milp):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        # Should have 2 constraints
        assert graph.con_features.shape == (2, 5)
        # Each constraint has [lhs_value, slack, sense_encoding, log_n_vars, log_n_cons]
        for j in range(2):
            assert graph.con_features[j].shape == (5,)

    def test_edge_indices_valid(self, simple_milp):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        n_edges = graph.edge_indices.shape[1]
        assert n_edges > 0

        # All variable indices in [0, n_vars)
        var_idx = graph.edge_indices[0]
        assert jnp.all(var_idx >= 0)
        assert jnp.all(var_idx < graph.n_vars)

        # All constraint indices in [0, n_cons)
        con_idx = graph.edge_indices[1]
        assert jnp.all(con_idx >= 0)
        assert jnp.all(con_idx < graph.n_cons)

    def test_graph_with_custom_bounds(self, simple_milp):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        custom_lb = np.array([0.1, 0.0])
        custom_ub = np.array([1.5, 1.0])
        graph = build_graph(simple_milp, solution, node_lb=custom_lb, node_ub=custom_ub)

        assert float(graph.var_features[0, 1]) == pytest.approx(0.1)  # lb
        assert float(graph.var_features[0, 2]) == pytest.approx(1.5)  # ub

    def test_graph_multiple_constraints(self, two_integer_model):
        from discopt._jax.problem_graph import build_graph

        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)

        assert graph.n_vars == 3
        assert graph.n_cons == 3
        assert graph.var_features.shape == (3, 7)
        assert graph.con_features.shape == (3, 5)

    def test_problem_size_features(self, two_integer_model):
        """Problem-size features (log_n_vars, log_n_cons) are appended."""
        from discopt._jax.problem_graph import build_graph

        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)

        # n_vars=3, n_cons=3 -> log1p(3) for both
        expected_log_nv = np.log1p(3)
        expected_log_nc = np.log1p(3)

        # Variable features columns 5,6 = log_n_vars, log_n_cons
        for i in range(graph.n_vars):
            assert float(graph.var_features[i, 5]) == pytest.approx(expected_log_nv)
            assert float(graph.var_features[i, 6]) == pytest.approx(expected_log_nc)

        # Constraint features columns 3,4 = log_n_vars, log_n_cons
        for j in range(graph.n_cons):
            assert float(graph.con_features[j, 3]) == pytest.approx(expected_log_nv)
            assert float(graph.con_features[j, 4]) == pytest.approx(expected_log_nc)

    def test_no_constraints(self):
        """Model with no constraints should produce empty con_features."""
        from discopt._jax.problem_graph import build_graph

        m = Model("unconstrained")
        m.continuous("x", lb=0, ub=1)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])

        solution = np.array([0.5, 0.5])
        graph = build_graph(m, solution)

        assert graph.n_cons == 0
        assert graph.con_features.shape == (0, 5)
        assert graph.edge_indices.shape == (2, 0)


# ─────────────────────────────────────────────────────────────
# GNN forward pass tests
# ─────────────────────────────────────────────────────────────


class TestGNNForward:
    def test_forward_output_shape(self, simple_milp):
        from discopt._jax.gnn_policy import gnn_forward, init_gnn_params
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        key = jax.random.PRNGKey(42)
        params = init_gnn_params(key)

        scores = gnn_forward(params, graph)
        assert scores.shape == (2,)  # one score per variable
        assert scores.dtype == jnp.float64

    def test_forward_output_shape_larger(self, two_integer_model):
        from discopt._jax.gnn_policy import gnn_forward, init_gnn_params
        from discopt._jax.problem_graph import build_graph

        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)

        key = jax.random.PRNGKey(0)
        params = init_gnn_params(key)

        scores = gnn_forward(params, graph)
        assert scores.shape == (3,)

    def test_forward_deterministic(self, simple_milp):
        from discopt._jax.gnn_policy import gnn_forward, init_gnn_params
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        key = jax.random.PRNGKey(42)
        params = init_gnn_params(key)

        scores1 = gnn_forward(params, graph)
        scores2 = gnn_forward(params, graph)
        np.testing.assert_array_equal(scores1, scores2)

    def test_forward_no_constraints(self):
        """GNN should work even with zero constraints (no edges)."""
        from discopt._jax.gnn_policy import gnn_forward, init_gnn_params
        from discopt._jax.problem_graph import build_graph

        m = Model("no_cons")
        m.continuous("x", lb=0, ub=1)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])

        solution = np.array([0.5, 0.5])
        graph = build_graph(m, solution)

        key = jax.random.PRNGKey(0)
        params = init_gnn_params(key)
        scores = gnn_forward(params, graph)

        assert scores.shape == (2,)
        # Should still produce finite scores
        assert jnp.all(jnp.isfinite(scores))

    def test_params_dict_keys(self):
        from discopt._jax.gnn_policy import N_ROUNDS, init_gnn_params

        key = jax.random.PRNGKey(0)
        params = init_gnn_params(key)

        # Check expected keys
        assert "var_embed_W" in params
        assert "var_embed_b" in params
        assert "con_embed_W" in params
        assert "con_embed_b" in params
        assert "readout_W" in params
        assert "readout_b" in params
        for r in range(N_ROUNDS):
            assert f"msg_v2c_W_{r}" in params
            assert f"msg_v2c_b_{r}" in params
            assert f"msg_c2v_W_{r}" in params
            assert f"msg_c2v_b_{r}" in params


# ─────────────────────────────────────────────────────────────
# Branching inference tests
# ─────────────────────────────────────────────────────────────


class TestBranchInference:
    def test_gnn_selects_valid_variable(self, two_integer_model):
        """GNN should select a fractional integer variable."""
        from discopt._jax.gnn_policy import init_gnn_params, select_branch_variable_gnn
        from discopt._jax.problem_graph import build_graph

        # y=1.5 (fractional integer), z=0.5 (fractional binary)
        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)

        key = jax.random.PRNGKey(42)
        params = init_gnn_params(key)

        result = select_branch_variable_gnn(graph, params=params)
        assert result is not None
        # Must be one of the fractional integer vars (index 1 or 2)
        assert result in (1, 2)

    def test_fallback_most_fractional(self, two_integer_model):
        """Without trained params, falls back to most-fractional."""
        from discopt._jax.gnn_policy import select_branch_variable_gnn
        from discopt._jax.problem_graph import build_graph

        # y=1.5 (frac=0.5, score=0.5), z=0.7 (frac=0.7, score=0.3)
        solution = np.array([2.0, 1.5, 0.7])
        graph = build_graph(two_integer_model, solution)

        result = select_branch_variable_gnn(graph, params=None)
        assert result is not None
        # y at index 1 has fractionality 0.5, z at index 2 has 0.3
        assert result == 1  # most fractional

    def test_no_branch_all_integral(self, two_integer_model):
        """Should return None when all integer vars are integral."""
        from discopt._jax.gnn_policy import select_branch_variable_gnn
        from discopt._jax.problem_graph import build_graph

        # y=2.0, z=1.0 (both integral)
        solution = np.array([2.0, 2.0, 1.0])
        graph = build_graph(two_integer_model, solution)

        result = select_branch_variable_gnn(graph, params=None)
        assert result is None

    def test_fallback_with_random_tiebreak(self, simple_milp):
        """Random tie-breaking produces valid results."""
        from discopt._jax.gnn_policy import select_branch_variable_gnn
        from discopt._jax.problem_graph import build_graph

        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        key = jax.random.PRNGKey(123)
        result = select_branch_variable_gnn(graph, params=None, key=key)
        # Only y (index 1) is integer and fractional
        assert result == 1


# ─────────────────────────────────────────────────────────────
# Strong branching tests
# ─────────────────────────────────────────────────────────────


class TestStrongBranching:
    def test_basic_strong_branching(self):
        from discopt._jax.strong_branching import evaluate_strong_branching

        n_vars = 3
        solution = np.array([2.0, 1.5, 0.5])
        node_lb = np.array([0.0, 0.0, 0.0])
        node_ub = np.array([10.0, 5.0, 1.0])
        is_integer = np.array([False, True, True])
        parent_bound = 5.0

        call_count = 0

        def mock_solve(lb, ub):
            nonlocal call_count
            call_count += 1
            # Simple mock: objective = sum(midpoint)
            mid = 0.5 * (np.clip(lb, -10, 10) + np.clip(ub, -10, 10))
            return float(np.sum(mid)), True

        result = evaluate_strong_branching(
            solution=solution,
            node_lb=node_lb,
            node_ub=node_ub,
            parent_bound=parent_bound,
            is_integer=is_integer,
            solve_fn=mock_solve,
            n_vars=n_vars,
        )

        # 2 fractional integer vars (y=1.5, z=0.5) -> 4 NLP solves
        assert call_count == 4
        assert len(result.results) == 2
        assert result.best_var_index in (1, 2)
        assert result.scores_array.shape == (n_vars,)

        # Scores should be normalized to [0, 1]
        assert result.scores_array.max() == pytest.approx(1.0)
        assert result.scores_array.min() >= 0.0

        # Non-integer variable should have 0 score
        assert result.scores_array[0] == 0.0

    def test_no_candidates(self):
        from discopt._jax.strong_branching import evaluate_strong_branching

        n_vars = 2
        solution = np.array([1.0, 0.0])  # all integral
        is_integer = np.array([True, True])

        def mock_solve(lb, ub):
            return 0.0, True

        result = evaluate_strong_branching(
            solution=solution,
            node_lb=np.zeros(2),
            node_ub=np.ones(2) * 5,
            parent_bound=0.0,
            is_integer=is_integer,
            solve_fn=mock_solve,
            n_vars=n_vars,
        )

        assert len(result.results) == 0
        assert result.best_var_index is None

    def test_infeasible_child(self):
        """Test scoring when one child is infeasible."""
        from discopt._jax.strong_branching import evaluate_strong_branching

        n_vars = 2
        solution = np.array([0.5, 1.5])
        is_integer = np.array([True, True])

        def mock_solve(lb, ub):
            # First variable left child infeasible
            if ub[0] < 0.1:
                return 1e30, False
            return float(np.sum(0.5 * (lb + ub))), True

        result = evaluate_strong_branching(
            solution=solution,
            node_lb=np.zeros(2),
            node_ub=np.ones(2) * 5,
            parent_bound=3.0,
            is_integer=is_integer,
            solve_fn=mock_solve,
            n_vars=n_vars,
        )

        assert len(result.results) == 2
        # All scores should be positive
        for r in result.results:
            assert r.score > 0

    def test_max_candidates_limit(self):
        """Test that max_candidates limits the number evaluated."""
        from discopt._jax.strong_branching import evaluate_strong_branching

        n_vars = 10
        solution = np.array([i + 0.3 for i in range(10)])
        is_integer = np.ones(10, dtype=bool)
        call_count = 0

        def mock_solve(lb, ub):
            nonlocal call_count
            call_count += 1
            return 0.0, True

        result = evaluate_strong_branching(
            solution=solution,
            node_lb=np.zeros(10),
            node_ub=np.ones(10) * 20,
            parent_bound=0.0,
            is_integer=is_integer,
            solve_fn=mock_solve,
            n_vars=n_vars,
            max_candidates=3,
        )

        assert len(result.results) == 3
        assert call_count == 6  # 3 candidates * 2 children each


# ─────────────────────────────────────────────────────────────
# Equinox BranchingGNN tests (gnn_branching module)
# ─────────────────────────────────────────────────────────────


class TestEquinoxBranchingGNN:
    """Tests for the Equinox-based BranchingGNN model."""

    def test_construction_default(self):
        from discopt._jax.gnn_branching import BranchingGNN

        gnn = BranchingGNN(key=jax.random.PRNGKey(0))
        assert gnn.hidden_dim == 64
        assert gnn.n_rounds == 2

    def test_construction_custom(self):
        from discopt._jax.gnn_branching import BranchingGNN

        gnn = BranchingGNN(hidden_dim=32, n_rounds=3, key=jax.random.PRNGKey(1))
        assert gnn.hidden_dim == 32
        assert gnn.n_rounds == 3
        assert len(gnn.msg_v2c) == 3
        assert len(gnn.msg_c2v) == 3

    def test_forward_shape(self, simple_milp):
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(2))
        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)
        scores = gnn(graph)
        assert scores.shape == (2,)
        assert scores.dtype == jnp.float64

    def test_forward_shape_larger(self, two_integer_model):
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(3))
        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)
        scores = gnn(graph)
        assert scores.shape == (3,)

    def test_jit_compatible(self, simple_milp):
        import equinox as eqx
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(4))
        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        jit_gnn = eqx.filter_jit(gnn)
        scores_jit = jit_gnn(graph)
        scores_raw = gnn(graph)
        np.testing.assert_allclose(np.asarray(scores_jit), np.asarray(scores_raw), atol=1e-12)

    def test_different_seeds_different_outputs(self, simple_milp):
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn1 = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(0))
        gnn2 = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(999))
        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)
        s1 = np.asarray(gnn1(graph))
        s2 = np.asarray(gnn2(graph))
        assert not np.allclose(s1, s2, atol=1e-6)

    def test_no_constraints_model(self):
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        m = Model("unconstrained_eqx")
        m.continuous("x", lb=0, ub=1)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])

        gnn = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(5))
        solution = np.array([0.5, 0.5])
        graph = build_graph(m, solution)
        assert graph.n_cons == 0
        scores = gnn(graph)
        assert scores.shape == (2,)
        assert jnp.all(jnp.isfinite(scores))


# ─────────────────────────────────────────────────────────────
# Equinox GNN inference latency tests
# ─────────────────────────────────────────────────────────────


class TestEquinoxGNNLatency:
    def test_latency_under_threshold(self, simple_milp):
        """Equinox GNN inference < 0.1 ms after JIT warmup."""
        import time as _time

        import equinox as eqx
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=16, n_rounds=2, key=jax.random.PRNGKey(10))
        solution = np.array([0.5, 0.5])
        graph = build_graph(simple_milp, solution)

        jit_gnn = eqx.filter_jit(gnn)

        # Warmup
        for _ in range(5):
            out = jit_gnn(graph)
            jax.block_until_ready(out)

        # Measure
        times = []
        for _ in range(50):
            t0 = _time.perf_counter()
            out = jit_gnn(graph)
            jax.block_until_ready(out)
            times.append(_time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        assert median_ms < 0.1, f"Equinox GNN inference too slow: {median_ms:.3f} ms"

    def test_latency_larger_model(self, two_integer_model):
        """Equinox GNN inference < 0.1 ms on larger model."""
        import time as _time

        import equinox as eqx
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=64, n_rounds=2, key=jax.random.PRNGKey(11))
        solution = np.array([2.0, 1.5, 0.5])
        graph = build_graph(two_integer_model, solution)

        jit_gnn = eqx.filter_jit(gnn)

        for _ in range(5):
            out = jit_gnn(graph)
            jax.block_until_ready(out)

        times = []
        for _ in range(50):
            t0 = _time.perf_counter()
            out = jit_gnn(graph)
            jax.block_until_ready(out)
            times.append(_time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        assert median_ms < 0.1, f"Equinox GNN inference too slow: {median_ms:.3f} ms"


# ─────────────────────────────────────────────────────────────
# Fractional variable detection tests
# ─────────────────────────────────────────────────────────────


class TestFractionalVarDetection:
    def test_all_integral(self):
        from discopt._jax.gnn_branching import _get_fractional_integer_vars

        solution = np.array([1.0, 2.0, 0.0, 1.0])
        result = _get_fractional_integer_vars(solution, [2, 3], [1, 1])
        assert result == []

    def test_one_fractional(self):
        from discopt._jax.gnn_branching import _get_fractional_integer_vars

        solution = np.array([1.0, 2.0, 0.5, 1.0])
        result = _get_fractional_integer_vars(solution, [2, 3], [1, 1])
        assert result == [2]

    def test_multiple_fractional(self):
        from discopt._jax.gnn_branching import _get_fractional_integer_vars

        solution = np.array([1.0, 2.0, 0.3, 0.7])
        result = _get_fractional_integer_vars(solution, [2, 3], [1, 1])
        assert 2 in result
        assert 3 in result

    def test_near_integer_excluded(self):
        from discopt._jax.gnn_branching import _get_fractional_integer_vars

        solution = np.array([1.0, 2.0, 0.999999, 0.000001])
        result = _get_fractional_integer_vars(solution, [2, 3], [1, 1])
        assert result == []

    def test_array_variable_group(self):
        from discopt._jax.gnn_branching import _get_fractional_integer_vars

        # One group of 3 binary vars starting at offset 0
        solution = np.array([0.5, 0.8, 0.2, 3.0])
        result = _get_fractional_integer_vars(solution, [0], [3])
        assert 0 in result  # 0.5
        assert 1 in result  # 0.8
        assert 2 in result  # 0.2


# ─────────────────────────────────────────────────────────────
# Strong branching data collection (model-level) tests
# ─────────────────────────────────────────────────────────────


class TestCollectStrongBranching:
    def test_produces_training_pairs(self):
        """collect_strong_branching_data returns pairs for MINLP."""
        from discopt._jax.gnn_branching import collect_strong_branching_data
        from discopt._jax.problem_graph import ProblemGraph

        m = Model("sb_test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        data = collect_strong_branching_data(m, max_nodes=10)
        assert len(data) >= 1

        for graph, best_var in data:
            assert isinstance(graph, ProblemGraph)
            assert isinstance(best_var, int)
            assert 0 <= best_var < graph.n_vars

    def test_best_var_is_integer(self):
        """Strong branching should always select an integer variable."""
        from discopt._jax.gnn_branching import collect_strong_branching_data

        m = Model("sb_int_check")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        data = collect_strong_branching_data(m, max_nodes=5)
        for graph, best_var in data:
            is_int = float(graph.var_features[best_var, 3])
            assert is_int > 0.5, f"Best var {best_var} not integer"


# ─────────────────────────────────────────────────────────────
# Imitation learning training tests
# ─────────────────────────────────────────────────────────────


class TestImitationLearning:
    def test_loss_decreases(self):
        """Loss should decrease over training epochs."""
        from discopt._jax.gnn_branching import (
            BranchingGNN,
            collect_strong_branching_data,
            train_branching_gnn,
        )

        # Model with many binaries and coupling constraints so that NLP
        # relaxation produces multiple fractional candidates per node
        m = Model("il_test")
        y = m.binary("y", shape=(6,))
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(
            x**2 + 1.5 * y[0] + 2.5 * y[1] + 0.5 * y[2] + 3.0 * y[3] + 1.0 * y[4] + 4.0 * y[5]
        )
        m.subject_to(y[0] + y[1] + y[2] >= 1.5)
        m.subject_to(y[3] + y[4] + y[5] >= 1.5)
        m.subject_to(y[0] + y[3] <= 1.2)
        m.subject_to(y[1] + y[4] <= 1.3)
        m.subject_to(y[2] + y[5] <= 1.1)
        m.subject_to(x + y[0] + y[1] >= 2)

        data = collect_strong_branching_data(m, max_nodes=30)
        # Filter to pairs with >= 2 candidates for nontrivial training
        import jax.numpy as jnp

        data = [
            (g, v)
            for g, v in data
            if int(jnp.sum((g.var_features[:, 3] > 0.5) & (g.var_features[:, 4] > 0.0))) >= 2
        ]
        if len(data) < 2:
            pytest.skip("Not enough training data with multiple candidates")

        gnn = BranchingGNN(hidden_dim=16, n_rounds=2, key=jax.random.PRNGKey(42))
        trained_gnn, loss_history = train_branching_gnn(
            data,
            gnn,
            n_epochs=20,
            lr=1e-3,
        )
        assert len(loss_history) == 20
        assert loss_history[-1] < loss_history[0], (
            f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"
        )

    def test_empty_data(self):
        """Training with empty data returns unchanged model and empty history."""
        from discopt._jax.gnn_branching import BranchingGNN, train_branching_gnn

        gnn = BranchingGNN(hidden_dim=16, key=jax.random.PRNGKey(42))
        trained_gnn, loss_history = train_branching_gnn([], gnn, n_epochs=10)
        assert len(loss_history) == 0

    def test_trained_model_produces_finite_scores(self):
        """Trained GNN should produce finite scores."""
        from discopt._jax.gnn_branching import (
            BranchingGNN,
            collect_strong_branching_data,
            train_branching_gnn,
        )
        from discopt._jax.problem_graph import build_graph

        m = Model("il_scores")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        data = collect_strong_branching_data(m, max_nodes=5)
        if len(data) < 1:
            pytest.skip("Not enough training data")

        gnn = BranchingGNN(hidden_dim=16, n_rounds=2, key=jax.random.PRNGKey(42))
        trained_gnn, _ = train_branching_gnn(data, gnn, n_epochs=5, lr=1e-3)

        solution = np.array([1.0, 0.3])
        graph = build_graph(m, solution)
        scores = trained_gnn(graph)
        assert scores.shape == (2,)
        assert jnp.all(jnp.isfinite(scores))


# ─────────────────────────────────────────────────────────────
# GNNBranchingPolicy tests
# ─────────────────────────────────────────────────────────────


class TestGNNBranchingPolicy:
    def test_fallback_most_fractional(self):
        """Untrained policy falls back to most-fractional."""
        from discopt._jax.gnn_branching import GNNBranchingPolicy

        m = Model("policy_test")
        m.continuous("x", lb=0, ub=5)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])
        m.subject_to(m._variables[0] + m._variables[1] >= 1)

        policy = GNNBranchingPolicy(m, hidden_dim=16, seed=0)
        # y (index 1) is the only integer variable
        solution = np.array([1.0, 0.3])
        node_lb = np.array([0.0, 0.0])
        node_ub = np.array([5.0, 1.0])

        var_idx = policy.select(solution, node_lb, node_ub)
        assert var_idx == 1  # y is the only candidate

    def test_returns_none_all_integral(self):
        """Returns None when all integer vars are integral."""
        from discopt._jax.gnn_branching import GNNBranchingPolicy

        m = Model("policy_none")
        m.continuous("x", lb=0, ub=5)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])

        policy = GNNBranchingPolicy(m, hidden_dim=16, seed=0)
        solution = np.array([1.0, 1.0])  # y=1.0 integral
        node_lb = np.array([0.0, 0.0])
        node_ub = np.array([5.0, 1.0])

        var_idx = policy.select(solution, node_lb, node_ub)
        assert var_idx is None

    def test_train_sets_trained_flag(self):
        """Training sets the _trained flag."""
        from discopt._jax.gnn_branching import GNNBranchingPolicy

        m = Model("policy_train")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        policy = GNNBranchingPolicy(m, hidden_dim=16, seed=0)
        assert not policy._trained

        policy.train(max_nodes=5, n_epochs=5, lr=1e-3)
        assert policy._trained

    def test_train_and_select_uses_gnn(self):
        """After training, select should use GNN scores."""
        from discopt._jax.gnn_branching import GNNBranchingPolicy

        m = Model("policy_gnn_select")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        policy = GNNBranchingPolicy(m, hidden_dim=16, seed=0)
        policy.train(max_nodes=5, n_epochs=5, lr=1e-3)

        solution = np.array([1.0, 0.3])
        node_lb = np.array([0.0, 0.0])
        node_ub = np.array([5.0, 1.0])
        var_idx = policy.select(solution, node_lb, node_ub)
        # Should select y (index 1, the only integer variable)
        assert var_idx == 1

    def test_inference_latency_method(self):
        """Policy latency measurement works and is sub-millisecond."""
        from discopt._jax.gnn_branching import GNNBranchingPolicy

        m = Model("policy_latency")
        m.continuous("x", lb=0, ub=5)
        m.binary("y")
        m.minimize(m._variables[0] + m._variables[1])

        policy = GNNBranchingPolicy(m, hidden_dim=16, seed=0)
        solution = np.array([1.0, 0.3])
        node_lb = np.array([0.0, 0.0])
        node_ub = np.array([5.0, 1.0])

        latency = policy.inference_latency(
            solution,
            node_lb,
            node_ub,
            n_warmup=3,
            n_measure=10,
        )
        assert latency > 0
        assert latency < 0.001, f"Latency too high: {latency * 1000:.3f} ms"


# ─────────────────────────────────────────────────────────────
# Solver integration tests
# ─────────────────────────────────────────────────────────────


class TestSolverGNNIntegration:
    def test_solve_with_gnn_policy(self):
        """branching_policy='gnn' should produce a correct result."""
        m = Model("solver_gnn")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result = m.solve(branching_policy="gnn", max_nodes=1000)
        assert result.status in ("optimal", "feasible")

    def test_solve_gnn_vs_fractional_same_answer(self):
        """GNN and fractional policies should give same optimal value."""
        m = Model("solver_compare")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result_frac = m.solve(branching_policy="fractional", max_nodes=1000)
        result_gnn = m.solve(branching_policy="gnn", max_nodes=1000)

        if result_frac.status == "optimal" and result_gnn.status == "optimal":
            np.testing.assert_allclose(
                result_gnn.objective,
                result_frac.objective,
                atol=1e-2,
            )


# ─────────────────────────────────────────────────────────────
# Variable-size graph handling
# ─────────────────────────────────────────────────────────────


class TestEquinoxVariableSizeGraphs:
    def test_different_problem_sizes(self, simple_milp, two_integer_model):
        """Same GNN instance handles different-size problems."""
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        gnn = BranchingGNN(hidden_dim=16, n_rounds=2, key=jax.random.PRNGKey(20))

        sol1 = np.array([0.5, 0.5])
        g1 = build_graph(simple_milp, sol1)
        s1 = gnn(g1)
        assert s1.shape == (2,)

        sol2 = np.array([2.0, 1.5, 0.5])
        g2 = build_graph(two_integer_model, sol2)
        s2 = gnn(g2)
        assert s2.shape == (3,)

    def test_single_variable_model(self):
        """GNN works with a single-variable model."""
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import build_graph

        m = Model("single_eqx")
        m.binary("y")
        m.minimize(m._variables[0])

        gnn = BranchingGNN(hidden_dim=16, n_rounds=2, key=jax.random.PRNGKey(21))
        sol = np.array([0.5])
        g = build_graph(m, sol)
        assert g.n_vars == 1
        scores = gnn(g)
        assert scores.shape == (1,)
