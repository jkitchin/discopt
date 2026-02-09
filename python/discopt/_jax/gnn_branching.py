"""
GNN branching policy with imitation learning for branch-and-bound.

Combines the bipartite graph builder (problem_graph.py) and GNN forward
pass (gnn_policy.py) with:
  1. An Equinox-based GNN module for clean parameter management
  2. Strong branching data collection for expert labels
  3. Imitation learning training loop (cross-entropy on expert labels)
  4. A policy wrapper that integrates with the solver B&B loop

References:
  Gasse et al. (2019), "Exact Combinatorial Optimization with Graph
  Convolutional Neural Networks", NeurIPS.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from discopt._jax.problem_graph import ProblemGraph, build_graph

INTEGRALITY_TOL = 1e-5


# -------------------------------------------------------------------
# Equinox GNN module
# -------------------------------------------------------------------


class BranchingGNN(eqx.Module):
    """Bipartite GNN for variable branching scores.

    Architecture: variable and constraint embedding MLPs, followed by
    ``n_rounds`` of bidirectional message passing (var -> con -> var),
    then a readout MLP that produces a score per variable.

    All layers use tanh activation for JIT-friendliness. The hidden
    dimension is kept small (default 64) to meet the < 0.1 ms latency
    target after JIT warmup.
    """

    var_encoder: eqx.nn.MLP
    con_encoder: eqx.nn.MLP
    msg_v2c: list[eqx.nn.Linear]
    msg_c2v: list[eqx.nn.Linear]
    readout: eqx.nn.Linear
    n_rounds: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        var_feat_dim: int = 7,
        con_feat_dim: int = 5,
        hidden_dim: int = 64,
        n_rounds: int = 2,
        *,
        key: jax.Array,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.hidden_dim = hidden_dim
        self.n_rounds = n_rounds

        self.var_encoder = eqx.nn.MLP(
            in_size=var_feat_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=k1,
        )
        self.con_encoder = eqx.nn.MLP(
            in_size=con_feat_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=k2,
        )

        msg_v2c = []
        msg_c2v = []
        for i in range(n_rounds):
            ki_v, ki_c, k4 = jax.random.split(k4, 3)
            msg_v2c.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=ki_v))
            msg_c2v.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=ki_c))
        self.msg_v2c = msg_v2c
        self.msg_c2v = msg_c2v

        k5, _ = jax.random.split(k4)
        self.readout = eqx.nn.Linear(hidden_dim, 1, key=k5)

    def __call__(self, graph: ProblemGraph) -> jnp.ndarray:
        """Compute branching scores for each variable node.

        Args:
            graph: Bipartite problem graph.

        Returns:
            scores: (n_vars,) branching score per variable (higher = better).
        """
        n_vars = graph.n_vars
        n_cons = graph.n_cons
        H = self.hidden_dim

        # Embed variable nodes
        h_var = jax.vmap(self.var_encoder)(graph.var_features)  # (n_vars, H)

        # Embed constraint nodes
        if n_cons > 0:
            h_con = jax.vmap(self.con_encoder)(graph.con_features)  # (n_cons, H)
        else:
            h_con = jnp.zeros((0, H), dtype=jnp.float64)

        edge_var = graph.edge_indices[0]
        edge_con = graph.edge_indices[1]
        n_edges = edge_var.shape[0]

        for r in range(self.n_rounds):
            if n_edges > 0 and n_cons > 0:
                # Variable -> Constraint
                msg_v = jax.vmap(self.msg_v2c[r])(h_var[edge_var])
                agg_c = jnp.zeros((n_cons, H), dtype=jnp.float64)
                agg_c = agg_c.at[edge_con].add(msg_v)
                h_con = jnp.tanh(h_con + agg_c)

                # Constraint -> Variable
                msg_c = jax.vmap(self.msg_c2v[r])(h_con[edge_con])
                agg_v = jnp.zeros((n_vars, H), dtype=jnp.float64)
                agg_v = agg_v.at[edge_var].add(msg_c)
                h_var = jnp.tanh(h_var + agg_v)

        # Readout: per-variable score
        scores: jax.Array = jax.vmap(self.readout)(h_var).squeeze(-1)  # (n_vars,)
        return scores


# -------------------------------------------------------------------
# Strong branching data collection
# -------------------------------------------------------------------


def _get_fractional_integer_vars(
    solution: np.ndarray,
    int_offsets: list[int],
    int_sizes: list[int],
) -> list[int]:
    """Return flat indices of fractional integer variables."""
    candidates = []
    for offset, size in zip(int_offsets, int_sizes):
        for k in range(size):
            idx = offset + k
            val = solution[idx]
            frac = val - np.floor(val)
            if INTEGRALITY_TOL < frac < 1.0 - INTEGRALITY_TOL:
                candidates.append(idx)
    return candidates


def collect_strong_branching_data(
    model,
    max_nodes: int = 200,
    nlp_solver: str = "ipm",
) -> list[tuple[ProblemGraph, int]]:
    """Collect expert branching data by evaluating strong branching scores.

    At each B&B node with fractional integer variables, this function:
      1. Tentatively branches on each candidate variable
      2. Solves the two child NLP relaxations (floor/ceil)
      3. Computes the score = min(lb_left - parent_lb, lb_right - parent_lb)
      4. Records (graph, best_variable_index) as a training pair

    This is expensive (2 NLP solves per candidate per node) but provides
    high-quality expert labels for imitation learning.

    Args:
        model: A discopt Model with integer variables.
        max_nodes: Maximum B&B nodes to collect from.
        nlp_solver: NLP backend ("ipm" or "ipopt").

    Returns:
        List of (ProblemGraph, best_variable_flat_index) pairs.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._rust import PyTreeManager
    from discopt.solver import (
        _extract_variable_info,
        _has_nl_repr,
        _infer_constraint_bounds,
        _infer_nl_constraint_bounds,
        _solve_node_nlp,
    )

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)

    tree = PyTreeManager(n_vars, lb.tolist(), ub.tolist(), int_offsets, int_sizes, "best_first")
    tree.initialize()

    evaluator: NLPEvaluator
    if _has_nl_repr(model):
        from discopt._jax.nl_evaluator import NLPEvaluatorFromNl

        evaluator = NLPEvaluatorFromNl(model)  # type: ignore[assignment]
        cl_list, cu_list = _infer_nl_constraint_bounds(model)
    else:
        evaluator = NLPEvaluator(model)
        cl_list, cu_list = _infer_constraint_bounds(model)

    constraint_bounds = list(zip(cl_list, cu_list)) if cl_list else None

    opts = {"print_level": 0, "max_iter": 3000, "tol": 1e-7}

    training_data: list[tuple[ProblemGraph, int]] = []
    nodes_processed = 0

    while nodes_processed < max_nodes:
        batch_lb, batch_ub, batch_ids = tree.export_batch(1)
        if len(batch_ids) == 0:
            break

        node_lb = np.array(batch_lb[0])
        node_ub = np.array(batch_ub[0])

        lb_clipped = np.clip(node_lb, -100.0, 100.0)
        ub_clipped = np.clip(node_ub, -100.0, 100.0)
        x0 = 0.5 * (lb_clipped + ub_clipped)

        parent_result = _solve_node_nlp(
            evaluator,
            x0,
            node_lb,
            node_ub,
            constraint_bounds,
            opts,
            nlp_solver=nlp_solver,
        )

        from discopt.solvers import SolveStatus

        if parent_result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
            # Infeasible node; import as infeasible and continue
            result_ids = np.array([int(batch_ids[0])], dtype=np.int64)
            result_lbs = np.array([1e30], dtype=np.float64)
            result_sols = x0.reshape(1, -1)
            result_feas = np.array([False])
            tree.import_results(result_ids, result_lbs, result_sols, result_feas)
            tree.process_evaluated()
            nodes_processed += 1
            continue

        parent_lb = parent_result.objective
        solution = parent_result.x

        # Find fractional integer variables
        candidates = _get_fractional_integer_vars(solution, int_offsets, int_sizes)

        if len(candidates) == 0:
            # Integer-feasible: import as feasible
            result_ids = np.array([int(batch_ids[0])], dtype=np.int64)
            result_lbs = np.array([parent_lb], dtype=np.float64)
            result_sols = solution.reshape(1, -1)
            result_feas = np.array([True])
            tree.import_results(result_ids, result_lbs, result_sols, result_feas)
            tree.process_evaluated()
            nodes_processed += 1
            continue

        # Strong branching: try branching on each candidate
        sb_scores = {}
        for var_idx in candidates:
            val = solution[var_idx]
            floor_val = np.floor(val)
            ceil_val = np.ceil(val)

            # Left child: x_var <= floor_val
            left_ub = node_ub.copy()
            left_ub[var_idx] = min(left_ub[var_idx], floor_val)
            lb_c = np.clip(node_lb, -100.0, 100.0)
            ub_c = np.clip(left_ub, -100.0, 100.0)
            x0_left = 0.5 * (lb_c + ub_c)

            left_result = _solve_node_nlp(
                evaluator,
                x0_left,
                node_lb,
                left_ub,
                constraint_bounds,
                opts,
                nlp_solver=nlp_solver,
            )

            # Right child: x_var >= ceil_val
            right_lb = node_lb.copy()
            right_lb[var_idx] = max(right_lb[var_idx], ceil_val)
            lb_c = np.clip(right_lb, -100.0, 100.0)
            ub_c = np.clip(node_ub, -100.0, 100.0)
            x0_right = 0.5 * (lb_c + ub_c)

            right_result = _solve_node_nlp(
                evaluator,
                x0_right,
                right_lb,
                node_ub,
                constraint_bounds,
                opts,
                nlp_solver=nlp_solver,
            )

            # Score: product of improvements (Gasse et al.)
            left_lb = (
                left_result.objective
                if left_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)
                else 1e30
            )
            right_lb = (
                right_result.objective
                if right_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)
                else 1e30
            )

            d_left = max(0.0, left_lb - parent_lb)
            d_right = max(0.0, right_lb - parent_lb)
            # Score = product (Gasse et al. use this weighting)
            score = (1e-6 + d_left) * (1e-6 + d_right)
            sb_scores[var_idx] = score

        # Best strong branching variable
        best_var = max(sb_scores, key=lambda k: sb_scores[k])

        # Build the graph for this node state
        if not _has_nl_repr(model):
            graph = build_graph(model, solution, node_lb, node_ub)
            training_data.append((graph, best_var))

        # Import results to Rust and let it branch (most-fractional)
        result_ids = np.array([int(batch_ids[0])], dtype=np.int64)
        result_lbs = np.array([parent_lb], dtype=np.float64)
        result_sols = solution.reshape(1, -1)
        result_feas = np.array([False])
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        nodes_processed += 1

        if tree.is_finished():
            break

    return training_data


# -------------------------------------------------------------------
# Imitation learning training
# -------------------------------------------------------------------


def train_branching_gnn(
    training_data: list[tuple[ProblemGraph, int]],
    gnn: BranchingGNN,
    n_epochs: int = 100,
    lr: float = 1e-3,
    *,
    key: Optional[jax.Array] = None,
) -> tuple[BranchingGNN, list[float]]:
    """Train GNN via imitation learning on strong branching expert labels.

    Loss: cross-entropy between GNN scores (over candidate fractional
    integer variables) and the strong branching expert choice.

    Args:
        training_data: List of (ProblemGraph, best_var_index) pairs.
        gnn: Initial BranchingGNN model.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        key: PRNG key (unused currently, reserved for data shuffling).

    Returns:
        Tuple of (trained_gnn, loss_history).
    """
    if len(training_data) == 0:
        return gnn, []

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(gnn, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(gnn_model, graph, target_idx):
        scores = gnn_model(graph)
        # Mask to integer variables only
        is_int = graph.var_features[:, 3]
        frac = graph.var_features[:, 4]
        mask = (is_int > 0.5) & (frac > 0.0)
        # Replace non-candidate scores with -inf for softmax
        masked_scores = jnp.where(mask, scores, -1e10)
        # Cross-entropy: log_softmax at the target index
        log_probs = jax.nn.log_softmax(masked_scores)
        return -log_probs[target_idx]

    @eqx.filter_jit
    def update_step(gnn_model, opt_state_inner, graph, target_idx):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(gnn_model, graph, target_idx)
        updates, new_opt_state = optimizer.update(
            grads, opt_state_inner, eqx.filter(gnn_model, eqx.is_array)
        )
        new_gnn = eqx.apply_updates(gnn_model, updates)
        return new_gnn, new_opt_state, loss

    loss_history = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for graph, best_var in training_data:
            target_idx = jnp.array(best_var, dtype=jnp.int32)
            gnn, opt_state, loss = update_step(gnn, opt_state, graph, target_idx)
            epoch_loss += float(loss)
        avg_loss = epoch_loss / len(training_data)
        loss_history.append(avg_loss)

    return gnn, loss_history


# -------------------------------------------------------------------
# Policy wrapper for solver integration
# -------------------------------------------------------------------


class GNNBranchingPolicy:
    """Wraps a trained BranchingGNN for use in the B&B solver loop.

    Usage:
        policy = GNNBranchingPolicy(model)
        policy.train(n_epochs=50)  # collect data + train
        var_idx = policy.select(solution, node_lb, node_ub)
    """

    def __init__(
        self,
        model,
        hidden_dim: int = 64,
        n_rounds: int = 2,
        seed: int = 0,
    ):
        self.model = model
        self._gnn = BranchingGNN(
            hidden_dim=hidden_dim,
            n_rounds=n_rounds,
            key=jax.random.PRNGKey(seed),
        )
        self._trained = False
        self._jit_forward: Optional[Callable[..., Any]] = None

    @property
    def gnn(self) -> BranchingGNN:
        return self._gnn

    def train(
        self,
        max_nodes: int = 200,
        n_epochs: int = 100,
        lr: float = 1e-3,
        nlp_solver: str = "ipm",
    ) -> list[float]:
        """Collect strong branching data and train via imitation learning.

        Returns:
            Loss history (one entry per epoch).
        """
        data = collect_strong_branching_data(
            self.model,
            max_nodes=max_nodes,
            nlp_solver=nlp_solver,
        )
        if len(data) == 0:
            return []
        self._gnn, loss_history = train_branching_gnn(
            data,
            self._gnn,
            n_epochs=n_epochs,
            lr=lr,
        )
        self._trained = True
        self._jit_forward = None  # reset JIT cache
        return loss_history

    def select(
        self,
        solution: np.ndarray,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
    ) -> Optional[int]:
        """Select branching variable using the GNN.

        Falls back to most-fractional if the GNN is not trained.

        Args:
            solution: Current relaxation solution (n_vars,).
            node_lb: Node lower bounds (n_vars,).
            node_ub: Node upper bounds (n_vars,).

        Returns:
            Flat variable index to branch on, or None if integer-feasible.
        """
        graph = build_graph(self.model, solution, node_lb, node_ub)

        # Identify fractional integer candidates
        is_int = np.asarray(graph.var_features[:, 3])
        frac = np.asarray(graph.var_features[:, 4])
        mask = (is_int > 0.5) & (frac > 0.0)

        if not np.any(mask):
            return None

        if self._trained:
            if self._jit_forward is None:
                self._jit_forward = eqx.filter_jit(self._gnn)
            forward = self._jit_forward
            assert forward is not None
            scores = np.asarray(forward(graph))
            masked_scores = np.where(mask, scores, -np.inf)
            return int(np.argmax(masked_scores))
        else:
            # Fallback: most fractional
            masked_frac = np.where(mask, frac, -1.0)
            return int(np.argmax(masked_frac))

    def inference_latency(
        self,
        solution: np.ndarray,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        n_warmup: int = 5,
        n_measure: int = 50,
    ) -> float:
        """Measure GNN inference latency in seconds (post-JIT).

        Args:
            solution, node_lb, node_ub: Sample inputs.
            n_warmup: JIT warmup iterations.
            n_measure: Measurement iterations.

        Returns:
            Median inference time in seconds.
        """
        graph = build_graph(self.model, solution, node_lb, node_ub)
        if self._jit_forward is None:
            self._jit_forward = eqx.filter_jit(self._gnn)
        forward = self._jit_forward
        assert forward is not None

        # Warmup
        for _ in range(n_warmup):
            _ = forward(graph)
            jax.block_until_ready(_)

        # Measure
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            out = forward(graph)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        return float(np.median(times))
