"""
GNN-based branching policy for branch-and-bound.

Implements a bipartite graph neural network for variable selection,
following the architecture of Gasse et al. (2019). Uses pure JAX
for the core computation; optionally uses Equinox for module
abstraction if available.

The GNN performs 2 rounds of message passing on the bipartite
variable-constraint graph, then produces a branching score per
variable node. The variable with the highest score among fractional
integer variables is selected for branching.

Fallback: if no trained model is available, defaults to most-fractional
branching (random tie-breaking).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from discopt._jax.problem_graph import ProblemGraph

# Feature dimensions
VAR_FEAT_DIM = 7  # [value, lb, ub, is_integer, fractionality, log_n_vars, log_n_cons]
CON_FEAT_DIM = 5  # [lhs_value, slack, sense_encoding, log_n_vars, log_n_cons]
HIDDEN_DIM = 16  # Hidden layer width (small for fast inference)
N_ROUNDS = 2  # Message-passing rounds

# Try to import Equinox; fall back to pure JAX if unavailable.
try:
    import importlib.util

    _HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
except (ImportError, ModuleNotFoundError):
    _HAS_EQUINOX = False


# ─────────────────────────────────────────────────────────────
# Pure JAX GNN (no equinox dependency)
# ─────────────────────────────────────────────────────────────


def init_gnn_params(key: jax.Array) -> dict:
    """Initialize GNN parameters with random weights.

    Architecture:
        - var_embed: (VAR_FEAT_DIM, HIDDEN_DIM) + bias
        - con_embed: (CON_FEAT_DIM, HIDDEN_DIM) + bias
        - For each round:
            - msg_v2c_W: (HIDDEN_DIM, HIDDEN_DIM) + bias
            - msg_c2v_W: (HIDDEN_DIM, HIDDEN_DIM) + bias
        - readout: (HIDDEN_DIM, 1) + bias

    Returns:
        dict of parameter arrays.
    """
    params = {}
    k1, k2, k3 = jax.random.split(key, 3)

    scale = 0.1

    # Embedding layers
    params["var_embed_W"] = scale * jax.random.normal(
        k1, (VAR_FEAT_DIM, HIDDEN_DIM), dtype=jnp.float64
    )
    params["var_embed_b"] = jnp.zeros(HIDDEN_DIM, dtype=jnp.float64)
    params["con_embed_W"] = scale * jax.random.normal(
        k2, (CON_FEAT_DIM, HIDDEN_DIM), dtype=jnp.float64
    )
    params["con_embed_b"] = jnp.zeros(HIDDEN_DIM, dtype=jnp.float64)

    # Message-passing layers
    for r in range(N_ROUNDS):
        kr1, kr2, k3 = jax.random.split(k3, 3)
        params[f"msg_v2c_W_{r}"] = scale * jax.random.normal(
            kr1, (HIDDEN_DIM, HIDDEN_DIM), dtype=jnp.float64
        )
        params[f"msg_v2c_b_{r}"] = jnp.zeros(HIDDEN_DIM, dtype=jnp.float64)
        params[f"msg_c2v_W_{r}"] = scale * jax.random.normal(
            kr2, (HIDDEN_DIM, HIDDEN_DIM), dtype=jnp.float64
        )
        params[f"msg_c2v_b_{r}"] = jnp.zeros(HIDDEN_DIM, dtype=jnp.float64)

    # Readout layer
    kr, _ = jax.random.split(k3)
    params["readout_W"] = scale * jax.random.normal(kr, (HIDDEN_DIM, 1), dtype=jnp.float64)
    params["readout_b"] = jnp.zeros(1, dtype=jnp.float64)

    return params


def gnn_forward(params: dict, graph: ProblemGraph) -> jnp.ndarray:
    """Forward pass of the bipartite GNN.

    Args:
        params: Parameter dict from init_gnn_params.
        graph: ProblemGraph with variable/constraint features and edges.

    Returns:
        scores: (n_vars,) branching score per variable.
    """
    n_vars = graph.n_vars
    n_cons = graph.n_cons

    # Embed variable and constraint features
    h_var = jnp.tanh(
        graph.var_features @ params["var_embed_W"] + params["var_embed_b"]
    )  # (n_vars, H)

    if n_cons > 0:
        h_con = jnp.tanh(
            graph.con_features @ params["con_embed_W"] + params["con_embed_b"]
        )  # (n_cons, H)
    else:
        h_con = jnp.zeros((0, HIDDEN_DIM), dtype=jnp.float64)

    edge_var = graph.edge_indices[0]  # variable indices
    edge_con = graph.edge_indices[1]  # constraint indices

    n_edges = edge_var.shape[0]

    for r in range(N_ROUNDS):
        # --- Variable -> Constraint messages ---
        if n_edges > 0 and n_cons > 0:
            # Gather variable embeddings at edges
            msg_v = h_var[edge_var]  # (n_edges, H)
            msg_v = msg_v @ params[f"msg_v2c_W_{r}"] + params[f"msg_v2c_b_{r}"]

            # Aggregate at constraint nodes (sum)
            agg_c = jnp.zeros((n_cons, HIDDEN_DIM), dtype=jnp.float64)
            agg_c = agg_c.at[edge_con].add(msg_v)

            h_con = jnp.tanh(h_con + agg_c)

            # --- Constraint -> Variable messages ---
            msg_c = h_con[edge_con]  # (n_edges, H)
            msg_c = msg_c @ params[f"msg_c2v_W_{r}"] + params[f"msg_c2v_b_{r}"]

            # Aggregate at variable nodes (sum)
            agg_v = jnp.zeros((n_vars, HIDDEN_DIM), dtype=jnp.float64)
            agg_v = agg_v.at[edge_var].add(msg_c)

            h_var = jnp.tanh(h_var + agg_v)

    # Readout: variable embeddings -> branching scores
    scores: jnp.ndarray = (h_var @ params["readout_W"] + params["readout_b"]).squeeze(
        -1
    )  # (n_vars,)
    return scores


# ─────────────────────────────────────────────────────────────
# Branching policy interface
# ─────────────────────────────────────────────────────────────


def select_branch_variable_gnn(
    graph: ProblemGraph,
    params: Optional[dict] = None,
    key: Optional[jax.Array] = None,
) -> Optional[int]:
    """Select the variable to branch on using GNN scores.

    If params is None, falls back to most-fractional branching with
    random tie-breaking.

    Args:
        graph: ProblemGraph for the current B&B node.
        params: GNN parameter dict (None = fallback).
        key: PRNG key for tie-breaking (optional).

    Returns:
        Flat variable index to branch on, or None if all integer
        variables are at integral values.
    """
    # Identify fractional integer variables
    is_int = graph.var_features[:, 3]  # is_integer column
    frac = graph.var_features[:, 4]  # fractionality column
    candidates = jnp.where((is_int > 0.5) & (frac > 0.0), 1.0, 0.0)

    if float(jnp.sum(candidates)) < 0.5:
        return None  # All integer vars are integral

    if params is not None:
        # GNN-based scoring
        scores = gnn_forward(params, graph)
        # Mask non-candidates with -inf
        masked_scores = jnp.where(candidates > 0.5, scores, -jnp.inf)
        return int(jnp.argmax(masked_scores))
    else:
        # Fallback: most fractional with optional random tie-breaking
        masked_frac = jnp.where(candidates > 0.5, frac, -1.0)
        if key is not None:
            # Add small noise for tie-breaking
            noise = 1e-8 * jax.random.uniform(key, shape=masked_frac.shape, dtype=jnp.float64)
            masked_frac = masked_frac + noise
        return int(jnp.argmax(masked_frac))
