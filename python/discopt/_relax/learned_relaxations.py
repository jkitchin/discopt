"""
Learned relaxations: ICNN-based convex/concave envelopes per operation.

Each LearnedRelaxation wraps a pair of ICNNs matching the McCormick
signature ``(x, lb, ub) -> (cv, cc)``. Runtime soundness enforcement
guarantees ``cv <= f(x) <= cc`` even if the network slightly violates.

Operations covered (6 total):
  1. exp   — univariate, input_dim=1
  2. log   — univariate, input_dim=1
  3. sqrt  — univariate, input_dim=1
  4. sin   — univariate, input_dim=1
  5. square — univariate, input_dim=1
  6. bilinear (x*y) — bivariate, input_dim=2
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from discopt._relax.icnn import ICNN, icnn_pair_create


class LearnedRelaxation(eqx.Module):
    """Learned relaxation for a single operation (e.g. exp, bilinear).

    Wraps a pair of ICNNs:
      - ``cv_net``: convex underestimator (ICNN output is convex by construction)
      - ``cc_net``: concave overestimator (via ``-ICNN(-features)``)

    Runtime soundness enforcement ensures ``cv <= f(x) <= cc``.
    """

    cv_net: ICNN
    cc_net: ICNN
    op_name: str = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)

    def __call__(
        self,
        x: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        true_val: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate learned relaxation with soundness enforcement.

        Args:
            x: Point(s) at which to evaluate.
            lb: Lower bound(s).
            ub: Upper bound(s).
            true_val: True function value ``f(x)`` for soundness clamping.

        Returns:
            ``(cv, cc)`` where ``cv <= f(x) <= cc``.
        """
        # Normalize x to [0, 1] for network stability
        width = jnp.maximum(ub - lb, 1e-15)
        x_norm = (x - lb) / width

        # Feature vector: [normalized_x, bound_width]
        if self.input_dim == 1:
            features = jnp.stack([x_norm, width])
        else:
            # Bivariate: x and y are already stacked by caller
            features = jnp.concatenate([x_norm, width])

        cv_pred = self.cv_net(features)
        cc_pred = -self.cc_net(-features)  # concave via negated ICNN

        # Runtime soundness enforcement (safety net)
        cv = jnp.minimum(cv_pred, true_val)  # cv <= f(x) guaranteed
        cc = jnp.maximum(cc_pred, true_val)  # cc >= f(x) guaranteed
        return cv, cc


# ---------------------------------------------------------------------------
# True function references for each operation
# ---------------------------------------------------------------------------

_TRUE_FNS: dict[str, Callable] = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "square": lambda x: x**2,
    "bilinear": lambda xy: xy[0] * xy[1],
}

# Operation configurations: (input_dim, hidden_dim)
_OP_CONFIGS: dict[str, tuple[int, int]] = {
    "exp": (1, 32),
    "log": (1, 32),
    "sqrt": (1, 32),
    "sin": (1, 32),
    "square": (1, 32),
    "bilinear": (2, 64),
}

# Feature dimensionality: input_dim * 2 (normalized x + width)
_FEATURE_DIMS: dict[str, int] = {
    "exp": 2,  # [x_norm, width]
    "log": 2,
    "sqrt": 2,
    "sin": 2,
    "square": 2,
    "bilinear": 4,  # [x_norm, y_norm, x_width, y_width]
}


def create_learned_relaxation(
    key: jax.Array,
    op_name: str,
    n_layers: int = 3,
) -> LearnedRelaxation:
    """Create an untrained LearnedRelaxation for the given operation.

    Args:
        key: JAX PRNG key.
        op_name: Operation name (one of: exp, log, sqrt, sin, square, bilinear).
        n_layers: Number of ICNN hidden layers.

    Returns:
        Initialized (untrained) LearnedRelaxation.
    """
    if op_name not in _OP_CONFIGS:
        raise ValueError(f"Unknown operation: {op_name!r}. Must be one of {list(_OP_CONFIGS)}")

    input_dim, hidden_dim = _OP_CONFIGS[op_name]
    feat_dim = _FEATURE_DIMS[op_name]
    cv_net, cc_net = icnn_pair_create(key, feat_dim, hidden_dim, n_layers)
    return LearnedRelaxation(cv_net=cv_net, cc_net=cc_net, op_name=op_name, input_dim=input_dim)


# ---------------------------------------------------------------------------
# Per-operation wrapper functions matching McCormick signatures
# ---------------------------------------------------------------------------


def relax_exp_learned(
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of exp(x) on [lb, ub]."""
    true_val = jnp.exp(x)
    return learned(x, lb, ub, true_val)


def relax_log_learned(
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of log(x) on [lb, ub]."""
    true_val = jnp.log(x)
    return learned(x, lb, ub, true_val)


def relax_sqrt_learned(
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of sqrt(x) on [lb, ub]."""
    true_val = jnp.sqrt(x)
    return learned(x, lb, ub, true_val)


def relax_sin_learned(
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of sin(x) on [lb, ub]."""
    true_val = jnp.sin(x)
    return learned(x, lb, ub, true_val)


def relax_square_learned(
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of x^2 on [lb, ub]."""
    true_val = x**2
    return learned(x, lb, ub, true_val)


def relax_bilinear_learned(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_lb: jnp.ndarray,
    x_ub: jnp.ndarray,
    y_lb: jnp.ndarray,
    y_ub: jnp.ndarray,
    learned: LearnedRelaxation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Learned relaxation of x*y given bounds on x and y."""
    true_val = x * y
    xy = jnp.stack([x, y])
    xy_lb = jnp.stack([x_lb, y_lb])
    xy_ub = jnp.stack([x_ub, y_ub])
    return learned(xy, xy_lb, xy_ub, true_val)


# ---------------------------------------------------------------------------
# Registry for storing/loading trained relaxations
# ---------------------------------------------------------------------------


class LearnedRelaxationRegistry:
    """Registry of trained per-operation relaxations.

    Stores :class:`LearnedRelaxation` instances keyed by operation name.
    Supports save/load via equinox serialization.
    """

    def __init__(self, relaxations: Optional[dict[str, LearnedRelaxation]] = None):
        self.relaxations: dict[str, LearnedRelaxation] = relaxations or {}

    def get(self, op_name: str) -> Optional[LearnedRelaxation]:
        """Get the learned relaxation for an operation, or None."""
        return self.relaxations.get(op_name)

    def register(self, op_name: str, relaxation: LearnedRelaxation) -> None:
        """Register a trained relaxation for an operation."""
        self.relaxations[op_name] = relaxation

    def save(self, directory: str) -> None:
        """Save all trained relaxations to a directory.

        Each operation is saved as ``{directory}/{op_name}.eqx``.
        """
        os.makedirs(directory, exist_ok=True)
        for op_name, relaxation in self.relaxations.items():
            path = os.path.join(directory, f"{op_name}.eqx")
            eqx.tree_serialise_leaves(path, relaxation)

    @staticmethod
    def load(directory: str) -> LearnedRelaxationRegistry:
        """Load a registry from a directory of ``.eqx`` files.

        Scans for files matching known operation names.
        """
        registry = LearnedRelaxationRegistry()
        if not os.path.isdir(directory):
            return registry

        for op_name in _OP_CONFIGS:
            path = os.path.join(directory, f"{op_name}.eqx")
            if os.path.isfile(path):
                # Create a skeleton model for deserialization
                skeleton = create_learned_relaxation(jax.random.PRNGKey(0), op_name)
                loaded = eqx.tree_deserialise_leaves(path, skeleton)
                registry.register(op_name, loaded)
        return registry

    def __len__(self) -> int:
        return len(self.relaxations)

    def __contains__(self, op_name: str) -> bool:
        return op_name in self.relaxations


def load_pretrained_registry(
    path: Optional[str] = None,
) -> LearnedRelaxationRegistry:
    """Load pretrained relaxation models.

    Args:
        path: Directory containing ``.eqx`` files. If None, uses the
            default location ``discopt/_relax/pretrained/``.

    Returns:
        Registry with any available pretrained relaxations.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "pretrained")

    if not os.path.isdir(path):
        warnings.warn(
            f"No pretrained relaxation models found at {path}. "
            "Train with icnn_trainer.train_all() first.",
            stacklevel=2,
        )
        return LearnedRelaxationRegistry()

    return LearnedRelaxationRegistry.load(path)
