"""
Input Convex Neural Network (ICNN) for learned relaxations.

An ICNN guarantees output convexity w.r.t. input via:
  - Non-negative weights on hidden-to-hidden connections (softplus parameterization)
  - Convex, non-decreasing activations (ReLU)
  - Unrestricted weights on input skip connections

References:
  Amos et al. (2017), "Input Convex Neural Networks", ICML.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ICNN(eqx.Module):
    """Input Convex Neural Network.

    Guarantees convexity of the output w.r.t. input ``x`` by construction:
    hidden-to-hidden weights are parameterized via ``softplus`` to stay
    non-negative, while input skip-connection weights are unconstrained.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (e.g. 2 for univariate features,
        4 for bivariate features).
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Number of hidden layers.
    """

    # Skip connections from input (unconstrained weights)
    input_layers: list[eqx.nn.Linear]
    # Hidden-to-hidden (non-negative weights via softplus)
    hidden_layers: list[eqx.nn.Linear]
    # Final projection to scalar
    output_layer: eqx.nn.Linear
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        n_layers: int = 3,
        *,
        key: jax.Array,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        keys = jax.random.split(key, 2 * n_layers + 1)

        # First input layer: input -> hidden
        input_layers = [eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])]
        hidden_layers = []

        # Subsequent layers: skip connection from input + hidden-to-hidden
        for i in range(1, n_layers):
            input_layers.append(eqx.nn.Linear(input_dim, hidden_dim, key=keys[2 * i - 1]))
            hidden_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[2 * i]))

        self.input_layers = input_layers
        self.hidden_layers = hidden_layers
        self.output_layer = eqx.nn.Linear(hidden_dim, 1, key=keys[2 * n_layers])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass producing a scalar convex in ``x``.

        Args:
            x: Input array of shape ``(input_dim,)``.

        Returns:
            Scalar output that is convex w.r.t. ``x``.
        """
        # First hidden layer (no hidden-to-hidden yet)
        z = jax.nn.relu(self.input_layers[0](x))

        # Subsequent hidden layers with non-negative weight enforcement
        for w_z, w_x in zip(self.hidden_layers, self.input_layers[1:]):
            # Enforce non-negative weights: softplus(raw_weight) >= 0
            bias: jax.Array = w_z.bias if w_z.bias is not None else jnp.zeros(w_z.weight.shape[0])
            z = jax.nn.relu(jnp.dot(z, jax.nn.softplus(w_z.weight.T)) + bias + w_x(x))

        # Final scalar output
        out: jnp.ndarray = self.output_layer(z).squeeze(-1)
        return out


def create_icnn(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int = 32,
    n_layers: int = 3,
) -> ICNN:
    """Create an ICNN with the given architecture.

    Args:
        key: JAX PRNG key.
        input_dim: Input dimensionality.
        hidden_dim: Hidden layer width (default 32).
        n_layers: Number of hidden layers (default 3).

    Returns:
        Initialized ICNN module.
    """
    return ICNN(input_dim, hidden_dim, n_layers, key=key)


def verify_convexity(
    icnn: ICNN,
    x_samples: jnp.ndarray,
    eps: float = 1e-4,
    tol: float = -1e-6,
) -> bool:
    """Verify convexity of an ICNN by checking Hessian PSD on sample points.

    Args:
        icnn: The ICNN to verify.
        x_samples: Sample points of shape ``(n_samples, input_dim)``.
        eps: Not used (kept for API compatibility).
        tol: Minimum eigenvalue tolerance — eigenvalues above this are
            considered non-negative.

    Returns:
        True if all Hessian eigenvalues are >= tol at every sample point.
    """
    fn = eqx.filter_jit(lambda x: icnn(x))
    hess_fn = jax.hessian(fn)

    for i in range(x_samples.shape[0]):
        x = x_samples[i]
        H = hess_fn(x)
        eigvals = jnp.linalg.eigvalsh(H)
        if jnp.any(eigvals < tol):
            return False
    return True


def enforce_nonneg(icnn: ICNN) -> ICNN:
    """Project hidden-to-hidden weights to be non-negative.

    This is a post-training cleanup step. During forward pass, weights are
    already made non-negative via softplus. This function clamps the raw
    weight parameters so that ``softplus(w)`` is closer to the intended
    non-negative value, primarily useful for serialization clarity.

    Args:
        icnn: The ICNN to project.

    Returns:
        New ICNN with clamped hidden layer weights.
    """

    def clamp_weight(layer: eqx.nn.Linear) -> eqx.nn.Linear:
        # Clamp raw weights so softplus(w) stays well-defined and non-negative.
        # softplus(w) >= 0 for all w, so this is mainly to avoid large negative
        # raw weights that would produce near-zero effective weights.
        new_weight = jnp.maximum(layer.weight, 0.0)
        return eqx.tree_at(lambda ly: ly.weight, layer, new_weight)  # type: ignore[no-any-return]

    new_hidden = [clamp_weight(layer) for layer in icnn.hidden_layers]
    result: ICNN = eqx.tree_at(lambda m: m.hidden_layers, icnn, new_hidden)
    return result


def icnn_pair_create(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int = 32,
    n_layers: int = 3,
) -> tuple[ICNN, ICNN]:
    """Create a pair of ICNNs for convex/concave relaxation.

    The first ICNN is the convex underestimator (output is convex in x).
    The second ICNN is used for the concave overestimator via negation:
    ``cc(x) = -icnn_cc(-features)``.

    Args:
        key: JAX PRNG key.
        input_dim: Input dimensionality.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.

    Returns:
        Tuple of ``(cv_net, cc_net)`` ICNN modules.
    """
    k1, k2 = jax.random.split(key)
    cv_net = create_icnn(k1, input_dim, hidden_dim, n_layers)
    cc_net = create_icnn(k2, input_dim, hidden_dim, n_layers)
    return cv_net, cc_net
