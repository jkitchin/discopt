"""Interval arithmetic bound propagation through neural networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.nn.network import Activation, NetworkDefinition

# Sentinel "unbounded" values matching Model.continuous defaults; used when no
# layer bounds were propagated so downstream vars stay effectively free.
_DEFAULT_LB = -9.999e19
_DEFAULT_UB = 9.999e19


@dataclass
class LayerBounds:
    """Pre- and post-activation bounds for a single layer."""

    pre_lb: np.ndarray
    pre_ub: np.ndarray
    post_lb: np.ndarray
    post_ub: np.ndarray


def _apply_activation_bounds(
    activation: Activation, pre_lb: np.ndarray, pre_ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply activation function to pre-activation bounds (monotone functions)."""
    if activation == Activation.LINEAR:
        return pre_lb.copy(), pre_ub.copy()
    elif activation == Activation.RELU:
        return np.maximum(pre_lb, 0.0), np.maximum(pre_ub, 0.0)
    elif activation == Activation.SIGMOID:
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return sig(pre_lb), sig(pre_ub)
    elif activation == Activation.TANH:
        return np.tanh(pre_lb), np.tanh(pre_ub)
    elif activation == Activation.SOFTPLUS:
        sp = lambda z: np.logaddexp(z, 0)  # noqa: E731
        return sp(pre_lb), sp(pre_ub)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def propagate_bounds(
    network: NetworkDefinition,
    *,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[LayerBounds]:
    """Propagate input bounds through the network via interval arithmetic.

    Uses natural interval extension: split weights into positive and negative
    parts to compute tight affine bounds, then apply monotone activation bounds.

    Parameters
    ----------
    network : NetworkDefinition
        Network with ``input_bounds`` set.
    input_bounds : tuple of (np.ndarray, np.ndarray), optional
        ``(lb, ub)`` box to propagate instead of ``network.input_bounds``. When
        given, layer 1 consumes *this* box; use it to propagate over the domain
        the layers actually see (e.g. the scaled input box when an
        :class:`~discopt.nn.scaling.OffsetScaling` is applied — see T-N0.2 /
        finding F1). Defaults to ``network.input_bounds`` (behavior-preserving).

    Returns
    -------
    list of LayerBounds
        Bounds for each layer (pre- and post-activation).

    Raises
    ------
    ValueError
        If neither ``input_bounds`` nor ``network.input_bounds`` is set.
    """
    if input_bounds is not None:
        lb, ub = input_bounds
    elif network.input_bounds is not None:
        lb, ub = network.input_bounds
    else:
        raise ValueError("input_bounds must be set for bound propagation")

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)

    result: list[LayerBounds] = []

    for layer in network.layers:
        W = layer.weights
        b = layer.biases

        # Natural interval extension for affine: zhat = W^T @ x + b
        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        pre_lb = W_pos.T @ lb + W_neg.T @ ub + b
        pre_ub = W_pos.T @ ub + W_neg.T @ lb + b

        post_lb, post_ub = _apply_activation_bounds(layer.activation, pre_lb, pre_ub)

        result.append(LayerBounds(pre_lb, pre_ub, post_lb, post_ub))

        # Post-activation bounds become input bounds for the next layer
        lb, ub = post_lb, post_ub

    return result


def scaled_output_bounds(
    layer_bounds: list[LayerBounds] | None,
    y_offset: np.ndarray,
    y_factor: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-safe bounds for affinely scaled network outputs (T-N0.4).

    The scaled output is ``outputs[j] = y_factor[j] * z[j] + y_offset[j]`` where
    ``z`` is the last layer's post-activation. Given that layer's propagated
    ``post_lb``/``post_ub``, return element-wise ``(lb, ub)`` accounting for the
    sign of ``y_factor``. When ``layer_bounds is None`` (no propagation), return
    the effectively-unbounded sentinel box, preserving prior behavior.

    This is a pure strengthening: it can never exclude a reachable prediction
    because it maps the exact interval hull of the last layer's outputs.
    """
    if layer_bounds is None:
        return (
            np.full(output_size, _DEFAULT_LB, dtype=np.float64),
            np.full(output_size, _DEFAULT_UB, dtype=np.float64),
        )
    last = layer_bounds[-1]
    yf = np.asarray(y_factor, dtype=np.float64)
    yo = np.asarray(y_offset, dtype=np.float64)
    a = yf * last.post_lb
    b = yf * last.post_ub
    lb = np.minimum(a, b) + yo
    ub = np.maximum(a, b) + yo
    return lb, ub
