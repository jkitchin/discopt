"""Import trained PyTorch models into discopt data structures."""

from __future__ import annotations

import numpy as np

from discopt.nn.network import Activation, DenseLayer, NetworkDefinition


def _linear_wb(linear) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(W, b)`` from a ``torch.nn.Linear`` as float64 arrays.

    ``W`` is transposed to ``(n_in, n_out)`` (discopt's convention). When the
    layer was built with ``bias=False`` (``linear.bias is None``), a zero bias
    of length ``n_out`` is returned instead of dereferencing ``None``.
    """
    W = linear.weight.detach().cpu().numpy().T.astype(np.float64)
    if linear.bias is None:
        b = np.zeros(W.shape[1], dtype=np.float64)
    else:
        b = linear.bias.detach().cpu().numpy().astype(np.float64)
    return W, b


def load_torch_sequential(
    model,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> NetworkDefinition:
    """Convert a trained ``torch.nn.Sequential`` to a NetworkDefinition.

    Supports models composed of ``nn.Linear`` layers interleaved with
    activation layers (``nn.ReLU``, ``nn.Sigmoid``, ``nn.Tanh``,
    ``nn.Softplus``).

    Parameters
    ----------
    model : torch.nn.Sequential
        Trained PyTorch sequential model.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on input features.

    Returns
    -------
    NetworkDefinition
    """
    import torch.nn as nn

    _TORCH_ACT_MAP = {
        nn.ReLU: Activation.RELU,
        nn.Sigmoid: Activation.SIGMOID,
        nn.Tanh: Activation.TANH,
        nn.Softplus: Activation.SOFTPLUS,
    }

    layers: list[DenseLayer] = []
    pending_linear = None

    for child in model.children():
        if isinstance(child, nn.Linear):
            if pending_linear is not None:
                # Previous linear had no activation -> LINEAR
                W, b = _linear_wb(pending_linear)
                layers.append(DenseLayer(W, b, Activation.LINEAR))
            pending_linear = child

        elif type(child) in _TORCH_ACT_MAP:
            if pending_linear is None:
                raise ValueError(f"Activation {type(child).__name__} without preceding Linear")
            W, b = _linear_wb(pending_linear)
            layers.append(DenseLayer(W, b, _TORCH_ACT_MAP[type(child)]))
            pending_linear = None

        elif isinstance(child, (nn.Flatten, nn.Dropout, nn.Identity)):
            continue
        else:
            raise ValueError(f"Unsupported layer type: {type(child).__name__}")

    # Handle trailing linear without activation
    if pending_linear is not None:
        W, b = _linear_wb(pending_linear)
        layers.append(DenseLayer(W, b, Activation.LINEAR))

    if not layers:
        raise ValueError("No Linear layers found in the Sequential model")

    return NetworkDefinition(layers, input_bounds=input_bounds)
