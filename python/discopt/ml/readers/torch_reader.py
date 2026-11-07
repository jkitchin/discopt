"""Import trained PyTorch models into discopt data structures."""

from __future__ import annotations

import math

import numpy as np

from discopt.ml.network import Activation, DenseLayer, NetworkDefinition


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


# `nn.Softplus(beta, threshold)` returns `x` exactly once `beta * x > threshold`
# instead of `log(1 + exp(beta x)) / beta`. The largest gap between the two is
# at the switch point, `log1p(exp(-threshold)) / beta` (2.1e-9 / beta at torch's
# default threshold of 20). discopt embeds the smooth function; it refuses a
# threshold whose linear tail would differ from it by more than this.
_SOFTPLUS_THRESHOLD_TOL = 1e-8


def _torch_activation(child, nn) -> tuple[Activation, float] | None:
    """``(Activation, beta)`` for a torch activation module, or ``None``.

    Reads every parameter that changes the function. A parameter discopt cannot
    represent raises instead of being dropped: an ignored ``beta`` embeds a
    different function and the model still solves -- to the wrong answer
    (#2135).
    """
    t = type(child)
    if t is nn.ReLU:
        return Activation.RELU, 1.0
    if t is nn.Sigmoid:
        return Activation.SIGMOID, 1.0
    if t is nn.Tanh:
        return Activation.TANH, 1.0
    if t is nn.SiLU:
        return Activation.SWISH, 1.0
    if t is nn.Softplus:
        beta = float(child.beta)
        threshold = float(child.threshold)
        if not (math.isfinite(beta) and beta > 0.0):
            raise ValueError(f"nn.Softplus with beta={beta!r} is not supported (need beta > 0)")
        gap = math.log1p(math.exp(-threshold)) / beta if threshold < 700 else 0.0
        if gap > _SOFTPLUS_THRESHOLD_TOL:
            raise ValueError(
                f"nn.Softplus(beta={beta}, threshold={threshold}) switches to its linear "
                f"tail where it differs from the smooth softplus by up to {gap:.3g}; "
                f"discopt embeds only the smooth function. Use a threshold with "
                f"log1p(exp(-threshold)) / beta <= {_SOFTPLUS_THRESHOLD_TOL:g} "
                f"(torch's default threshold=20 qualifies)."
            )
        return Activation.SOFTPLUS, beta
    if t is nn.GELU:
        approximate = getattr(child, "approximate", "none")
        if approximate != "none":
            raise ValueError(
                f"nn.GELU(approximate={approximate!r}) is not supported; only the exact "
                f"erf form (approximate='none') is embedded."
            )
        return Activation.GELU, 1.0
    return None


def load_torch_sequential(
    model,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> NetworkDefinition:
    """Convert a trained ``torch.nn.Sequential`` to a NetworkDefinition.

    Supports models composed of ``nn.Linear`` layers interleaved with
    activation layers (``nn.ReLU``, ``nn.Sigmoid``, ``nn.Tanh``,
    ``nn.Softplus`` -- its ``beta`` is carried into ``DenseLayer.beta`` --,
    ``nn.SiLU`` and exact ``nn.GELU``).

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
    from discopt._openmp_guard import check_openmp_compatible

    # On macOS a process that already holds cyipopt's libomp aborts when torch
    # loads its own (#2205); refuse with an explanation instead.
    check_openmp_compatible("torch", feature="load_torch_sequential")
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "load_torch_sequential requires PyTorch, which discopt does not install. "
            'Install it with `pip install "discopt[torch]"` (or `pip install torch`). '
            "The `nn` extra is the ONNX toolchain only and does not include torch."
        ) from exc

    layers: list[DenseLayer] = []
    pending_linear = None

    for child in model.children():
        if isinstance(child, nn.Linear):
            if pending_linear is not None:
                # Previous linear had no activation -> LINEAR
                W, b = _linear_wb(pending_linear)
                layers.append(DenseLayer(W, b, Activation.LINEAR))
            pending_linear = child

        elif (act := _torch_activation(child, nn)) is not None:
            if pending_linear is None:
                raise ValueError(f"Activation {type(child).__name__} without preceding Linear")
            W, b = _linear_wb(pending_linear)
            activation, beta = act
            layers.append(DenseLayer(W, b, activation, beta=beta))
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
