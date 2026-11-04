"""Neural network data model for embedding trained models in discopt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Activation(Enum):
    """Supported activation functions.

    SOFTPLUS, SQUAREPLUS, SWISH and GELU take a sharpness parameter ``beta``
    (``DenseLayer.beta``); see :mod:`discopt.ml.activations` for the formulas.
    SWISH (a.k.a. SiLU) and GELU are not monotone.
    """

    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTPLUS = "softplus"
    SQUAREPLUS = "squareplus"
    SWISH = "swish"
    GELU = "gelu"

    @classmethod
    def _missing_(cls, value):
        # "silu" is the PyTorch name for Swish at beta = 1.
        if isinstance(value, str) and value.lower() == "silu":
            return cls.SWISH
        return None


@dataclass
class DenseLayer:
    """A fully-connected layer with weights, biases, and an activation.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of shape ``(n_in, n_out)``.
    biases : np.ndarray
        Bias vector of shape ``(n_out,)``.
    activation : Activation or str
        Activation function applied after the affine transform. A plain string
        (``"tanh"``, ``"relu"``, ...) is accepted and normalised to the enum.
    beta : float, default 1.0
        Sharpness of a SOFTPLUS / SQUAREPLUS / SWISH / GELU activation:
        ``f_beta(z) = g(beta * z) / beta`` (e.g. softplus becomes
        ``log(1 + exp(beta z)) / beta``, matching ``torch.nn.Softplus(beta)``).
        Any other activation refuses ``beta != 1``.
    """

    weights: np.ndarray
    biases: np.ndarray
    activation: Activation
    beta: float = 1.0

    def __post_init__(self) -> None:
        # Normalise a string activation to the enum. Without this,
        # `DenseLayer(..., activation="tanh")` -- the form the readers and the
        # docstring use -- stored the bare string, and every downstream
        # membership test against a set of `Activation` members then reported the
        # layer as UNSUPPORTED. `FullSpaceFormulation` did exactly that and then
        # crashed formatting its own error (`a.value` on a `str`), so a valid
        # smooth network was refused with an `AttributeError` instead of being
        # built.
        if not isinstance(self.activation, Activation):
            try:
                self.activation = Activation(self.activation)
            except ValueError:
                valid = ", ".join(repr(a.value) for a in Activation)
                raise ValueError(
                    f"Unknown activation {self.activation!r}; expected one of {valid} "
                    f"or an Activation member."
                ) from None
        from discopt.ml.activations import check_beta

        self.beta = check_beta(self.activation, self.beta)
        if self.weights.ndim != 2:
            raise ValueError(f"weights must be 2-D, got shape {self.weights.shape}")
        if self.biases.ndim != 1:
            raise ValueError(f"biases must be 1-D, got shape {self.biases.shape}")
        if self.weights.shape[1] != self.biases.shape[0]:
            raise ValueError(
                f"weights columns ({self.weights.shape[1]}) must match "
                f"biases length ({self.biases.shape[0]})"
            )

    @property
    def n_inputs(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_outputs(self) -> int:
        return int(self.weights.shape[1])


@dataclass
class NetworkDefinition:
    """A sequential feedforward neural network.

    Parameters
    ----------
    layers : list of DenseLayer
        Ordered list of dense layers from input to output.
    input_bounds : tuple of (np.ndarray, np.ndarray) or None
        ``(lb, ub)`` bounds on each input feature. Required for formulations
        that need bound propagation (e.g. ReluBigM).
    """

    layers: list[DenseLayer]
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Network must have at least one layer")
        for i in range(1, len(self.layers)):
            prev_out = self.layers[i - 1].n_outputs
            curr_in = self.layers[i].n_inputs
            if prev_out != curr_in:
                raise ValueError(
                    f"Layer {i} expects {curr_in} inputs but layer {i - 1} has {prev_out} outputs"
                )
        if self.input_bounds is not None:
            lb, ub = self.input_bounds
            if lb.shape != (self.input_size,) or ub.shape != (self.input_size,):
                raise ValueError(
                    f"input_bounds must have shape ({self.input_size},), "
                    f"got lb={lb.shape}, ub={ub.shape}"
                )

    @property
    def input_size(self) -> int:
        return self.layers[0].n_inputs

    @property
    def output_size(self) -> int:
        return self.layers[-1].n_outputs

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the network on input ``x`` using numpy (for testing)."""
        from discopt.ml.activations import evaluate

        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = h @ layer.weights + layer.biases
            h = evaluate(layer.activation, h, layer.beta)
        return h
