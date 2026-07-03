"""Lean full-space formulation: one variable per layer, affine+activation fused."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.nn.bounds import propagate_bounds, scaled_output_bounds
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

_ACTIVATION_FN = {
    Activation.LINEAR: lambda x: x,
    Activation.SIGMOID: dm.sigmoid,
    Activation.TANH: dm.tanh,
    Activation.SOFTPLUS: dm.softplus,
}


class ReducedSpaceFormulation:
    """A lean full-space variant: one variable per layer.

    For each layer this introduces a single post-activation variable ``z_k``
    and one equality constraint that fuses the affine transform and the
    activation into a single expression
    (``z_k[j] == act(sum_i W[i,j] * z_{k-1}[i] + b[j])``) — the pre-activation
    is *not* materialized as its own variable, unlike
    :class:`~discopt.nn.formulations.full_space.FullSpaceFormulation`, which is
    the only sense in which this formulation is "reduced". Supports all smooth
    activations plus ReLU (via ``dm.maximum``).

    Each ``z_k`` receives finite bounds from interval propagation when the
    network has ``input_bounds`` (in the *scaled* domain when a scaling is
    applied — see T-N0.2), which keeps the McCormick relaxation and NLP solves
    well-posed. The last layer's expressions are emitted directly into the
    output constraints, avoiding one redundant variable.

    True single-expression nesting (collapsing the whole network into one
    nested DAG with no per-layer variables) is deliberately *not* done: deeply
    nested DAGs slow the JAX compile, and the McCormick relaxation machinery
    needs a bounded box per intermediate anyway.
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling: OffsetScaling | None,
    ) -> None:
        self._model = model
        self._network = network
        self._prefix = prefix
        self._scaling = scaling

    def build(self) -> tuple[Variable, Variable]:
        """Add variables and constraints to the model.

        Returns (inputs, outputs) variable handles.
        """
        m = self._model
        net = self._network
        pfx = self._prefix

        # Create input variables
        if net.input_bounds is not None:
            lb, ub = net.input_bounds
            inputs = m.continuous(f"{pfx}_input", shape=(net.input_size,), lb=lb, ub=ub)
        else:
            inputs = m.continuous(f"{pfx}_input", shape=(net.input_size,))

        # Handle input scaling. The layers consume the *scaled* input, so
        # intermediate-variable bounds must be propagated over the scaled box
        # (F1 / T-N0.2); over the raw box when there is no scaling.
        layer_bounds = None
        if self._scaling is not None:
            sc = self._scaling
            if net.input_bounds is not None:
                s_lb = (net.input_bounds[0] - sc.x_offset) / sc.x_factor
                s_ub = (net.input_bounds[1] - sc.x_offset) / sc.x_factor
                s_lo = np.minimum(s_lb, s_ub)
                s_hi = np.maximum(s_lb, s_ub)
                scaled_in = m.continuous(
                    f"{pfx}_scaled_input", shape=(net.input_size,), lb=s_lo, ub=s_hi
                )
                layer_bounds = propagate_bounds(net, input_bounds=(s_lo, s_hi))
            else:
                scaled_in = m.continuous(f"{pfx}_scaled_input", shape=(net.input_size,))
            for j in range(net.input_size):
                m.subject_to(
                    scaled_in[j] == (inputs[j] - sc.x_offset[j]) / sc.x_factor[j],
                    name=f"{pfx}_scale_in_{j}",
                )
            prev = scaled_in
        else:
            if net.input_bounds is not None:
                layer_bounds = propagate_bounds(net)
            prev = inputs

        # Build one fused affine+activation variable per layer. The last
        # layer's expressions are emitted directly into the output constraints
        # (skipping a redundant final z var).
        n_layers = len(net.layers)
        last_exprs: list = []
        for k, layer in enumerate(net.layers):
            W = np.asarray(layer.weights, dtype=np.float64)
            b = np.asarray(layer.biases, dtype=np.float64)
            n_out = layer.n_outputs

            # Compute affine + activation as expressions
            new_exprs = []
            for j in range(n_out):
                # zhat_j = sum(W[i,j] * prev[i]) + b[j]
                zhat_j = (
                    dm.sum(
                        lambda i, _j=j, _W=W: _W[i, _j] * prev[i],
                        over=range(layer.n_inputs),
                    )
                    + b[j]
                )

                # Apply activation
                if layer.activation == Activation.RELU:
                    new_exprs.append(dm.maximum(zhat_j, 0))
                elif layer.activation in _ACTIVATION_FN:
                    new_exprs.append(_ACTIVATION_FN[layer.activation](zhat_j))
                else:
                    raise ValueError(f"Unsupported activation: {layer.activation}")

            if k == n_layers - 1:
                # Defer the last layer to the output constraints below.
                last_exprs = new_exprs
                break

            # Intermediate layer: one bounded variable per neuron.
            if layer_bounds is not None:
                z_lb = layer_bounds[k].post_lb
                z_ub = layer_bounds[k].post_ub
                z = m.continuous(f"{pfx}_z_{k}", shape=(n_out,), lb=z_lb, ub=z_ub)
            else:
                z = m.continuous(f"{pfx}_z_{k}", shape=(n_out,))
            for j in range(n_out):
                m.subject_to(z[j] == new_exprs[j], name=f"{pfx}_layer_{k}_{j}")

            prev = z

        # Output layer: fuse the last layer's expressions (and any output
        # scaling) directly into the output variable's defining constraints.
        if self._scaling is not None:
            sc = self._scaling
            out_lb, out_ub = scaled_output_bounds(
                layer_bounds, sc.y_offset, sc.y_factor, net.output_size
            )
            outputs = m.continuous(f"{pfx}_output", shape=(net.output_size,), lb=out_lb, ub=out_ub)
            for j in range(net.output_size):
                m.subject_to(
                    outputs[j] == last_exprs[j] * sc.y_factor[j] + sc.y_offset[j],
                    name=f"{pfx}_scale_out_{j}",
                )
        else:
            if layer_bounds is not None:
                out_lb = layer_bounds[-1].post_lb
                out_ub = layer_bounds[-1].post_ub
                outputs = m.continuous(
                    f"{pfx}_output", shape=(net.output_size,), lb=out_lb, ub=out_ub
                )
            else:
                outputs = m.continuous(f"{pfx}_output", shape=(net.output_size,))
            for j in range(net.output_size):
                m.subject_to(outputs[j] == last_exprs[j], name=f"{pfx}_layer_out_{j}")

        return inputs, outputs
