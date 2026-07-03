"""Big-M MILP formulation for ReLU neural networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.nn.bounds import propagate_bounds, scaled_output_bounds
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

# Activations supported in non-ReLU layers within a ReluBigM network
_SUPPORTED = {
    Activation.LINEAR,
    Activation.RELU,
    Activation.SIGMOID,
    Activation.TANH,
    Activation.SOFTPLUS,
}

_SMOOTH_FN = {
    Activation.SIGMOID: dm.sigmoid,
    Activation.TANH: dm.tanh,
    Activation.SOFTPLUS: dm.softplus,
}


class ReluBigMFormulation:
    """Big-M formulation for networks containing ReLU activations.

    For ReLU neurons, introduces binary variables and linear big-M constraints.
    Smooth activations in non-ReLU layers are handled as in FullSpaceFormulation.

    Requires ``network.input_bounds`` for big-M constant computation via
    interval arithmetic bound propagation.
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

        if network.input_bounds is None:
            raise ValueError("ReluBigMFormulation requires input_bounds for big-M constants")

        unsupported = {
            layer.activation for layer in network.layers if layer.activation not in _SUPPORTED
        }
        if unsupported:
            raise ValueError(f"Unsupported activations: {[a.value for a in unsupported]}")

    def build(self) -> tuple[Variable, Variable]:
        """Add variables and constraints to the model.

        Returns (inputs, outputs) variable handles.
        """
        m = self._model
        net = self._network
        pfx = self._prefix

        # Create input variables
        lb, ub = net.input_bounds  # type: ignore[misc]
        inputs = m.continuous(f"{pfx}_input", shape=(net.input_size,), lb=lb, ub=ub)

        # Handle input scaling. The ReLU big-M constants must come from the box
        # the layers actually consume: the *scaled* input when a scaling is
        # applied, else the raw input box (F1 / T-N0.2).
        if self._scaling is not None:
            sc = self._scaling
            s_lb = (lb - sc.x_offset) / sc.x_factor
            s_ub = (ub - sc.x_offset) / sc.x_factor
            s_lo = np.minimum(s_lb, s_ub)
            s_hi = np.maximum(s_lb, s_ub)
            layer_bounds = propagate_bounds(net, input_bounds=(s_lo, s_hi))
            scaled_in = m.continuous(
                f"{pfx}_scaled_input", shape=(net.input_size,), lb=s_lo, ub=s_hi
            )
            for j in range(net.input_size):
                m.subject_to(
                    scaled_in[j] == (inputs[j] - sc.x_offset[j]) / sc.x_factor[j],
                    name=f"{pfx}_scale_in_{j}",
                )
            prev_z = scaled_in
        else:
            layer_bounds = propagate_bounds(net)
            prev_z = inputs

        for k, layer in enumerate(net.layers):
            n_out = layer.n_outputs
            W = np.asarray(layer.weights, dtype=np.float64)
            b = np.asarray(layer.biases, dtype=np.float64)
            bounds = layer_bounds[k]

            # Pre-activation variables
            zhat = m.continuous(
                f"{pfx}_zhat_{k}",
                shape=(n_out,),
                lb=bounds.pre_lb,
                ub=bounds.pre_ub,
            )

            # Affine constraints: zhat = W^T @ prev_z + b
            for j in range(n_out):
                lhs = dm.sum(
                    lambda i, _j=j, _W=W: _W[i, _j] * prev_z[i],
                    over=range(layer.n_inputs),
                )
                m.subject_to(zhat[j] == lhs + b[j], name=f"{pfx}_affine_{k}_{j}")

            if layer.activation == Activation.RELU:
                z = self._add_relu_constraints(m, zhat, bounds.pre_lb, bounds.pre_ub, k, n_out, pfx)
            elif layer.activation == Activation.LINEAR:
                z = zhat
            else:
                # Smooth activation (sigmoid, tanh, softplus)
                z = m.continuous(
                    f"{pfx}_z_{k}",
                    shape=(n_out,),
                    lb=bounds.post_lb,
                    ub=bounds.post_ub,
                )
                act_fn = _SMOOTH_FN[layer.activation]
                for j in range(n_out):
                    m.subject_to(
                        z[j] == act_fn(zhat[j]),
                        name=f"{pfx}_act_{k}_{j}",
                    )

            prev_z = z

        # Handle output scaling
        if self._scaling is not None:
            sc = self._scaling
            # Free output bounds (T-N0.4) from the last layer's post-activation.
            out_lb, out_ub = scaled_output_bounds(
                layer_bounds, sc.y_offset, sc.y_factor, net.output_size
            )
            outputs = m.continuous(f"{pfx}_output", shape=(net.output_size,), lb=out_lb, ub=out_ub)
            for j in range(net.output_size):
                m.subject_to(
                    outputs[j] == prev_z[j] * sc.y_factor[j] + sc.y_offset[j],
                    name=f"{pfx}_scale_out_{j}",
                )
        else:
            outputs = prev_z

        return inputs, outputs

    @staticmethod
    def _add_relu_constraints(
        m: Model,
        zhat: Variable,
        pre_lb: np.ndarray,
        pre_ub: np.ndarray,
        k: int,
        n_out: int,
        pfx: str,
    ) -> Variable:
        """Add big-M ReLU constraints for a layer.

        For each neuron j:
        - If lb >= 0: always active → z = zhat
        - If ub <= 0: always inactive → z = 0
        - Otherwise: binary variable q with 4 big-M constraints
        """
        z_lb = np.maximum(pre_lb, 0.0)
        z_ub = np.maximum(pre_ub, 0.0)
        z = m.continuous(f"{pfx}_z_{k}", shape=(n_out,), lb=z_lb, ub=z_ub)

        for j in range(n_out):
            lb_j = float(pre_lb[j])
            ub_j = float(pre_ub[j])

            if lb_j >= 0:
                # Always active
                m.subject_to(z[j] == zhat[j], name=f"{pfx}_relu_active_{k}_{j}")
            elif ub_j <= 0:
                # Always inactive
                m.subject_to(z[j] == 0, name=f"{pfx}_relu_inactive_{k}_{j}")
            else:
                # Mixed: need binary variable
                q = m.binary(f"{pfx}_q_{k}_{j}")
                m.subject_to(z[j] >= 0, name=f"{pfx}_relu_lb_{k}_{j}")
                m.subject_to(z[j] >= zhat[j], name=f"{pfx}_relu_ge_{k}_{j}")
                m.subject_to(z[j] <= ub_j * q, name=f"{pfx}_relu_ub_{k}_{j}")
                m.subject_to(
                    z[j] <= zhat[j] - lb_j * (1 - q),
                    name=f"{pfx}_relu_bigm_{k}_{j}",
                )

        return z
