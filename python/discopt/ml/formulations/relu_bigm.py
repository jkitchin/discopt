"""Big-M MILP formulation for ReLU neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.ml.activations import symbolic
from discopt.ml.bounds import LayerBounds, propagate_bounds, scaled_output_bounds
from discopt.ml.formulations.full_space import _family_name
from discopt.ml.network import Activation, NetworkDefinition
from discopt.ml.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

# Activations supported in non-ReLU layers within a ReluBigM network
_SUPPORTED = {
    Activation.LINEAR,
    Activation.RELU,
    Activation.SIGMOID,
    Activation.TANH,
    Activation.SOFTPLUS,
    Activation.SQUAREPLUS,
    Activation.SWISH,
    Activation.GELU,
}


@dataclass
class ReluLayerRecord:
    """One embedded ReLU layer, as the ideal-cut separator needs it.

    ``binaries[j]`` is the indicator of neuron ``j`` (``None`` for a stable
    neuron, which has none). ``in_lb``/``in_ub`` is the box of ``x`` the layer
    reads; every Anderson cut is derived from it, and it is a *global* box (the
    bounds the variables carry), so the cuts are globally valid.
    """

    layer_index: int
    z: Variable
    x: Variable
    weights: np.ndarray
    biases: np.ndarray
    in_lb: np.ndarray
    in_ub: np.ndarray
    pre_lb: np.ndarray
    pre_ub: np.ndarray
    binaries: list


def _find_binaries(m: Model, pfx: str, k: int, bounds: LayerBounds) -> list:
    """Look up the per-neuron indicators ``{pfx}_q_{k}_{j}`` just created."""
    by_name = {v.name: v for v in m._variables}
    out = []
    for j in range(len(bounds.pre_lb)):
        out.append(by_name.get(f"{pfx}_q_{k}_{j}"))
    return out


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
            m.subject_to(
                scaled_in == (inputs - sc.x_offset) / sc.x_factor,
                name=_family_name(f"{pfx}_scale_in", net.input_size),
            )
            prev_z = scaled_in
            in_lo, in_hi = s_lo, s_hi
        else:
            layer_bounds = propagate_bounds(net)
            prev_z = inputs
            in_lo = np.asarray(lb, dtype=np.float64)
            in_hi = np.asarray(ub, dtype=np.float64)
        self.layer_bounds = layer_bounds

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

            # Affine constraints: zhat = W^T @ prev_z + b, as ONE array body.
            #
            # `dm.sum(W.T * prev_z, axis=1)` and NOT `W.T @ prev_z`: same
            # mathematics, and the matmul form relaxes more weakly -- on
            # `full_space` it turned a certified optimum into an uncertified
            # feasible point (#1215 §55).
            m.subject_to(
                zhat == dm.sum(W.T * prev_z, axis=1) + b,
                name=_family_name(f"{pfx}_affine_{k}", n_out),
            )

            if layer.activation == Activation.RELU:
                z = self._relu_layer(m, zhat, prev_z, W, b, in_lo, in_hi, bounds, k)
            elif layer.activation == Activation.LINEAR:
                z = zhat
            else:
                # Smooth activation (sigmoid, tanh, softplus, squareplus, swish,
                # gelu); the post-activation box is the exact interval image,
                # including the interior minimum of the non-monotone ones.
                z = m.continuous(
                    f"{pfx}_z_{k}",
                    shape=(n_out,),
                    lb=bounds.post_lb,
                    ub=bounds.post_ub,
                )
                m.subject_to(
                    z == symbolic(layer.activation, zhat, layer.beta),
                    name=_family_name(f"{pfx}_act_{k}", n_out),
                )

            prev_z = z
            # The box the next layer consumes: this layer's post-activation
            # interval bounds, which are exactly the bounds `z` carries.
            in_lo, in_hi = bounds.post_lb, bounds.post_ub

        # Handle output scaling
        if self._scaling is not None:
            sc = self._scaling
            # Free output bounds (T-N0.4) from the last layer's post-activation.
            out_lb, out_ub = scaled_output_bounds(
                layer_bounds, sc.y_offset, sc.y_factor, net.output_size
            )
            outputs = m.continuous(f"{pfx}_output", shape=(net.output_size,), lb=out_lb, ub=out_ub)
            m.subject_to(
                outputs == prev_z * sc.y_factor + sc.y_offset,
                name=_family_name(f"{pfx}_scale_out", net.output_size),
            )
        else:
            outputs = prev_z

        return inputs, outputs

    def _relu_layer(
        self,
        m: Model,
        zhat: Variable,
        prev_z: Variable,
        W: np.ndarray,
        b: np.ndarray,
        in_lb: np.ndarray,
        in_ub: np.ndarray,
        bounds: LayerBounds,
        k: int,
    ) -> Variable:
        """Encode one ReLU layer; returns its post-activation variable.

        The hook subclasses override to change the per-neuron encoding
        (:mod:`discopt.ml.formulations.relu_partition`). ``prev_z`` is the
        layer's input variable, ``in_lb``/``in_ub`` the box it lives in (the
        bounds ``prev_z`` carries), ``W``/``b`` the layer's affine map. Big-M
        needs only the pre-activation bounds.
        """
        z = self._add_relu_constraints(
            m, zhat, bounds.pre_lb, bounds.pre_ub, k, zhat.shape[0], self._prefix
        )
        self._record_relu_layer(k, z, prev_z, W, b, in_lb, in_ub, bounds)
        return z

    def _record_relu_layer(
        self,
        k: int,
        z: Variable,
        prev_z: Variable,
        W: np.ndarray,
        b: np.ndarray,
        in_lb: np.ndarray,
        in_ub: np.ndarray,
        bounds: LayerBounds,
    ) -> None:
        """Remember a ReLU layer's handles for the Anderson cut separator."""
        if not hasattr(self, "relu_layers"):
            self.relu_layers: list[ReluLayerRecord] = []
        self.relu_layers.append(
            ReluLayerRecord(
                layer_index=k,
                z=z,
                x=prev_z,
                weights=W,
                biases=b,
                in_lb=np.asarray(in_lb, dtype=np.float64),
                in_ub=np.asarray(in_ub, dtype=np.float64),
                pre_lb=np.asarray(bounds.pre_lb, dtype=np.float64),
                pre_ub=np.asarray(bounds.pre_ub, dtype=np.float64),
                binaries=_find_binaries(self._model, self._prefix, k, bounds),
            )
        )

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
