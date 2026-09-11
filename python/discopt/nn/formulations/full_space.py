"""Full-space formulation for smooth activation neural networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.nn.bounds import propagate_bounds, scaled_output_bounds
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

_SMOOTH_ACTIVATIONS = {Activation.LINEAR, Activation.SIGMOID, Activation.TANH, Activation.SOFTPLUS}

_ACTIVATION_FN = {
    Activation.SIGMOID: dm.sigmoid,
    Activation.TANH: dm.tanh,
    Activation.SOFTPLUS: dm.softplus,
}


def _family_name(prefix: str, n_rows: int) -> str:
    """Family name that expands to the SAME row names the per-element loop wrote.

    A family of `n` rows named `c` is written as `c_0 ... c_{n-1}`, so passing the
    bare prefix reproduces the loop's names exactly -- except when the family has
    a single row, which the writers emit under the family name unsuffixed. That
    would silently rename `pred_affine_2_0` to `pred_affine_2` in LP/MPS/GAMS for
    any single-output layer, so the index is appended explicitly in that case.

    `.nl` carries no row names and is unaffected either way; this exists so the
    three formats that DO carry them keep byte-identical output (#1215 §55).
    """
    return f"{prefix}_0" if n_rows == 1 else prefix


class FullSpaceFormulation:
    """Full-space formulation with explicit pre/post-activation variables.

    For each layer, creates:
    - ``zhat`` (pre-activation) variables with affine constraints
    - ``z`` (post-activation) variables with activation constraints

    Supports LINEAR, SIGMOID, TANH, SOFTPLUS activations only.
    For ReLU, use :class:`ReluBigMFormulation`.
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

        unsupported = {
            layer.activation
            for layer in network.layers
            if layer.activation not in _SMOOTH_ACTIVATIONS
        }
        if unsupported:
            raise ValueError(
                f"FullSpaceFormulation does not support activations: "
                f"{sorted(str(getattr(a, 'value', a)) for a in unsupported)}. "
                f"Use 'relu_bigm' for ReLU."
            )

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

        # Handle input scaling. The layers consume the *scaled* input, so bounds
        # must be propagated over the scaled box, not net.input_bounds (F1 /
        # T-N0.2). When scaling is None, the layers consume the raw inputs.
        layer_bounds = None
        if self._scaling is not None:
            sc = self._scaling
            if net.input_bounds is not None:
                s_lb = (lb - sc.x_offset) / sc.x_factor
                s_ub = (ub - sc.x_offset) / sc.x_factor
                # Handle negative factors (swaps lb/ub)
                s_lo = np.minimum(s_lb, s_ub)
                s_hi = np.maximum(s_lb, s_ub)
                scaled_in = m.continuous(
                    f"{pfx}_scaled_input", shape=(net.input_size,), lb=s_lo, ub=s_hi
                )
                layer_bounds = propagate_bounds(net, input_bounds=(s_lo, s_hi))
            else:
                scaled_in = m.continuous(f"{pfx}_scaled_input", shape=(net.input_size,))
            # One array-valued body for the whole family, not one Constraint per
            # unit (#1215). The family name expands to the same per-row names the
            # loop wrote (`{pfx}_scale_in_0`, `_1`, ...), so LP/MPS/GAMS output is
            # unchanged and `.nl` -- which carries no names -- is byte-identical.
            m.subject_to(
                scaled_in == (inputs - sc.x_offset) / sc.x_factor,
                name=_family_name(f"{pfx}_scale_in", net.input_size),
            )
            prev_z = scaled_in
        else:
            if net.input_bounds is not None:
                layer_bounds = propagate_bounds(net)
            prev_z = inputs

        # Build each layer
        for k, layer in enumerate(net.layers):
            n_out = layer.n_outputs
            W = layer.weights
            b = layer.biases

            # Pre-activation bounds
            if layer_bounds is not None:
                zhat_lb = layer_bounds[k].pre_lb
                zhat_ub = layer_bounds[k].pre_ub
                z_lb = layer_bounds[k].post_lb
                z_ub = layer_bounds[k].post_ub
            else:
                zhat_lb, zhat_ub = None, None
                z_lb, z_ub = None, None

            # Pre-activation variables
            zhat = m.continuous(
                f"{pfx}_zhat_{k}",
                shape=(n_out,),
                lb=zhat_lb if zhat_lb is not None else -1e20,
                ub=zhat_ub if zhat_ub is not None else 1e20,
            )

            # Affine constraints: zhat = W^T @ prev_z + b
            W_const = np.asarray(W, dtype=np.float64)
            b_const = np.asarray(b, dtype=np.float64)
            # The layer's affine map as ONE array-valued body. The loop this
            # replaces built `n_out * n_inputs` Python expression objects per
            # layer, which is why a 128-wide layer cost 77 us/row to build and
            # 274 to write (#1215 §55).
            #
            # Written as a broadcast product reduced along the input axis rather
            # than as `W_const.T @ prev_z`, which is the same mathematics and
            # LOSES THE CERTIFICATE: a `MatMulExpression` relaxes more weakly than
            # the equivalent expanded sum, so the 1x1x1 sigmoid net in
            # `test_nn_equivalence` went from `optimal` in 1 node to `feasible` in
            # 121 with an identical incumbent. Same objective, weaker bound --
            # exactly the bound-changing effect CLAUDE.md §5 says must not ship by
            # accident. This form gives the same arena node count as the
            # per-element loop and certifies identically.
            m.subject_to(
                zhat == dm.sum(W_const.T * prev_z, axis=1) + b_const,
                name=_family_name(f"{pfx}_affine_{k}", n_out),
            )

            # Post-activation variables and constraints
            if layer.activation == Activation.LINEAR:
                # z = zhat, no separate variable needed
                z = zhat
            else:
                z = m.continuous(
                    f"{pfx}_z_{k}",
                    shape=(n_out,),
                    lb=z_lb if z_lb is not None else -1e20,
                    ub=z_ub if z_ub is not None else 1e20,
                )
                act_fn = _ACTIVATION_FN[layer.activation]
                # Every activation in `_ACTIVATION_FN` is elementwise over an
                # array body, so the whole layer is one constraint.
                m.subject_to(z == act_fn(zhat), name=_family_name(f"{pfx}_act_{k}", n_out))

            prev_z = z

        # Handle output scaling
        if self._scaling is not None:
            sc = self._scaling
            # Free output bounds (T-N0.4): the last layer's propagated
            # post-activation bounds map through the affine output scaling.
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
