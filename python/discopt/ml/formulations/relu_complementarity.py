"""Complementarity formulation for ReLU networks (#2167, slice 1).

Each *unstable* ReLU neuron ``y = max(0, z)`` (pre-activation interval
``l < 0 < u``) is encoded as a first-class
:class:`~discopt.mpec.Complementarity` relation instead of an indicator binary
and big-M rows::

    y - s == z,   y >= 0,   s >= 0,   0 <= y  ⊥  s >= 0

``s`` is the negative part of ``z``. The pair ``(y, s)`` with ``y·s = 0`` and
``y - s = z`` has exactly one solution, ``y = max(0, z)``, ``s = max(0, -z)``,
so the feasible set is the graph of the network -- the same set ``relu_bigm``
encodes -- with no binary and no big-M constant.

Stable neurons keep ``relu_bigm``'s treatment (``y = z`` when ``l >= 0``,
``y = 0`` when ``u <= 0``), and non-ReLU layers are emitted exactly as there.

The strategy only **declares** the relations; it does not lower them. They are
registered on the model (``Model._complementarities``), so a plain
``Model.solve()`` refuses loudly until a lowering is chosen -- never solves the
network as if the ReLU were absent. Choose the solve mode downstream::

    nn = NNFormulation(m, net, strategy="relu_complementarity")
    nn.formulate()
    res = solve_mpec(m, nn.relations, method="sos1")      # exact (binaries)
    res = solve_mpec(m, nn.relations, method="gdp")       # exact (disjunctions)
    res = solve_mpec(m, nn.relations, method="scholtes")  # local, no certificate

Bounds. Unlike ``relu_bigm`` the encoding needs no finite bound to be correct,
so ``network.input_bounds`` may be ``None`` (every pre-activation is then
unbounded and every ReLU neuron unstable). An *exact* lowering of today
(``sos1``/``gdp``) still needs finite operand bounds and refuses otherwise; the
local Scholtes mode needs none.

The relation's role is :attr:`~discopt.mpec.ComplementarityRole.FROM_ACTIVATION`
with ``parent`` naming the network layer and ``index`` the neuron. Role is
provenance only: every lowering branches on the declared bounds ``(0, inf)``,
so ReLU pairs need no special case downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from discopt import mpec
from discopt.ml.activations import activation_interval
from discopt.ml.bounds import LayerBounds
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.network import NetworkDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

#: The magnitude ``Model.continuous`` treats as "unbounded" (its default bound).
_UNBOUNDED = 9.999e19


def _var_bound(a: np.ndarray) -> np.ndarray:
    """Map ``±inf`` onto the model's unbounded sentinel for a variable declaration."""
    return np.clip(np.asarray(a, dtype=np.float64), -_UNBOUNDED, _UNBOUNDED)


def _to_inf(a: np.ndarray) -> np.ndarray:
    """Map sentinel-magnitude entries (``|v| >= 9.999e19``) onto ``±inf``."""
    a = np.asarray(a, dtype=np.float64).copy()
    a[a >= _UNBOUNDED] = np.inf
    a[a <= -_UNBOUNDED] = -np.inf
    return a


def _interval_affine(
    W: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Natural interval extension of ``W^T x + b`` that is safe for infinite boxes.

    ``propagate_bounds`` computes ``W_pos.T @ lb``, which is ``nan`` as soon as a
    zero weight meets an infinite bound (``0 * inf``). A zero weight contributes
    exactly nothing, so it is masked out here before the sum.
    """
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    with np.errstate(invalid="ignore"):
        lo_terms = np.where(W_pos != 0, W_pos * lb[:, None], 0.0) + np.where(
            W_neg != 0, W_neg * ub[:, None], 0.0
        )
        hi_terms = np.where(W_pos != 0, W_pos * ub[:, None], 0.0) + np.where(
            W_neg != 0, W_neg * lb[:, None], 0.0
        )
    # A column holding both +inf and -inf terms cannot occur: within one sum all
    # lower terms are <= finite or -inf, all upper terms >= finite or +inf.
    return lo_terms.sum(axis=0) + b, hi_terms.sum(axis=0) + b


def _propagate_unbounded_safe(
    network: NetworkDefinition, lb: np.ndarray, ub: np.ndarray
) -> list[LayerBounds]:
    """Per-layer interval bounds over a box that may be (partly) infinite.

    Identical in meaning to :func:`discopt.ml.bounds.propagate_bounds`; it only
    differs where that function would produce ``nan``. A ``nan`` that survives
    (an activation whose interval image is undefined on an infinite argument) is
    refused rather than declared as a bound.
    """
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    out: list[LayerBounds] = []
    for k, layer in enumerate(network.layers):
        W = np.asarray(layer.weights, dtype=np.float64)
        b = np.asarray(layer.biases, dtype=np.float64)
        pre_lb, pre_ub = _interval_affine(W, b, lb, ub)
        with np.errstate(invalid="ignore", over="ignore"):
            post_lb, post_ub = activation_interval(layer.activation, pre_lb, pre_ub, layer.beta)
        post_lb = np.asarray(post_lb, dtype=np.float64)
        post_ub = np.asarray(post_ub, dtype=np.float64)
        for arr, what in ((pre_lb, "pre"), (pre_ub, "pre"), (post_lb, "post"), (post_ub, "post")):
            if np.isnan(arr).any():
                raise ValueError(
                    f"relu_complementarity: interval bounds of layer {k} "
                    f"({layer.activation.value}) are undefined ({what}-activation nan) "
                    "over the given input box; give finite input_bounds."
                )
        out.append(LayerBounds(pre_lb, pre_ub, post_lb, post_ub))
        lb, ub = post_lb, post_ub
    return out


@dataclass
class ReluComplementarityRecord:
    """One unstable ReLU neuron, as encoded by :class:`ReluComplementarityFormulation`.

    ``y`` is the post-activation element, ``s`` the negative-part variable,
    ``zhat`` the pre-activation element, and ``relation`` the declared
    ``0 <= y ⊥ s >= 0``. ``pre_lb``/``pre_ub`` are the interval bounds the
    neuron's pre-activation was declared on (possibly infinite).
    """

    layer_index: int
    neuron: int
    y: object
    s: Variable
    zhat: object
    relation: mpec.Complementarity
    pre_lb: float
    pre_ub: float


class ReluComplementarityFormulation(ReluBigMFormulation):
    """ReLU networks with one complementarity relation per unstable neuron.

    See the module docstring. After :meth:`build`, :attr:`relations` lists the
    declared relations (in layer, then neuron order) and :attr:`neurons` the
    per-neuron records.
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling,
    ) -> None:
        super().__init__(model, network, prefix, scaling)
        self.relations: list[mpec.Complementarity] = []
        self.neurons: list[ReluComplementarityRecord] = []

    # ── bound hooks: no finite box is required ──

    def _require_input_bounds(self) -> None:
        """No refusal: the complementarity encoding is exact on any box."""

    def _input_box(self) -> tuple[np.ndarray, np.ndarray]:
        net = self._network
        if net.input_bounds is None:
            n = net.input_size
            return np.full(n, -_UNBOUNDED), np.full(n, _UNBOUNDED)
        lb, ub = net.input_bounds
        return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)

    def _propagate(
        self, *, input_bounds: tuple[np.ndarray, np.ndarray] | None
    ) -> list[LayerBounds]:
        """Interval bounds over a box whose sentinel entries mean "unbounded".

        Sentinel-magnitude entries (``Model.continuous``'s ``±9.999e19`` default,
        or a scaled image of it) become true infinities for the arithmetic, and
        every infinite result is mapped back onto the sentinel, so each variable
        the emitter declares from these bounds reads as "free", not as a literal
        ``inf`` and not as a huge finite number posing as a bound.
        """
        lb, ub = self._input_box() if input_bounds is None else input_bounds
        lb = _to_inf(lb)
        ub = _to_inf(ub)
        out = _propagate_unbounded_safe(self._network, lb, ub)
        return [
            LayerBounds(
                _var_bound(L.pre_lb),
                _var_bound(L.pre_ub),
                _var_bound(L.post_lb),
                _var_bound(L.post_ub),
            )
            for L in out
        ]

    def build(self) -> tuple[Variable, Variable]:
        inputs, outputs = super().build()
        mpec.register_relations(self._model, self.relations)
        return inputs, outputs

    # ── the per-layer encoding ──

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
        pfx = self._prefix
        pre_lb = np.asarray(bounds.pre_lb, dtype=np.float64)
        pre_ub = np.asarray(bounds.pre_ub, dtype=np.float64)
        n_out = zhat.shape[0]
        y = m.continuous(
            f"{pfx}_z_{k}",
            shape=(n_out,),
            lb=np.maximum(pre_lb, 0.0),
            ub=np.maximum(pre_ub, 0.0),
        )
        parent = f"{pfx}.layer[{k}]"
        for j in range(n_out):
            lb_j = float(pre_lb[j])
            ub_j = float(pre_ub[j])
            if lb_j >= 0:
                m.subject_to(y[j] == zhat[j], name=f"{pfx}_relu_active_{k}_{j}")
            elif ub_j <= 0:
                m.subject_to(y[j] == 0, name=f"{pfx}_relu_inactive_{k}_{j}")
            else:
                # s = max(0, -z) lives on [0, -l]; an unbounded l leaves it free above.
                s = m.continuous(f"{pfx}_s_{k}_{j}", lb=0.0, ub=-lb_j)
                m.subject_to(y[j] - s == zhat[j], name=f"{pfx}_relu_split_{k}_{j}")
                rel = mpec.Complementarity(
                    y[j],
                    s,
                    name=f"{pfx}_relu_cc_{k}_{j}",
                    role=mpec.ComplementarityRole.FROM_ACTIVATION,
                    parent=parent,
                    index=(j,),
                )
                self.relations.append(rel)
                self.neurons.append(
                    ReluComplementarityRecord(
                        layer_index=k,
                        neuron=j,
                        y=y[j],
                        s=s,
                        zhat=zhat[j],
                        relation=rel,
                        pre_lb=lb_j,
                        pre_ub=ub_j,
                    )
                )
        return y
