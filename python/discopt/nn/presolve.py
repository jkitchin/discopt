"""Neural-network-embedded MINLP presolve (D6 of issue #53).

discopt-distinctive. Most solvers can't reason structurally about NN
constraints embedded in a MINLP — they fall back on generic interval
arithmetic. This pass closes the loop by treating an embedded
:class:`NetworkDefinition` as a graph the orchestrator can tighten:

1. **Tighten activation bounds.** Run interval propagation through the
   network using the *current* (possibly tightened) input variable
   box, not the network's static ``input_bounds``. This lets upstream
   FBBT, OBBT, or reverse-AD tightening on the input variables flow
   into smaller activation envelopes than the network's a-priori
   bounds permit.
2. **Detect dead ReLUs.** A ReLU is *dead-zero* if its pre-activation
   upper bound is ≤ 0 (always 0 output) and *dead-active* if its
   pre-activation lower bound is ≥ 0 (output equals pre-activation).
   Dead ReLUs become linear and should be eliminated from any big-M
   encoding to drop the corresponding binary variable.
3. **Re-emit tightened input bounds for any subsequent sweeps.** The
   pass writes the tightened input box back into the model_repr via
   ``tighten_var_bounds``, matching the A3 handshake convention so the
   next Rust sweep picks them up.

The pass is registered as a Python-side :class:`PresolvePass` so it
participates in the orchestrator's fixed-point loop alongside FBBT,
implied-bounds, reduced-cost fixing, and B2 (reverse-AD).

Dependencies: A3 for orchestrator participation; B2 (reverse-AD)
strongly recommended because reverse-AD on the JAX-DAG side closes the
loop on input-side constraints that drive the NN's input box.

References
----------
- Tjeng, Xiao, Tedrake (2019), *Evaluating robustness of neural networks
  with mixed integer programming*, ICLR.
- Grimstad, Andersson (2019), *ReLU networks as surrogate models in
  mixed-integer linear programs*, Comput. Chem. Eng. 131.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from discopt._jax.presolve.protocol import make_python_delta
from discopt.nn.bounds import LayerBounds, propagate_bounds
from discopt.nn.network import Activation, NetworkDefinition

logger = logging.getLogger(__name__)


@dataclass
class DeadReluLayer:
    """Dead-ReLU mask for one layer.

    For layer ``L`` with pre-activation lower bound ``z_lb`` and upper
    bound ``z_ub``:
    - ``dead_zero[i] = True`` iff ``z_ub[i] <= 0`` (output is identically 0)
    - ``dead_active[i] = True`` iff ``z_lb[i] >= 0`` (output equals z_i)

    ``dead_zero`` and ``dead_active`` are mutually exclusive. Both
    states drop the binary indicator from a big-M encoding; the
    formulation can short-circuit the constraint to a linear pass-through
    or zero-fill.
    """

    layer_index: int
    dead_zero: np.ndarray  # bool, shape (n_neurons,)
    dead_active: np.ndarray  # bool, shape (n_neurons,)


@dataclass
class NNPresolveResult:
    """Outcome of one D6 presolve pass over a network."""

    layer_bounds: list[LayerBounds]
    dead_relus: list[DeadReluLayer] = field(default_factory=list)
    input_lb: Optional[np.ndarray] = None
    input_ub: Optional[np.ndarray] = None
    n_neurons_dead: int = 0


def detect_dead_relus(
    network: NetworkDefinition,
    layer_bounds: Sequence[LayerBounds],
) -> list[DeadReluLayer]:
    """Mark dead ReLU neurons given pre-/post-activation bounds.

    Iterates over every ReLU layer and classifies each neuron as
    dead-zero (``z_ub <= 0``), dead-active (``z_lb >= 0``), or live.
    Linear / sigmoid / tanh / softplus layers contribute no entries —
    "deadness" is only well-defined for piecewise-linear gates.
    """
    out: list[DeadReluLayer] = []
    for li, (layer, lb) in enumerate(zip(network.layers, layer_bounds)):
        if layer.activation != Activation.RELU:
            continue
        dead_zero = lb.pre_ub <= 0.0
        dead_active = lb.pre_lb >= 0.0
        # Both can't be true unless pre_lb == pre_ub == 0 (an exactly
        # collapsed neuron). Treat that case as dead_zero, matching the
        # big-M convention that lb=ub=0 fixes the output to 0.
        dead_active = dead_active & ~dead_zero
        if dead_zero.any() or dead_active.any():
            out.append(DeadReluLayer(layer_index=li, dead_zero=dead_zero, dead_active=dead_active))
    return out


def tighten_network(
    network: NetworkDefinition,
    *,
    input_lb: Optional[np.ndarray] = None,
    input_ub: Optional[np.ndarray] = None,
) -> NNPresolveResult:
    """Run one round of NN-presolve.

    :param network: the network whose layer bounds we want to refresh.
    :param input_lb: optional input lower-bound vector overriding
        ``network.input_bounds``. Used by the orchestrator to feed the
        *current* (possibly tightened) input bounds rather than the
        static declared box.
    :param input_ub: optional input upper-bound vector overriding
        ``network.input_bounds``; must be provided together with
        ``input_lb``.
    """
    used_lb: Optional[np.ndarray]
    used_ub: Optional[np.ndarray]
    if input_lb is not None or input_ub is not None:
        if input_lb is None or input_ub is None:
            raise ValueError("input_lb and input_ub must both be set or both None")
        # Feed the caller's box directly via the ``input_bounds`` override
        # (T-N0.2) instead of mutating and restoring network.input_bounds.
        used_lb = np.asarray(input_lb, dtype=np.float64)
        used_ub = np.asarray(input_ub, dtype=np.float64)
        layer_bounds = propagate_bounds(network, input_bounds=(used_lb, used_ub))
    else:
        layer_bounds = propagate_bounds(network)
        used_lb = (
            np.asarray(network.input_bounds[0], dtype=np.float64)
            if network.input_bounds is not None
            else None
        )
        used_ub = (
            np.asarray(network.input_bounds[1], dtype=np.float64)
            if network.input_bounds is not None
            else None
        )

    dead = detect_dead_relus(network, layer_bounds)
    n_dead = sum(int(d.dead_zero.sum() + d.dead_active.sum()) for d in dead)

    return NNPresolveResult(
        layer_bounds=list(layer_bounds),
        dead_relus=dead,
        input_lb=used_lb,
        input_ub=used_ub,
        n_neurons_dead=n_dead,
    )


class NNPresolvePass:
    """A presolve pass that tightens an embedded NN's activation bounds
    and reports dead ReLUs.

    Constructor binds the pass to a specific network and to the variable
    block index housing the network's input vector. Use this when the
    NN is embedded in a Python ``Model`` whose variables include the
    network's input layer.

    The pass is *informational* in v0 and is not wired into any solve
    path: no orchestrator constructs it, and ``run()`` performs **no
    bound writeback** to the model. It computes activation bounds and
    dead-ReLU masks and surfaces them on the returned delta under the
    ``nn_implications``/``nn_neurons_dead`` keys (one entry per dead
    neuron) purely for inspection; the orchestrator treats these unknown
    keys as inert. Wiring this into ``run_root_presolve(python_passes=...)``
    so dead-ReLU implications can fix big-M binaries after root
    tightening is future work (v1), tracked in the nn-module plan
    (T-N2.2); it requires the entry experiment that measures whether
    root FBBT/OBBT tightens NN input boxes enough to kill neurons.

    :param network: the embedded :class:`NetworkDefinition`.
    :param input_block_index: variable block index of the input layer
        in the ``model_repr`` passed at runtime. Set to ``None`` if the
        network's inputs are not represented as a single contiguous
        variable block (e.g., spread across multiple variables); in that
        case the pass uses ``network.input_bounds`` only and does no
        writeback.
    """

    name = "nn_presolve"

    def __init__(
        self,
        network: NetworkDefinition,
        *,
        input_block_index: Optional[int] = None,
    ) -> None:
        self.network = network
        self.input_block_index = input_block_index
        self.last_result: Optional[NNPresolveResult] = None

    def run(self, model_repr) -> dict:
        delta = make_python_delta(self.name)

        # Pull the current input box from the model_repr if the caller
        # told us where the network's inputs live, otherwise fall back
        # to the network's declared input_bounds.
        input_lb = None
        input_ub = None
        if (
            self.input_block_index is not None
            and 0 <= self.input_block_index < model_repr.n_var_blocks
        ):
            input_lb = np.asarray(model_repr.var_lb(self.input_block_index), dtype=np.float64)
            input_ub = np.asarray(model_repr.var_ub(self.input_block_index), dtype=np.float64)
            if input_lb.size != self.network.input_size:
                # Block size doesn't match the network — fall back.
                input_lb = None
                input_ub = None

        try:
            if input_lb is not None and input_ub is not None:
                result = tighten_network(self.network, input_lb=input_lb, input_ub=input_ub)
            else:
                result = tighten_network(self.network)
        except ValueError as exc:
            # The only expected failures are a missing input_bounds and
            # weight/bias/box shape mismatches, both of which surface as
            # ValueError (propagate_bounds and numpy affine ops). Abstain
            # rather than break the orchestrator, but leave a trace.
            logger.debug("NNPresolvePass abstaining: %s", exc)
            return delta

        self.last_result = result
        delta["work_units"] = len(self.network.layers)

        # Surface dead-ReLU implications in the structural manifest.
        # Each dead-zero neuron is encoded as an implication of the
        # form (layer_index, neuron_index): the binary indicator
        # downstream big-M would introduce can be fixed to 0.
        impls: list[dict] = []
        for d in result.dead_relus:
            for j in np.flatnonzero(d.dead_zero):
                impls.append(
                    {
                        "layer": int(d.layer_index),
                        "neuron": int(j),
                        "state": "dead_zero",
                    }
                )
            for j in np.flatnonzero(d.dead_active):
                impls.append(
                    {
                        "layer": int(d.layer_index),
                        "neuron": int(j),
                        "state": "dead_active",
                    }
                )
        # Stash on the delta for downstream consumers. The orchestrator's
        # protocol doesn't carry a typed nn-implication slot in v0, so
        # we extend the dict directly with an "nn_implications" key —
        # delta_made_progress treats unknown keys as inert.
        delta["nn_implications"] = impls
        delta["nn_neurons_dead"] = int(result.n_neurons_dead)
        return delta


__all__ = [
    "DeadReluLayer",
    "NNPresolveResult",
    "NNPresolvePass",
    "detect_dead_relus",
    "tighten_network",
]
