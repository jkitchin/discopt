"""Input-convex ReLU networks embedded as an LP: no binaries (``strategy="relu_icnn"``).

An input-convex neural network (ICNN; Amos, Xu & Kolter, ICML 2017) restricted to
discopt's sequential :class:`~discopt.ml.network.NetworkDefinition` is a ReLU
network whose first layer is unconstrained and whose every later weight
(hidden-to-hidden and output) is nonnegative. Each later unit is then a
nonnegative combination of convex functions passed through the convex
nondecreasing ReLU, so every output is convex in the input.

For such a network the ReLU binaries are unnecessary. Per neuron, the epigraph rows

    z >= 0,   z >= zhat

describe ``{(x, y): y >= f(x)}`` once projected onto the input and output, because
each later layer is nondecreasing in its inputs: the least ``y`` reachable from
``x`` is the one with every ``z = max(zhat, 0)``, i.e. ``f(x)``. The rows are
therefore a *relaxation* of the network's graph ``{y = f(x)}``, exact wherever a
smaller ``y`` is preferred (``f`` minimised, or bounded above). The idea is from
J. Kitchin, "MIP surrogates" (2026-10-28,
https://kitchingroup.cheme.cmu.edu/posts/mip-surrogates/), where pycse's
``JAXICNNRegressor(formulation="convex")`` drops the binaries. The question was
raised by PWLNet (Sanyal, Demir, Türkay, Aydın, arXiv:2611.25969). Entry
experiment E2 (``docs/dev/nn-module-plan.md`` §6): median 23x faster than the
big-M MILP on 8 ICNNs, with the same certified minimum on all 24 solves.

Soundness
---------
Whether a smaller output is preferred is a property of how the *rest of the
model* uses it, which this module cannot see when it builds the rows. So it is
not assumed; it is checked on each result by :meth:`ICNNEpigraph.check`, which
``Model.solve`` runs. The check rests on two facts about a relaxation:

* its bound is a valid bound on the network's model, and its infeasibility proves
  the network's model infeasible;
* a point of the relaxation that is also on the graph, with the same objective,
  is optimal for the network's model whenever it is optimal for the relaxation.

The check therefore repairs every network variable to its forward value at the
reported input (``zhat``, ``z`` and the output, as ``forward`` computes them),
re-verifies the *whole* model at the repaired point, and requires the objective
to be unchanged. If any of that fails, the output was used where a larger value
helps (maximised, bounded below, or inside an expression that rewards it). Then
the relaxation's answer says nothing about the network, and ``Model.solve``
raises :class:`ICNNExactnessError` rather than return it. An ``unbounded``
relaxation is refused for the same reason. ``relu_bigm`` encodes the graph
exactly and is the formulation to use for such a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.ml.bounds import LayerBounds
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.network import Activation, NetworkDefinition
from discopt.ml.scaling import OffsetScaling
from discopt.modeling.core import Model, Variable

#: Relative tolerance on the objective at the repaired point. The repaired
#: point is the relaxation's own point moved onto the graph, so on an exact use
#: the two objectives differ only by the LP's feasibility error carried through
#: the layers.
OBJECTIVE_RTOL = 1e-6


class ICNNExactnessError(RuntimeError):
    """A ``relu_icnn`` solve whose answer does not hold for the network itself."""


def input_convexity_violation(
    network: NetworkDefinition, scaling: OffsetScaling | None = None
) -> str | None:
    """Why ``network`` (with ``scaling``) is not input-convex in the LP's sense,
    or ``None`` if it is.

    The test is structural and sufficient, not necessary: ReLU or linear
    activations everywhere, every weight after the first layer nonnegative, and
    a positive output scale factor. A convex network that fails it (e.g. one
    with a negative weight cancelled elsewhere) is refused, never guessed at.
    """
    for k, layer in enumerate(network.layers):
        if layer.activation not in (Activation.RELU, Activation.LINEAR):
            return (
                f"layer {k} has activation {layer.activation.value!r}; the LP "
                "encodes only 'relu' and 'linear'"
            )
        if k > 0:
            W = np.asarray(layer.weights, dtype=np.float64)
            if np.any(W < 0.0):
                i, j = np.argwhere(W < 0.0)[0]
                return (
                    f"layer {k} has a negative weight ({W[i, j]:.6g} at [{i}, {j}]); "
                    "every weight after the first layer must be >= 0"
                )
    if scaling is not None and np.any(np.asarray(scaling.y_factor, dtype=np.float64) <= 0.0):
        return "the output scaling has a factor <= 0, which makes the output concave"
    return None


@dataclass
class ICNNEpigraph:
    """What :meth:`check` needs to put a result back on the network's graph.

    Plain data (the network, the scaling and variable *names*), so the model
    stays copyable and picklable. Registered on ``Model._solve_result_checks``.
    """

    network: NetworkDefinition
    scaling: OffsetScaling | None
    input_name: str
    scaled_input_name: str | None
    zhat_names: list[str]
    z_names: list[str | None]
    output_name: str | None

    def forward_values(self, x_in: np.ndarray) -> dict[str, np.ndarray]:
        """Every network variable's value at the raw input ``x_in``."""
        out: dict[str, np.ndarray] = {}
        h = np.asarray(x_in, dtype=np.float64).reshape(-1)
        if self.scaling is not None:
            h = (h - self.scaling.x_offset) / self.scaling.x_factor
            assert self.scaled_input_name is not None
            out[self.scaled_input_name] = h
        for k, layer in enumerate(self.network.layers):
            pre = h @ np.asarray(layer.weights, dtype=np.float64) + layer.biases
            out[self.zhat_names[k]] = pre
            h = np.maximum(pre, 0.0) if layer.activation == Activation.RELU else pre
            name = self.z_names[k]
            if name is not None:
                out[name] = h
        if self.output_name is not None:
            assert self.scaling is not None
            out[self.output_name] = h * self.scaling.y_factor + self.scaling.y_offset
        return out

    def check(self, model: Model, result) -> None:
        """Put ``result`` on the network's graph, or raise
        :class:`ICNNExactnessError` (see the module docstring)."""
        if result.status == "unbounded":
            raise ICNNExactnessError(
                "the relu_icnn relaxation is unbounded, which says nothing about the "
                "network: its output is probably maximised or bounded below. Use "
                "strategy='relu_bigm', which encodes the network exactly."
            )
        if not result.x:
            return
        from discopt.validation.feasibility import verify_point

        repaired = dict(result.x)
        for name, value in self.forward_values(result.x[self.input_name]).items():
            repaired[name] = value.reshape(np.shape(result.x[name]))
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(repaired[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        verdict = verify_point(model, flat, with_objective=True)
        why = None
        if not verdict.ok:
            why = f"the repaired point is infeasible ({verdict.reason})"
        elif result.objective is not None:
            obj, rep = float(result.objective), float(verdict.objective)
            if abs(rep - obj) > OBJECTIVE_RTOL * max(1.0, abs(obj)):
                why = f"the objective changes from {obj!r} to {rep!r}"
        if why is not None:
            raise ICNNExactnessError(
                "relu_icnn: putting the reported point back on the network's graph "
                f"(every neuron at its forward value) fails: {why}. The relaxation "
                "is exact only where a smaller network output is preferred; this "
                "model rewards a larger one somewhere. Use strategy='relu_bigm', "
                "which encodes the network exactly."
            )
        result.x = repaired


class ReluICNNFormulation(ReluBigMFormulation):
    """The epigraph LP of an input-convex ReLU network (see the module docstring).

    Refuses a network that is not input-convex by :func:`input_convexity_violation`.
    """

    #: The variable boxes come from interval arithmetic; there is no big-M for
    #: LP tightening to improve (T-N3.2).
    LP_BIGM_DEFAULT = False

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling: OffsetScaling | None,
    ) -> None:
        why = input_convexity_violation(network, scaling)
        if why is not None:
            raise ValueError(
                f"strategy='relu_icnn' needs an input-convex network: {why}. Use "
                "strategy='relu_bigm' for a general ReLU network."
            )
        super().__init__(model, network, prefix, scaling)
        self._z_names: list[str | None] = [None] * len(network.layers)

    def build(self) -> tuple[Variable, Variable]:
        inputs, outputs = super().build()
        pfx = self._prefix
        scaled = self._scaling is not None
        self._model._solve_result_checks.append(
            ICNNEpigraph(
                network=self._network,
                scaling=self._scaling,
                input_name=inputs.name,
                scaled_input_name=f"{pfx}_scaled_input" if scaled else None,
                zhat_names=[f"{pfx}_zhat_{k}" for k in range(len(self._network.layers))],
                z_names=self._z_names,
                output_name=outputs.name if scaled else None,
            )
        )
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
        """``z >= zhat`` as one array row, with ``z`` boxed in ``[max(lb, 0),
        max(ub, 0)]``.

        The box supplies ``z >= 0`` and holds ``max(zhat, 0)`` for every ``zhat``
        in the pre-activation box, so the graph stays inside. The one row is the
        epigraph for every neuron: an unstable one gets ``z >= max(zhat, 0)``; a
        stable-active one (``lb >= 0``) gets ``z >= zhat``, its own epigraph; a
        stable-inactive one (``ub <= 0``) is boxed to ``[0, 0]``, where the row
        always holds. One array body rather than a row per neuron, which E2
        measured as the cheaper build.
        """
        pfx = self._prefix
        z = m.continuous(
            f"{pfx}_z_{k}",
            shape=(zhat.shape[0],),
            lb=np.maximum(bounds.pre_lb, 0.0),
            ub=np.maximum(bounds.pre_ub, 0.0),
        )
        m.subject_to(z >= zhat, name=f"{pfx}_relu_epi_{k}")
        self._z_names[k] = z.name
        return z
