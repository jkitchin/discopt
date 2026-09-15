"""Formulation wrappers for embedding trained ML models into discopt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

from discopt.ml.formulations.full_space import FullSpaceFormulation
from discopt.ml.formulations.reduced_space import ReducedSpaceFormulation
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.network import NetworkDefinition
from discopt.ml.scaling import OffsetScaling
from discopt.ml.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


class _StrategyProtocol(Protocol):
    def build(self) -> tuple[Variable, Variable]: ...


_StrategyCls = Union[
    type[FullSpaceFormulation], type[ReluBigMFormulation], type[ReducedSpaceFormulation]
]

_STRATEGIES: dict[str, _StrategyCls] = {
    "full_space": FullSpaceFormulation,
    "relu_bigm": ReluBigMFormulation,
    "reduced_space": ReducedSpaceFormulation,
}


class NNFormulation:
    """Embed a trained neural network into a discopt optimization model.

    Follows the builder pattern: takes an existing Model, adds variables and
    constraints to represent the neural network, and exposes input/output
    variable handles for connecting to the rest of the optimization model.

    Parameters
    ----------
    model : discopt.Model
        The optimization model to embed the network into.
    network : NetworkDefinition
        The trained network to embed.
    strategy : str
        Formulation strategy. One of ``"full_space"`` (smooth activations),
        ``"relu_bigm"`` (ReLU via big-M MILP), or ``"reduced_space"``
        (nested expressions, no intermediate variables).
    prefix : str
        Name prefix for created variables and constraints.
    scaling : OffsetScaling or None
        Optional input/output scaling.

    Example
    -------
    >>> nn = NNFormulation(model, network, strategy="relu_bigm")
    >>> nn.formulate()
    >>> model.minimize(dm.sum(nn.outputs))
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        strategy: str = "full_space",
        prefix: str = "nn",
        scaling: OffsetScaling | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGIES)}")
        self._model = model
        self._network = network
        self._prefix = prefix
        self._scaling = scaling
        self._strategy_cls: _StrategyCls = _STRATEGIES[strategy]
        self._inputs: Variable | None = None
        self._outputs: Variable | None = None
        self._formulated = False

    @property
    def inputs(self) -> Variable:
        """Input variables (unscaled, in the user's domain)."""
        if self._inputs is None:
            raise RuntimeError("Call formulate() before accessing inputs")
        return self._inputs

    @property
    def outputs(self) -> Variable:
        """Output variables (unscaled, in the user's domain)."""
        if self._outputs is None:
            raise RuntimeError("Call formulate() before accessing outputs")
        return self._outputs

    def formulate(self) -> None:
        """Add all variables and constraints to the model.

        This method can only be called once per NNFormulation instance.
        """
        if self._formulated:
            raise RuntimeError("formulate() has already been called")

        strategy = self._strategy_cls(self._model, self._network, self._prefix, self._scaling)
        self._inputs, self._outputs = strategy.build()
        self._formulated = True


class TreeFormulation:
    """Embed a trained tree ensemble into a discopt optimization model.

    Parameters
    ----------
    model : discopt.Model
        The optimization model to embed the tree ensemble into.
    ensemble : TreeEnsembleDefinition
        The trained tree ensemble.
    prefix : str
        Name prefix for created variables and constraints.
    scaling : OffsetScaling or None
        Optional input/output scaling.
    split_eps : float
        Epsilon for encoding strict inequality splits.

    Example
    -------
    >>> tf = TreeFormulation(model, ensemble)
    >>> tf.formulate()
    >>> model.minimize(tf.outputs[0])
    """

    def __init__(
        self,
        model: Model,
        ensemble: TreeEnsembleDefinition,
        prefix: str = "tree",
        scaling: OffsetScaling | None = None,
        split_eps: float = 1e-6,
    ) -> None:
        self._model = model
        self._ensemble = ensemble
        self._prefix = prefix
        self._scaling = scaling
        self._split_eps = split_eps
        self._inputs: Variable | None = None
        self._outputs: Variable | None = None
        self._formulated = False

    @property
    def inputs(self) -> Variable:
        """Input variables."""
        if self._inputs is None:
            raise RuntimeError("Call formulate() before accessing inputs")
        return self._inputs

    @property
    def outputs(self) -> Variable:
        """Output variables."""
        if self._outputs is None:
            raise RuntimeError("Call formulate() before accessing outputs")
        return self._outputs

    def formulate(self) -> None:
        """Add all variables and constraints to the model."""
        if self._formulated:
            raise RuntimeError("formulate() has already been called")

        from discopt.ml.formulations.tree_ensemble import TreeEnsembleFormulation

        strategy = TreeEnsembleFormulation(
            self._model, self._ensemble, self._prefix, self._scaling, self._split_eps
        )
        self._inputs, self._outputs = strategy.build()
        self._formulated = True
