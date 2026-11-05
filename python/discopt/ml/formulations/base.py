"""Formulation wrappers for embedding trained ML models into discopt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

from discopt.ml.formulations.full_space import FullSpaceFormulation
from discopt.ml.formulations.reduced_space import ReducedSpaceFormulation
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.formulations.relu_complementarity import ReluComplementarityFormulation
from discopt.ml.formulations.relu_partition import (
    AndersonCutSeparator,
    PartitionFn,
    ReluExtendedFormulation,
    ReluPartitionFormulation,
)
from discopt.ml.network import NetworkDefinition
from discopt.ml.scaling import OffsetScaling
from discopt.ml.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


class _StrategyProtocol(Protocol):
    def build(self) -> tuple[Variable, Variable]: ...


_StrategyCls = Union[
    type[FullSpaceFormulation],
    type[ReluBigMFormulation],
    type[ReducedSpaceFormulation],
    type[ReluPartitionFormulation],
    type[ReluExtendedFormulation],
    type[ReluComplementarityFormulation],
]

_STRATEGIES: dict[str, _StrategyCls] = {
    "full_space": FullSpaceFormulation,
    "relu_bigm": ReluBigMFormulation,
    "reduced_space": ReducedSpaceFormulation,
    "relu_partition": ReluPartitionFormulation,
    "relu_extended": ReluExtendedFormulation,
    "relu_complementarity": ReluComplementarityFormulation,
}

#: Strategies that encode ReLU neurons with indicator binaries, and so can carry
#: Anderson et al.'s ideal cuts.
_RELU_MILP = ("relu_bigm", "relu_partition", "relu_extended")


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
        ``"relu_bigm"`` (ReLU via big-M MILP), ``"relu_partition"`` (the
        partition-based MILP of Tsay et al. 2021, between big-M and the convex
        hull), ``"relu_extended"`` (the per-neuron convex hull of Anderson et al.
        2020), or ``"reduced_space"`` (nested expressions, no intermediate
        variables). The three ``relu_*`` MILPs have the same integer-feasible
        set; their LP relaxations are ordered big-M <= partition <= extended.
        ``"relu_complementarity"`` (#2167) encodes each unstable ReLU neuron as a
        complementarity relation ``0 <= y ⊥ s >= 0`` with ``y - s = z``, with no
        binary and no big-M; it needs no ``input_bounds``. The relations are
        declared, not lowered: pass :attr:`relations` to
        :func:`discopt.mpec.solve_mpec` with ``method="sos1"``/``"gdp"`` (exact)
        or ``"scholtes"`` (local, no certificate). ``Model.solve()`` refuses the
        model until then.
    prefix : str
        Name prefix for created variables and constraints.
    scaling : OffsetScaling or None
        Optional input/output scaling.
    n_partitions : int, optional
        ``relu_partition`` only: partitions per unstable neuron (default 2;
        clipped to its number of nonzero weights). ``1`` is big-M.
    partition_strategy : str or callable, optional
        ``relu_partition`` only: ``"equal_size"`` (default, weight-sorted
        blocks), ``"contiguous"``, or ``f(weights, P) -> index arrays``; see
        :func:`~discopt.ml.formulations.relu_partition.partition_indices`.
    ideal_cuts : bool
        ``relu_*`` strategies only. Build an
        :class:`~discopt.ml.formulations.relu_partition.AndersonCutSeparator`
        for the embedded neurons, available after :meth:`formulate` as
        :attr:`cut_callback`; pass it as ``model.solve(cut_callback=...)``.
        On a pure MILP the inequalities are then separated natively in Rust
        (#2157): over repeated root LP solves on the default HiGHS route, at
        every fractional node on the Rust MILP engine.

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
        *,
        n_partitions: int | None = None,
        partition_strategy: str | PartitionFn | None = None,
        ideal_cuts: bool = False,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGIES)}")
        if strategy != "relu_partition" and (
            n_partitions is not None or partition_strategy is not None
        ):
            raise ValueError(
                "n_partitions / partition_strategy apply only to strategy='relu_partition'"
            )
        if ideal_cuts and strategy not in _RELU_MILP:
            raise ValueError(f"ideal_cuts=True needs a ReLU MILP strategy, one of {_RELU_MILP}")
        if n_partitions is not None and int(n_partitions) < 1:
            raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")
        self._strategy_kwargs: dict = {}
        if strategy == "relu_partition":
            self._strategy_kwargs = {
                "n_partitions": 2 if n_partitions is None else n_partitions,
                "partition_strategy": (
                    "equal_size" if partition_strategy is None else partition_strategy
                ),
            }
        self._ideal_cuts = bool(ideal_cuts)
        self._cut_callback: AndersonCutSeparator | None = None
        self._model = model
        self._network = network
        self._prefix = prefix
        self._scaling = scaling
        self._strategy_cls: _StrategyCls = _STRATEGIES[strategy]
        self._inputs: Variable | None = None
        self._outputs: Variable | None = None
        self._relations: list = []
        self._formulated = False

    @property
    def relations(self) -> list:
        """The complementarity relations the strategy declared (#2167).

        One :class:`~discopt.mpec.Complementarity` per unstable ReLU neuron for
        ``strategy="relu_complementarity"``, in layer-then-neuron order; empty for
        every other strategy. Hand them to :func:`discopt.mpec.solve_mpec`.
        """
        if not self._formulated:
            raise RuntimeError("Call formulate() before accessing relations")
        return list(self._relations)

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

        strategy = self._strategy_cls(
            self._model, self._network, self._prefix, self._scaling, **self._strategy_kwargs
        )
        self._inputs, self._outputs = strategy.build()
        self._relations = list(getattr(strategy, "relations", []))
        if self._ideal_cuts:
            self._cut_callback = AndersonCutSeparator(getattr(strategy, "relu_layers", []))
        self._formulated = True

    @property
    def cut_callback(self) -> AndersonCutSeparator:
        """The Anderson ideal-cut separator (``ideal_cuts=True`` only)."""
        if not self._ideal_cuts:
            raise RuntimeError("cut_callback needs NNFormulation(..., ideal_cuts=True)")
        if self._cut_callback is None:
            raise RuntimeError("Call formulate() before accessing cut_callback")
        return self._cut_callback


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
        Half-width of the separation band around each split threshold
        (default 1e-5). It must exceed the solver's absolute feasibility
        tolerance (1e-6): a tree is discontinuous at its thresholds, so
        tolerance-sized slack becomes an O(1) prediction error. The default
        matches :class:`~discopt.ml.formulations.tree_ensemble.TreeEnsembleFormulation`
        (#1447, #1620); see that class for the measured rationale.

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
        split_eps: float = 1e-5,
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
