"""Convenience dispatcher for embedding trained ML models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

# Bounds whose magnitude reaches this threshold are treated as effectively
# unbounded (not "finite") when harvesting from a user variable. The modeling
# API's default variable bounds are +/-9.999e19 (see ``Model.continuous``),
# which are sentinels for "no bound", not usable big-M box edges.
_BOUND_INF_THRESHOLD = 1e15


def add_predictor(
    model: Model,
    inputs: Variable,
    predictor: object,
    *,
    method: str = "auto",
    prefix: str = "pred",
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Variable, Union[NNFormulation, TreeFormulation]]:
    """Embed a trained ML model as constraints in a discopt Model.

    Auto-detects the predictor type and converts it to the appropriate
    internal representation, then formulates it as optimization constraints.

    Parameters
    ----------
    model : discopt.Model
        The optimization model.
    inputs : Variable
        Existing input variables to connect the predictor to. Its length must
        equal the predictor's feature count. When neither ``input_bounds`` nor
        the predictor carries feature bounds, this variable's declared
        (finite) ``lb``/``ub`` are harvested as the input box.
    predictor : object
        A trained ML model. Supported types:

        - :class:`NetworkDefinition` or :class:`TreeEnsembleDefinition`
        - ``sklearn.neural_network.MLPRegressor`` / ``MLPClassifier``
        - ``sklearn.tree.DecisionTreeRegressor`` / ``DecisionTreeClassifier``
        - ``sklearn.ensemble.GradientBoostingRegressor`` / ``RandomForestRegressor``
        - ``torch.nn.Sequential``
        - A file path (str or Path) to an ONNX model
    method : str
        Formulation method. ``"auto"`` selects based on the model type.
        For neural networks: ``"relu_bigm"``, ``"full_space"``, or
        ``"reduced_space"``. Tree ensembles always use MILP encoding.
    prefix : str
        Name prefix for created variables and constraints.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on features. Required for big-M and
        tree formulations. Overrides bounds already set on the predictor.
        If ``None`` and the predictor carries no bounds, the bounds are
        harvested from ``inputs`` (see above).

    Returns
    -------
    outputs : Variable
        Output variable(s) of the embedded predictor.
    formulation : NNFormulation or TreeFormulation
        The formulation object (for inspection or further use).

    Notes
    -----
    The formulation creates its *own* input variables and this function links
    them to the user's ``inputs`` with equality constraints, rather than
    formulating directly onto ``inputs``. This duplication is intentional: it
    keeps the formulation self-contained (bounds/structure derived from the
    predictor, not the surrounding model) and the link equalities are trivially
    eliminated by presolve, so the duplicate variables cost nothing at solve
    time.
    """
    definition = _convert(predictor, input_bounds)

    n_features = _expected_features(definition)
    _validate_input_length(inputs, n_features)

    # Harvest bounds from the user's inputs variable when none were supplied and
    # the predictor carries none, so the informative error is raised here rather
    # than surfacing from deep inside the formulation.
    if input_bounds is None and definition.input_bounds is None:
        input_bounds = _harvest_input_bounds(inputs, n_features)

    if isinstance(definition, NetworkDefinition):
        if input_bounds is not None:
            definition = NetworkDefinition(
                definition.layers,
                input_bounds=input_bounds,
            )
        if method == "auto":
            has_relu = any(layer.activation == Activation.RELU for layer in definition.layers)
            method = "relu_bigm" if has_relu else "full_space"

        form = NNFormulation(model, definition, strategy=method, prefix=prefix)
        form.formulate()
        _link_inputs(model, inputs, form.inputs, prefix)
        return form.outputs, form

    # TreeEnsembleDefinition
    if input_bounds is not None:
        definition = TreeEnsembleDefinition(
            trees=definition.trees,
            n_features=definition.n_features,
            base_score=definition.base_score,
            input_bounds=input_bounds,
        )
    form_t = TreeFormulation(model, definition, prefix=prefix)
    form_t.formulate()
    _link_inputs(model, inputs, form_t.inputs, prefix)
    return form_t.outputs, form_t


def _expected_features(
    definition: NetworkDefinition | TreeEnsembleDefinition,
) -> int:
    """Expected input feature count of a converted definition."""
    if isinstance(definition, NetworkDefinition):
        return definition.input_size
    return definition.n_features


def _validate_input_length(inputs: Variable, n_features: int) -> None:
    """Check the user's ``inputs`` length matches the predictor's feature count."""
    n_inputs = inputs.shape[0] if inputs.shape else 1
    if n_inputs != n_features:
        raise ValueError(
            f"inputs has length {n_inputs} but the predictor expects "
            f"{n_features} features. Pass an inputs variable with shape "
            f"({n_features},)."
        )


def _harvest_input_bounds(inputs: Variable, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Harvest finite ``(lb, ub)`` from the user's ``inputs`` variable.

    Raises a clear :class:`ValueError` naming the three ways to supply bounds
    when the variable's declared bounds are non-finite/absent.
    """
    lb = np.asarray(inputs.lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(inputs.ub, dtype=np.float64).reshape(-1)
    finite = (
        np.isfinite(lb)
        & np.isfinite(ub)
        & (np.abs(lb) < _BOUND_INF_THRESHOLD)
        & (np.abs(ub) < _BOUND_INF_THRESHOLD)
    )
    if bool(np.all(finite)):
        return lb, ub
    raise ValueError(
        "Input bounds are required for this predictor's formulation but none "
        "are available. Supply them in one of three ways: (1) set the "
        "predictor's own input_bounds, (2) pass input_bounds=(lower, upper) to "
        "add_predictor, or (3) declare finite lb/ub on the inputs variable "
        "(e.g. m.continuous('x', shape=(n,), lb=..., ub=...))."
    )


def _link_inputs(model: Model, user_inputs: Variable, form_inputs: Variable, prefix: str) -> None:
    """Add equality constraints linking user's input variables to formulation inputs."""
    n = form_inputs.shape[0] if form_inputs.shape else 1
    constraints = []
    for j in range(n):
        constraints.append(user_inputs[j] == form_inputs[j])
    model.subject_to(constraints, name=f"{prefix}_link_inputs")


def _convert(
    predictor: object,
    input_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> NetworkDefinition | TreeEnsembleDefinition:
    """Convert a predictor object to a discopt definition."""
    if isinstance(predictor, (NetworkDefinition, TreeEnsembleDefinition)):
        return predictor

    if isinstance(predictor, (str, Path)):
        p = Path(predictor)
        if not p.exists():
            raise FileNotFoundError(f"Predictor file not found: {predictor}")
        if p.suffix == ".onnx":
            from discopt.nn.readers.onnx_reader import load_onnx

            return load_onnx(str(predictor), input_bounds=input_bounds)
        raise TypeError(f"Unsupported predictor file format: {predictor}")

    mod = type(predictor).__module__ or ""

    # sklearn MLP
    if hasattr(predictor, "coefs_") and hasattr(predictor, "intercepts_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp

        return load_sklearn_mlp(predictor, input_bounds=input_bounds)

    # sklearn single tree
    if hasattr(predictor, "tree_") and not hasattr(predictor, "estimators_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_tree

        return load_sklearn_tree(predictor, input_bounds=input_bounds)

    # sklearn ensemble (GBR, RF)
    if hasattr(predictor, "estimators_") and hasattr(predictor, "n_features_in_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble

        return load_sklearn_ensemble(predictor, input_bounds=input_bounds)

    # PyTorch Sequential
    if mod.startswith("torch"):
        from discopt.nn.readers.torch_reader import load_torch_sequential

        return load_torch_sequential(predictor, input_bounds=input_bounds)

    raise TypeError(
        f"Cannot auto-detect predictor type for {type(predictor).__name__}. "
        "Pass a NetworkDefinition, TreeEnsembleDefinition, sklearn model, "
        "torch.nn.Sequential, or path to an ONNX file."
    )
