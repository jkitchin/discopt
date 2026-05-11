"""Neural network embedding for discopt optimization models.

Embed trained neural networks as algebraic constraints in MINLP models,
enabling optimization over ML surrogates with global optimality guarantees.

Example
-------
>>> import discopt.modeling as dm
>>> from discopt.nn import NNFormulation, NetworkDefinition, DenseLayer, Activation
>>>
>>> m = dm.Model("nn_opt")
>>> net = NetworkDefinition([
...     DenseLayer(W1, b1, Activation.RELU),
...     DenseLayer(W2, b2, Activation.LINEAR),
... ], input_bounds=(lb, ub))
>>>
>>> nn = NNFormulation(m, net, strategy="relu_bigm")
>>> nn.formulate()
>>> m.minimize(dm.sum(nn.outputs))
>>> m.subject_to(nn.inputs[0] >= 1.0)
>>> result = m.solve()
"""

from discopt.nn.bounds import LayerBounds, propagate_bounds
from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.presolve import (
    DeadReluLayer,
    NNPresolvePass,
    NNPresolveResult,
    detect_dead_relus,
    tighten_network,
)
from discopt.nn.scaling import OffsetScaling
from discopt.nn.tree import DecisionTree, TreeEnsembleDefinition

__all__ = [
    "Activation",
    "DeadReluLayer",
    "DecisionTree",
    "DenseLayer",
    "LayerBounds",
    "NNFormulation",
    "NNPresolvePass",
    "NNPresolveResult",
    "NetworkDefinition",
    "OffsetScaling",
    "TreeEnsembleDefinition",
    "TreeFormulation",
    "detect_dead_relus",
    "propagate_bounds",
    "tighten_network",
]


# Lazy imports for optional dependencies


def load_onnx(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load an ONNX model. Requires ``pip install discopt[nn]``."""
    from discopt.nn.readers.onnx_reader import load_onnx as _load_onnx

    return _load_onnx(*args, **kwargs)


def load_sklearn_mlp(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load sklearn MLPRegressor/Classifier. Requires scikit-learn."""
    from discopt.nn.readers.sklearn_reader import load_sklearn_mlp as _f

    return _f(*args, **kwargs)


def load_sklearn_tree(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load sklearn DecisionTree. Requires scikit-learn."""
    from discopt.nn.readers.sklearn_reader import load_sklearn_tree as _f

    return _f(*args, **kwargs)


def load_sklearn_ensemble(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load sklearn ensemble (GBR, RF). Requires scikit-learn."""
    from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble as _f

    return _f(*args, **kwargs)


def load_torch_sequential(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load torch.nn.Sequential. Requires PyTorch."""
    from discopt.nn.readers.torch_reader import load_torch_sequential as _f

    return _f(*args, **kwargs)


def add_predictor(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Embed a trained ML model as constraints. See :func:`discopt.nn.predictor.add_predictor`."""
    from discopt.nn.predictor import add_predictor as _f

    return _f(*args, **kwargs)
