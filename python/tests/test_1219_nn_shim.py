"""Regression pins for the ``discopt.nn`` → :mod:`discopt.ml` rename (issue #1219).

The package was renamed because it embeds ML predictors generally — trees,
ensembles and arbitrary ``Surrogate`` implementations as much as neural
networks. ``discopt.nn`` stays as a deprecated alias, and these tests pin the
three properties downstream code depends on:

1. it still imports, and warns;
2. every submodule path still resolves (``discopt.nn.network``,
   ``discopt.nn.formulations.base``, ``discopt.nn.readers.sklearn_reader``, …);
3. it forwards by *identity*, so ``isinstance`` and class-identity checks hold
   across the two spellings — a shim that re-exported copies would silently
   break ``isinstance(f, discopt.ml.NNFormulation)`` for objects built through
   the old path.

Property 3 is what makes the alias a rename rather than a fork, so every test
that loops over names or submodules records each comparison and asserts the
recorded count equals the expected one before asserting the comparisons passed
(CLAUDE.md §6: a loop that silently iterates nothing reads as a pass). The
counts are per-test, so they stay valid when pytest-xdist splits this file
across workers.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

# Names forwarded by the shim, checked one by one for object identity.
_PUBLIC_NAMES = (
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
    "Surrogate",
    "TrainableDense",
    "TrainableKernelExpansion",
    "TrainableNetwork",
    "TreeEnsembleDefinition",
    "TreeFormulation",
    "add_predictor",
    "detect_dead_relus",
    "load_onnx",
    "load_sklearn_ensemble",
    "load_sklearn_mlp",
    "load_sklearn_tree",
    "load_torch_sequential",
    "propagate_bounds",
    "tighten_network",
    "train",
)

# Submodule paths that tests, docs and downstream code import directly.
_SUBMODULES = (
    "bounds",
    "formulations",
    "formulations.base",
    "formulations.full_space",
    "formulations.reduced_space",
    "formulations.relu_bigm",
    "formulations.tree_ensemble",
    "network",
    "predictor",
    "presolve",
    "readers",
    "readers.onnx_reader",
    "readers.sklearn_reader",
    "readers.torch_reader",
    "scaling",
    "surrogate",
    "trainable",
    "tree",
)


class _IdentityChecks:
    """Records one ``old is new`` comparison per call, so a loop that iterated
    nothing fails on the count instead of passing silently."""

    def __init__(self) -> None:
        self.mismatches: list[str] = []
        self.count = 0

    def same(self, old: object, new: object, what: str) -> None:
        self.count += 1
        if old is not new:
            self.mismatches.append(what)

    def assert_ran(self, expected: int) -> None:
        assert self.count == expected, (
            f"{self.count} identity comparisons executed, expected {expected} — "
            f"the check degraded to a no-op"
        )
        assert not self.mismatches, f"not forwarded by identity: {self.mismatches}"


@pytest.mark.unit
def test_import_discopt_nn_warns_deprecation():
    """The alias imports, and says it is deprecated."""
    # A module already in sys.modules does not re-execute, so drop the alias
    # (and its submodule aliases) before re-importing to observe the warning.
    for name in [n for n in sys.modules if n == "discopt.nn" or n.startswith("discopt.nn.")]:
        del sys.modules[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("discopt.nn")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, f"no DeprecationWarning; got {[str(w.message) for w in caught]}"
    message = str(deprecations[0].message)
    assert "discopt.ml" in message, message


@pytest.mark.unit
def test_public_names_are_the_same_objects():
    """Every forwarded name is the ``discopt.ml`` object itself, not a copy."""
    import discopt.ml as ml
    import discopt.nn as nn

    checks = _IdentityChecks()
    for name in _PUBLIC_NAMES:
        checks.same(getattr(nn, name), getattr(ml, name), f"discopt.nn.{name}")
    checks.assert_ran(len(_PUBLIC_NAMES))


@pytest.mark.unit
def test_submodule_paths_still_resolve_to_the_real_modules():
    """``import discopt.nn.<sub>`` yields the ``discopt.ml.<sub>`` module object."""
    checks = _IdentityChecks()
    for suffix in _SUBMODULES:
        checks.same(
            importlib.import_module(f"discopt.nn.{suffix}"),
            importlib.import_module(f"discopt.ml.{suffix}"),
            f"discopt.nn.{suffix}",
        )
    checks.assert_ran(len(_SUBMODULES))


@pytest.mark.unit
def test_from_import_of_a_submodule_member_matches():
    """The spelling used by the notebooks and by ``modeling/examples.py``."""
    from discopt.ml.formulations.base import NNFormulation as MlFormulation
    from discopt.ml.network import DenseLayer as MlDenseLayer
    from discopt.nn.formulations.base import NNFormulation as NnFormulation
    from discopt.nn.network import DenseLayer as NnDenseLayer

    assert NnFormulation is MlFormulation
    assert NnDenseLayer is MlDenseLayer


@pytest.mark.unit
def test_isinstance_holds_across_both_spellings():
    """The property a copying shim would break."""
    import discopt.ml as ml
    import discopt.nn as nn
    import numpy as np

    layer = nn.DenseLayer(
        np.array([[1.0]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        nn.Activation.RELU,
    )
    net = nn.NetworkDefinition([layer], input_bounds=(np.array([0.0]), np.array([1.0])))

    assert isinstance(layer, ml.DenseLayer)
    assert isinstance(net, ml.NetworkDefinition)
    assert type(net) is ml.NetworkDefinition


@pytest.mark.unit
def test_shim_forwards_names_added_to_discopt_ml_later():
    """``__getattr__`` forwarding, so the alias cannot drift from the package."""
    import discopt.ml as ml
    import discopt.nn as nn

    sentinel = object()
    ml.issue_1219_probe = sentinel  # type: ignore[attr-defined]
    try:
        assert nn.issue_1219_probe is sentinel  # type: ignore[attr-defined]
    finally:
        del ml.issue_1219_probe  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        nn.definitely_not_a_real_name

    # Private names are deliberately NOT forwarded: resolving dunders such as
    # ``__path__`` or ``__all__`` against discopt.ml would make the alias lie
    # about its own identity to importlib, pickle and inspect.
    ml._issue_1219_private = sentinel  # type: ignore[attr-defined]
    try:
        with pytest.raises(AttributeError):
            nn._issue_1219_private  # type: ignore[attr-defined]
    finally:
        del ml._issue_1219_private  # type: ignore[attr-defined]


@pytest.mark.unit
def test_forwarded_names_cover_the_documented_public_surface():
    """``_PUBLIC_NAMES`` must not fall behind ``discopt.ml``'s own ``__all__``."""
    import discopt.ml as ml

    missing = sorted(set(ml.__all__) - set(_PUBLIC_NAMES))
    assert not missing, f"discopt.ml.__all__ names not pinned by this test: {missing}"
