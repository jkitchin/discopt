"""Tests for ``discopt.nn.predictor.add_predictor`` ergonomics (task T-N2.3).

Covers the four improvements landed under T-N2.3:

* bounds harvested from the user's ``inputs`` variable when neither
  ``input_bounds`` nor the predictor carries any;
* an up-front, clear error when ``inputs`` length != the predictor's feature
  count;
* an informative ``ValueError`` (naming the three ways to supply bounds) when
  no bounds are available anywhere;
* ``FileNotFoundError`` (not ``TypeError``) for a missing predictor file path.

Core cases build a :class:`NetworkDefinition` directly from numpy weights so no
optional ML dependency is required.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.predictor import add_predictor

# Make the sibling equivalence harness importable regardless of pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_relu_net(n_in=2, n_hidden=3, n_out=1, with_bounds=False):
    """Small two-layer ReLU network (relu_bigm target; needs input bounds)."""
    rng = np.random.RandomState(0)
    w1 = rng.randn(n_in, n_hidden)
    b1 = rng.randn(n_hidden)
    w2 = rng.randn(n_hidden, n_out)
    b2 = rng.randn(n_out)
    bounds = (-np.ones(n_in), np.ones(n_in)) if with_bounds else None
    return NetworkDefinition(
        [
            DenseLayer(w1, b1, Activation.RELU),
            DenseLayer(w2, b2, Activation.LINEAR),
        ],
        input_bounds=bounds,
    )


# ---------------------------------------------------------------------------
# Bounds harvested from the inputs variable
# ---------------------------------------------------------------------------


def test_bounds_harvested_from_inputs_variable():
    """No input_bounds and no net bounds -> harvest finite lb/ub from inputs."""
    net = _make_relu_net(with_bounds=False)
    m = dm.Model("harvest")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)

    outputs, form = add_predictor(m, x, net, method="auto", prefix="p")

    # The formulation (relu_bigm, which needs a box) built successfully using the
    # harvested bounds.
    lb, ub = form._network.input_bounds
    np.testing.assert_allclose(lb, [-1.0, -1.0])
    np.testing.assert_allclose(ub, [1.0, 1.0])

    # End-to-end: pin the inputs to a point and confirm the embedded output
    # matches the network's forward pass.
    pt = np.array([0.3, -0.4])
    for j in range(2):
        m.subject_to(x[j] == float(pt[j]), name=f"pin_{j}")
    m.minimize(outputs[0])
    r = m.solve()
    assert r.status == "optimal"
    got = float(np.asarray(r.value(outputs)).ravel()[0])
    expected = float(np.asarray(net.forward(pt)).ravel()[0])
    assert abs(got - expected) < 1e-5


def test_harvested_bounds_match_equivalence_harness():
    """The harvested-bounds net is embedding-equivalent to its forward pass."""
    eq = pytest.importorskip("test_nn_equivalence")
    net = _make_relu_net(with_bounds=True)
    # relu_bigm exact encoding -> 1e-6 tolerance per the harness convention.
    eq.assert_embedding_matches(net, strategy="relu_bigm", tol=1e-6)


# ---------------------------------------------------------------------------
# Input-length validation
# ---------------------------------------------------------------------------


def test_input_length_mismatch_raises_clear_message():
    """inputs length != n_features -> clear ValueError up front."""
    net = _make_relu_net(n_in=2, with_bounds=True)
    m = dm.Model("mismatch")
    x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)  # wrong: 3 != 2
    with pytest.raises(ValueError, match=r"length 3.*expects 2 features"):
        add_predictor(m, x, net)


# ---------------------------------------------------------------------------
# No bounds anywhere
# ---------------------------------------------------------------------------


def test_no_bounds_anywhere_raises_informative_error():
    """No net bounds, no arg, non-finite inputs bounds -> the 3-ways ValueError."""
    net = _make_relu_net(with_bounds=False)
    m = dm.Model("nobounds")
    x = m.continuous("x", shape=(2,))  # default +/-9.999e19 sentinels
    with pytest.raises(ValueError, match=r"three ways"):
        add_predictor(m, x, net)


def test_input_bounds_arg_satisfies_requirement():
    """Explicit input_bounds arg is accepted even with unbounded inputs var."""
    net = _make_relu_net(with_bounds=False)
    m = dm.Model("argbounds")
    x = m.continuous("x", shape=(2,))  # unbounded
    outputs, form = add_predictor(m, x, net, input_bounds=(-np.ones(2), np.ones(2)))
    lb, ub = form._network.input_bounds
    np.testing.assert_allclose(lb, [-1.0, -1.0])
    np.testing.assert_allclose(ub, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Missing file path
# ---------------------------------------------------------------------------


def test_missing_onnx_path_raises_file_not_found(tmp_path):
    m = dm.Model("missingfile")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    missing = tmp_path / "does_not_exist.onnx"
    with pytest.raises(FileNotFoundError):
        add_predictor(m, x, str(missing))


def test_missing_plain_path_raises_file_not_found(tmp_path):
    """A missing non-onnx path is also FileNotFoundError (existence checked first)."""
    m = dm.Model("missingfile2")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        add_predictor(m, x, missing)


def test_existing_non_onnx_path_raises_type_error(tmp_path):
    """An existing but unsupported file format still raises TypeError."""
    m = dm.Model("badformat")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    bad = tmp_path / "model.txt"
    bad.write_text("not a model")
    with pytest.raises(TypeError, match=r"[Uu]nsupported"):
        add_predictor(m, x, str(bad))
