"""Equivalence-test harness for embedded NN / tree predictors (task T-N0.1).

The product of ``discopt.nn`` is the equivalence
``solve(embedded model) == trained predictor``. This module provides two
reusable, importable helpers that every later nn-fix regression test builds on:

* :func:`assert_embedding_matches` — *fixed-input* equivalence: pin the
  formulation's inputs to sampled points, solve, and compare the output
  variables to the reference prediction.
* :func:`assert_optimum_matches` — *optimization* equivalence: minimize/maximize
  an embedded output over the input box and compare the certified optimum against
  dense enumeration of the reference predictor (this is the harness that must
  detect the F1/F2 "cut-the-optimum" failure mode).

Reference-prediction semantics (see nn-module-plan.md §2):
``predict(x) = y_factor * net.forward((x - x_offset)/x_factor) + y_offset``
elementwise (identity when ``scaling is None``); for trees the reference is
``ensemble.predict(x)`` (scaling is not wired into the tree formulation).

This PR is harness + a green baseline over the *current, unmodified* nn module:
only the currently-correct regimes are exercised (identity/no scaling, in-box
tree thresholds). Non-identity scaling and out-of-box thresholds are the F1/F2
regression cases and belong to the fix PRs (T-N0.2 / T-N0.3).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools

import numpy as np
import pytest
from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.tree import DecisionTree, TreeEnsembleDefinition

# ---------------------------------------------------------------------------
# Reference predictor + input-box helpers
# ---------------------------------------------------------------------------


def _is_ensemble(obj) -> bool:
    return isinstance(obj, TreeEnsembleDefinition)


def _input_box(obj, feature_bounds):
    """Return ``(lb, ub)`` float arrays for the predictor's input box."""
    if feature_bounds is not None:
        lb, ub = feature_bounds
    else:
        ib = obj.input_bounds
        if ib is None:
            raise ValueError("predictor has no input_bounds; pass feature_bounds=")
        lb, ub = ib
    return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)


def _n_features(obj) -> int:
    return int(obj.n_features) if _is_ensemble(obj) else int(obj.input_size)


def _reference_predict(obj, x, scaling) -> np.ndarray:
    """Reference prediction as a 1-D array (matches ``outputs`` var shape)."""
    x = np.asarray(x, dtype=np.float64)
    if _is_ensemble(obj):
        # Scaling is intentionally not wired into the tree formulation.
        return np.array([obj.predict(x)], dtype=np.float64)
    if scaling is None:
        return np.asarray(obj.forward(x), dtype=np.float64)
    xs = (x - scaling.x_offset) / scaling.x_factor
    return scaling.y_factor * np.asarray(obj.forward(xs), dtype=np.float64) + scaling.y_offset


def _default_strategy(obj) -> str:
    """relu_bigm if the net has any ReLU layer, else full_space."""
    if _is_ensemble(obj):
        return "tree"
    has_relu = any(layer.activation == Activation.RELU for layer in obj.layers)
    return "relu_bigm" if has_relu else "full_space"


def _make_formulation(m, obj, strategy, scaling):
    """Add the appropriate formulation to model ``m``; return the form object."""
    if _is_ensemble(obj):
        form = TreeFormulation(m, obj, prefix="e")
    else:
        form = NNFormulation(m, obj, strategy=strategy, prefix="e", scaling=scaling)
    form.formulate()
    return form


def _tree_thresholds_by_feature(ens, lb, ub) -> dict[int, list[float]]:
    """In-box split thresholds, grouped by feature (for the eps-exclusion band)."""
    out: dict[int, list[float]] = {}
    for tree in ens.trees:
        for node in range(len(tree.feature)):
            if int(tree.feature[node]) == -1:
                continue
            j = int(tree.feature[node])
            thr = float(tree.threshold[node])
            if lb[j] < thr < ub[j]:
                out.setdefault(j, []).append(thr)
    return out


# ---------------------------------------------------------------------------
# Public helpers (imported by later regression tests)
# ---------------------------------------------------------------------------


def assert_embedding_matches(
    net_or_ensemble,
    *,
    strategy=None,
    scaling=None,
    n_points=6,
    tol=1e-6,
    seed=0,
    feature_bounds=None,
):
    """Fixed-input equivalence: embedded outputs == reference prediction.

    Samples ``n_points`` inputs uniformly inside the input box, pins the
    formulation's inputs to each sample with equality constraints, solves, and
    asserts the output variables match the reference prediction to ``tol``.

    ``tol`` is chosen by the caller: ``1e-6`` for MILP-exact encodings whose
    output is a continuous function of the pinned inputs (relu_bigm, linear
    full_space), ``1e-5`` for the tree encoding (its output reads leaf-selection
    binaries back directly, so it inherits the solver's integrality tolerance),
    and ``1e-4`` for smooth-activation NLP encodings (sigmoid/tanh/softplus).
    """
    import discopt.modeling as dm

    obj = net_or_ensemble
    if strategy is None:
        strategy = _default_strategy(obj)
    lb, ub = _input_box(obj, feature_bounds)
    n = _n_features(obj)
    rng = np.random.RandomState(seed)

    band = None
    if _is_ensemble(obj):
        band = _tree_thresholds_by_feature(obj, lb, ub)

    def _accept(x) -> bool:
        if not band:
            return True
        for j, thrs in band.items():
            for thr in thrs:
                if abs(x[j] - thr) < 1e-3:
                    return False
        return True

    points = []
    attempts = 0
    while len(points) < n_points:
        attempts += 1
        if attempts > 1000 * n_points:
            raise RuntimeError("could not sample points outside the eps-exclusion band")
        x = lb + rng.random_sample(n) * (ub - lb)
        if _accept(x):
            points.append(x)

    for x in points:
        m = dm.Model("nn_fixed")
        form = _make_formulation(m, obj, strategy, scaling)
        for j in range(n):
            m.subject_to(form.inputs[j] == float(x[j]), name=f"pin_{j}")
        m.minimize(form.outputs[0])
        r = m.solve()
        assert r.status == "optimal", f"solve status {r.status!r} at x={x}"
        got = np.asarray(r.value(form.outputs), dtype=np.float64)
        ref = _reference_predict(obj, x, scaling)
        np.testing.assert_allclose(
            got,
            ref,
            atol=tol,
            rtol=0,
            err_msg=f"embedded output != reference at x={x} (strategy={strategy})",
        )


def assert_optimum_matches(
    net_or_ensemble,
    *,
    strategy=None,
    scaling=None,
    sense="min",
    tol=1e-4,
    grid=21,
    n_random=200,
    seed=0,
    feature_bounds=None,
):
    """Optimization equivalence: certified optimum == enumerated optimum.

    Embeds the predictor once, minimizes/maximizes ``outputs[0]`` over the input
    box, and compares the certified optimum against dense enumeration of the
    reference prediction (per-feature ``grid`` points plus ``n_random`` random
    points).

    Two guarantees are checked:

    (a) *incumbent consistency* — the reported optimum must be an achievable
        prediction: ``predict(r.value(inputs))[0] == r.objective`` to ``tol``.
    (b) *no cut optimum* — for ``min`` the certified optimum must not sit above
        the best sampled value: ``r.objective <= ref_min_sampled + tol`` (and the
        mirror for ``max``). A certified optimum strictly *below* every enumerated
        point beyond ``tol`` is the F1/F2 cut-the-optimum failure — guarantee (a)
        catches that (the incumbent would not re-evaluate to the objective).

    Use tiny predictors (1-2 inputs) so the grid stays cheap.
    """
    import discopt.modeling as dm

    obj = net_or_ensemble
    if strategy is None:
        strategy = _default_strategy(obj)
    lb, ub = _input_box(obj, feature_bounds)
    n = _n_features(obj)

    m = dm.Model("nn_opt")
    form = _make_formulation(m, obj, strategy, scaling)
    if sense == "min":
        m.minimize(form.outputs[0])
    elif sense == "max":
        m.maximize(form.outputs[0])
    else:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    r = m.solve()
    assert r.status == "optimal", f"solve status {r.status!r}"

    # Enumerate the reference predictor over a dense grid + random points.
    axes = [np.linspace(lb[j], ub[j], grid) for j in range(n)]
    ref_vals = [
        float(_reference_predict(obj, np.array(pt), scaling)[0]) for pt in itertools.product(*axes)
    ]
    rng = np.random.RandomState(seed)
    for _ in range(n_random):
        x = lb + rng.random_sample(n) * (ub - lb)
        ref_vals.append(float(_reference_predict(obj, x, scaling)[0]))
    ref_min = min(ref_vals)
    ref_max = max(ref_vals)

    # (a) incumbent consistency: reported optimum is an achievable prediction.
    x_star = np.asarray(r.value(form.inputs), dtype=np.float64)
    incumbent = float(_reference_predict(obj, x_star, scaling)[0])
    assert abs(incumbent - r.objective) <= tol, (
        f"incumbent {incumbent} != reported objective {r.objective} "
        f"(inputs {x_star} do not re-evaluate through the predictor)"
    )

    # (b) certificate not above the best sampled value (no cut optimum).
    if sense == "min":
        assert r.objective <= ref_min + tol, (
            f"certified min {r.objective} > enumerated min {ref_min} + {tol}"
        )
    else:
        assert r.objective >= ref_max - tol, (
            f"certified max {r.objective} < enumerated max {ref_max} - {tol}"
        )


# ---------------------------------------------------------------------------
# Small synthetic predictor builders (adapted from test_nn_formulations.py)
# ---------------------------------------------------------------------------


def _make_linear_net(n_in=2, n_out=1):
    rng = np.random.RandomState(0)
    W = rng.randn(n_in, n_out)
    b = rng.randn(n_out)
    return NetworkDefinition(
        [DenseLayer(W, b, Activation.LINEAR)],
        input_bounds=(-np.ones(n_in), np.ones(n_in)),
    )


def _make_relu_net():
    """Two-layer ReLU -> LINEAR net (2 inputs, 3 hidden, 1 output)."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(2, 3)
    b1 = rng.randn(3)
    W2 = rng.randn(3, 1)
    b2 = rng.randn(1)
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.RELU),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    )


def _make_relu_net_1d():
    """One-input ReLU -> LINEAR net (1 input, 3 hidden, 1 output)."""
    rng = np.random.RandomState(11)
    W1 = rng.randn(1, 3)
    b1 = rng.randn(3)
    W2 = rng.randn(3, 1)
    b2 = rng.randn(1)
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.RELU),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0]), np.array([1.0])),
    )


def _make_sigmoid_net():
    """Tiny sigmoid -> LINEAR net (1 input, 1 hidden, 1 output).

    Kept 1-D with small, well-conditioned weights on purpose: smooth activations
    are relaxed with McCormick envelopes and solved by spatial branch-and-bound,
    whose cost explodes with input dimension and pre-activation width. This net
    keeps every fixed-input solve well under a second (measured ~0.1-0.7s); large
    random weights make the same 1-D solve take minutes.
    """
    W1 = np.array([[0.8]])
    b1 = np.array([0.1])
    W2 = np.array([[1.2]])
    b2 = np.array([-0.3])
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.SIGMOID),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0]), np.array([1.0])),
    )


def _make_tanh_net():
    """Tiny single-layer tanh net (1 input, 1 output). See _make_sigmoid_net."""
    W = np.array([[1.0]])
    b = np.array([0.0])
    return NetworkDefinition(
        [DenseLayer(W, b, Activation.TANH)],
        input_bounds=(np.array([-1.0]), np.array([1.0])),
    )


def _make_simple_tree():
    return DecisionTree(
        n_features=2,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 3.0]),
    )


def _make_tree_ensemble():
    """Two-tree ensemble with 2 features (in-box thresholds)."""
    tree1 = _make_simple_tree()
    tree2 = DecisionTree(
        n_features=2,
        feature=np.array([1, -1, -1]),
        threshold=np.array([0.0, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, -1.0, 2.0]),
    )
    return TreeEnsembleDefinition(
        trees=[tree1, tree2],
        n_features=2,
        base_score=0.5,
        input_bounds=(np.array([0.0, -1.0]), np.array([1.0, 1.0])),
    )


# ---------------------------------------------------------------------------
# Green baseline: fixed-input equivalence over currently-correct regimes
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_relu_bigm_fixed_input():
    assert_embedding_matches(_make_relu_net(), strategy="relu_bigm", tol=1e-6)


@pytest.mark.smoke
def test_linear_full_space_fixed_input():
    assert_embedding_matches(_make_linear_net(), strategy="full_space", tol=1e-6)


def test_sigmoid_full_space_fixed_input():
    assert_embedding_matches(_make_sigmoid_net(), strategy="full_space", tol=1e-4)


def test_tanh_reduced_space_fixed_input():
    assert_embedding_matches(_make_tanh_net(), strategy="reduced_space", tol=1e-4)


def test_relu_reduced_space_fixed_input():
    assert_embedding_matches(_make_relu_net(), strategy="reduced_space", tol=1e-4)


@pytest.mark.smoke
def test_tree_ensemble_fixed_input():
    # The tree output is a direct linear read-back of leaf-selection binaries
    # (outputs == sum(z_leaf * leaf_value)). Those binaries are integral only to
    # the solver's integrality tolerance (1e-5, conftest), so the read-back
    # inherits that floor even though the correct leaf is always selected
    # (a wrong leaf would differ by an O(1) leaf-value gap, not 1e-6). 1e-5 is
    # therefore the tightest sound tolerance for this exact MILP encoding.
    assert_embedding_matches(_make_tree_ensemble(), tol=1e-5)


# ---------------------------------------------------------------------------
# Green baseline: optimization equivalence
# ---------------------------------------------------------------------------


def test_relu_1d_optimum():
    assert_optimum_matches(_make_relu_net_1d(), strategy="relu_bigm", sense="min", tol=1e-4)


def test_tanh_optimum():
    assert_optimum_matches(_make_tanh_net(), strategy="reduced_space", sense="min", tol=1e-4)


def test_tree_ensemble_optimum():
    assert_optimum_matches(_make_tree_ensemble(), sense="min", tol=1e-4)
