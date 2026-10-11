"""Tests for linear model decision tree embedding (oblique splits, affine leaves).

The reference for every MILP answer is independent of the encoding: the tree
is evaluated by walking the v1 spec directly (``_spec_predict``), and the true
optimum of a linear objective over the tree is the best of one LP per leaf
(``_brute_force``), each solved with SciPy on the leaf polytope.
"""

from __future__ import annotations

import json

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ml import LinearTreeDefinition, LinearTreeFormulation
from discopt.ml.predictor import add_predictor
from scipy.optimize import linprog

LB = np.array([0.0, -1.0])
UB = np.array([2.0, 1.0])


def _spec(seed: int = 0, depth: int = 2, n_targets: int = 2, bounds=True) -> dict:
    """A random complete oblique tree in the v1 hyperplane-tree spec layout."""
    rng = np.random.default_rng(seed)
    p = 2
    n_int = 2**depth - 1
    a = rng.normal(size=(n_int, p))
    centre = (LB + UB) / 2
    t = a @ centre + rng.normal(scale=0.3, size=n_int)
    T = np.hstack([np.eye(p), a.T])
    n_t = T.shape[1]

    is_leaf, feature, threshold, left, right, params = [], [], [], [], [], []

    def emit(node, level):
        rid = len(is_leaf)
        is_leaf.append(None)
        feature.append(-1)
        threshold.append(None)
        left.append(-1)
        right.append(-1)
        params.append(np.zeros((1 + n_t, n_targets)))
        if level == depth:
            is_leaf[rid] = True
            P = np.zeros((1 + n_t, n_targets))
            P[0] = rng.normal(scale=2.0, size=n_targets)
            P[1 : 1 + p] = rng.normal(size=(p, n_targets))
            params[rid] = P
        else:
            is_leaf[rid] = False
            feature[rid] = p + node
            threshold[rid] = float(t[node])
            left[rid] = emit(2 * node + 1, level + 1)
            right[rid] = emit(2 * node + 2, level + 1)
        return rid

    emit(0, 0)
    spec = {
        "version": 1,
        "kind": "hyperplane_tree",
        "n_features_in": p,
        "n_features_transformed": n_t,
        "n_targets": n_targets,
        "transform_matrix": T.tolist(),
        "categorical_features": [],
        "linear_features": list(range(p)),
        "tree": {
            "node_count": len(is_leaf),
            "is_leaf": is_leaf,
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "leaf_params": [P.tolist() for P in params],
        },
    }
    if bounds:
        spec["difftree"] = {"bounds": [LB.tolist(), UB.tolist()]}
    return spec


def _spec_predict(spec: dict, x: np.ndarray) -> np.ndarray:
    """Walk the spec directly (the reference evaluator)."""
    T = np.asarray(spec["transform_matrix"])
    tr = spec["tree"]
    xt = x @ T
    node = 0
    while not tr["is_leaf"][node]:
        go_left = xt[tr["feature"][node]] <= tr["threshold"][node]
        node = tr["left"][node] if go_left else tr["right"][node]
    P = np.asarray(tr["leaf_params"][node])
    return np.concatenate([[1.0], xt]) @ P


def _brute_force(defn: LinearTreeDefinition, c: np.ndarray, split_eps: float, sense="max"):
    """Exact optimum of c . y over the tree: best of one LP per leaf."""
    best = -np.inf if sense == "max" else np.inf
    for leaf in defn.leaves:
        h = leaf.h.copy()
        ns = leaf.n_split_rows
        if ns:
            h[:ns] -= split_eps * np.linalg.norm(leaf.G[:ns], axis=1)
        obj = c @ leaf.A
        sign = -1.0 if sense == "max" else 1.0
        r = linprog(sign * obj, A_ub=leaf.G, b_ub=h, bounds=[(None, None)] * defn.n_features)
        if r.status == 2:  # infeasible leaf region
            continue
        assert r.status == 0, r.message
        val = obj @ r.x + c @ leaf.b
        best = max(best, val) if sense == "max" else min(best, val)
    return best


def _solve_max(defn, c, encoding, split_eps=1e-5, sense="max"):
    m = dm.Model("ltree")
    form = LinearTreeFormulation(m, defn, encoding=encoding, split_eps=split_eps)
    form.formulate()
    y = form.outputs
    obj = sum(float(c[k]) * y[k] for k in range(defn.n_outputs))
    m.maximize(obj) if sense == "max" else m.minimize(obj)
    r = m.solve(time_limit=60)
    assert r.status == "optimal", r.status
    x = np.asarray(r.x["ltree_input"]).ravel()
    yv = np.asarray(r.x["ltree_output"]).ravel()
    return float(r.objective), x, yv


# ─────────────────────────────────────────────────────────────
# Definition
# ─────────────────────────────────────────────────────────────


def test_from_spec_structure_and_predict():
    spec = _spec(seed=3)
    d = LinearTreeDefinition.from_spec(spec)
    assert d.n_leaves == 4 and d.n_features == 2 and d.n_outputs == 2
    for leaf in d.leaves:
        assert leaf.n_split_rows == 2
        assert leaf.G.shape == (2 + 4, 2)  # 2 splits + box
    rng = np.random.default_rng(0)
    for x in rng.uniform(LB, UB, size=(200, 2)):
        np.testing.assert_allclose(d.predict(x), _spec_predict(spec, x), rtol=0, atol=1e-12)


def test_from_spec_accepts_json_text_and_file(tmp_path):
    spec = _spec(seed=1)
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(spec))
    x = np.array([0.7, 0.2])
    want = _spec_predict(spec, x)
    for src in (json.dumps(spec), path, str(path)):
        np.testing.assert_allclose(LinearTreeDefinition.from_spec(src).predict(x), want)


def test_from_spec_rejects_bad_input():
    with pytest.raises(ValueError, match="hyperplane_tree"):
        LinearTreeDefinition.from_spec({**_spec(), "kind": "axis_tree"})
    with pytest.raises(NotImplementedError):
        LinearTreeDefinition.from_spec({**_spec(), "categorical_features": [0]})
    with pytest.raises(ValueError, match="input_bounds is required"):
        LinearTreeDefinition.from_spec(_spec(bounds=False))
    d = LinearTreeDefinition.from_spec(_spec(bounds=False), input_bounds=(LB, UB))
    assert np.array_equal(d.input_bounds[0], LB)
    with pytest.raises(ValueError, match="finite"):
        LinearTreeDefinition.from_spec(_spec(), input_bounds=(LB, np.array([np.inf, 1.0])))


def test_unknown_encoding_rejected():
    d = LinearTreeDefinition.from_spec(_spec())
    with pytest.raises(ValueError, match="Unknown encoding"):
        LinearTreeFormulation(dm.Model("x"), d, encoding="sos1")


# ─────────────────────────────────────────────────────────────
# Encodings against brute force
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("encoding", ["hull", "bigm"])
def test_smoke_single_solve(encoding):
    d = LinearTreeDefinition.from_spec(_spec(seed=0))
    c = np.array([1.0, -0.5])
    obj, x, y = _solve_max(d, c, encoding)
    assert obj == pytest.approx(_brute_force(d, c, 1e-5), abs=1e-6)


@pytest.mark.parametrize("encoding", ["hull", "bigm"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sense", ["max", "min"])
def test_optimum_matches_brute_force(encoding, seed, sense):
    """The MILP optimum equals the best per-leaf LP, and the tree evaluated at
    the returned x reproduces the returned y (the claimed leaf contains x)."""
    spec = _spec(seed=seed, depth=2 + seed % 2)
    d = LinearTreeDefinition.from_spec(spec)
    c = np.random.default_rng(100 + seed).normal(size=d.n_outputs)
    obj, x, y = _solve_max(d, c, encoding, sense=sense)
    assert obj == pytest.approx(_brute_force(d, c, 1e-5, sense=sense), abs=1e-6)
    np.testing.assert_allclose(y, _spec_predict(spec, x), atol=1e-6)


@pytest.mark.parametrize("encoding", ["hull", "bigm"])
def test_fixed_input_reproduces_prediction(encoding):
    """With the input pinned, the only feasible output is the tree's."""
    spec = _spec(seed=5, depth=3)
    d = LinearTreeDefinition.from_spec(spec)
    rng = np.random.default_rng(7)
    checked = 0
    for x0 in rng.uniform(LB, UB, size=(40, 2)):
        # Skip points inside the excluded split band.
        k = d.leaf_of(x0)
        leaf = d.leaves[k]
        slack = leaf.h[: leaf.n_split_rows] - leaf.G[: leaf.n_split_rows] @ x0
        if np.any(np.abs(slack) < 1e-3):
            continue
        m = dm.Model("pin")
        xin = m.continuous("u", shape=(2,), lb=x0, ub=x0)
        out, _ = add_predictor(m, xin, d, method=encoding, prefix="lt")
        m.minimize(out[0])
        r = m.solve(time_limit=30)
        assert r.status == "optimal"
        np.testing.assert_allclose(
            np.asarray(r.x["lt_output"]).ravel(), _spec_predict(spec, x0), atol=1e-6
        )
        checked += 1
    assert checked >= 20  # the probe really ran


def test_hull_and_bigm_agree():
    d = LinearTreeDefinition.from_spec(_spec(seed=4, depth=3))
    c = np.array([1.0, 2.0])
    obj_h, *_ = _solve_max(d, c, "hull")
    obj_b, *_ = _solve_max(d, c, "bigm")
    assert obj_h == pytest.approx(obj_b, abs=1e-6)


def test_add_predictor_accepts_spec_dict_and_json(tmp_path):
    spec = _spec(seed=2)
    path = tmp_path / "t.json"
    path.write_text(json.dumps(spec))
    for pred in (spec, path):
        m = dm.Model("ap")
        x = m.continuous("x", shape=(2,), lb=LB, ub=UB)
        out, form = add_predictor(m, x, pred, prefix="p")
        assert isinstance(form, LinearTreeFormulation)
        m.maximize(out[0])
        r = m.solve(time_limit=30)
        assert r.status == "optimal"
        d = LinearTreeDefinition.from_spec(spec)
        assert float(r.objective) == pytest.approx(
            _brute_force(d, np.array([1.0, 0.0]), 1e-5), abs=1e-6
        )
