"""#1568: in-tree FBBT expands array-valued rows element by element.

``discopt.ml``'s full-space form writes each layer as ONE vector row
(``zhat - (W.T @ x + b) == 0``, ``z - sigmoid(zhat) == 0``) over whole-array
references. The per-scalar FBBT view (#1513) pointed every whole-array
reference at a hull proxy that is never tightened, so a branch on an input
moved nothing downstream: measured on a 2-10-1 sigmoid net with ``x0`` branched
to [0, 1] and ``x1`` to [-1, 0], the vector rows tightened 2 half-bounds where
the same rows written per element tightened 26. ``DISCOPT_FBBT_ARRAY_ROWS``
(the ``expand_array_rows`` kwarg) appends one scalar row per element.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._rust import model_to_repr
from discopt.export._arrays import scalarize_body
from discopt.modeling.core import Constraint

pytest.importorskip("discopt.ml")
from discopt.ml import add_predictor  # noqa: E402
from discopt.ml.network import DenseLayer, NetworkDefinition  # noqa: E402


def _net_model(scalar_rows: bool) -> dm.Model:
    rng = np.random.default_rng(0)
    w1, b1 = rng.normal(0, 1 / np.sqrt(2), size=(2, 10)), rng.normal(0, 0.1, size=10)
    w2, b2 = rng.normal(0, 1 / np.sqrt(10), size=(10, 1)), rng.normal(0, 0.1, size=1)
    m = dm.Model("nn")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    net = NetworkDefinition(
        [DenseLayer(w1, b1, "sigmoid"), DenseLayer(w2, b2, "linear")],
        input_bounds=(np.full(2, -1.0), np.full(2, 1.0)),
    )
    y, _ = add_predictor(m, x, net, method="full_space")
    m.minimize(y[0])
    if scalar_rows:
        m._constraints = [
            Constraint(body=b, sense=c.sense, rhs=0.0)
            for c in m._constraints
            for b in scalarize_body(c.body)
        ]
    return m


def _branched_box(repr_):
    lb = np.concatenate([np.atleast_1d(repr_.var_lb(i)) for i in range(repr_.n_var_blocks)])
    ub = np.concatenate([np.atleast_1d(repr_.var_ub(i)) for i in range(repr_.n_var_blocks)])
    lb, ub = lb.astype(float), ub.astype(float)
    lb[0], ub[1] = 0.0, 0.0  # x0 in [0, 1], x1 in [-1, 0]
    return lb, ub


def _run(m, expand):
    r = model_to_repr(m, getattr(m, "_builder", None))
    lb, ub = _branched_box(r)
    return r.in_tree_presolve(lb, ub, node_depth=0, depth_stride=1, expand_array_rows=expand)


def test_vector_rows_match_scalar_rows_when_expanded():
    vec, sca = _net_model(False), _net_model(True)
    assert len(vec._constraints) < len(sca._constraints)
    ref = _run(sca, False)
    off = _run(vec, False)
    on = _run(vec, True)
    assert ref["ran"] and off["ran"] and on["ran"]
    # Before: the vector rows move (almost) nothing -- every hidden unit is a
    # hull proxy.
    assert int(off["bounds_tightened"]) < int(ref["bounds_tightened"]) // 4
    assert int(off["array_rows_added"]) == 0
    # After: exactly the box of the per-element rows.
    assert int(on["array_rows_added"]) > 0
    assert int(on["array_rows_on_hull"]) == 0
    np.testing.assert_allclose(on["lb"], ref["lb"], atol=1e-9)
    np.testing.assert_allclose(on["ub"], ref["ub"], atol=1e-9)
    # And never looser than OFF.
    assert np.all(on["lb"] >= off["lb"] - 1e-12)
    assert np.all(on["ub"] <= off["ub"] + 1e-12)


def test_expansion_never_cuts_a_point_on_the_network():
    """Feasible-point sampling: points ON the net, inside random sub-boxes."""
    m = _net_model(False)
    r = model_to_repr(m, getattr(m, "_builder", None))
    rlb, rub = _branched_box(r)
    rlb[0], rub[1] = -1.0, 1.0
    rng = np.random.default_rng(1568)
    # Recover the weights the model was built from (same seed as _net_model).
    g = np.random.default_rng(0)
    w1, b1 = g.normal(0, 1 / np.sqrt(2), size=(2, 10)), g.normal(0, 0.1, size=10)
    w2, b2 = g.normal(0, 1 / np.sqrt(10), size=(10, 1)), g.normal(0, 0.1, size=1)
    checked = 0
    for _ in range(200):
        x = rng.uniform(-1, 1, size=2)
        zh = x @ w1 + b1
        z = 1 / (1 + np.exp(-zh))
        y = z @ w2 + b2
        # Slot order follows the variable blocks.
        vals = {"x": x}
        for v in m._variables:
            if v.name == "x":
                continue
            if v.size == 2:
                vals[v.name] = x
            elif v.size == 1:
                vals[v.name] = y
            elif "zhat" in v.name:
                vals[v.name] = zh
            else:
                vals[v.name] = z
        p = np.concatenate([np.ravel(vals[v.name]) for v in m._variables])
        assert p.shape == rlb.shape
        lb = p - (p - rlb) * rng.uniform(size=p.size)
        ub = p + (rub - p) * rng.uniform(size=p.size)
        d = r.in_tree_presolve(lb, ub, node_depth=0, depth_stride=1, expand_array_rows=True)
        assert not d["infeasible"], p
        assert np.all(d["lb"] <= p + 1e-7) and np.all(p <= d["ub"] + 1e-7), p
        checked += 1
    assert checked == 200


def test_flag_parsing_default_on_with_opt_out(monkeypatch):
    """Graduated default ON (§5 panel in the #1568 PR); ``=0`` opts out."""
    monkeypatch.delenv("DISCOPT_FBBT_ARRAY_ROWS", raising=False)
    assert S._fbbt_array_rows_enabled() is True
    for v in ("0", "false", "off", "no"):
        monkeypatch.setenv("DISCOPT_FBBT_ARRAY_ROWS", v)
        assert S._fbbt_array_rows_enabled() is False
    monkeypatch.setenv("DISCOPT_FBBT_ARRAY_ROWS", "1")
    assert S._fbbt_array_rows_enabled() is True


def test_default_solve_uses_expanded_rows_and_opt_out_does_not(monkeypatch):
    """End to end: the default solve reaches the expansion in the node loop and
    certifies the 2-10-1 net; the ``=0`` opt-out never fires it."""
    monkeypatch.delenv("DISCOPT_FBBT_ARRAY_ROWS", raising=False)
    r_on = _net_model(False).solve(time_limit=60)
    assert S._in_tree_array_row_calls() > 0
    assert r_on.status == "optimal" and r_on.gap_certified
    assert r_on.bound <= r_on.objective + 1e-9
    monkeypatch.setenv("DISCOPT_FBBT_ARRAY_ROWS", "0")
    r_off = _net_model(False).solve(time_limit=5)
    assert S._in_tree_array_row_calls() == 0
    assert r_off.objective is not None
    assert r_off.bound <= r_on.objective + 1e-9
