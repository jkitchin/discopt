"""Functional coverage of node-level solve paths in ``discopt.solver`` (#87).

Third battery: Ipopt node NLPs (skipped without cyipopt), NLP-based B&B on
nonconvex models (heuristic mode) and its node callback, integer convex-QP
relaxation nodes, and the integer-multilinear reformulation route. Known
optima throughout; each solve is seconds.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


def _bilinear_binary_model():
    m = Model("node_mi")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + b)
    m.minimize(x + y + 2.0 * b)
    return m


def test_ipopt_node_nlp_backend():
    pytest.importorskip("cyipopt", reason="Ipopt node-NLP path requires cyipopt")
    m = _bilinear_binary_model()
    res = m.solve(nlp_solver="ipopt", time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_ipopt_backend_on_convex_nlp():
    pytest.importorskip("cyipopt", reason="requires cyipopt")
    m = Model("cvx_ipopt")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 1.0)
    m.minimize(x**2 + y**2)
    res = m.solve(nlp_solver="ipopt", time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.5, abs=1e-5)  # x=y=0.5


def test_nlp_bb_heuristic_mode_on_nonconvex():
    # NLP-based B&B on a nonconvex model runs in heuristic mode: it may not
    # certify (status 'feasible'), but the incumbent must be truly feasible
    # and can never beat the global optimum from below.
    m = _bilinear_binary_model()
    res = m.solve(nlp_bb=True, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective >= 2.0 - 1e-6
    assert res.objective == pytest.approx(2.0, abs=1e-3)
    assert res.x["x"] * res.x["y"] >= 1.0 - 1e-6


def test_nlp_bb_node_callback_fires():
    m = Model("nlpbb_cb")
    i = m.integer("i", lb=0, ub=5)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(i + z >= 2.5)
    m.minimize((i - 2.3) ** 2 + (z - 0.4) ** 2)
    seen = []

    def node_cb(ctx, model):
        seen.append(ctx.node_count)

    res = m.solve(nlp_bb=True, node_callback=node_cb, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    # Optimum: try i=2, z max(0.5, 0.4->but need i+z>=2.5 so z>=0.5): (2, .5)
    # -> .09+.01=.10 ; i=3, z=0.4 -> .49+0 = .49. So 0.10.
    assert res.objective == pytest.approx(0.10, abs=1e-4)
    assert seen, "node callback must fire in the NLP-B&B loop"


def test_integer_convex_qp_nodes():
    # Pure-integer convex QP: exercised through the QP relaxation node path.
    m = Model("iqp")
    i = m.integer("i", lb=-3, ub=3)
    j = m.integer("j", lb=-3, ub=3)
    m.subject_to(i + j >= 1)
    m.minimize((i - 1.3) ** 2 + (j - 0.7) ** 2)
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.18, abs=1e-6)  # (1, 1)


def test_integer_multilinear_reformulation_route():
    # A pure-integer trilinear product triggers the integer-multilinear
    # binarization reform when profitable; the certified optimum must match
    # exhaustive enumeration (-4 at e.g. (2, 0, 2)).
    m = Model("iml")
    i = m.integer("i", lb=0, ub=2)
    j = m.integer("j", lb=0, ub=2)
    k = m.integer("k", lb=0, ub=2)
    m.subject_to(i + j + k >= 3)
    m.minimize(i * j * k - i - j - k)
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(-4.0, abs=1e-6)


def test_integer_bilinear_maximize_certifies_boundary():
    # max i*j over integers with a coupling constraint: the boundary point
    # must win, guarding the neg-product decomposition (cf. amp helper docs).
    m = Model("ibmax")
    i = m.integer("i", lb=0, ub=4)
    j = m.integer("j", lb=0, ub=4)
    m.subject_to(i + j <= 5)
    m.maximize(i * j)
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(6.0, abs=1e-6)  # (2,3) or (3,2)
