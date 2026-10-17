"""#1743: post-HiGHS dual recovery reads builder-resident rows without an evaluator.

A per-element LP built with ``Model.constraint(Set, rule)`` takes the linear fast
path, so its rows live in ``model._builder_linear_blocks`` (#840), not in
``model._constraints``. ``_affine_constraint_rows`` used to decline on any such
row, and once the model was past the tape's node cap the HiGHS route built the
JAX evaluator -- a trace over 1M terms -- just to read an affine Jacobian.

These tests hold the affine walk to the evaluator's own ``J``, ``g(0)`` and row
map on builder rows of every sense, and check that the HiGHS route's named duals
are unchanged while no evaluator is built.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
from discopt import Model


def _per_element_transport(n: int, seed: int = 0) -> Model:
    """The #1743 repro shape: per-element rule rows over a 2-D set."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(50, 100, n), rng.uniform(10, 40, n), rng.uniform(1, 10, n * n)
    m = Model("transport")
    P = m.set("P", [f"p{k}" for k in range(n)])
    K = m.set("K", [f"m{k}" for k in range(n)])
    A = {p: a[i] for i, p in enumerate(P)}
    B = {k: b[j] for j, k in enumerate(K)}
    C = {(p, k): c[i * n + j] for i, p in enumerate(P) for j, k in enumerate(K)}
    x = m.continuous("x", over=P * K, lb=0, ub=1e4)
    m.minimize(dm.sum(C[(p, k)] * x[p, k] for p in P for k in K))
    m.constraint(P, lambda p: dm.sum(x[p, k] for k in K) <= A[p], name="sup")
    m.constraint(K, lambda k: dm.sum(x[p, k] for p in P) >= B[k], name="dem")
    return m


def _mixed_with_builder_rows() -> Model:
    """Expression constraints first, then builder rows of all three senses, on a
    variable that is not the first in the flat layout (a nonzero column offset)."""
    rng = np.random.default_rng(3)
    m = Model("mixed")
    z = m.continuous("z", shape=(2,), lb=-5, ub=5)
    y = m.continuous("y", shape=(4,), lb=-5, ub=5)
    m.minimize(dm.sum(rng.normal(size=4) * y) + z[0] - z[1])
    m.subject_to(z[0] + 2.0 * y[1] <= 3.0, name="expr_le")
    m.subject_to(y - z[1] >= -1.0, name="expr_vec")
    m.add_linear_constraints(rng.normal(size=(2, 4)), y, "<=", np.array([1.0, 2.0]), name="le")
    m.add_linear_constraints(rng.normal(size=(2, 4)), y, ">=", np.array([-3.0, -4.0]), name="ge")
    m.add_linear_constraints(np.ones((1, 4)), y, "==", 0.5, name="eq")
    assert m._builder_linear_blocks  # the probe exercises builder rows
    return m


def _assert_rows_match_evaluator(m: Model) -> int:
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solver import _affine_constraint_rows, _csr_rows

    n = sum(v.size for v in m._variables)
    got = _affine_constraint_rows(m, n)
    assert got is not None, "affine walk declined a model with builder rows"
    J, g0, row_map = got
    ev = make_evaluator(m)
    x0 = np.zeros(n)
    np.testing.assert_allclose(
        J.toarray(), _csr_rows(ev.evaluate_sparse_jacobian(x0), n).toarray(), atol=0, rtol=1e-15
    )
    np.testing.assert_allclose(
        g0, np.asarray(ev.evaluate_constraints(x0)).reshape(-1), atol=0, rtol=1e-15
    )

    def _key(rm):
        return [(a, b, c.name, c.sense) for a, b, c in rm]

    want = list(ev.constraint_row_map())
    assert _key(row_map) == _key(want)
    return len(want)


def test_affine_rows_match_evaluator_on_builder_rows():
    assert _assert_rows_match_evaluator(_mixed_with_builder_rows()) == 7


def test_affine_rows_match_evaluator_on_per_element_family():
    m = _per_element_transport(4)
    assert m._builder_linear_blocks  # Model.constraint took the fast path
    assert _assert_rows_match_evaluator(m) == 8


def _solve_duals(m: Model):
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.constraint_duals
    return r


def test_highs_duals_unchanged_and_no_evaluator_built(monkeypatch):
    """Baseline: the evaluator path (affine walk forced to decline). New: the
    affine walk, with every evaluator builder made to raise. Same named duals."""
    with monkeypatch.context() as mp:
        mp.setattr(solver_mod, "_affine_constraint_rows", lambda *_a, **_k: None)
        want = _solve_duals(_per_element_transport(5)).constraint_duals

    import discopt._tape_nlp_evaluator as tne

    def _boom(*_a, **_k):
        raise AssertionError("an evaluator was built for post-solve dual recovery")

    monkeypatch.setattr(tne, "build_evaluator", _boom)
    monkeypatch.setattr(solver_mod, "_make_evaluator", _boom)
    got = _solve_duals(_per_element_transport(5)).constraint_duals

    assert set(got) == set(want) and len(got) == 10
    for name in want:
        np.testing.assert_allclose(got[name], want[name], rtol=1e-12, atol=1e-12)
