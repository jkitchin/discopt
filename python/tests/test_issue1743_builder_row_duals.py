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

    # Only dual recovery is guarded: incumbent verification builds its own
    # evaluator and is not what this issue is about.
    inside = {"depth": 0, "calls": 0}
    real_matched = solver_mod._matched_row_constraint_duals
    real_build, real_make = tne.build_evaluator, solver_mod._make_evaluator

    def _matched(*a, **k):
        inside["depth"] += 1
        inside["calls"] += 1
        try:
            return real_matched(*a, **k)
        finally:
            inside["depth"] -= 1

    def _guard(real):
        def _wrapped(*a, **k):
            if inside["depth"]:
                raise AssertionError("an evaluator was built for post-solve dual recovery")
            return real(*a, **k)

        return _wrapped

    monkeypatch.setattr(solver_mod, "_matched_row_constraint_duals", _matched)
    monkeypatch.setattr(tne, "build_evaluator", _guard(real_build))
    monkeypatch.setattr(solver_mod, "_make_evaluator", _guard(real_make))
    got = _solve_duals(_per_element_transport(5)).constraint_duals

    assert inside["calls"] >= 1  # the probe fired: duals came through the matcher
    assert set(got) == set(want)
    assert {f"{f}_{i}" for f in ("sup", "dem") for i in range(5)} <= set(got)
    for name in want:
        np.testing.assert_allclose(got[name], want[name], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The post-solve consumers: AffineEvaluator instead of a tape/JAX build.
# ---------------------------------------------------------------------------


def _assert_matches_tape(m: Model) -> None:
    from discopt._affine_evaluator import AffineEvaluator
    from discopt._tape_nlp_evaluator import make_evaluator

    aff = AffineEvaluator.try_build(m)
    assert aff is not None
    ev = make_evaluator(m)
    assert type(ev).__name__ == "TapeNLPEvaluator"  # the reference is the tape
    n = sum(v.size for v in m._variables)
    assert aff.n_variables == ev.n_variables == n
    assert aff.n_constraints == ev.n_constraints
    rng = np.random.default_rng(11)
    for _ in range(3):
        x = rng.uniform(-3, 3, n)
        np.testing.assert_allclose(
            aff.evaluate_objective(x), ev.evaluate_objective(x), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            aff.evaluate_gradient(x), ev.evaluate_gradient(x), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            aff.evaluate_constraints(x), ev.evaluate_constraints(x), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            aff.evaluate_sparse_jacobian(x).toarray(), ev.evaluate_jacobian(x), rtol=0, atol=0
        )
    assert [(a, b, c.name, c.sense) for a, b, c in aff.constraint_row_map()] == [
        (a, b, c.name, c.sense) for a, b, c in ev.constraint_row_map()
    ]
    np.testing.assert_array_equal(aff.variable_bounds[0], ev.variable_bounds[0])
    np.testing.assert_array_equal(aff.variable_bounds[1], ev.variable_bounds[1])


def test_affine_evaluator_matches_tape_min():
    _assert_matches_tape(_mixed_with_builder_rows())


def test_affine_evaluator_matches_tape_max_and_canonical_sum():
    m = _per_element_transport(6)
    x = m._variables[0]
    m.maximize(-2.0 * dm.sum(x) + 7.0)
    _assert_matches_tape(m)
    assert m._builder_linear_blocks


def test_affine_evaluator_declines_non_affine():
    from discopt._affine_evaluator import AffineEvaluator

    m = _mixed_with_builder_rows()
    y = m._variables[1]
    m.minimize(dm.sum(y * y))  # a QP objective: rows are affine, the objective not
    assert AffineEvaluator.try_build(m) is None
    m2 = _mixed_with_builder_rows()
    m2.subject_to(dm.exp(m2._variables[0][0]) <= 3.0)
    assert AffineEvaluator.try_build(m2) is None


def test_verify_point_rejects_infeasible_point_on_affine_evaluator():
    from discopt._affine_evaluator import AffineEvaluator
    from discopt.validation.feasibility import verify_point

    m = _per_element_transport(4)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    x = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    aff = AffineEvaluator.try_build(m)
    assert verify_point(m, x, evaluator=aff).ok
    assert not verify_point(m, np.zeros_like(x), evaluator=aff).ok  # demand rows violated
    bumped = x.copy()
    bumped[0] += 200.0  # supply row 0 violated (a_0 <= 100)
    assert not verify_point(m, bumped, evaluator=aff).ok


def test_lp_solve_builds_no_evaluator(monkeypatch):
    """The whole HiGHS LP solve -- duals, backstop, repair, reconciliation -- on the
    #1743 shape builds neither a tape nor a JAX evaluator, and still certifies."""
    import discopt._tape_nlp_evaluator as tne

    calls = []
    real = tne.build_evaluator

    def _count(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(tne, "build_evaluator", _count)
    r = _per_element_transport(5).solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    assert r.constraint_duals
    assert calls == []
