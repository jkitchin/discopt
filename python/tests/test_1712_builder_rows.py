"""#1712: builder rows reach the tape evaluator without building their expressions.

``Model._builder_linear_constraints(lazy=True)`` defers each row's body to its first
read, and ``compile_builder_row`` lowers the row from its coefficients. Both must be
indistinguishable from the eager path: same body when read, same tape when lowered.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt._nl_expr_compiler import compile_builder_row, compile_to_nl_array
from discopt.export._common import iter_builder_linear_rows
from discopt.modeling.core import Constraint, builder_row_of


def _model(rng) -> Model:
    m = Model("rows")
    s = m.continuous("s", lb=-2, ub=2)  # scalar variable
    x = m.continuous("x", shape=(7,), lb=-5, ub=5)
    y = m.continuous("y", shape=(3, 4), lb=0, ub=3)
    m.minimize(dm.sum(x) + dm.sum(y) + s)
    for b, (var, n) in enumerate([(x, 7), (y, 12), (x, 7), (s, 1)]):
        rows = int(rng.integers(3, 6))
        A = rng.integers(-3, 4, size=(rows, n)).astype(float)
        A[0, :] = 0.0
        A[0, : min(n, 2)] = 1.0  # a short row: <= 2 terms -> generic fallback
        coo = sp.coo_matrix(A)
        # Repeated columns, and a coefficient equal to the right-hand side.
        r_ = np.concatenate([coo.row, [rows - 1, rows - 1]])
        c_ = np.concatenate([coo.col, [0, 0]])
        v_ = np.concatenate([coo.data, [0.5, 2.0]])
        A = sp.coo_matrix((v_, (r_, c_)), shape=A.shape)
        rhs = rng.uniform(-3, 3, rows)
        rhs[-1] = 2.0
        m.add_linear_constraints(A, var, ("<=", ">=", "==", "<=")[b], rhs, name=f"b{b}")
    S = m.set("S", ["a", "b", "c"])
    m.constraint(S, lambda k: dm.sum(x[i] for i in range(5)) + 2.0 * s >= -9.0, name="fam")
    return m


def _rows_and_eager(m):
    return list(zip(iter_builder_linear_rows(m), m._builder_linear_constraints()))


@pytest.mark.smoke
def test_lazy_constraints_build_the_eager_body():
    m = _model(np.random.default_rng(0))
    eager = m._builder_linear_constraints()
    lazy = m._builder_linear_constraints(lazy=True)
    assert len(eager) == len(lazy) > 0
    for e, lz in zip(eager, lazy):
        assert type(lz) is Constraint
        assert builder_row_of(lz) is not None  # nothing built yet
        assert (lz.name, lz.sense, lz.rhs) == (e.name, e.sense, e.rhs)
        assert repr(lz.body) == repr(e.body)
        assert builder_row_of(lz) is None  # built once, now an ordinary attribute
        assert lz.body is lz.body


@pytest.mark.smoke
def test_builder_row_tape_matches_materialized():
    """The direct lowering is the generic lowering of the materialized body."""
    rng = np.random.default_rng(1712)
    direct = fallback = 0
    for _ in range(8):
        m = _model(rng)
        n = sum(v.size for v in m._variables)
        for row, eager in _rows_and_eager(m):
            built = compile_builder_row(row, m)
            if built is None:
                assert len(row.terms) <= 2
                fallback += 1
                continue
            ref = compile_to_nl_array(eager.body, m).reshape(-1)
            assert built.shape == ref.shape == (1,)
            a, b = built[0], ref[0]
            assert repr(a) == repr(b)
            assert a.depth == b.depth
            for _ in range(3):
                pt = rng.normal(size=n) * float(rng.choice([1e-3, 1.0, 1e5]))
                assert np.float64(a.eval(pt)).tobytes() == np.float64(b.eval(pt)).tobytes()
                ga = np.asarray(a.gradient(pt), dtype=np.float64)
                gb = np.asarray(b.gradient(pt), dtype=np.float64)
                assert ga.tobytes() == gb.tobytes()
            direct += 1
    assert direct > 50 and fallback > 0  # both paths exercised


@pytest.mark.smoke
def test_tape_evaluator_unchanged_by_lazy_rows(monkeypatch):
    """End to end: the evaluator's rows, Jacobian and row map are the same whether
    builder rows are lowered directly or compiled from their bodies."""
    import discopt.modeling.core as core
    from discopt._tape_nlp_evaluator import make_evaluator

    n = sum(v.size for v in _model(np.random.default_rng(7))._variables)
    pts = [np.random.default_rng(k).normal(size=n) for k in range(3)]

    def snapshot():
        # A fresh model each time: the evaluator cache hangs off the model.
        ev = make_evaluator(_model(np.random.default_rng(7)))
        out = [
            (
                np.asarray(ev.evaluate_constraints(p)).tobytes(),
                np.asarray(ev.evaluate_jacobian(p)).tobytes(),
            )
            for p in pts
        ]
        return out, [(a, b, c.name, c.sense) for a, b, c in ev.constraint_row_map()]

    import discopt._tape_nlp_evaluator as tne

    direct = []
    real = tne.compile_builder_row
    monkeypatch.setattr(
        tne, "compile_builder_row", lambda row, mdl: direct.append(1) or real(row, mdl)
    )
    fast = snapshot()
    assert len(direct) > 10  # the direct lowering ran
    with monkeypatch.context() as mp:
        mp.setattr(core, "builder_row_of", lambda c: None)  # force the generic compile
        slow = snapshot()
    assert fast == slow


def test_lp_solve_builds_no_builder_row_body(monkeypatch):
    """No consumer on the HiGHS LP path reads a builder row's body."""
    import discopt.modeling.core as core

    built = []
    real = core._builder_row_constraint
    monkeypatch.setattr(core, "_builder_row_constraint", lambda row: built.append(1) or real(row))
    rng = np.random.default_rng(0)
    m = Model("lp")
    x = m.continuous("x", shape=(30,), lb=0, ub=10)
    m.minimize(dm.sum(rng.uniform(1, 2, 30) * x))
    A = sp.random(12, 30, density=0.4, random_state=1) + sp.eye(12, 30)
    m.add_linear_constraints(A, x, ">=", 1.0, name="cov")
    r = m.solve()
    assert r.status == "optimal" and r.gap_certified
    assert "cov" in (r.constraint_duals or {})
    assert built == []
