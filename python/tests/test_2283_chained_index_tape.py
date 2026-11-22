"""#2283: a chained integer index ``x[k][s]`` lowers to ONE tape slot, on both paths.

``y = x[k]; ... y[s]`` is how every time-stepped transcription addresses a state.
Before #2283 neither tape lowering resolved it:

* the Python walk (``_nl_expr_compiler._static_scalar_slot``) accepted only a
  ``Variable`` base, so each ``y[s]`` fell to the general path, which
  materializes the WHOLE variable (one ``E.var`` per slot) before indexing --
  O(n_variables) per row, O(rows * n_variables) per build. Measured on the
  issue's reproducer (14,336 rows, 14,374 variables): 60.3 s -> 0.63 s;
* the Rust ``tape_program`` emitted ``OP_UNSUPPORTED`` for the same leaf, which
  refused the arena tape for the whole model and sent it to that slow walk.

Both are bound-neutral marshaling changes, so the bar is exact equality with the
``x[k, s]`` spelling and between the two paths, not a tolerance.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

pounce = pytest.importorskip("pounce")

from discopt._nl_expr_compiler import (  # noqa: E402
    _Budget,
    _lower,
    _static_scalar_slot,
    compile_to_nl_expr,
)


def _stepped_model(n_steps: int = 6, n_states: int = 5):
    """Implicit-Euler rows written with the ``y = x[k + 1]`` idiom."""
    m = dm.Model("chained_index")
    x = m.continuous("x", shape=(n_steps + 1, n_states), lb=-1, ub=2)
    p = m.continuous("p", shape=(3,), lb=-1, ub=1)
    for k in range(n_steps):
        y = x[k + 1]
        for s in range(n_states):
            rhs = dm.exp(p[s % 3]) * y[s] * y[(s + 1) % n_states] - y[(s + 2) % n_states]
            m.subject_to(y[s] - x[k, s] - 0.1 * rhs == 0)
    m.minimize(dm.sum([(x[n_steps, s] - 0.5) ** 2 for s in range(n_states)]))
    return m, x


@pytest.mark.parametrize(
    "chained, direct",
    [
        (lambda x: x[2][3], lambda x: x[2, 3]),
        (lambda x: x[-1][-2], lambda x: x[6, 3]),
        (lambda x: x[(4,)][(0,)], lambda x: x[4, 0]),
        (lambda x: x[np.int64(1)][2], lambda x: x[1, 2]),
    ],
)
def test_chained_index_names_the_same_slot(chained, direct):
    m, x = _stepped_model()
    assert _static_scalar_slot(chained(x), m) == _static_scalar_slot(direct(x), m) is not None


def test_chained_index_does_not_materialize_the_variable():
    """The failing-before property: lowering ``x[k][s]`` builds no array of ``x``.

    The general path charges the node budget ``x.size`` when it materializes the
    variable; the fast path charges nothing.
    """
    m, x = _stepped_model()
    budget = _Budget()
    _lower(x[3][1], m, pounce.NlExpr, {}, budget)
    assert budget.count == 0
    # Control: a SLICE in the chain must still decline (its parts do not simply
    # concatenate: x[1:][0] is x[1], not x[1:, 0]) and so does materialize.
    assert _static_scalar_slot(x[1:][0], m) is None
    budget = _Budget()
    _lower(x[1:][0], m, pounce.NlExpr, {}, budget)
    assert budget.count >= x.size


@pytest.mark.parametrize("k,s", [(1, 0), (1, 1), (5, 4), (0, 2)])
def test_chained_slice_then_int_matches_numpy(k, s):
    """``x[1:][k][s]`` is declined by the fast path, and the general path agrees with numpy."""
    m, x = _stepped_model()
    n = sum(int(v.size) for v in m._variables)
    xv = np.linspace(-0.7, 1.3, n)
    expected = xv[: x.size].reshape(x.shape)[1:][k][s]
    assert float(compile_to_nl_expr(x[1:][k][s], m).eval(list(xv))) == expected


def _tape_evaluator(model, arena: bool, monkeypatch):
    import discopt._tape_nlp_evaluator as T

    used = []
    real = T.try_build_arena_tape

    def spy(*a):
        out = real(*a)
        used.append(out is not None)
        return out

    monkeypatch.setattr(T, "try_build_arena_tape", spy)
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "1" if arena else "0")
    ev = T.TapeNLPEvaluator(model)
    monkeypatch.undo()
    return ev, used


def test_arena_tape_accepts_chained_index_and_matches_walk(monkeypatch):
    m, _ = _stepped_model()
    arena, used = _tape_evaluator(m, True, monkeypatch)
    # Failing-before: tape_program emitted OP_UNSUPPORTED for `y[s]`, so the
    # arena lowering returned None for this model.
    assert used == [True]
    walk, used = _tape_evaluator(m, False, monkeypatch)
    assert used == []

    rng = np.random.default_rng(2283)
    checks = 0
    for _ in range(4):
        xv = rng.uniform(-1.0, 1.0, arena.n_variables)
        for name in ("evaluate_objective", "evaluate_gradient", "evaluate_constraints"):
            a = np.asarray(getattr(arena, name)(xv))
            b = np.asarray(getattr(walk, name)(xv))
            assert np.array_equal(a, b), name
            checks += 1
        ja, jb = arena.evaluate_jacobian(xv), walk.evaluate_jacobian(xv)
        assert np.array_equal(np.asarray(ja), np.asarray(jb))
        checks += 1
    for a, b in zip(arena.jacobian_structure(), walk.jacobian_structure()):
        assert np.array_equal(a, b)
    for a, b in zip(arena.hessian_structure(), walk.hessian_structure()):
        assert np.array_equal(a, b)
    assert checks == 16


def test_chained_and_direct_spellings_build_identical_tapes(monkeypatch):
    """The same model written with ``x[k + 1, s]`` instead of ``y[s]`` -- same tape."""
    m1, _ = _stepped_model()

    m2 = dm.Model("direct_index")
    x = m2.continuous("x", shape=(7, 5), lb=-1, ub=2)
    p = m2.continuous("p", shape=(3,), lb=-1, ub=1)
    for k in range(6):
        for s in range(5):
            rhs = dm.exp(p[s % 3]) * x[k + 1, s] * x[k + 1, (s + 1) % 5] - x[k + 1, (s + 2) % 5]
            m2.subject_to(x[k + 1, s] - x[k, s] - 0.1 * rhs == 0)
    m2.minimize(dm.sum([(x[6, s] - 0.5) ** 2 for s in range(5)]))

    for arena in (True, False):
        e1, _ = _tape_evaluator(m1, arena, monkeypatch)
        e2, _ = _tape_evaluator(m2, arena, monkeypatch)
        xv = np.linspace(-0.9, 0.9, e1.n_variables)
        assert np.array_equal(e1.evaluate_constraints(xv), e2.evaluate_constraints(xv))
        assert np.array_equal(
            np.asarray(e1.evaluate_jacobian(xv)), np.asarray(e2.evaluate_jacobian(xv))
        )
