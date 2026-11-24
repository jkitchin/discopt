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

Since #2293 the ``[]`` operator itself builds ``x[k][s]`` as ``x[k, s]``, so a
user-written chain no longer reaches these lowerings as a chain. They still meet
genuine nested nodes built with ``IndexExpression(...)`` directly (graph
machinery, deserialization), so the chains here are built that way: ``_ix``.
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
from discopt.modeling.core import IndexExpression  # noqa: E402


def _ix(base, *parts):
    """A genuine nested index chain (the ``[]`` operator now joins integer chains)."""
    for part in parts:
        base = IndexExpression(base, part)
    return base


def _stepped_model(n_steps: int = 6, n_states: int = 5):
    """Implicit-Euler rows written with the ``y = x[k + 1]`` idiom."""
    m = dm.Model("chained_index")
    x = m.continuous("x", shape=(n_steps + 1, n_states), lb=-1, ub=2)
    p = m.continuous("p", shape=(3,), lb=-1, ub=1)
    for k in range(n_steps):
        y = x[k + 1]

        def at(s, y=y):
            return IndexExpression(y, s)  # genuine chain, not joined by `[]`

        for s in range(n_states):
            rhs = dm.exp(p[s % 3]) * at(s) * at((s + 1) % n_states) - at((s + 2) % n_states)
            m.subject_to(at(s) - x[k, s] - 0.1 * rhs == 0)
    m.minimize(dm.sum([(x[n_steps, s] - 0.5) ** 2 for s in range(n_states)]))
    return m, x


@pytest.mark.parametrize(
    "chained, direct",
    [
        (lambda x: _ix(x, 2, 3), lambda x: x[2, 3]),
        (lambda x: _ix(x, -1, -2), lambda x: x[6, 3]),
        (lambda x: _ix(x, (4,), (0,)), lambda x: x[4, 0]),
        (lambda x: _ix(x, np.int64(1), 2), lambda x: x[1, 2]),
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
    _lower(_ix(x, 3, 1), m, pounce.NlExpr, {}, budget)
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

    # Both arena entry points: the evaluator lowers through `ArenaPrepared`
    # (the `_split` lowering, prepared once, #2324) when its POUNCE accepts
    # `con_linear=` (#2293), through the plain one otherwise.
    def spying(real):
        def spy(*a):
            out = real(*a)
            used.append(out is not None)
            return out

        return spy

    monkeypatch.setattr(T, "try_build_arena_tape", spying(T.try_build_arena_tape))
    from discopt._arena_tape import ArenaPrepared

    monkeypatch.setattr(ArenaPrepared, "lower_split", spying(ArenaPrepared.lower_split))
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


def _cube_model():
    """A 3-D variable addressed through every chain split, so concatenating MORE than two
    parts is pinned (the 2-D cases above only ever join two)."""
    m = dm.Model("chained_index_3d")
    x = m.continuous("x", shape=(2, 3, 4), lb=-1, ub=2)
    w = m.continuous("w", lb=-1, ub=1)
    for a in range(2):
        for b in range(3):
            for c in range(4):
                m.subject_to(
                    _ix(x, a, b, c) - w * _ix(x, a, (b, c)) + _ix(x, (a, b), c) ** 2
                    == 0.1 * (a + b + c)
                )
    m.minimize(
        dm.sum(
            [_ix(x, a - 2, b - 3, c - 4) ** 2 for a in range(2) for b in range(3) for c in range(4)]
        )
    )
    return m, x


@pytest.mark.parametrize(
    "chained",
    [
        lambda x, a, b, c: _ix(x, a, b, c),
        lambda x, a, b, c: _ix(x, a, (b, c)),
        lambda x, a, b, c: _ix(x, (a, b), c),
        lambda x, a, b, c: _ix(x, a - 2, b - 3, c - 4),
    ],
)
def test_three_part_chains_name_the_direct_slot(chained):
    m, x = _cube_model()
    checks = 0
    for a in range(2):
        for b in range(3):
            for c in range(4):
                assert _static_scalar_slot(chained(x, a, b, c), m) == _static_scalar_slot(
                    x[a, b, c], m
                )
                checks += 1
    assert checks == 24


@pytest.mark.parametrize(
    "declined",
    [
        lambda x: x[0][1:][0],
        lambda x: x[1:][0][0],
        lambda x: x[0][0][0:1],
        lambda x: x[0][None][0][0],
    ],
)
def test_three_part_chain_with_a_non_integer_part_declines(declined):
    m, x = _cube_model()
    assert _static_scalar_slot(declined(x), m) is None


def test_three_part_chains_arena_matches_walk(monkeypatch):
    m, _ = _cube_model()
    arena, used = _tape_evaluator(m, True, monkeypatch)
    assert used == [True]  # tape_program resolved every 3-part chain
    walk, _ = _tape_evaluator(m, False, monkeypatch)
    rng = np.random.default_rng(22830)
    checks = 0
    for _ in range(3):
        xv = rng.uniform(-1.0, 1.0, arena.n_variables)
        for name in ("evaluate_objective", "evaluate_gradient", "evaluate_constraints"):
            assert np.array_equal(
                np.asarray(getattr(arena, name)(xv)), np.asarray(getattr(walk, name)(xv))
            )
            checks += 1
        assert np.array_equal(arena.evaluate_jacobian(xv), walk.evaluate_jacobian(xv))
        checks += 1
    assert checks == 12


def test_shared_problem_releases_the_nlexpr_roots(monkeypatch):
    """#2283: once the shared NlProblem is built, the evaluator drops its NlExpr roots.

    POUNCE builds its own tapes and does not keep the NlExpr DAG alive, so holding
    the roots pinned the whole DAG for the evaluator's lifetime: 66 MiB of live
    heap on a 28,672-row method-of-lines model. The per-thread fallback still
    needs them (it builds one problem per thread on first use) and must keep them.
    Both paths must evaluate identically.
    """
    import discopt._tape_nlp_evaluator as T

    m, _ = _stepped_model()
    shared = T.TapeNLPEvaluator(m)
    assert shared._shared_problem is not None, "probe should report thread-safe here"
    assert shared._obj_expr is None and shared._con_exprs is None  # failing-before

    monkeypatch.setattr(T, "_nlproblem_is_thread_safe", lambda: False)
    fallback = T.TapeNLPEvaluator(m)
    monkeypatch.undo()
    assert fallback._shared_problem is None
    assert fallback._obj_expr is not None and fallback._con_exprs is not None

    rng = np.random.default_rng(2283)
    lam = rng.normal(size=shared.n_constraints)
    checks = 0
    for _ in range(3):
        xv = rng.uniform(-1.0, 1.0, shared.n_variables)
        for name in ("evaluate_objective", "evaluate_gradient", "evaluate_constraints"):
            assert np.array_equal(
                np.asarray(getattr(shared, name)(xv)), np.asarray(getattr(fallback, name)(xv))
            )
            checks += 1
        assert np.array_equal(shared.evaluate_jacobian(xv), fallback.evaluate_jacobian(xv))
        assert np.array_equal(
            shared.evaluate_hessian_values(xv, 0.7, lam),
            fallback.evaluate_hessian_values(xv, 0.7, lam),
        )
        checks += 2
    assert checks == 15


def test_parameter_rebind_after_the_roots_are_released():
    """A rebuild lowers the roots afresh from the model, so a released evaluator
    still picks up a new ``Parameter.value`` -- and releases them again."""
    m = dm.Model("released_rebind")
    x = m.continuous("x", shape=(3, 2), lb=0.1, ub=5.0)
    p = m.parameter("p", value=2.0)
    for k in range(2):
        y = x[k + 1]
        m.subject_to(y[0] - x[k][0] - p * y[1] == 0)
    m.minimize(p * x[2][0] * x[2][0])
    import discopt._tape_nlp_evaluator as T

    ev = T.TapeNLPEvaluator(m)
    assert ev._con_exprs is None
    pt = np.linspace(0.5, 1.5, ev.n_variables)
    slot = 2 * 2 + 0  # x[2, 0]
    assert ev.evaluate_objective(pt) == pytest.approx(2.0 * pt[slot] ** 2)
    p.value = 7.0
    assert ev.evaluate_objective(pt) == pytest.approx(7.0 * pt[slot] ** 2), "stale tape"
    g = np.asarray(ev.evaluate_constraints(pt))
    assert g[0] == pytest.approx(pt[2] - pt[0] - 7.0 * pt[3])
    assert ev._con_exprs is None, "rebuild must release the roots again"
