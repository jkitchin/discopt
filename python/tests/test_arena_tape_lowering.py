"""The arena-lowered AD tape must be bit-identical to the Python-DAG tape.

``_arena_tape`` builds the POUNCE tape by scanning the Rust expression arena that
``model_to_repr`` has already produced, instead of walking the Python expression
DAG a second time. Measured at foundation scale, that walk is 93-94% of a tape
build which is itself 59% of the whole build -> solve-ready pipeline (#1215).

Because the tape is the derivative source for the entire NLP path, this is a
CLAUDE.md §5 *bound-neutral* change: the bar is not agreement within a tolerance
but **exact equality** of objective, gradient, constraint values and Jacobian.
A tape that is merely plausible yields wrong gradients, hence wrong incumbents,
hence a wrong certificate.

The refusal tests matter as much as the equality ones. The arena encoding does
not cover array-valued bodies, builder-resident rows or ``dm.custom``; for those
it must return ``None`` so the caller falls back, and must never build a partial
tape or raise where the old path succeeded.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

pounce = pytest.importorskip("pounce")


def _bounds(model):
    lo, hi = [], []
    for v in model._variables:
        n = int(v.size)
        for dest, bound in ((lo, v.lb), (hi, v.ub)):
            if np.ndim(bound):
                dest.extend(np.asarray(bound, float).reshape(-1).tolist())
            else:
                dest.extend([float(bound)] * n)
    return np.asarray(lo), np.asarray(hi)


def _python_tape(model):
    from discopt._nl_expr_compiler import compile_to_nl_array, compile_to_nl_expr

    obj = compile_to_nl_expr(model._objective.expression, model)
    cons = []
    for c in model._constraints:
        cons.extend(compile_to_nl_array(c.body, model).reshape(-1).tolist())
    return obj, cons


def _problem(model, obj, cons):
    lo, hi = _bounds(model)
    return (
        pounce.build_nl_problem(lo.size, obj, constraints=cons, x_l=list(lo), x_u=list(hi)),
        lo,
        hi,
    )


# ── models ────────────────────────────────────────────────────────────────


def _flowsheet(n=40):
    m = dm.Model(f"fs{n}")
    outs = []
    for u in range(n):
        f = m.continuous(f"f{u}", shape=(3,), lb=0.1, ub=10.0)
        t = m.continuous(f"T{u}", lb=300.0, ub=600.0)
        m.subject_to(f[0] + f[1] - f[2] == 0.0, name=f"mb{u}")
        m.subject_to(f[2] * dm.exp(-2000.0 / t) <= 5.0, name=f"rate{u}")
        m.subject_to(dm.log(f[0] + 1.0) + 0.01 * t <= 12.0, name=f"nrg{u}")
        outs.append(f[2])
    m.minimize(dm.sum(outs))
    return m


def _long_chain(n=6000):
    m = dm.Model("chain")
    x = m.continuous("x", shape=(5,), lb=0.3, ub=3.0)
    m.subject_to(sum(x[i % 5] * float(i + 1) for i in range(n)) <= 1e9, name="c")
    m.minimize(sum(dm.exp(x[i % 5] * 0.01) for i in range(200)))
    return m


def _short_chains():
    """Below the flattening threshold the original binary nesting must survive.

    ``a + (b + c)`` and ``(a + b) + c`` differ in IEEE, so a flattening applied
    where the Python path does not apply one shows up here as a last-bit drift.
    Coefficients span 1e-8..1e8, where summation order matters most.
    """
    m = dm.Model("short")
    x = m.continuous("x", shape=(4,), lb=1e-8, ub=1e8)
    a, b, c, d = (x[i] for i in range(4))
    m.subject_to(a + (b + c) <= 1e9, name="right_nested")
    m.subject_to((a - b) + c <= 1e9, name="left_nested")
    m.subject_to(a + (b - (c + d)) <= 1e9, name="mixed")
    m.minimize(a * 1e-8 + b * 1e8 - c)
    return m


def _funcs():
    m = dm.Model("funcs")
    x = m.continuous("x", shape=(6,), lb=0.2, ub=0.9)
    m.subject_to(dm.sqrt(x[0]) + dm.log(x[1]) + dm.sin(x[2]) <= 5.0, name="f0")
    m.subject_to(dm.cos(x[3]) + dm.tan(x[4]) + dm.atan(x[5]) <= 5.0, name="f1")
    m.subject_to(dm.exp(x[0]) / (x[1] + 1.0) - x[2] ** 1.7 <= 9.0, name="f2")
    m.subject_to(abs(x[3] - x[4]) <= 1.0, name="f3")
    m.minimize(-dm.sum([x[i] for i in range(6)]))
    return m


def _shared():
    """A chain interior that is ALSO an operand of a non-additive parent.

    The lowering skips chain interiors so a long chain does not rebuild the deep
    binary nesting; an interior reached from a ``*`` must still get its own node.
    """
    m = dm.Model("shared")
    x = m.continuous("x", shape=(4,), lb=0.5, ub=2.0)
    common = x[0] + x[1] + x[2] + x[3]
    for i in range(20):
        m.subject_to(common * dm.exp(x[i % 4] * 0.1) <= 100.0, name=f"c{i}")
    m.minimize(common + common * common)
    return m


BUILDERS = {
    "flowsheet": _flowsheet,
    "long_chain": _long_chain,
    "short_chains": _short_chains,
    "funcs": _funcs,
    "shared_subexpr": _shared,
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_arena_tape_is_bit_identical(name):
    from discopt._arena_tape import try_build_arena_tape

    model = BUILDERS[name]()
    built = try_build_arena_tape(model, pounce.NlExpr)
    assert built is not None, f"{name} should lower through the arena"
    ar_obj, ar_cons = built
    py_obj, py_cons = _python_tape(model)
    assert len(ar_cons) == len(py_cons)

    p_py, lo, hi = _problem(model, py_obj, py_cons)
    p_ar, _, _ = _problem(model, ar_obj, ar_cons)
    rng = np.random.default_rng(7)
    compared = 0
    for _ in range(4):
        x = lo + rng.uniform(0.05, 0.95, size=lo.size) * (hi - lo)
        assert float(p_py.objective(x)) == float(p_ar.objective(x))
        np.testing.assert_array_equal(np.asarray(p_py.gradient(x)), np.asarray(p_ar.gradient(x)))
        np.testing.assert_array_equal(
            np.asarray(p_py.constraints(x)), np.asarray(p_ar.constraints(x))
        )
        np.testing.assert_array_equal(np.asarray(p_py.jacobian(x)), np.asarray(p_ar.jacobian(x)))
        compared += 4
    assert compared == 16, "probe compared nothing"


def test_long_chain_stays_shallow_on_the_arena_path():
    """The arena path must flatten additive chains exactly as the DAG path does.

    Emitting a binary ``+`` chain here would rebuild the depth-N nesting that
    ``NlExpr.max_depth`` refuses at 10 000 terms -- reintroducing the crash the
    flattening fixed, on a path the crash tests do not cover.
    """
    from discopt._arena_tape import try_build_arena_tape

    model = _long_chain(12000)
    built = try_build_arena_tape(model, pounce.NlExpr)
    assert built is not None
    _, cons = built
    assert cons[0].depth <= 4, f"chain lowered to depth {cons[0].depth}"


def test_array_body_is_refused_not_guessed():
    """An array-valued body fans out to many rows; the arena path cannot, so it refuses."""
    from discopt._arena_tape import try_build_arena_tape

    m = dm.Model("arr")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=2.0)
    m.subject_to(x <= 1.5, name="arr")
    m.minimize(-x[0])
    assert try_build_arena_tape(m, pounce.NlExpr) is None


def test_builder_rows_are_refused():
    """Builder rows sit ahead of expression rows in the arena and after them in the
    evaluator, so aligning by index would permute every row's dual."""
    from discopt._arena_tape import try_build_arena_tape

    m = dm.Model("blk")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=2.0)
    m.subject_to(x[0] * dm.exp(x[1]) <= 4.0, name="nl")
    m.add_linear_constraints(np.array([[1.0, 1.0, 1.0]]), x, "<=", np.array([2.0]))
    m.minimize(-x[0])
    assert try_build_arena_tape(m, pounce.NlExpr) is None


def test_custom_call_is_refused_by_both_paths():
    """``dm.custom`` has no tape equivalent; the arena path must not invent one."""
    from discopt._arena_tape import try_build_arena_tape
    from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_expr

    m = dm.Model("custom")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=2.0)
    f = dm.custom(lambda v: v * 2.0, name="double")
    m.subject_to(f(x[0]) + x[1] <= 5.0, name="c")
    m.minimize(-x[0])
    with pytest.raises(UnsupportedForTape):
        compile_to_nl_expr(m._constraints[0].body, m)
    assert try_build_arena_tape(m, pounce.NlExpr) is None


def test_evaluator_agrees_with_the_opt_out(monkeypatch):
    """The evaluator must produce the same numbers with the lowering off.

    Also pins the row map: ``_constraint_flat_sizes`` drives dual attribution and
    feasibility reporting, so a row-count difference between the arms would
    mis-attribute every row without changing any value.
    """
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    model = _flowsheet(20)
    rng = np.random.default_rng(5)
    lo, hi = _bounds(model)
    xs = [lo + rng.uniform(0.05, 0.95, size=lo.size) * (hi - lo) for _ in range(3)]

    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "1")
    on = TapeNLPEvaluator(model)
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "0")
    off = TapeNLPEvaluator(model)

    assert on.n_constraints == off.n_constraints
    np.testing.assert_array_equal(on._constraint_flat_sizes, off._constraint_flat_sizes)
    compared = 0
    for x in xs:
        assert on.evaluate_objective(x) == off.evaluate_objective(x)
        np.testing.assert_array_equal(
            np.asarray(on.evaluate_gradient(x)), np.asarray(off.evaluate_gradient(x))
        )
        compared += 2
    assert compared == 6, "probe compared nothing"


def test_solve_matches_across_the_two_paths(monkeypatch):
    """End to end: the certified objective must not move."""
    results = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("DISCOPT_ARENA_TAPE", flag)
        m = _flowsheet(6)
        res = m.solve(time_limit=60)
        assert res.status in ("optimal", "feasible"), f"flag={flag} status={res.status}"
        results[flag] = float(res.objective)
    assert results["1"] == pytest.approx(results["0"], rel=1e-9, abs=1e-9)
