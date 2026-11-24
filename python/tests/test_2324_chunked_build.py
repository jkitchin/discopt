"""#2324: the shared POUNCE problem is built a chunk of rows at a time, unchanged.

``pounce.NlProblemBuilder`` (pounce#1263) copies each chunk of constraint rows
into the problem as it is added, so only one chunk of ``NlExpr`` is alive at a
time. ``TapeNLPEvaluator`` lowers the arena chunk by chunk, keeping between
chunks only the nodes a later chunk (or the objective) still reaches.

The cut is never free. POUNCE shares a body when it counts two references to
it; the builder counts per chunk, so a body whose references straddle chunks,
reached once by the first, is inlined where one ``build_nl_problem`` call would
share it -- and the shared-prelude Hessian then moves in the last bits.
``_arena_tape.plan_chunks`` only takes cuts where that cannot happen, which
makes this a memory change and not a numerical one. These tests hold the
chunked evaluator to the one-call evaluator (``DISCOPT_TAPE_CHUNKED_BUILD=0``)
**bit for bit** -- values, gradient, Jacobian, Hessian and both structures --
and show, with a negative control, that this comparison does see a bad cut.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pounce = pytest.importorskip("pounce")

import discopt._arena_tape as A  # noqa: E402
import discopt._rust as R  # noqa: E402
import discopt._tape_nlp_evaluator as T  # noqa: E402

needs_builder = pytest.mark.skipif(
    not (T.pounce_has_problem_builder() and T.pounce_takes_con_linear()),
    reason="this POUNCE has no NlProblemBuilder (needs pounce#1263)",
)


# -- fixtures ---------------------------------------------------------------


def _deep_model(n_rows: int = 60, depth: int = 14):
    """Three deep non-polynomial bodies, each a whole row's summand in every
    third row. The POUNCE analogue (pounce#1263's ``deep_shared_rows``) is the
    shape on which a split cut loses the hybrid shared-CSE tape."""
    m = dm.Model("deep2324")
    x = m.continuous("x", shape=(30,), lb=-0.5, ub=0.5)
    pool = []
    for i in range(3):
        e = x[i]
        for j in range(depth):
            e = dm.exp(e * x[(i + j) % 10]) * 0.5
        pool.append(e)
    for i in range(n_rows):
        m.subject_to(pool[i % 3] + x[10 + i % 20] == 0)
    m.minimize(dm.sum([x[i] ** 2 for i in range(30)]))
    return m


def _ode_model(n_steps: int, kind: str, *, objective_shares_a_row: bool = False):
    """#2293's reproducer: implicit-Euler steps x 28 states, one row each."""
    S, h = 28, 0.01
    m = dm.Model(f"ode2324_{kind}")
    x = m.continuous("x", shape=(n_steps + 1, S), lb=-1, ub=2)
    p = m.continuous("p", shape=(10,), lb=-1, ub=1)
    for k in range(n_steps):
        y = x[k + 1]
        if kind == "shared":
            rates = [
                dm.exp(p[j % 10]) * y[j % S] * y[(j + 1) % S]
                - dm.exp(p[(j + 1) % 10]) * y[(j + 2) % S]
                for j in range(5)
            ]
        for s in range(S):
            if kind == "linear":
                rhs = y[(s + 1) % S]
            elif kind == "nonlinear":
                rhs = (
                    dm.exp(p[s % 10]) * y[s] * y[(s + 1) % S]
                    - dm.exp(p[(s + 1) % 10]) * y[(s + 2) % S]
                )
            else:
                rhs = rates[s % 5] - rates[(s + 1) % 5] + 2 * rates[(s + 2) % 5]
            m.subject_to(y[s] - x[k, s] - h * rhs == 0)
    obj = dm.sum([(x[n_steps, s] - 0.5) ** 2 for s in range(S)])
    if objective_shares_a_row:
        # The arena dedups this product with the last step's row s=0 term.
        obj = obj + dm.exp(p[0]) * x[n_steps, 0]
    m.minimize(obj)
    return m


def _poly_model():
    """Polynomial rows only, all reaching one product body ``x0 * x1``."""
    m = dm.Model("poly2324")
    x = m.continuous("x", shape=(400,), lb=-1, ub=1)
    b = x[0] * x[1]
    for i in range(400):
        m.subject_to(b * x[i] + x[(i + 1) % 400] <= 1)
    m.minimize(dm.sum([x[i] ** 2 for i in range(400)]))
    return m


def _pair(build, monkeypatch, chunk_rows: int):
    """``(chunked, one_call)`` evaluators of two fresh copies of one model."""
    monkeypatch.setattr(T, "CHUNK_ROWS", chunk_rows)
    monkeypatch.setenv("DISCOPT_TAPE_CHUNKED_BUILD", "1")
    chunked = T.TapeNLPEvaluator(build())
    monkeypatch.setenv("DISCOPT_TAPE_CHUNKED_BUILD", "0")
    whole = T.TapeNLPEvaluator(build())
    assert whole._n_build_chunks is None
    return chunked, whole


def _differences(a, b, seed: int = 2324, points: int = 4):
    """Per-quantity bit-equality of two evaluators, and how many were compared."""
    rng = np.random.default_rng(seed)
    lo, hi = a.variable_bounds
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    out: dict[str, bool] = {}
    count = 0

    def put(name, u, v):
        nonlocal count
        count += 1
        # equal_nan: a corpus point can overflow; NaN in the same place is the
        # same evaluation.
        same = np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)
        out[name] = out.get(name, True) and same

    for s, t in zip(a.jacobian_structure(), b.jacobian_structure()):
        put("jacobian_structure", s, t)
    for s, t in zip(a.hessian_structure(), b.hessian_structure()):
        put("hessian_structure", s, t)
    for _ in range(points):
        x = rng.uniform(lo, hi)
        lam = rng.normal(size=a.n_constraints)
        put("objective", a.evaluate_objective(x), b.evaluate_objective(x))
        put("gradient", a.evaluate_gradient(x), b.evaluate_gradient(x))
        put("constraints", a.evaluate_constraints(x), b.evaluate_constraints(x))
        put("jacobian", a.evaluate_jacobian_values(x), b.evaluate_jacobian_values(x))
        put(
            "hessian",
            a.evaluate_hessian_values(x, 0.7, lam),
            b.evaluate_hessian_values(x, 0.7, lam),
        )
    return out, count


def _assert_bit_identical(a, b):
    diff, count = _differences(a, b)
    assert count == 4 + 5 * 4, "the comparison did not run"
    assert all(diff.values()), f"chunked build differs from one call: {diff}"


# -- the cut ----------------------------------------------------------------


def test_planner_cuts_where_every_shared_body_is_established():
    """Needs no ``NlProblemBuilder``: the plan is discopt's, so it is pinned on
    any POUNCE. Each deep body needs two of its rows in the first chunk that
    reaches it, so rows 0-5 share a chunk; after that every boundary is free.
    An objective sharing a row's product, and a body only polynomial rows
    reach, have no exact cut."""
    planned = A.plan_chunks(*A._arena_inputs(_deep_model(), split_linear=True), 4)
    assert planned is not None
    chunks, first_row, last_row = planned
    assert chunks[0] == range(0, 6)
    assert [len(r) for r in chunks[1:]] == [4] * 13 + [2]
    assert first_row.shape == last_row.shape and int(last_row.max()) == 60

    m = _ode_model(40, "nonlinear", objective_shares_a_row=True)
    assert A.plan_chunks(*A._arena_inputs(m, split_linear=True), 100) is None
    assert A.plan_chunks(*A._arena_inputs(_poly_model(), split_linear=True), 10) is None
    # Nonlinear rows share only the parameters' exp(p_j), established in the
    # first chunk, so every boundary is exact. Shared rows share per-step
    # bodies, among them `2 * rates[k]`, which the arena dedups and which a
    # step's row s and row s+5 both reach: rows 196-205 of step 7 must share a
    # chunk, so the cut due at row 200 moves to 206.
    m = _ode_model(40, "nonlinear")
    chunks = A.plan_chunks(*A._arena_inputs(m, split_linear=True), 100)[0]
    assert all(len(r) == 100 for r in chunks[:-1])
    m = _ode_model(40, "shared")
    chunks = A.plan_chunks(*A._arena_inputs(m, split_linear=True), 100)[0]
    assert chunks[0] == range(0, 100) and chunks[1] == range(100, 206)


@needs_builder
def test_deep_shared_bodies_planned_cut_is_bit_identical(monkeypatch):
    chunked, whole = _pair(_deep_model, monkeypatch, 4)
    assert chunked._n_build_chunks == 15
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_negative_control_a_naive_cut_moves_the_hessian(monkeypatch):
    """The instrument check for every bit-identity test here.

    Cut one row a chunk, ignoring the planner, and the deep bodies are reached
    once per chunk: POUNCE inlines them, drops the hybrid tape, and the Hessian
    moves in the last bits while values and Jacobian do not (measured 4.7e-15
    relative). If this stops failing to match, POUNCE no longer distinguishes
    the cuts and the comparisons above can no longer see a bad plan.
    """
    real = R.tape_chunk_plan

    def one_row_a_chunk(ptr, idx, poly, leaf, row_root, obj, target):
        _, first, last = real(ptr, idx, poly, leaf, row_root, obj, target)
        n = len(row_root)
        return np.arange(n + 1, dtype=np.int64), first, last

    monkeypatch.setattr(R, "tape_chunk_plan", one_row_a_chunk)
    chunked, whole = _pair(_deep_model, monkeypatch, 4)
    assert chunked._n_build_chunks == 60
    diff, count = _differences(chunked, whole)
    assert count == 24
    assert not diff["hessian"], "a split cut no longer moves the Hessian (see docstring)"
    assert all(v for k, v in diff.items() if k != "hessian"), diff


@needs_builder
@pytest.mark.parametrize("kind", ["nonlinear", "shared"])
def test_2293_reproducer_rows_chunk_by_step_bit_identical(kind, monkeypatch):
    chunked, whole = _pair(lambda: _ode_model(40, kind), monkeypatch, 100)
    assert chunked._n_build_chunks is not None and chunked._n_build_chunks > 1
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_lowering_cache_keeps_only_what_later_chunks_reach(monkeypatch):
    """#2324 item 1: if the lowering cache kept every chunk, nothing would be
    freed and the builder would save nothing. Count the live cache entries at
    each chunk against everything lowered."""
    monkeypatch.setattr(T, "CHUNK_ROWS", 100)
    prep = A.prepare_arena_tape(_ode_model(40, "nonlinear"))
    plan = prep.plan(pounce.NlExpr, 100)
    assert plan is not None and len(plan.plan) >= 10
    gen = plan.chunks()
    live = []
    for _rows, _csr in gen:
        cache = gen.gi_frame.f_locals["cache"]
        live.append(sum(e is not None for e in cache))
    lowered = int((np.asarray(plan._first_bounds)[-1]))
    assert len(live) == len(plan.plan)
    # Every chunk holds its own nodes plus what later chunks reach (the
    # parameters' exp(p_j), the variables): a small fraction of the whole.
    assert max(live) < lowered / 4, (max(live), lowered)
    assert plan.objective() is not None


# -- where no cut is taken ----------------------------------------------------


@needs_builder
def test_objective_sharing_a_row_body_builds_in_one_call(monkeypatch):
    """The builder materialises the objective after every chunk, so a body the
    objective shares with a single row is inlined by the builder and shared by
    one call. No cut fixes that: the planner declines and one call builds it."""
    chunked, whole = _pair(
        lambda: _ode_model(40, "nonlinear", objective_shares_a_row=True), monkeypatch, 100
    )
    assert chunked._n_build_chunks is None
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_polynomial_rows_sharing_a_body_are_not_cut(monkeypatch):
    """A row POUNCE's quadratic recognizer keeps as its own tree counts no
    references, and which polynomial rows it keeps is not decided here. So a
    body only polynomial rows reach is never certified shared in a first chunk,
    and every row from its first reach on stays together."""

    chunked, whole = _pair(_poly_model, monkeypatch, 10)
    assert chunked._n_build_chunks is None
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_linear_only_rows_are_not_chunked(monkeypatch):
    """Chunking frees ``NlExpr`` between chunks; coefficient rows have none, and
    the builder holds their ``con_linear`` less compactly (measured +130-150
    B/row stranded on #2293's linear rows). So they take the one-call path."""
    chunked, whole = _pair(lambda: _ode_model(40, "linear"), monkeypatch, 100)
    assert chunked._n_build_chunks is None
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_unsupported_row_in_a_later_chunk_falls_back_to_the_walk(monkeypatch):
    """``tanh`` has no arena lowering. Placed in the last row, the chunked
    lowering meets it only after earlier chunks went into a builder; that
    builder is dropped and the Python walk builds the model, as it would have."""

    def build():
        m = _ode_model(20, "nonlinear")
        x = m._variables[0]
        m.subject_to(dm.tanh(x[1, 0]) - x[1, 1] == 0)
        return m

    planned = []
    real_plan = A.ArenaPrepared.plan

    def spy(self, E, target_rows=A.CHUNK_ROWS):
        out = real_plan(self, E, target_rows)
        planned.append(out is not None)
        return out

    monkeypatch.setattr(A.ArenaPrepared, "plan", spy)
    chunked, whole = _pair(build, monkeypatch, 100)
    assert planned == [True], "the chunked lowering must have been attempted"
    assert chunked._n_build_chunks is None
    _assert_bit_identical(chunked, whole)


@needs_builder
def test_per_thread_fallback_and_opt_out_build_in_one_call(monkeypatch):
    monkeypatch.setattr(T, "CHUNK_ROWS", 100)
    monkeypatch.setattr(T, "_nlproblem_is_thread_safe", lambda: False)
    ev = T.TapeNLPEvaluator(_ode_model(40, "nonlinear"))
    assert ev._n_build_chunks is None and ev._con_exprs is not None
    monkeypatch.undo()

    monkeypatch.setattr(T, "CHUNK_ROWS", 100)
    monkeypatch.setenv("DISCOPT_TAPE_CHUNKED_BUILD", "0")
    assert T.TapeNLPEvaluator(_ode_model(40, "nonlinear"))._n_build_chunks is None
    monkeypatch.setenv("DISCOPT_TAPE_CHUNKED_BUILD", "1")
    assert T.TapeNLPEvaluator(_ode_model(40, "nonlinear"))._n_build_chunks > 1


@needs_builder
def test_parameter_rebind_rebuilds_in_chunks(monkeypatch):
    """A ``Parameter.value`` change rebuilds the tape from the model, chunked
    again, and the rebuilt rows see the new value."""
    monkeypatch.setattr(T, "CHUNK_ROWS", 50)
    m = dm.Model("rebind2324")
    x = m.continuous("x", shape=(201,), lb=0.1, ub=2.0)
    k = m.parameter("k", value=2.0)
    for i in range(200):
        m.subject_to(x[i + 1] - x[i] - 0.01 * k * dm.exp(x[i]) == 0)
    m.minimize(dm.sum([x[i] ** 2 for i in range(201)]))
    ev = T.TapeNLPEvaluator(m)
    assert ev._n_build_chunks is not None and ev._n_build_chunks > 1
    pt = np.linspace(0.5, 1.5, ev.n_variables)
    g = np.asarray(ev.evaluate_constraints(pt))
    assert g[3] == pytest.approx(pt[4] - pt[3] - 0.02 * np.exp(pt[3]))
    k.value = 7.0
    g = np.asarray(ev.evaluate_constraints(pt))
    assert g[3] == pytest.approx(pt[4] - pt[3] - 0.07 * np.exp(pt[3])), "stale tape"
    assert ev._n_build_chunks is not None and ev._n_build_chunks > 1


@pytest.mark.skipif(
    not T.pounce_takes_keep_expressions(),
    reason="this POUNCE's build_nl_problem has no keep_expressions= (pounce#1262)",
)
def test_one_call_build_drops_the_expression_trees(monkeypatch):
    """#2324 item 3: nothing in the evaluator reads ``NlProblem.con_expr``, so the
    shared one-call build passes ``keep_expressions=False``."""
    seen = []
    real = pounce.build_nl_problem

    def spy(*a, **kw):
        seen.append(kw.get("keep_expressions"))
        return real(*a, **kw)

    # The evaluator reads the keyword off the signature; keep it visible.
    spy.__text_signature__ = real.__text_signature__  # type: ignore[attr-defined]
    monkeypatch.setattr(pounce, "build_nl_problem", spy)
    ev = T.TapeNLPEvaluator(_ode_model(3, "nonlinear"))
    assert ev._n_build_chunks is None
    assert seen and seen[0] is False


@needs_builder
def test_corpus_chunked_at_four_rows_is_bit_identical(monkeypatch):
    """Every in-repo MINLPLib instance, chunked as finely as the planner allows
    (four rows), against one call. Measured: 16 of 74 instances take the
    chunked branch (heatexch_gen3 in 128 chunks), the rest are too small or
    have no exact cut; all are bit-identical."""
    from pathlib import Path

    from discopt.modeling.core import from_nl

    corpus = Path(__file__).parent / "data" / "minlplib_nl"
    compared = chunked_instances = 0
    for path in sorted(corpus.glob("*.nl")):
        chunked, whole = _pair(lambda p=path: from_nl(str(p)), monkeypatch, 4)
        if whole.n_constraints == 0:
            continue
        diff, count = _differences(chunked, whole, points=2)
        assert all(diff.values()), (path.stem, diff)
        compared += count
        chunked_instances += chunked._n_build_chunks is not None
    # The probe must have fired, and on the branch it exists for.
    assert compared > 500, compared
    assert chunked_instances >= 10, chunked_instances
