"""The array half of the tape lowering must reproduce the JAX path exactly.

Issue #75. ``_nl_expr_compiler`` previously lowered only *scalar* DAG nodes: an
array ``Variable``, ``SumExpression``, ``MatMulExpression`` and every slice /
partial index raised ``UnsupportedForTape``, so any model written against the
Python API with ``shape=(...)`` variables — and every ``DAEBuilder`` collocation
model — fell back to the JAX evaluator. The entry experiment (CLAUDE.md §4) that
scoped this measured the residual on the smoke corpus after static indexing
landed::

    EVALUATOR_DECISIONS=255  TAPE=247  JAX=8
          6  array variable '?' (size N); the tape path is scalar
          1  SumExpression (array reduction) has no scalar tape lowering
          1  MatMulExpression is not yet lowered

i.e. the entire residual was this one class. It is now lowered by carrying a
numpy object array of scalar tape nodes through the walk.

What these tests pin, in order of what would hurt most if it broke:

* **row identity** — an array-valued constraint body is ONE ``Constraint`` and
  MANY rows, and the tape's rows must be the JAX arm's rows *in the same order*,
  or the two backends' duals, row maps and feasibility reports refer to different
  constraints (#908's failure mode, from the other side);
* **numerical agreement** with ``_relax/dag_compiler`` on value and Jacobian;
* **the reductions are n-ary, not chains** — a ``+`` chain of depth N is not
  merely slow, POUNCE refuses past ``NlExpr.max_depth`` with a ``ValueError``
  that ``try_compile`` does not catch, so it escapes the JAX fallback entirely;
* **the scalar fast path survives** — ``_static_scalar_slot`` must still resolve
  ``x[i]`` without materializing the base, or issue #654's quadratic build cost
  comes back.
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._nl_expr_compiler import (
    UnsupportedForTape,
    compile_to_nl_array,
    compile_to_nl_expr,
)

pytest.importorskip("pounce")


def _eval_rows(arr, x):
    return np.array([e.eval(list(x)) for e in np.asarray(arr).reshape(-1)], dtype=float)


def _jac_rows(arr, x):
    return np.array(
        [np.asarray(e.gradient(list(x)), dtype=float) for e in np.asarray(arr).reshape(-1)]
    )


def _array_model():
    """One model carrying every array shape the cases below need."""
    m = Model("arrays")
    x = m.continuous("x", shape=(4,), lb=0.1, ub=3.0)
    Y = m.continuous("Y", shape=(2, 3), lb=0.1, ub=3.0)
    s = m.continuous("s", lb=0.1, ub=3.0)
    m.minimize(s)
    return m, x, Y, s


# Every array-valued DAG form the lowering claims to cover. `A` is a plain numpy
# constant matrix, which is how `dae/collocation.py` builds its blocks.
_A = np.arange(12, dtype=float).reshape(3, 4) / 7.0 + 0.5

ARRAY_CASES = {
    "array_variable": lambda x, Y, s: x,
    "array_variable_2d": lambda x, Y, s: Y,
    "sum_all": lambda x, Y, s: dm.sum(x),
    "sum_axis0": lambda x, Y, s: dm.sum(Y, axis=0),
    "sum_axis1": lambda x, Y, s: dm.sum(Y, axis=1),
    "matmul_const_vec": lambda x, Y, s: _A @ x,
    "matmul_var_vec": lambda x, Y, s: Y @ x[:3],
    "matmul_vec_mat": lambda x, Y, s: x[:2] @ Y,
    "matmul_mat_mat": lambda x, Y, s: Y @ np.eye(3),
    "slice": lambda x, Y, s: x[1:],
    "negative_slice": lambda x, Y, s: x[-3:-1],
    "partial_index": lambda x, Y, s: Y[0],
    "column": lambda x, Y, s: Y[:, 1],
    "strided_2d": lambda x, Y, s: Y[1, ::2],
    "broadcast_scalar": lambda x, Y, s: x * s,
    "broadcast_row": lambda x, Y, s: Y + x[:3],
    "elementwise_unary": lambda x, Y, s: dm.exp(x),
    "elementwise_over_matmul": lambda x, Y, s: dm.sin(_A @ x),
    "abs_over_array": lambda x, Y, s: abs(x - 1.0),
    "nested_reduction": lambda x, Y, s: dm.sum((_A @ x) * x[:3]),
    "prod": lambda x, Y, s: dm.prod(x),
    "norm1": lambda x, Y, s: dm.norm(x, 1),
    "norm2": lambda x, Y, s: dm.norm(x, 2),
    "norminf": lambda x, Y, s: dm.norm(x, float("inf")),
    "norm3": lambda x, Y, s: dm.norm(x, 3),
    "scalar_still_scalar": lambda x, Y, s: x[0] * x[1] + dm.log(s),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ARRAY_CASES))
def test_array_form_matches_jax_value_and_jacobian(name):
    """Tape and JAX must agree on the row COUNT, the row ORDER, and the numbers.

    The Jacobian comparison is what makes this more than a smoke test: for an
    indexing or reduction node each row of the Jacobian names the flat slots that
    row actually reads, so a lowering that produced the right shape from the wrong
    elements fails here and passes on values alone.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from discopt._relax.dag_compiler import compile_expression

    m, x, Y, s = _array_model()
    expr = ARRAY_CASES[name](x, Y, s)
    n = sum(v.size for v in m._variables)

    tape = compile_to_nl_array(expr, m)
    jfn = compile_expression(expr, m)

    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    compared = 0
    for _ in range(5):
        pt = rng.uniform(0.3, 1.7, size=n)
        jax_value = np.asarray(jfn(jnp.asarray(pt))).reshape(-1)
        tape_value = _eval_rows(tape, pt)
        assert tape_value.shape == jax_value.shape, (
            f"{name}: tape gave {tape_value.shape} rows, JAX gave {jax_value.shape}"
        )
        np.testing.assert_allclose(tape_value, jax_value, rtol=1e-12, atol=1e-12)

        jax_jac = np.asarray(jax.jacobian(lambda z: jnp.reshape(jfn(z), (-1,)))(jnp.asarray(pt)))
        np.testing.assert_allclose(_jac_rows(tape, pt), jax_jac, rtol=1e-11, atol=1e-11)
        compared += 1

    # §6: a case that compared nothing would pass vacuously.
    assert compared == 5, f"only {compared} points compared for {name}"


@pytest.mark.unit
def test_array_constraint_body_fans_out_into_rows_in_jax_order():
    """One ``Constraint``, many rows — in ``reshape(-1)`` order, like the JAX arm.

    This is the property #908 was about, seen from the tape side. The evaluator's
    ``constraint_row_map`` is consumed by the incumbent verifiers, the dual
    extraction and the examiner; if the tape's row order differed from the JAX
    arm's, every one of them would attribute a violation to the wrong constraint
    while every count still matched.
    """
    pytest.importorskip("jax")
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._tape_nlp_evaluator import try_build

    def build(name):
        m = Model(name)
        v = m.continuous("v", shape=(2, 3), lb=0.0, ub=5.0)
        w = m.continuous("w", shape=(3,), lb=0.0, ub=5.0)
        m.minimize(dm.sum(v @ w))
        m.subject_to(v @ w <= 4.0)  # ONE Constraint, TWO rows
        m.subject_to(dm.sum(v, axis=0) >= 0.5)  # ONE Constraint, THREE rows
        m.subject_to(w[0] + w[1] == 1.0)  # ONE Constraint, ONE row
        return m

    tape = try_build(build("tape"))
    assert tape is not None, "array-bodied model still degrades to the JAX evaluator"
    jax_eval = NLPEvaluator(build("jax"))

    assert tape.n_constraints == jax_eval.n_constraints == 6, (
        f"tape {tape.n_constraints} rows, JAX {jax_eval.n_constraints} rows"
    )
    spans_tape = [(a, b) for a, b, _ in tape.constraint_row_map()]
    spans_jax = [(a, b) for a, b, _ in jax_eval.constraint_row_map()]
    assert spans_tape == spans_jax == [(0, 2), (2, 5), (5, 6)], spans_tape

    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(4):
        pt = rng.uniform(0.0, 2.0, size=tape.n_variables)
        np.testing.assert_allclose(
            tape.evaluate_constraints(pt),
            np.asarray(jax_eval.evaluate_constraints(pt)),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            tape.evaluate_jacobian(pt),
            np.asarray(jax_eval.evaluate_jacobian(pt)),
            rtol=1e-11,
            atol=1e-11,
        )
        checked += 1
    assert checked == 4, f"only {checked} points compared"


@pytest.mark.unit
def test_wide_reduction_is_one_nary_node_not_a_chain():
    """A wide sum must not nest, or POUNCE refuses it outright.

    ``NlExpr.max_depth`` is 10000 and a left-leaning ``+`` chain has depth N, so
    ``np.sum`` over an object array raised ``ValueError: expression nesting would
    reach depth 10001`` at N = 50000 — and ``ValueError`` is not what
    ``try_compile`` catches, so it escaped the JAX fallback rather than degrading
    to it. Both the reduction and the matmul contraction must be n-ary.
    """
    import pounce

    n = 4 * int(pounce.NlExpr.max_depth)  # comfortably past the chain limit

    m = Model("wide")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=2.0)
    m.minimize(x[0])

    node = compile_to_nl_expr(dm.sum(x * x), m)
    assert node.depth <= 4, f"sum lowered to depth {node.depth}; expected an n-ary node"

    pt = np.linspace(0.2, 1.9, n)
    assert node.eval(list(pt)) == pytest.approx(float(np.sum(pt * pt)), rel=1e-12)

    # The matmul contraction is the same shape of risk with the same fix.
    m2 = Model("wide_matmul")
    y = m2.continuous("y", shape=(n,), lb=0.1, ub=2.0)
    m2.minimize(y[0])
    row = np.linspace(0.5, 1.5, n)
    dot = compile_to_nl_expr(row @ y, m2)
    assert dot.depth <= 4, f"matmul lowered to depth {dot.depth}; expected an n-ary node"
    assert dot.eval(list(pt)) == pytest.approx(float(row @ pt), rel=1e-12)


@pytest.mark.unit
def test_wide_sum_over_a_term_list_lowers():
    """``dm.sum([...])`` over many *scalar* terms is a ``SumOverExpression``.

    A first cut mapped this with ``np.frompyfunc`` over one array per term, and
    ``np.frompyfunc`` refuses past 64 operands with a ``ValueError`` — again not
    an exception ``try_compile`` catches, so it escaped the fallback and took the
    solve down instead of degrading to JAX. The adversarial suite caught it at
    1100 terms (``test_large_dense_jacobian_no_crash``); this pins the lowering
    directly. Broadcasting terms are included because that is the case an
    element-count-only fix would still get wrong.
    """
    n = 1100
    m = Model("wide_sum_over")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    m.minimize(dm.sum(xs))

    node = compile_to_nl_expr(dm.sum(xs), m)
    pt = np.linspace(0.0, 2.0, n)
    assert node.eval(list(pt)) == pytest.approx(float(pt.sum()), rel=1e-12)
    assert node.depth <= 4, f"sum_over lowered to depth {node.depth}"

    # ...and the same with an array-valued term in the list, which must broadcast.
    m2 = Model("wide_sum_over_array")
    v = m2.continuous("v", shape=(3,), lb=0.0, ub=5.0)
    ys = [m2.continuous(f"y{i}", lb=0.0, ub=5.0) for i in range(80)]
    m2.minimize(v[0])
    rows = compile_to_nl_array(dm.sum([v, *ys]), m2)
    assert rows.shape == (3,), rows.shape
    pt2 = np.arange(83, dtype=float) / 10.0
    expected = pt2[:3] + pt2[3:].sum()
    np.testing.assert_allclose(_eval_rows(rows, pt2), expected, rtol=1e-12)


@pytest.mark.unit
def test_chain_only_reductions_refuse_past_the_depth_limit():
    """``prod`` and ``norminf`` have no n-ary opcode, so they refuse rather than crash.

    Refusing is the sound direction — the caller degrades to JAX. Letting POUNCE
    raise ``ValueError`` instead would escape ``try_compile`` and take the solve
    down with it.
    """
    import pounce

    n = 2 * int(pounce.NlExpr.max_depth)
    m = Model("too_deep")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=2.0)
    m.minimize(x[0])

    refused = 0
    for expr in (dm.prod(x), dm.norm(x, float("inf"))):
        with pytest.raises(UnsupportedForTape, match="max_depth"):
            compile_to_nl_array(expr, m)
        refused += 1
    assert refused == 2, f"only {refused} reductions checked"


@pytest.mark.unit
def test_node_budget_refuses_a_pathologically_dense_body():
    """The blowup guard fires, and fires as a refusal rather than an exhaustion.

    ``_MAX_TAPE_NODES`` bounds one build at ~0.2 s and ~2e6 live objects (0.10
    us/node, measured flat from 1e3 to 1e6 nodes). Past it the vectorized path is
    the right shape, so the tape refuses and the caller falls back.
    """
    from discopt import _nl_expr_compiler as C

    m = Model("dense")
    k = 1200
    a = m.continuous("a", shape=(k, k), lb=0.0, ub=1.0)
    b = m.continuous("b", shape=(k,), lb=0.0, ub=1.0)
    m.minimize(b[0])

    # k*k leaves for `a`, k for `b`, then k*(k+1) for the contraction: 2.88e6.
    expected = k * k + k + k * (k + 1)
    assert expected > C._MAX_TAPE_NODES, f"case is not past the cap ({expected})"
    with pytest.raises(UnsupportedForTape, match="tape nodes"):
        compile_to_nl_array(a @ b, m)

    # ...and the cap must not fire on a matmul that is merely large. This is the
    # over-charge that an `n*k * k*m` bound would have refused: 1000 products.
    m2 = Model("wide_dot")
    v = m2.continuous("v", shape=(1000,), lb=0.0, ub=1.0)
    m2.minimize(v[0])
    assert compile_to_nl_expr(np.ones(1000) @ v, m2) is not None


@pytest.mark.unit
def test_scalar_indexing_does_not_materialize_the_base():
    """``x[i]`` must stay O(1) per leaf — issue #654's cost, from the array side.

    The general path materializes the base as ``size`` nodes; the fast path
    resolves the flat slot arithmetically. If the fast path regressed, lowering
    one scalar reference into a huge variable would go from microseconds to
    seconds, and nothing about the *result* would look wrong.
    """
    m = Model("huge")
    x = m.continuous("x", shape=(2_000_000,), lb=0.0, ub=1.0)
    m.minimize(x[0])

    t0 = time.perf_counter()
    node = compile_to_nl_expr(x[1_999_999] * x[7], m)
    elapsed = time.perf_counter() - t0

    # Materializing the base would build 4e6 nodes at ~0.10 us each (~0.4 s), and
    # would in fact hit the node cap first. One second is far above the fast
    # path's cost (microseconds) and far below the general path's.
    assert elapsed < 1.0, f"scalar indexing took {elapsed:.3f}s; the fast path regressed"

    grad = np.zeros(x.size)
    pt = np.zeros(x.size)
    pt[1_999_999], pt[7] = 0.5, 0.25
    grad[1_999_999], grad[7] = 0.25, 0.5
    assert node.eval(list(pt)) == pytest.approx(0.125)
    np.testing.assert_allclose(np.asarray(node.gradient(list(pt)), dtype=float), grad)


@pytest.mark.unit
def test_matrix_norm_is_refused_rather_than_approximated():
    """``jnp.linalg.norm`` of a 2-D argument is the INDUCED norm, not a fold.

    ``ord=2`` is the largest singular value and ``ord=1`` the max column sum;
    neither is expressible as a reduction over elements. Lowering it as the
    entrywise norm would be the same class of defect as the old variadic-``*``
    ``prod``: a plausible shape computing a different function.
    """
    m = Model("matrix_norm")
    Y = m.continuous("Y", shape=(2, 3), lb=0.1, ub=2.0)
    m.minimize(Y[0, 0])

    with pytest.raises(UnsupportedForTape, match="MATRIX norm"):
        compile_to_nl_array(dm.norm(Y, 2), m)


@pytest.mark.unit
def test_scalar_entry_point_refuses_an_array_expression():
    """``compile_to_nl_expr`` must not silently pick an element or a sum.

    Its callers — objective, separation tangent, Gauss-Newton residual — each need
    exactly one scalar. Reducing an array for them would compute a different
    function, so the refusal is what keeps the array lowering from leaking into
    paths that cannot use it.
    """
    m, x, Y, s = _array_model()
    with pytest.raises(UnsupportedForTape, match="array-valued"):
        compile_to_nl_expr(x, m)
    with pytest.raises(UnsupportedForTape, match="array-valued"):
        compile_to_nl_expr(dm.sum(Y, axis=0), m)
    # ...but a genuinely scalar reduction still passes through it.
    assert compile_to_nl_expr(dm.sum(x), m) is not None


@pytest.mark.unit
def test_dae_collocation_model_solves_on_the_tape_and_matches_jax(monkeypatch):
    """The class this increment existed for, end to end.

    ``DAEBuilder`` emits one array-valued body per collocation block, which is why
    *every* collocation model used to fall back to JAX. Both arms are solved and
    both are checked against the analytic ``exp(-t)`` — comparing the two backends
    to each other alone would pass if they were wrong the same way.
    """
    pytest.importorskip("discopt.dae")
    from discopt._tape_nlp_evaluator import try_build
    from discopt.dae import ContinuousSet, DAEBuilder

    def build(name):
        m = Model(name)
        dae = DAEBuilder(m, ContinuousSet("t", bounds=(0, 2), nfe=4, ncp=3))
        dae.add_state("x", initial=1.0, bounds=(-5.0, 5.0))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()
        m.minimize(0 * dae.get_state("x")[0, 0])
        return m, dae

    m, _ = build("dae_tape")
    assert try_build(m) is not None, "DAE collocation still degrades to the JAX evaluator"

    checked = 0
    for backend in ("tape", "jax"):
        monkeypatch.setenv("DISCOPT_NLP_EVAL", backend)
        ms, ds = build(f"dae_{backend}")
        result = ms.solve()
        t_pts, x_vals = ds.extract_solution(result, "x")
        assert result.status == "optimal", f"{backend}: status {result.status}"
        err = float(np.max(np.abs(x_vals - np.exp(-t_pts))))
        assert err < 1e-3, f"{backend}: max|x - exp(-t)| = {err:.3e}"
        checked += 1
    assert checked == 2
