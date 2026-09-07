"""Symbolic (analytical, sparse) extraction of the objective's quadratic form.

The QP extractor recovered ``Q`` by finite-difference probing -- one model
evaluation per variable *pair*, O(|support|^2). ``ModelRepr`` already holds the
full expression DAG and Rust already walks it to answer ``is_quadratic``; these
tests cover the walk that emits the coefficients that walk sees, as a sparse COO
triplet, in O(nodes). The probe is now deleted and this walk is the first rung of
every extraction ladder, so these are no longer tests of an alternative -- they
are tests of the default path.

Two properties matter and both are asserted here:

1. **Exactness.** Every coefficient is a sum of products of literals in the DAG,
   so there is no subtractive cancellation. The probe identity
   ``f(e_i+e_j) - f(e_i) - f(e_j) + f(0)`` is a difference of nearly-equal
   floats; on ``min (x - 1e10)**2`` it loses the quadratic term entirely and
   certified a false optimum (#866). The symbolic walk must get that case right.
2. **Cost.** The walk is O(nodes), and the DAG is orders of magnitude smaller
   than the pair space it replaced.

Where a test used to compare the walk against the probe, it now compares the walk
against the **AD tape** -- the other analytical arm, and the rung the ladder falls
through to. Two independent exact derivations of the same coefficients is a
stronger cross-check than agreeing with a differencing scheme was.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import problem_classifier as PC


def _repr_of(model):
    from discopt._rust import model_to_repr

    return model_to_repr(model, getattr(model, "_builder", None))


def _eval_form(form, x):
    """Evaluate a ``(qi, qj, qd, ci, cd, const)`` COO form at ``x``."""
    qi, qj, qd, ci, cd, const = form
    v = float(const)
    for i, c in zip(ci, cd):
        v += c * x[i]
    for i, j, c in zip(qi, qj, qd):
        v += c * x[i] * x[j]
    return v


def test_coefficients_are_exact_and_sparse():
    m = dm.Model()
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.minimize(2 * x * x + 3 * x * y - y + 5)

    form = _repr_of(m).objective_quadratic_form()
    assert form is not None
    qi, qj, qd, ci, cd, const = form
    assert const == 5.0
    assert list(zip(qi, qj, qd)) == [(0, 0, 2.0), (0, 1, 3.0)]
    assert list(zip(ci, cd)) == [(1, -1.0)]


def test_matches_the_evaluator_on_random_points():
    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    m.minimize(x**2 - 3 * y + x * y / 4 + 7 * z * z - 2)

    r = _repr_of(m)
    form = r.objective_quadratic_form()
    assert form is not None

    rng = np.random.default_rng(1208)
    checked = 0
    for _ in range(25):
        pt = rng.uniform(-5, 5, size=r.n_vars)
        want = r.evaluate_objective(pt)
        got = _eval_form(form, pt)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
        checked += 1
    assert checked == 25


def test_survives_the_866_cancellation_that_defeats_the_probe():
    """``min (x - 1e10)**2``: the probe loses the quadratic term to
    cancellation (d = 1e20, ulp(1e20) ~ 16384) and returns Q = 0, turning a
    sum of squares into a linear objective. The symbolic walk never subtracts
    two large numbers, so it must recover the exact coefficients."""
    m = dm.Model()
    x = m.continuous("x", lb=-1e11, ub=1e11)
    m.minimize((x - 1e10) ** 2)

    form = _repr_of(m).objective_quadratic_form()
    assert form is not None
    qi, qj, qd, ci, cd, const = form
    # (x - 1e10)^2 = x^2 - 2e10 x + 1e20
    assert list(zip(qi, qj, qd)) == [(0, 0, 1.0)]
    assert list(zip(ci, cd)) == [(0, -2e10)]
    assert const == 1e20


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda m, x, y: dm.exp(x) + y, id="transcendental"),
        pytest.param(lambda m, x, y: x / y, id="divide-by-variable"),
        pytest.param(lambda m, x, y: x * x * y, id="degree-3"),
        pytest.param(lambda m, x, y: x**3, id="cubic-power"),
    ],
)
def test_declines_rather_than_approximating(build):
    """A form the walk cannot represent must return None, never a wrong Q.
    Declining is what lets the dispatcher fall through safely."""
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=5)
    y = m.continuous("y", lb=1, ub=5)
    m.minimize(build(m, x, y))
    assert _repr_of(m).objective_quadratic_form() is None


def test_extractor_agrees_with_the_autodiff_rung():
    """The two rungs of the ladder that can both handle this model must agree
    exactly. This is the bound-neutrality check: the walk is what runs, the
    autodiff rung is what runs when the walk declines, and a disagreement means
    the answer depends on which one happened to fire."""
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-3, ub=3) for i in range(6)]
    expr = sum((i + 1) * xs[i] * xs[(i + 3) % 6] for i in range(6))
    expr = expr + sum(0.5 * (i + 1) * xs[i] * xs[i] for i in range(6))
    expr = expr - xs[2] + 11.0
    m.minimize(expr)

    symbolic = PC._extract_qp_data_symbolic(m)
    autodiff = PC._extract_qp_data_autodiff(m)

    np.testing.assert_allclose(
        PC.dense_Q(symbolic.Q), PC.dense_Q(autodiff.Q), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(symbolic.c, autodiff.c, rtol=1e-12, atol=1e-12)
    assert symbolic.obj_const == pytest.approx(autodiff.obj_const, rel=1e-12)


def test_the_symbolic_rung_is_first_and_unconditional():
    """It used to sit behind ``DISCOPT_QP_SYMBOLIC``, default-off, below a numeric
    probe. Both the flag and the probe are gone, so the walk is what a default
    solve runs -- and nothing may put it back behind a switch. A spy rather than a
    result comparison: the ladder's lower rungs return the same numbers, so
    equality alone would not notice the first rung being skipped."""
    assert not hasattr(PC, "_qp_symbolic_enabled"), "the opt-in gate is back"
    assert not hasattr(PC, "_extract_qp_data_from_repr"), "the numeric probe is back"

    m = dm.Model()
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize(2 * x * x + 3 * x * y - y + 5)
    m.subject_to(x + y <= 4)

    calls = []
    real = PC._extract_qp_data_symbolic

    def _spy(model):
        try:
            out = real(model)
        except Exception as exc:
            calls.append(("declined", exc))
            raise
        calls.append(("returned", out))
        return out

    PC._extract_qp_data_symbolic = _spy
    try:
        data = PC.extract_qp_data(m)
    finally:
        PC._extract_qp_data_symbolic = real

    assert len(calls) == 1, f"the symbolic rung ran {len(calls)} times, expected 1"
    # Not just "it was called" -- "it produced the answer". The lower rungs return
    # the same numbers, so a symbolic rung that raises on every quadratic
    # objective would leave every value in this test unchanged while the walk
    # never ran. That is not hypothetical: splitting the objective out of the
    # constraint extractor was a fix for exactly that, and this assertion is what
    # would have caught it. (``_assemble_qp_from_repr`` called the LP extractor
    # for its constraints; once the LP objective arm started refusing a nonlinear
    # objective instead of projecting it, the QP path raised on every QP.)
    assert calls[0][0] == "returned", f"the symbolic rung declined: {calls[0][1]!r}"
    assert PC.dense_Q(data.Q)[0, 0] == pytest.approx(4.0)


def test_dag_is_orders_of_magnitude_smaller_than_the_pair_space():
    """The cost argument, asserted rather than asserted-in-prose: the walk is
    O(nodes) and the probe it replaced was O(|support|^2)."""
    m = dm.Model()
    n = 120
    xs = [m.continuous(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(sum(xs[i] * xs[(i + 1) % n] for i in range(n)))

    r = _repr_of(m)
    form = r.objective_quadratic_form()
    assert form is not None
    pairs = n * (n - 1) // 2
    assert r.n_nodes < pairs, f"{r.n_nodes} nodes vs {pairs} pairs"
    # The recovered form is sparse: n cross terms, not n^2.
    assert len(form[0]) == n


# ---------------------------------------------------------------------------
# The same defect, one dimension lower and one dimension wider: the LP row
# extractor and the quadratic *constraint* extractor were numerical probes too,
# and are now the same arena walk.
# ---------------------------------------------------------------------------


class _NoEvalRepr:
    """Forwards the symbolic accessors; raises on any numerical evaluation.

    This is the executed proof (CLAUDE.md §6) that the symbolic arm did the work:
    if the extractor touches ``evaluate_objective`` / ``evaluate_constraint`` at
    all, the test fails loudly instead of silently measuring something else.
    """

    def __init__(self, inner):
        self._inner = inner
        self.n_vars = inner.n_vars
        self.n_constraints = inner.n_constraints

    def objective_quadratic_form(self, *a, **k):
        return self._inner.objective_quadratic_form(*a, **k)

    def constraint_quadratic_form(self, i, *a, **k):
        return self._inner.constraint_quadratic_form(i, *a, **k)

    def evaluate_objective(self, *a, **k):
        raise AssertionError("the extractor evaluated the model")

    def evaluate_constraint(self, *a, **k):
        raise AssertionError("the extractor evaluated the model")


def _lp_model(n=6):
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n)]
    m.minimize(sum((i + 1) * xs[i] for i in range(n)) + 7)
    for r in range(3):
        m.subject_to(sum((r + 2) * xs[i] for i in range(n)) <= 20 + r)
    return m


def test_linear_rows_match_the_tape_jacobian():
    """The walk's rows must equal the AD tape's Jacobian row for row.

    The tape is the independent oracle here: it derives the same coefficients by
    a different mechanism (reverse-mode AD over the compiled tape) and is the rung
    ``extract_lp_data`` falls through to, so a disagreement is a real fork in what
    the solver sees. Compared against the tape's Jacobian at the origin, which for
    an affine body IS the row.
    """
    from discopt._tape_nlp_evaluator import try_build

    m = _lp_model()
    r = _repr_of(m)
    n = r.n_vars
    tape = try_build(m)
    assert tape is not None, "the tape declined a plain LP"
    x0 = np.zeros(n, dtype=np.float64)
    jac = np.asarray(tape.evaluate_jacobian(x0), dtype=np.float64)
    body0 = np.asarray(tape.evaluate_constraints(x0), dtype=np.float64).reshape(-1)

    compared = 0
    for i in range(r.n_constraints):
        terms, const = PC._linear_terms_from_repr(r, n, i)
        row = np.zeros(n, dtype=np.float64)
        for j, v in terms.items():
            row[j] = v
        np.testing.assert_allclose(row, jac[i], rtol=1e-12, atol=1e-12)
        assert const == pytest.approx(float(body0[i]), rel=1e-12, abs=1e-12)
        compared += 1 + len(terms)

    obj_terms, obj_const = PC._linear_terms_from_repr(r, n, None)
    grad = np.asarray(tape.evaluate_gradient(x0), dtype=np.float64)
    for j in range(n):
        assert obj_terms.get(j, 0.0) == pytest.approx(float(grad[j]), rel=1e-12, abs=1e-12)
    assert obj_const == pytest.approx(float(tape.evaluate_objective(x0)), rel=1e-12)
    compared += 1 + n

    assert compared > 0, "no comparison executed"
    print(f"EXECUTED_COMPARISONS={compared}")


def test_linear_extraction_does_not_evaluate_the_model():
    """CLAUDE.md §6: no unit-vector evaluation happens at all."""
    m = _lp_model()
    guarded = _NoEvalRepr(_repr_of(m))
    terms, const = PC._linear_terms_from_repr(guarded, guarded.n_vars, 0)
    # The repr body is `sum(2*x_i) - 20`, so g(0) is the negated rhs.
    assert terms == {i: 2.0 for i in range(6)}
    assert const == -20.0


def test_a_declining_row_refuses_instead_of_projecting():
    """A row the walk cannot prove linear must RAISE, not return its linear part.

    The probe used to answer here: it evaluated ``g(e_j) - g(0)`` and returned
    ``y``'s coefficient while silently dropping ``log(x)``, which is a wrong row
    presented as a right one. The walk declines instead, and ``extract_lp_data``
    falls through to the tape rung -- so the model still extracts, by an arm that
    can actually represent it. Refusing loudly is the fix (CLAUDE.md §3).
    """
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=5)
    y = m.continuous("y", lb=1, ub=5)
    m.minimize(x + y)
    m.subject_to(dm.log(x) + y <= 3)
    r = _repr_of(m)
    assert r.constraint_quadratic_form(0) is None, "log should decline the walk"

    with pytest.raises(PC._NotQuadraticError) as exc:
        PC._linear_terms_from_repr(r, r.n_vars, 0)
    assert "constraint 0" in str(exc.value), str(exc.value)


def test_quadratic_constraint_rows_match_the_tape():
    """The QCP twin: per-row Q, c, d agree between the walk and its fallback.

    ``_tape_quadratic_coefficients`` is what runs when the walk declines a row, so
    the two must be interchangeable. This is also where a factor-of-two convention
    error would surface -- see that function's docstring."""
    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x * x + y)
    m.subject_to(x * x + 2 * x * y + 3 * y * y <= 9)
    m.subject_to(2 * x + y <= 4)
    r = _repr_of(m)
    n = r.n_vars

    from discopt._tape_nlp_evaluator import try_build

    tape = try_build(m)
    assert tape is not None, "the tape declined a plain QCP"

    compared = 0
    for target in [None, 0, 1]:
        Qw, cw, dw = PC._quadratic_coefficients(r, n, target)
        Qt, ct, dt = PC._tape_quadratic_coefficients(tape, n, target)
        np.testing.assert_allclose(PC.dense_Q(Qt), PC.dense_Q(Qw), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(ct), np.asarray(cw), rtol=1e-12, atol=1e-12)
        assert dt == pytest.approx(dw, rel=1e-12, abs=1e-12)
        compared += 1

    assert compared == 3
    print(f"EXECUTED_COMPARISONS={compared}")


def test_quadratic_constraint_extraction_does_not_probe():
    """CLAUDE.md §6 for the QCP twin."""
    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x + y)
    m.subject_to(x * y <= 4)
    guarded = _NoEvalRepr(_repr_of(m))
    Q, c, d = PC._quadratic_coefficients(guarded, guarded.n_vars, 0)
    # 0.5 * (Q[0,1] + Q[1,0]) * x*y == 1.0 * x*y  =>  Q[0,1] == Q[1,0] == 1.0.
    assert PC.dense_Q(Q)[0, 1] == 1.0
    assert PC.dense_Q(Q)[1, 0] == 1.0


# ---------------------------------------------------------------------------
# The probe budget gate (#1187) is gone with the probe it budgeted.
# ---------------------------------------------------------------------------
#
# There were tests here pinning that the gate's decision was a pure function of
# the model rather than of machine speed, and that a model declined on cost still
# received an extraction. Both properties were about keeping a wrong algorithm
# survivable. Budgeting an O(|support|^2) reconstruction of coefficients the arena
# holds exactly was a band-aid; the fix was to delete it, which makes the gate,
# its determinism argument and its fallback all unreachable code.
#
# What replaced those guarantees, and where each is now asserted:
#
#   * "the decision does not depend on machine speed" -- there is no decision.
#     ``test_the_symbolic_rung_is_first_and_unconditional`` asserts the flag and
#     the probe are both absent, so no path can reintroduce one.
#   * "a declined model still gets an extraction" -- the ladder still has its
#     lower rungs, and ``test_a_declining_row_refuses_instead_of_projecting``
#     plus ``test_875_sparse_qcp_extraction.py`` cover declining into the tape.
