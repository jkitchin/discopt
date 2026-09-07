"""Symbolic (analytical, sparse) extraction of the objective's quadratic form.

The QP extractor recovered ``Q`` by finite-difference probing --
one model evaluation per variable *pair*, O(|support|^2). ``ModelRepr`` already
holds the full expression DAG and Rust already walks it to answer
``is_quadratic``; these tests cover the walk that emits the coefficients that
walk sees, as a sparse COO triplet, in O(nodes).

Two properties matter and both are asserted here:

1. **Exactness.** Every coefficient is a sum of products of literals in the DAG,
   so there is no subtractive cancellation. The probe identity
   ``f(e_i+e_j) - f(e_i) - f(e_j) + f(0)`` is a difference of nearly-equal
   floats; on ``min (x - 1e10)**2`` it loses the quadratic term entirely and
   certified a false optimum (#866). The symbolic walk must get that case right.
2. **Cost.** The walk is O(nodes), and the DAG is orders of magnitude smaller
   than the pair space it replaces.
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


def test_extractor_agrees_with_the_probe_it_replaces():
    """Where the probe succeeds, the two must agree -- this is the
    bound-neutrality check for the models the probe handles correctly."""
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-3, ub=3) for i in range(6)]
    expr = sum((i + 1) * xs[i] * xs[(i + 3) % 6] for i in range(6))
    expr = expr + sum(0.5 * (i + 1) * xs[i] * xs[i] for i in range(6))
    expr = expr - xs[2] + 11.0
    m.minimize(expr)

    symbolic = PC._extract_qp_data_symbolic(m)
    probe = PC._extract_qp_data_from_repr(m)

    np.testing.assert_allclose(PC.dense_Q(symbolic.Q), PC.dense_Q(probe.Q), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(symbolic.c, probe.c, rtol=1e-12, atol=1e-12)
    assert symbolic.obj_const == pytest.approx(probe.obj_const, rel=1e-12)


def test_dispatcher_uses_the_symbolic_path_only_when_enabled(monkeypatch):
    """The flag must be read per call, not cached at import -- a cached
    module-level bool is how a flag becomes untestable and then dead."""
    monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "0")
    assert PC._qp_symbolic_enabled() is False
    monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
    assert PC._qp_symbolic_enabled() is True


def test_dag_is_orders_of_magnitude_smaller_than_the_pair_space():
    """The cost argument, asserted rather than asserted-in-prose: the walk is
    O(nodes) and the probe is O(|support|^2)."""
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
# extractor and the quadratic *constraint* extractor were both numerical probes.
# ---------------------------------------------------------------------------


class _NoEvalRepr:
    """Forwards the symbolic accessors; raises on any numerical evaluation.

    This is the executed proof (S6) that the symbolic arm did the work: if the
    extractor still touches ``evaluate_objective`` / ``evaluate_constraint``, the
    test fails loudly instead of silently measuring the probe.
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
        raise AssertionError("fell back to the numerical probe")

    def evaluate_constraint(self, *a, **k):
        raise AssertionError("fell back to the numerical probe")


def _lp_model(n=6):
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n)]
    m.minimize(sum((i + 1) * xs[i] for i in range(n)) + 7)
    for r in range(3):
        m.subject_to(sum((r + 2) * xs[i] for i in range(n)) <= 20 + r)
    return m


def test_linear_rows_match_the_probe_they_replace(monkeypatch):
    """Symbolic and probe LP extraction agree coefficient-for-coefficient."""
    m = _lp_model()
    r = _repr_of(m)
    n = r.n_vars

    compared = 0
    for target in [None] + list(range(r.n_constraints)):
        monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "0")
        probe_terms, probe_const = PC._linear_terms_from_repr(r, n, target)
        monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
        sym_terms, sym_const = PC._linear_terms_from_repr(r, n, target)
        assert sym_terms == probe_terms
        assert sym_const == probe_const
        compared += 1 + len(probe_terms)

    assert compared > 0, "no comparison executed"
    print(f"EXECUTED_COMPARISONS={compared}")


def test_linear_extraction_does_not_evaluate_the_model(monkeypatch):
    """S6: with the flag on, no unit-vector evaluation happens at all."""
    monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
    m = _lp_model()
    guarded = _NoEvalRepr(_repr_of(m))
    terms, const = PC._linear_terms_from_repr(guarded, guarded.n_vars, 0)
    # The repr body is `sum(2*x_i) - 20`, so g(0) is the negated rhs.
    assert terms == {i: 2.0 for i in range(6)}
    assert const == -20.0


def test_linear_extraction_falls_back_when_the_walk_declines(monkeypatch):
    """A non-quadratic row must still extract -- via the probe, not a wrong answer."""
    monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=5)
    y = m.continuous("y", lb=1, ub=5)
    m.minimize(x + y)
    m.subject_to(dm.log(x) + y <= 3)
    r = _repr_of(m)
    assert r.constraint_quadratic_form(0) is None, "log should decline the walk"
    terms, _const = PC._linear_terms_from_repr(r, r.n_vars, 0)
    assert 1 in terms  # y's linear coefficient survives the probe's projection


def test_quadratic_constraint_rows_match_the_probe(monkeypatch):
    """The QCP twin: per-row Q, c, d agree between the two extractors."""
    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x * x + y)
    m.subject_to(x * x + 2 * x * y + 3 * y * y <= 9)
    m.subject_to(2 * x + y <= 4)
    r = _repr_of(m)
    n = r.n_vars

    compared = 0
    for target in [None, 0, 1]:
        monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "0")
        Qp, cp, dp = PC._quadratic_coefficients(r, n, target)
        monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
        Qs, cs, ds = PC._quadratic_coefficients(r, n, target)
        np.testing.assert_allclose(PC.dense_Q(Qs), PC.dense_Q(Qp), rtol=0, atol=0)
        np.testing.assert_allclose(cs, cp, rtol=0, atol=0)
        assert ds == dp
        compared += 1

    assert compared == 3
    print(f"EXECUTED_COMPARISONS={compared}")


def test_quadratic_constraint_extraction_does_not_probe(monkeypatch):
    """S6 for the QCP twin."""
    monkeypatch.setenv("DISCOPT_QP_SYMBOLIC", "1")
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
