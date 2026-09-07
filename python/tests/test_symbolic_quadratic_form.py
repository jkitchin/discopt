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


# ---------------------------------------------------------------------------
# The probe budget gate (#1187 / test_912_wall_budget_inventory)
# ---------------------------------------------------------------------------
#
# The gate that declines an unaffordable probe sweep is denominated in PROBES,
# not seconds. The first revision priced the sweep by timing the diagonal pass
# and multiplying by the pair count, which made *which extractor runs* a
# function of machine speed -- the exact construction #1187 exists to prevent.
# The two extractors agree to ~1e-16 but not bitwise, so the fast and slow
# machine could disagree about Q and branch differently at an identical node
# count under ``deterministic=True``.
#
# These tests pin the property that replaced it: the decision is a pure function
# of the model, so it is the same on every machine and in every run.


def _dense_support_model(n):
    """A QP whose objective support is all ``n`` variables, fully coupled.

    Every variable carries a square term *and* every pair is present, so the
    pair count the gate counts is exactly ``n*(n-1)/2`` with nothing to infer.

    The square terms are load-bearing, not decoration. The probe identifies the
    support from the *diagonal* sweep alone, so a purely bilinear objective
    (``sum_{i<j} x_i x_j``, no squares) has ``f(e_j) == f(-e_j) == d`` and a zero
    diagonal for every variable: the support comes back empty, ``Q`` comes back
    zero, and the #866 verification rejects the whole extraction. That is the
    probe declining safely rather than answering wrongly -- but it means a
    bilinear-only model never reaches the budget gate, which is what these tests
    are about.
    """
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(n)]
    obj = 0.0
    for i in range(n):
        obj = obj + xs[i] * xs[i]
        for j in range(i + 1, n):
            obj = obj + xs[i] * xs[j]
    m.minimize(obj)
    return m


@pytest.mark.parametrize("n", [6, 9])
def test_probe_budget_gate_trips_exactly_at_the_pair_count(n):
    """The gate fires iff ``pairs > budget``, and the boundary is exact.

    Both sides of the boundary are asserted, one probe apart. A gate priced in
    seconds cannot be tested this way at all -- which is the point.
    """
    pairs = n * (n - 1) // 2
    model = _dense_support_model(n)
    checks = 0

    # One probe under the pair count: refused, and the message states the count.
    with pytest.raises(PC._ProbeBudgetExceeded) as exc:
        PC._extract_qp_data_from_repr(model, probe_budget=pairs - 1)
    assert f"{pairs:,} probes" in str(exc.value), str(exc.value)
    checks += 1
    assert "DISCOPT_QP_PROBE_MAX_PROBES" in str(exc.value), str(exc.value)
    checks += 1

    # Exactly at the pair count: affordable, so it runs and returns real data.
    data = PC._extract_qp_data_from_repr(model, probe_budget=pairs)
    assert data.Q is not None
    checks += 1

    # ``0`` means "no budget", not "budget of zero" -- the unbudgeted default.
    data0 = PC._extract_qp_data_from_repr(model, probe_budget=0)
    assert data0.Q is not None
    checks += 1

    assert checks == 4


def test_probe_budget_decision_is_a_pure_function_of_the_model():
    """Same model, same budget, same verdict -- with no clock in the loop.

    ``problem_classifier`` no longer imports ``time`` at all, so there is no
    clock for a slow machine to read differently. Asserting the absence of the
    import is what keeps a future revision from quietly reintroducing one and
    restoring the machine-speed dependence; ``test_912_wall_budget_inventory``
    is the package-wide version of the same guard.
    """
    import inspect

    checks = 0
    src = inspect.getsource(PC)
    assert "import time" not in src, "a clock read is back in the QP extractor"
    checks += 1

    model = _dense_support_model(7)
    budget = 7 * 6 // 2 - 1
    verdicts = []
    for _ in range(3):
        try:
            PC._extract_qp_data_from_repr(model, probe_budget=budget)
            verdicts.append("ran")
        except PC._ProbeBudgetExceeded:
            verdicts.append("declined")
    assert verdicts == ["declined"] * 3, verdicts
    checks += 1

    assert checks == 2


def test_a_model_declined_on_cost_still_gets_an_extraction():
    """Declining is a statement about cost, never a refusal to answer.

    ``extract_qp_data`` routes a declined model to the tape extractor, so the
    caller gets the same ``Q`` by a cheaper route. If that contract broke, a
    budget set too low would turn into a solve failure rather than a fallback.
    """
    model = _dense_support_model(8)
    checks = 0

    # Unbudgeted: the probe runs and its answer is the reference.
    ref = PC.extract_qp_data(model)
    assert ref.Q is not None
    checks += 1

    # A budget of one probe declines every non-trivial model, so this exercises
    # the fallback rather than the happy path.
    with pytest.raises(PC._ProbeBudgetExceeded):
        PC._extract_qp_data_from_repr(model, probe_budget=1)
    checks += 1

    monkey = PC._QP_PROBE_MAX_PROBES
    try:
        PC._QP_PROBE_MAX_PROBES = 1
        got = PC.extract_qp_data(model)
    finally:
        PC._QP_PROBE_MAX_PROBES = monkey

    assert got.Q is not None
    checks += 1

    ref_Q = ref.Q.toarray() if hasattr(ref.Q, "toarray") else np.asarray(ref.Q)
    got_Q = got.Q.toarray() if hasattr(got.Q, "toarray") else np.asarray(got.Q)
    assert got_Q.shape == ref_Q.shape, (got_Q.shape, ref_Q.shape)
    checks += 1
    np.testing.assert_allclose(got_Q, ref_Q, rtol=0, atol=1e-12)
    checks += 1
    np.testing.assert_allclose(np.asarray(got.c), np.asarray(ref.c), rtol=0, atol=1e-12)
    checks += 1

    assert checks == 6
