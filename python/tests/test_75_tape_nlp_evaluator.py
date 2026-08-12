"""``TapeNLPEvaluator`` must reproduce the JAX ``NLPEvaluator`` it replaces.

Issue #75, Stage 3. ``_relax/nlp_evaluator.py`` is the single trigger that imports
JAX on every nonlinear solve, so replacing it is what makes the solve path
JAX-free. This pins the properties that decide whether it can:

* **numerical agreement** on f, grad f, g, J and the Lagrangian Hessian, against
  the incumbent, on real corpus instances;
* **conventions**, which the callers depend on and which differ between methods —
  ``evaluate_lagrangian_hessian`` is FULL dense, ``hessian_structure`` /
  ``evaluate_hessian_values`` are LOWER-TRIANGLE COO;
* **loud refusal** on a model with no tape lowering, so the JAX path is kept
  rather than silently approximated;
* **the parameter hazard** — the tape bakes ``Parameter.value`` in as a constant
  where JAX reads it live, so a stale tape returns wrong derivatives with no
  error. This is the one failure mode that is invisible without a test.

Entry experiment recorded when this landed: 66 in-repo corpus instances, 5 points
each, max rel drift f 5.48e-16, grad 3.77e-16, g 4.55e-13, J 3.06e-15, H 7.82e-13.
"""

import logging
import math
import sys
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._tape_nlp_evaluator import (
    TapeNLPEvaluator,
    build_evaluator,
    tape_backend_requested,
    try_build,
)

pytest.importorskip("pounce")

DATA = Path(__file__).parent / "data" / "minlplib_nl"

# Small, structurally varied, and fast to trace on the JAX side (the comparison
# is only as cheap as the incumbent). Nonlinear so the Hessian is not trivially 0.
INSTANCES = ["alan", "ex1221", "ex1225", "gbd", "st_e13", "nvs12", "st_test1"]


def _jax_evaluator(model):
    from discopt._relax.nlp_evaluator import NLPEvaluator

    return NLPEvaluator(model)


def _points(ev, n_pts, seed):
    """Points strictly inside the box, with infinite sides clipped."""
    lb, ub = ev.variable_bounds
    lo = np.where(np.isfinite(lb), lb, -5.0)
    hi = np.where(np.isfinite(ub), ub, 5.0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    rng = np.random.default_rng(seed)
    return [lo + (0.05 + 0.9 * rng.random(len(lo))) * (hi - lo) for _ in range(n_pts)]


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(a))))


@pytest.mark.unit
@pytest.mark.parametrize("name", INSTANCES)
def test_matches_the_jax_evaluator(name):
    """f, grad f, g, J and the Lagrangian Hessian, at sampled points."""
    from discopt.modeling.core import from_nl

    path = DATA / f"{name}.nl"
    if not path.exists():
        pytest.skip(f"{name}.nl not in the in-repo corpus")

    model = from_nl(str(path))
    jax_ev = _jax_evaluator(model)
    tape = try_build(model)
    assert tape is not None, f"{name} should be tape-representable"

    assert tape.n_variables == jax_ev.n_variables
    assert tape.n_constraints == jax_ev.n_constraints

    n, m = tape.n_variables, tape.n_constraints
    compared = 0
    for x in _points(jax_ev, 4, seed=abs(hash(name)) % (2**32)):
        jf = float(jax_ev.evaluate_objective(x))
        jg = np.asarray(jax_ev.evaluate_gradient(x), dtype=float)
        if not (np.isfinite(jf) and np.all(np.isfinite(jg))):
            continue  # incumbent out of domain here; nothing to compare
        compared += 1

        assert _rel([jf], [tape.evaluate_objective(x)]) <= 1e-10
        assert _rel(jg, tape.evaluate_gradient(x)) <= 1e-10

        if m:
            jc = np.asarray(jax_ev.evaluate_constraints(x), dtype=float)
            assert _rel(jc, tape.evaluate_constraints(x)) <= 1e-10
            jj = np.asarray(jax_ev.evaluate_jacobian(x), dtype=float).reshape(m, n)
            assert _rel(jj, tape.evaluate_jacobian(x)) <= 1e-10

        lam = np.linspace(0.3, 1.7, m) if m else np.zeros(0)
        jh = np.asarray(jax_ev.evaluate_lagrangian_hessian(x, 1.0, lam), dtype=float).reshape(n, n)
        th = tape.evaluate_lagrangian_hessian(x, 1.0, lam)
        assert _rel(jh, th) <= 1e-8, f"Lagrangian Hessian drift on {name}"

    # §6: every point skipped would pass vacuously.
    assert compared >= 2, f"only {compared} points compared for {name}"


@pytest.mark.unit
def test_lagrangian_hessian_is_full_and_values_are_lower_triangle():
    """The two Hessian methods use DIFFERENT conventions, and callers rely on it.

    ``nlp_ipopt.hessian`` indexes the dense result with ``h[rows, cols]`` against
    its own lower-triangle structure; returning a bare triangle from the dense
    method would silently zero half of every off-diagonal term.
    """
    m, x, y = _xy_model()
    tape = TapeNLPEvaluator(m)
    pt = np.array([0.7, 1.3])

    dense = tape.evaluate_lagrangian_hessian(pt, 1.0, np.zeros(tape.n_constraints))
    assert dense.shape == (2, 2)
    np.testing.assert_allclose(dense, dense.T, rtol=0, atol=0), "dense must be symmetric"
    # x*y in the objective => a genuinely nonzero off-diagonal to detect.
    assert abs(dense[0, 1]) > 1e-9
    assert dense[0, 1] == pytest.approx(dense[1, 0])

    rows, cols = tape.hessian_structure()
    assert np.all(np.asarray(rows) >= np.asarray(cols)), "structure must be lower triangle"


def _xy_model():
    m = Model()
    x = m.continuous("x", lb=0.2, ub=3.0)
    y = m.continuous("y", lb=0.2, ub=3.0)
    m.subject_to(x * y >= 0.5)
    m.minimize(x * y + x * x)
    return m, x, y


def _curvature(build, pt):
    """d2/dx2 of a 1-var objective, through the evaluator the solver calls."""
    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    m.minimize(build(x))
    m.subject_to(x <= 1e309)
    tape = TapeNLPEvaluator(m)
    h = tape.evaluate_lagrangian_hessian(np.array([pt]), 1.0, np.zeros(tape.n_constraints))
    return float(np.asarray(h)[0, 0])


def _sigmoid_at_minus_abs(a):
    """``sigmoid(-|a|)`` without cancellation -- it never rounds to 1.0.

    Writing the reference as ``s = sigmoid(a); s*(1-s)`` silently underflows for
    ``a >~ 37``: ``s`` becomes exactly 1.0, ``1-s`` is 0, and the reference then
    claims 0 for a true 4.248e-18 -- which reads as a defect in whatever it is
    checking. An earlier revision of this probe did exactly that and reported
    three false failures.
    """
    t = math.exp(-abs(a))
    return t / (1.0 + t)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,build,pt,exact",
    [
        # log1p'' = -1/(1+a)^2. The `u == 1` arm of the Kahan form is LINEAR, so
        # it has no curvature: before the series branch this returned exactly 0.0
        # at 1e-17 and -1.0039 at 1e-13, against a true -1.
        ("log1p@1e-17", dm.log1p, 1e-17, -1.0),
        ("log1p@-1e-17", dm.log1p, -1e-17, -1.0),
        ("log1p@1e-13", dm.log1p, 1e-13, -1.0),
        ("log1p@0.5", dm.log1p, 0.5, -1.0 / 2.25),
        # sigmoid'' = s(1-s)(1-2s), odd in a. The naive `1/(1+exp(-a))` overflows
        # in the LEFT tail at second order -- `exp(745)` is inf, the quotient rule
        # forms inf/inf -- giving nan at -745 and a SIGN-FLIPPED -5.148e-131 at
        # -300. Orders 0 and 1 were finite there, which is why it went unnoticed.
        ("sigmoid@-745", dm.sigmoid, -745.0, None),
        ("sigmoid@-300", dm.sigmoid, -300.0, None),
        ("sigmoid@300", dm.sigmoid, 300.0, None),
        ("sigmoid@0", dm.sigmoid, 0.0, 0.0),
        # softplus'' = s(1-s); already correct, pinned so the log1p rewrite it
        # depends on cannot regress it.
        ("softplus@40", dm.softplus, 40.0, None),
        ("softplus@-40", dm.softplus, -40.0, None),
    ],
)
def test_second_derivatives_match_analytic_truth_in_the_tails(name, build, pt, exact):
    """The NLP subsolve consumes the LAGRANGIAN HESSIAN, not just f and grad f.

    The 1c54b726 hardening verified orders 0 and 1 only, and two rewrites were
    wrong at order 2 while looking clean at orders 0 and 1. `dm.sigmoid` is the
    SIGMOID activation in all four `nn/formulations/`, so this is a live path.

    Truth here is ANALYTIC, not JAX: at these points JAX itself underflows (it
    returns 0.0 for sigmoid''(300) where the true value is -5.148e-131), so
    asserting against JAX would pin the wrong answer.
    """
    if exact is None:
        s = _sigmoid_at_minus_abs(pt)
        if build is dm.softplus:
            exact = s * (1.0 - s)
        else:
            mag = s * (1.0 - s) * (1.0 - 2.0 * s)
            exact = mag if pt <= 0 else -mag

    got = _curvature(build, pt)
    assert math.isfinite(got), f"{name}: curvature is {got}"
    assert got == pytest.approx(exact, rel=1e-6, abs=1e-330), f"{name}: {got} vs exact {exact}"


@pytest.mark.unit
@pytest.mark.parametrize("a", [1e-4, 2e-4, 4.9e-4, 5e-4, 5.1e-4, 1e-3, 1.778e-3, 1e-2])
def test_log1p_curvature_across_the_taylor_crossover(a):
    """``_LOG1P_TAYLOR`` is a MEASURED minimax point; pin it so it stays one.

    log1p's curvature error peaks at the crossover from both sides: the series'
    truncation error grows like ``5a**4`` and Kahan's derivative distortion decays
    like ``eps/a`` (its ``u - 1`` differs from ``a`` by a rounding gap that double
    differentiation amplifies). The worst case over the whole line therefore sits
    wherever the two curves cross, and moving the constant moves the peak.

    The tolerance is deliberately 1e-13, NOT the 1e-6 used for the tail cases. At
    the previous 1e-4 crossover the measured worst-case relative error was
    1.092e-11, so a 1e-6 assertion would have passed pre-fix and been decorative.
    1e-13 fails on that value while leaving ~3 orders of headroom over the
    4.608e-16 measured now.
    """
    exact = -1.0 / (1.0 + a) ** 2
    got = _curvature(dm.log1p, a)
    assert math.isfinite(got), f"log1p'' at {a} is {got}"
    assert got == pytest.approx(exact, rel=1e-13), f"log1p'' at {a}: {got} vs {exact}"


@pytest.mark.unit
def test_parameter_rebind_rebuilds_the_tape():
    """The tape bakes ``Parameter.value`` in; JAX reads it live.

    ``evaluator_fingerprint`` deliberately excludes parameter values, so a tape
    cached under it would serve derivatives for the OLD value with no error and
    no exception. This is the failure mode that has no symptom.
    """
    import discopt.modeling as dm

    m = Model()
    x = m.continuous("x", lb=0.1, ub=5.0)
    p = m.parameter("p", value=2.0) if hasattr(m, "parameter") else None
    if p is None:
        pytest.skip("modeling API exposes no Parameter constructor")
    m.minimize(p * x * x)

    tape = TapeNLPEvaluator(m)
    pt = np.array([1.5])
    assert tape.evaluate_objective(pt) == pytest.approx(2.0 * 1.5**2)
    assert tape.evaluate_gradient(pt)[0] == pytest.approx(2 * 2.0 * 1.5)

    p.value = 7.0
    assert tape.evaluate_objective(pt) == pytest.approx(7.0 * 1.5**2), "stale tape after re-bind"
    assert tape.evaluate_gradient(pt)[0] == pytest.approx(2 * 7.0 * 1.5)
    _ = dm  # imported for parity with the modeling-API idiom


@pytest.mark.unit
def test_evaluator_is_usable_from_multiple_threads():
    """A shared tape across threads produced a FALSE `infeasible`. Never again.

    ``pounce.NlProblem`` is ``#[pyclass(unsendable)]``: touching one from a
    thread other than its creator raises a Rust ``PanicException``, which derives
    directly from ``BaseException`` and so slips past every ``except Exception``
    in the solver. Measured on clay0303hfsg before the fix: the JAX arm returned
    `feasible` (obj 26669.1) and the tape arm returned **`infeasible`** — a wrong
    certificate, the one outcome that is never acceptable (CLAUDE.md §1).

    The evaluator now keeps one tape per thread. This asserts every thread gets
    the SAME values, not merely that nothing raised: a per-thread tape built from
    the wrong expression would also "work".
    """
    import threading

    m, _x, _y = _xy_model()
    ev = TapeNLPEvaluator(m)
    pt = np.array([0.7, 1.3])
    want_f = ev.evaluate_objective(pt)
    want_g = np.asarray(ev.evaluate_gradient(pt), dtype=float)

    results: dict[int, object] = {}

    def worker(i):
        try:
            results[i] = (
                ev.evaluate_objective(pt),
                np.asarray(ev.evaluate_gradient(pt), dtype=float),
                ev.evaluate_lagrangian_hessian(pt, 1.0, np.zeros(ev.n_constraints)),
            )
        except BaseException as exc:  # noqa: BLE001 - PanicException is not an Exception
            results[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4, "a worker died without recording anything"
    for i, got in sorted(results.items()):
        assert not isinstance(got, str), f"thread {i} raised: {got}"
        f_i, g_i, h_i = got
        assert f_i == pytest.approx(want_f, rel=1e-15), f"thread {i} objective differs"
        np.testing.assert_allclose(g_i, want_g, rtol=1e-15)
        assert np.all(np.isfinite(h_i))


@pytest.mark.unit
def test_per_thread_fallback_is_exercised_and_agrees(monkeypatch):
    """The OLD-pounce path must keep working, and must be tested on new pounce.

    pounce #477/#478 made ``NlProblem`` sendable, so on a current build the
    evaluator shares one problem and the per-thread fallback never runs. That is
    precisely when a fallback rots: using it wrongly costs a false
    ``infeasible``. Forcing the capability probe to ``False`` keeps it covered,
    and asserts the two paths agree rather than merely both running.

    #477/#478 shipped in ``pounce-solver`` 0.10.0 — the version this project now
    floors at — so every in-contract install takes the shared path and the
    fallback covers only a below-floor pounce. It is retained rather than
    deleted because the capability is *probed*, not assumed from a version
    string: a build where the probe says "not sendable" must still solve.
    """
    import threading

    import discopt._tape_nlp_evaluator as T

    m, _x, _y = _xy_model()
    pt = np.array([0.7, 1.3])

    shared = TapeNLPEvaluator(m)
    assert shared._shared_problem is not None, "probe should report thread-safe here"
    want_f = shared.evaluate_objective(pt)
    want_g = np.asarray(shared.evaluate_gradient(pt), dtype=float)

    monkeypatch.setattr(T, "_nlproblem_is_thread_safe", lambda: False)
    fallback = TapeNLPEvaluator(m)
    assert fallback._shared_problem is None, "fallback path was not taken"

    assert fallback.evaluate_objective(pt) == pytest.approx(want_f, rel=1e-15)
    np.testing.assert_allclose(fallback.evaluate_gradient(pt), want_g, rtol=1e-15)

    seen: dict[int, object] = {}

    def worker(i):
        try:
            seen[i] = fallback.evaluate_objective(pt)
        except BaseException as exc:  # noqa: BLE001 - PanicException is not an Exception
            seen[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 3
    for i, got in sorted(seen.items()):
        assert not isinstance(got, str), f"fallback thread {i} raised: {got}"
        assert got == pytest.approx(want_f, rel=1e-15)


@pytest.mark.unit
def test_solve_degrades_to_jax_when_pounce_is_missing():
    """A missing or too-old POUNCE must fall back, not crash.

    ``pyproject`` requires ``pounce-solver>=0.10``, but a minimal install need not
    have it — CI's AMP-coverage lane installed jax/numpy/scipy/highspy and
    nothing else. That was invisible while the tape backend was opt-in and became
    a hard ``ModuleNotFoundError`` on every solve the moment it graduated to
    default-ON, because ``try_build`` catches only ``UnsupportedForTape``.

    Runs in a SUBPROCESS with the import blocked, since ``pounce`` is already
    imported in this process and cannot be un-imported.
    """
    import subprocess
    import sys

    script = """
import sys, builtins
_real = builtins.__import__


def block(name, *a, **k):
    if name == "pounce" or name.startswith("pounce."):
        raise ModuleNotFoundError("No module named 'pounce'")
    return _real(name, *a, **k)


builtins.__import__ = block

from discopt import Model
import discopt.modeling as dm

m = Model()
x = m.continuous("x", lb=0.2, ub=4.0)
y = m.continuous("y", lb=0.2, ub=4.0)
m.subject_to(dm.exp(x) + y * y <= 20.0)
m.subject_to(x * y >= 1.0)
m.minimize(x * x + y + dm.log(x))
r = m.solve()
assert str(r.status) == "optimal", r.status
assert "jax" in sys.modules, "should have fallen back to the JAX evaluator"
print("FALLBACK-OK")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, f"solve crashed without pounce\n{out.stderr[-2500:]}"
    assert "FALLBACK-OK" in out.stdout


@pytest.mark.unit
def test_pounce_usable_reports_false_on_a_build_without_nlexpr(monkeypatch):
    """A POUNCE too old to have ``NlExpr`` must also be refused.

    Importability is not the contract — a build predating pounce #470 imports
    fine and explodes on attribute access deep inside a solve. That exact stale
    build silently disabled a whole test file behind ``importorskip`` earlier in
    this work, so the probe checks the surface it needs.
    """
    import types

    import discopt._tape_nlp_evaluator as T

    monkeypatch.setattr(T, "_POUNCE_USABLE", None)
    fake = types.ModuleType("pounce")  # no NlExpr, no build_nl_problem
    monkeypatch.setitem(sys.modules, "pounce", fake)
    assert T.pounce_usable() is False
    assert T.tape_backend_requested() is False

    monkeypatch.setattr(T, "_POUNCE_USABLE", None)
    fake.NlExpr = object()
    fake.build_nl_problem = lambda *a, **k: None
    assert T.pounce_usable() is True


# ── Gauss-Newton on the tape (was blocker #3 in docs/dev/jax-removal-plan.md) ──
#
# This used to be a loud refusal: `2 JᵀJ` is a different matrix from the exact
# Hessian, and swapping one for the other silently is a wrong derivative. The
# refusal was correct but incomplete — pounce has no "residual" concept, but its
# sparse *constraint* Jacobian is the same object under another name, so taping
# r(x) as the constraint rows of an auxiliary NlProblem yields ∂r/∂x directly.
# These tests are what makes the swap safe rather than silent.


def _gn_ls_model(with_constraint: bool = False):
    """Nonlinear least squares: shared parameters, so ``JᵀJ`` is genuinely dense."""
    m = Model()
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    c = m.continuous("c", lb=-5, ub=5)
    ts = [0.1, 0.5, 1.0, 1.7, 2.4]
    ys = [1.2, 1.9, 2.3, 2.1, 1.5]
    # Builtin sum() deliberately: it seeds with int 0, so this is the `0 + r₀² + …`
    # form that used to make extract_residuals decline and silently test the exact
    # path here. Now recognized (a constant term has zero curvature), and the
    # is_gauss_newton assertions below are what keep that honest.
    m.minimize(sum((a * dm.exp(-b * t) + c - y) ** 2 for t, y in zip(ts, ys)))
    if with_constraint:
        m.subject_to(a * b + c**2 <= 4.0)
    m._gauss_newton_hessian = True
    return m


@pytest.mark.unit
def test_gauss_newton_is_supported_not_refused():
    """Regression: the tape used to raise ``UnsupportedForTape`` and degrade to JAX."""
    m = _gn_ls_model()
    tape = TapeNLPEvaluator(m)
    assert tape.is_gauss_newton is True
    assert try_build(m) is not None


@pytest.mark.unit
def test_gauss_newton_objective_hessian_matches_jax():
    """``2 JᵀJ`` from the aux tape vs the JAX arm — the entry experiment, pinned."""
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _gn_ls_model()
    tape = TapeNLPEvaluator(m)
    jax_ev = NLPEvaluator(m, gauss_newton=True)
    assert jax_ev.is_gauss_newton is True

    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(5):
        pt = rng.uniform(-1.5, 1.5, tape.n_variables)
        lam = np.zeros(tape.n_constraints)
        H_tape = tape.evaluate_lagrangian_hessian(pt, 1.0, lam)
        H_jax = np.asarray(jax_ev.evaluate_lagrangian_hessian(pt, 1.0, lam))
        np.testing.assert_allclose(H_tape, H_jax, rtol=1e-9, atol=1e-9)
        checked += 1
    assert checked == 5, "probe must have compared something (CLAUDE.md §6)"


@pytest.mark.unit
def test_gauss_newton_keeps_exact_constraint_curvature():
    """Only the OBJECTIVE term is approximated; ``Σ λᵢ ∇²gᵢ`` stays exact."""
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _gn_ls_model(with_constraint=True)
    tape = TapeNLPEvaluator(m)
    jax_ev = NLPEvaluator(m, gauss_newton=True)
    assert tape.n_constraints == 1

    rng = np.random.default_rng(12)
    pt = rng.uniform(-1.0, 1.0, tape.n_variables)
    lam = np.array([0.85])

    # obj_factor=0 isolates the constraint block, which must be EXACT.
    H_tape = tape.evaluate_lagrangian_hessian(pt, 0.0, lam)
    H_jax = np.asarray(jax_ev.evaluate_lagrangian_hessian(pt, 0.0, lam))
    np.testing.assert_allclose(H_tape, H_jax, rtol=1e-9, atol=1e-9)
    assert np.abs(H_tape).max() > 1e-9, "constraint curvature must be nonzero here"

    # And with both terms present.
    np.testing.assert_allclose(
        tape.evaluate_lagrangian_hessian(pt, 1.0, lam),
        np.asarray(jax_ev.evaluate_lagrangian_hessian(pt, 1.0, lam)),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.unit
def test_gauss_newton_declared_structure_loses_no_value():
    """The silent-drop guard: every GN nonzero must land inside the DECLARED COO.

    ``hessian_structure`` is declared once. ``2 JᵀJ`` fills wherever two residuals
    share a variable, so a value outside the declared pattern is dropped with no
    error — a wrong Hessian that reads as a pass. Reconstructing the dense matrix
    from the COO and comparing against the dense path is what catches that.
    """
    m = _gn_ls_model(with_constraint=True)
    tape = TapeNLPEvaluator(m)
    rng = np.random.default_rng(13)
    pt = rng.uniform(-1.0, 1.0, tape.n_variables)
    lam = np.array([0.4])

    rows, cols = tape.hessian_structure()
    assert np.all(np.asarray(rows) >= np.asarray(cols)), "structure must be lower triangle"
    vals = tape.evaluate_hessian_values(pt, 1.0, lam)
    assert len(vals) == len(rows)

    dense = tape.evaluate_lagrangian_hessian(pt, 1.0, lam)
    n = tape.n_variables
    rebuilt = np.zeros((n, n))
    np.add.at(rebuilt, (np.asarray(rows), np.asarray(cols)), vals)
    rebuilt = rebuilt + np.tril(rebuilt, -1).T
    np.testing.assert_allclose(rebuilt, dense, rtol=0, atol=0)
    # Nothing was dropped: the dense GN Hessian is not all-zero off the diagonal.
    assert np.abs(dense - np.diag(np.diag(dense))).max() > 1e-9


@pytest.mark.unit
def test_gauss_newton_hessian_is_psd_and_differs_from_exact():
    """``2 JᵀJ`` is PSD by construction, and away from a zero residual it is a
    genuinely different matrix — so the approximation is really being applied."""
    m = _gn_ls_model()
    gn = TapeNLPEvaluator(m)

    exact_model = _gn_ls_model()
    exact_model._gauss_newton_hessian = False
    exact = TapeNLPEvaluator(exact_model)
    assert exact.is_gauss_newton is False

    pt = np.array([1.0, 0.3, 0.5])
    lam = np.zeros(gn.n_constraints)
    H_gn = gn.evaluate_lagrangian_hessian(pt, 1.0, lam)
    H_ex = exact.evaluate_lagrangian_hessian(pt, 1.0, lam)
    assert np.min(np.linalg.eigvalsh(H_gn)) >= -1e-9
    assert np.abs(H_gn - H_ex).max() > 1e-6


@pytest.mark.unit
@pytest.mark.parametrize("reason", ["maximize", "not_sum_of_squares"])
def test_gauss_newton_declines_to_exact_hessian(reason):
    """When GN does not apply the tape uses the EXACT Hessian, matching the JAX
    arm's rules — so ``DISCOPT_NLP_EVAL`` cannot change which models get it."""
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = Model()
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    if reason == "maximize":
        m.maximize((x - 1) ** 2 + (y - 2) ** 2)
    else:
        m.minimize(dm.exp(x) + y**2)
    m._gauss_newton_hessian = True

    tape = TapeNLPEvaluator(m)
    assert tape.is_gauss_newton is False
    assert NLPEvaluator(m, gauss_newton=True).is_gauss_newton is False


@pytest.mark.unit
def test_gauss_newton_decline_warns_once_not_per_rebuild(caplog):
    """Audible, but not once per B&B node.

    Declining must WARN — the caller set ``gauss_newton=True`` and it did nothing,
    which at INFO is invisible. But ``_build`` re-runs whenever a ``Parameter``
    moves, which in a fitting loop is every iteration, so the warning is guarded
    to once per evaluator.
    """
    m = Model()
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(dm.exp(x) + y**2)  # not a sum of squares
    m._gauss_newton_hessian = True

    with caplog.at_level(logging.WARNING, logger="discopt._tape_nlp_evaluator"):
        tape = TapeNLPEvaluator(m)
        assert tape.is_gauss_newton is False
        tape._build()  # simulate the parameter-moved rebuild
        tape._build()

    hits = [r for r in caplog.records if "gauss_newton" in r.message]
    assert len(hits) == 1, f"expected exactly one warning, got {[r.message for r in hits]}"
    assert hits[0].levelno >= logging.WARNING


@pytest.mark.unit
def test_custom_call_model_falls_back_to_jax():
    """``dm.custom`` has no tape equivalent, so the JAX evaluator must be kept."""
    import discopt.modeling as dm

    m = Model()
    x = m.continuous("x", lb=0.2, ub=2.0)

    @dm.custom
    def opaque(a):
        return a * 2.0

    m.minimize(opaque(x) + x)
    assert try_build(m) is None

    sentinel = object()
    assert build_evaluator(m, lambda: sentinel) is sentinel


@pytest.mark.unit
def test_backend_is_on_by_default_with_a_working_opt_out(monkeypatch):
    """Default ON since the §5 panel; ``DISCOPT_NLP_EVAL=jax`` still opts out.

    Was ``test_backend_is_off_by_default``. The premise changed deliberately —
    the panel passed both bars (cert-clean on every check; wall 10 faster / 0
    slower, median 1.80×, with node counts 44-of-46 identical) — so §5's
    graduation rule flips the default and keeps the opt-out and legacy path
    intact. This asserts BOTH halves; a default flip that quietly broke the
    escape hatch would make the legacy path unreachable.
    """
    m, _x, _y = _xy_model()
    sentinel = object()

    monkeypatch.delenv("DISCOPT_NLP_EVAL", raising=False)
    assert tape_backend_requested() is True
    assert isinstance(build_evaluator(m, lambda: sentinel), TapeNLPEvaluator)

    monkeypatch.setenv("DISCOPT_NLP_EVAL", "jax")
    assert tape_backend_requested() is False
    assert build_evaluator(m, lambda: sentinel) is sentinel, "opt-out must reach the JAX path"

    # An unrecognised value must not silently disable the graduated default.
    monkeypatch.setenv("DISCOPT_NLP_EVAL", "tape")
    assert tape_backend_requested() is True
    assert isinstance(build_evaluator(m, lambda: sentinel), TapeNLPEvaluator)


@pytest.mark.unit
def test_full_derivative_set_never_imports_jax():
    """The point of Stage 3: derivatives without JAX.

    Runs in a SUBPROCESS so ``sys.modules`` starts clean — this test file itself
    imports JAX to build the comparison, so an in-process check would be
    meaningless. Asserts on ``sys.modules`` and never on a source grep:
    ``dag_compiler.py:225`` uses ``__import__("jax")``, which no ``import jax``
    grep matches.
    """
    import subprocess
    import sys

    script = """
import sys
import numpy as np
from discopt import Model
from discopt._tape_nlp_evaluator import TapeNLPEvaluator

m = Model()
x = m.continuous("x", lb=0.2, ub=3.0)
y = m.continuous("y", lb=0.2, ub=3.0)
m.subject_to(x * y >= 0.5)
m.minimize(x * y + x * x)

ev = TapeNLPEvaluator(m)
pt = np.array([0.7, 1.3])
ev.evaluate_objective(pt)
ev.evaluate_gradient(pt)
ev.evaluate_constraints(pt)
ev.evaluate_jacobian(pt)
ev.evaluate_jacobian_values(pt)
ev.evaluate_hessian(pt)
ev.evaluate_hessian_values(pt, 1.0, np.array([1.0]))
ev.evaluate_lagrangian_hessian(pt, 1.0, np.array([1.0]))
ev.jacobian_structure()
ev.hessian_structure()

assert "jax" not in sys.modules, sorted(k for k in sys.modules if k.startswith("jax"))
print("JAXFREE-OK")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr[-2000:]}"
    assert "JAXFREE-OK" in out.stdout


@pytest.mark.unit
def test_row_map_and_bounds_agree_with_the_jax_evaluator():
    """A desynchronised row map makes every feasibility check read the wrong row (#908)."""
    from discopt.modeling.core import from_nl

    path = DATA / "alan.nl"
    if not path.exists():
        pytest.skip("alan.nl not in the in-repo corpus")
    model = from_nl(str(path))
    jax_ev = _jax_evaluator(model)
    tape = TapeNLPEvaluator(model)

    jmap = jax_ev.constraint_row_map()
    tmap = tape.constraint_row_map()
    assert len(jmap) == len(tmap)
    for (js, je, jc), (ts, te, tc) in zip(jmap, tmap):
        assert (js, je) == (ts, te)
        assert jc is tc

    jlb, jub = jax_ev.variable_bounds
    tlb, tub = tape.variable_bounds
    np.testing.assert_allclose(jlb, tlb)
    np.testing.assert_allclose(jub, tub)
