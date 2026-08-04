"""``TapeNLPEvaluator`` must reproduce the JAX ``NLPEvaluator`` it replaces.

Issue #75, Stage 3. ``_jax/nlp_evaluator.py`` is the single trigger that imports
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

from pathlib import Path

import numpy as np
import pytest
from discopt import Model
from discopt._nl_expr_compiler import UnsupportedForTape
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
    from discopt._jax.nlp_evaluator import NLPEvaluator

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
    precisely when a fallback rots: it is retained for users on
    ``pounce-solver>=0.9`` builds predating the fix, where using it wrongly costs
    a false ``infeasible``. Forcing the capability probe to ``False`` keeps it
    covered, and asserts the two paths agree rather than merely both running.
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
def test_gauss_newton_is_refused_not_approximated():
    """``2 J^T J`` is a different matrix from the exact Hessian; never silently swap."""
    m, _x, _y = _xy_model()
    m._gauss_newton_hessian = True
    with pytest.raises(UnsupportedForTape, match="Gauss-Newton"):
        TapeNLPEvaluator(m)
    assert try_build(m) is None


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
