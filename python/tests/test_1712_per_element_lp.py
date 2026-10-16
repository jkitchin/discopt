"""#1712: a per-element LP must not pay NLP-sized preprocessing before HiGHS.

Three changes, each checked here for *exact* neutrality rather than speed alone:

* the interval evaluator encloses ``dm.sum(c * x[i] for i ...)`` in one vectorised
  pass -- bit-identical to the generic per-node walk it replaces;
* the convex kernel refuses a continuous or exactly-linear model before building
  an NLP tape (it refused those same models after building one);
* ``solve_model`` does not classify an LP's convexity (the LP route never read it).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax.convexity import interval_eval
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval

# Values chosen to hit every branch of ``Interval.__mul__``/``__add__``: signed
# zeros (#957 exact0), products that underflow to zero, infinities (``0 * inf``
# NaN cleanup), cancellation, and ordinary magnitudes.
_EDGE = [0.0, -0.0, 1e-200, -1e-200, 1e-170, 3.0, -2.5, 1e16, -1e16, 7.25e-3, np.inf, -np.inf]


def _generic(expr, model, box, monkeypatch):
    with monkeypatch.context() as mp:
        mp.setattr(interval_eval, "_scalar_affine_sum", lambda terms, box: None)
        return evaluate_interval(expr, model, box)


def _same(a: Interval, b: Interval) -> None:
    # Bitwise: ``array_equal`` would let -0.0 == 0.0 and NaN != NaN through.
    for x, y in ((a.lo, b.lo), (a.hi, b.hi)):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        assert x.shape == y.shape == ()
        assert x.tobytes() == y.tobytes(), (float(x), float(y))


def _model(n, rng, *, point: bool):
    m = Model("affine")
    x = m.continuous("x", shape=(n,), lb=-1e3, ub=1e3)
    s = m.continuous("s", lb=-4.0, ub=9.0)
    terms = []
    for i in range(n):
        kind = i % 5
        c = float(rng.choice(_EDGE[:10])) if rng.random() < 0.5 else float(rng.normal() * 10)
        if kind == 0:
            terms.append(c * x[i])
        elif kind == 1:
            terms.append(x[i] * c)
        elif kind == 2:
            terms.append(x[i])
        elif kind == 3:
            terms.append(c * s)
        else:
            terms.append(m.parameter(f"p{i}", c) * x[i])
    expr = dm.sum(t for t in terms)
    box = {}
    if point:
        xv = np.array([rng.choice(_EDGE) if rng.random() < 0.4 else rng.normal() for _ in range(n)])
        box[x] = Interval.point(xv)
        box[s] = Interval.point(np.float64(rng.choice(_EDGE)))
    return m, expr, box


@pytest.mark.smoke
@pytest.mark.parametrize("point", [True, False])
def test_interval_eval_affine_sum_fast_path(point, monkeypatch):
    """The vectorised enclosure equals the generic walk bit for bit."""
    calls = {"fast": 0}
    real = interval_eval._scalar_affine_sum

    def counting(terms, box):
        r = real(terms, box)
        if r is not None:
            calls["fast"] += 1
        return r

    monkeypatch.setattr(interval_eval, "_scalar_affine_sum", counting)
    rng = np.random.default_rng(1712)
    compared = 0
    for trial in range(150):
        m, expr, box = _model(int(rng.integers(1, 40)), rng, point=point)
        fast = evaluate_interval(expr, m, box)
        slow = _generic(expr, m, box, monkeypatch)
        _same(fast, slow)
        compared += 1
    # Prove the probe fired: every trial took the fast path and was compared.
    assert compared == 150
    assert calls["fast"] == 150


def test_interval_eval_affine_sum_declines_other_terms(monkeypatch):
    """A term outside ``leaf`` / ``coef * leaf`` falls back to the generic walk."""
    m = Model("nl")
    x = m.continuous("x", shape=(3,), lb=0.5, ub=2.0)
    expr = dm.sum(t for t in [2.0 * x[0], x[1] * x[2], x[0]])
    assert interval_eval._scalar_affine_sum(expr.terms, {}) is None
    _same(evaluate_interval(expr, m), _generic(expr, m, {}, monkeypatch))


def test_tape_affine_sum_fast_path_matches_generic(monkeypatch):
    """The bulk tape lowering builds the same tape as the generic traversal."""
    from discopt import _nl_expr_compiler as nlc

    calls = {"fast": 0}
    real_lower = nlc._lower_scalar_affine_sum

    def counting(*a, **k):
        calls["fast"] += 1
        return real_lower(*a, **k)

    rng = np.random.default_rng(17)
    compared = 0
    for trial in range(60):
        n = int(rng.integers(2, 40))
        m, expr, _ = _model(n, rng, point=False)
        if trial % 2:
            # The ``+``/``-`` chain shape (builtin ``sum``, builder-row bodies): the
            # same terms folded left with random signs and a constant leaf.
            chain = expr.terms[0]
            for t in expr.terms[1:]:
                chain = chain - t if rng.random() < 0.4 else chain + t
            expr = chain - 5.0 if n >= 3 else chain + chain + chain + chain
        # A sum that shares nodes with the rest of the DAG: the memo must be honoured.
        x0 = m._variables[0]
        outer = expr * expr + 3.0 * x0[0]
        with monkeypatch.context() as mp:
            mp.setattr(nlc, "_lower_scalar_affine_sum", counting)
            fast = nlc.compile_to_nl_expr(outer, m)
        with monkeypatch.context() as mp:
            mp.setattr(nlc, "_scalar_affine_plan", lambda *a: None)
            slow = nlc.compile_to_nl_expr(outer, m)
        assert repr(fast) == repr(slow)
        assert fast.depth == slow.depth
        n_flat = sum(v.size for v in m._variables)
        for _ in range(4):
            xv = rng.normal(size=n_flat) * float(rng.choice([1e-3, 1.0, 1e6]))
            assert np.float64(fast.eval(xv)).tobytes() == np.float64(slow.eval(xv)).tobytes()
            gf = np.asarray(fast.gradient(xv), dtype=np.float64)
            gs = np.asarray(slow.gradient(xv), dtype=np.float64)
            assert gf.tobytes() == gs.tobytes()
        compared += 1
    assert compared == 60 and calls["fast"] == 60


def _per_element_lp(n: int, *, integer: bool = False):
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(50, 100, n), rng.uniform(10, 40, n), rng.uniform(1, 10, (n, n))
    m = Model("transport")
    P = m.set("P", [f"p{k}" for k in range(n)])
    K = m.set("K", [f"m{k}" for k in range(n)])
    ip, ik = {p: i for i, p in enumerate(P)}, {k: j for j, k in enumerate(K)}
    if integer:
        x = m.integer("x", over=P * K, lb=0, ub=100)
    else:
        x = m.continuous("x", over=P * K, lb=0, ub=1e4)
    m.minimize(dm.sum(c[ip[p], ik[k]] * x[p, k] for p in P for k in K))
    m.constraint(P, lambda p: dm.sum(x[p, k] for k in K) <= a[ip[p]], name="sup")
    m.constraint(K, lambda k: dm.sum(x[p, k] for p in P) >= b[ik[k]], name="dem")
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("integer", [False, True])
def test_convex_kernel_refuses_linear_model_without_a_tape(integer, monkeypatch):
    """LP and linear MILP are refused before ``make_evaluator`` (they always were refused)."""
    import discopt._tape_nlp_evaluator as tne
    from discopt.solvers._convex_kernel import build_convex_spec

    built = []
    real = tne.make_evaluator
    monkeypatch.setattr(tne, "make_evaluator", lambda *a, **k: built.append(1) or real(*a, **k))
    assert build_convex_spec(_per_element_lp(4, integer=integer)) is None
    assert built == []


def test_convex_kernel_still_builds_for_nonlinear_minlp():
    """The early refusal is linear-only: a convex MINLP still gets a spec."""
    m = Model("cvx")
    x = m.continuous("x", lb=0, ub=4)
    y = m.binary("y")
    m.minimize(-x + y)
    m.subject_to(x**2 <= 4 + 5 * y)
    assert build_convex_spec_ok(m)


def build_convex_spec_ok(m) -> bool:
    from discopt.solvers._convex_kernel import build_convex_spec

    return build_convex_spec(m) is not None


@pytest.mark.smoke
def test_lp_solve_skips_convexity_classification(monkeypatch):
    """An LP solve never classifies convexity, and its certificate is unchanged."""
    import discopt.solver as S

    m = _per_element_lp(6)
    calls = []
    real = S._classify_model_convexity
    monkeypatch.setattr(
        S, "_classify_model_convexity", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    r = m.solve()
    assert calls == []
    assert r.status == "optimal" and r.gap_certified
    assert r.algorithm_route is not None and r.algorithm_route.startswith("highs-lp")
    # #1551 guard ran on the fast path and measured the error (not skipped).
    assert "certificate/objective_eval_error" in (r.solver_stats or {})


def _record_linear_skips(monkeypatch):
    """Record calls to the entropy rewrite and the rules handed to the domain pass."""
    import discopt._relax.factorable_reform as fr
    import discopt._relax.nonlinear_bound_tightening as nbt

    seen = {"entropy": 0, "rules": []}
    real_entropy, real_tighten = fr.canonicalize_entropy, nbt.tighten_nonlinear_bounds

    def entropy(model):
        seen["entropy"] += 1
        return real_entropy(model)

    def tighten(*a, **k):
        if "rules" in k:  # the solve's domain pass; the declared-box pass uses defaults
            seen["rules"].append(tuple(type(r).__name__ for r in k["rules"]))
        return real_tighten(*a, **k)

    monkeypatch.setattr(fr, "canonicalize_entropy", entropy)
    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", tighten)
    return seen


@pytest.mark.smoke
@pytest.mark.parametrize("integer", [False, True])
def test_linear_model_skips_entropy_and_domain_rules(integer, monkeypatch):
    """LP/MILP: no entropy rewrite, and the domain pass runs with no rules -- the
    pass itself (box checks, crossover snapping) still runs."""
    seen = _record_linear_skips(monkeypatch)
    r = _per_element_lp(5, integer=integer).solve()
    assert r.status == "optimal"
    assert seen["entropy"] == 0
    assert seen["rules"] == [()]


def test_nonlinear_model_keeps_entropy_and_domain_rules(monkeypatch):
    """The skips are linear-only: an entropy objective still gets both."""
    seen = _record_linear_skips(monkeypatch)
    m = Model("ent")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=2.0)
    m.minimize(dm.sum(x[i] * dm.log(x[i]) for i in range(3)))
    m.subject_to(dm.sum(x[i] for i in range(3)) >= 1.5)
    m.solve(time_limit=20)
    assert seen["entropy"] >= 1
    assert seen["rules"] and all(r for r in seen["rules"])

