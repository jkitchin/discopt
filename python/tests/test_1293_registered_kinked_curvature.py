"""A registered function with a kink must not get a smooth curvature verdict (#1293).

``RegisteredFunction._derived`` takes f'' symbolically, and ``symbolic_diff``
differentiates ``abs``/``sign`` by subgradient, so f'' of ``-abs(t)`` is
identically 0 and the verdict read "convex" for a concave function. The envelope
then excluded the graph: the root relaxation of a feasible model was LP-infeasible,
and separable concave minimizations were certified above their optimum.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.uniform_relax import build_uniform_relaxation
from scipy.optimize import linprog

negabs = dm.register_function("t1293_negabs", lambda x: -dm.abs(x))
tent = dm.register_function("t1293_tent", lambda x: 1 - dm.abs(x - 0.3))
step = dm.register_function("t1293_sign", lambda x: dm.sign(x))
root = dm.register_function("t1293_pow", lambda x: x**2.5)


def _verdict_holds(verdict, fn, lo, hi):
    """Midpoint-convexity on a grid of pairs: a definitional check, no derivatives."""
    ts = np.linspace(lo, hi, 201)
    a, b = np.meshgrid(ts, ts)
    mid = fn(0.5 * (a + b))
    avg = 0.5 * (fn(a) + fn(b))
    slack = 1e-9 * (1 + np.abs(avg))
    if verdict == "convex":
        return bool(np.all(mid <= avg + slack))
    return bool(np.all(mid >= avg - slack))


@pytest.mark.parametrize(
    "reg, lo, hi",
    [
        (negabs, -1.0, 2.0),
        (negabs, -1e-3, 1e-3),
        (negabs, 0.0, 1.0),
        (tent, -2.0, 2.0),
        (step, -1.0, 1.0),
        (root, -1.0, 1.0),
    ],
)
def test_box_containing_the_kink_abstains(reg, lo, hi):
    assert reg._derived()[2](lo, hi) is None


@pytest.mark.parametrize(
    "reg, fn, lo, hi",
    [
        (negabs, lambda t: -np.abs(t), 0.5, 2.0),
        (negabs, lambda t: -np.abs(t), -2.0, -0.5),
        (tent, lambda t: 1 - np.abs(t - 0.3), 0.4, 2.0),
        (root, lambda t: t**2.5, 0.5, 2.0),
    ],
)
def test_box_away_from_the_kink_keeps_a_true_verdict(reg, fn, lo, hi):
    verdict = reg._derived()[2](lo, hi)
    assert verdict is not None
    assert _verdict_holds(verdict, fn, lo, hi)


def test_root_relaxation_of_a_feasible_model_is_feasible():
    m = dm.Model("t1293_root")
    x = m.continuous("x", lb=-1, ub=2)
    y = m.continuous("y", lb=-1, ub=2)
    m.minimize(negabs(x) + negabs(y) + 0.3 * x * y)
    uses = negabs.use_count
    M = build_uniform_relaxation(m).model
    A = M._A_ub.toarray() if hasattr(M._A_ub, "toarray") else np.asarray(M._A_ub)
    lp = linprog(M._c, A_ub=A, b_ub=M._b_ub, bounds=M._bounds, method="highs")
    assert lp.status == 0, lp.message
    # optimum -3.6 at (2, -1)
    assert lp.fun + M._obj_offset <= -3.6 + 1e-6
    # the registered envelope entry was consulted, so this exercised its verdict
    assert negabs.use_count > uses


def _instances(n_inst, nv, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_inst):
        L = rng.uniform(-3, 0, nv)
        U = L + rng.uniform(1, 5, nv)
        a = rng.uniform(L, U)
        b = rng.uniform(-0.9, 0.9, nv)
        w = rng.uniform(0.5, 2.0, nv)
        K = float(w @ (L + rng.uniform(0.3, 0.7) * (U - L)))
        yield L, U, a, b, w, K


def _vertex_optimum(L, U, a, b, w, K):
    """Concave objective: its minimum over the polytope is at a vertex."""
    nv = len(L)

    def f(v):
        return float((-np.abs(v - a) + b * v).sum())

    best = np.inf
    for bits in itertools.product((0, 1), repeat=nv):
        v = np.where(np.array(bits) == 1, U, L)
        if w @ v <= K + 1e-12:
            best = min(best, f(v))
    for j in range(nv):
        rest = [i for i in range(nv) if i != j]
        for bits in itertools.product((0, 1), repeat=nv - 1):
            v = np.zeros(nv)
            v[rest] = np.where(np.array(bits) == 1, U[rest], L[rest])
            v[j] = (K - w[rest] @ v[rest]) / w[j]
            if L[j] <= v[j] <= U[j]:
                best = min(best, f(v))
    return best


@pytest.mark.parametrize("k", [4, 7])
def test_separable_concave_knapsack_is_certified_at_the_optimum(k):
    L, U, a, b, w, K = list(_instances(k + 1, 6, 3))[k]
    opt = _vertex_optimum(L, U, a, b, w, K)
    m = dm.Model(f"t1293_sep{k}")
    xs = [m.continuous(f"x{i}", lb=L[i], ub=U[i]) for i in range(6)]
    m.subject_to(sum(float(w[i]) * xs[i] for i in range(6)) <= K)
    m.minimize(sum(negabs(xs[i] - float(a[i])) + float(b[i]) * xs[i] for i in range(6)))
    r = m.solve(time_limit=60)
    tol = 1e-4 * (1 + abs(opt))
    assert r.status in ("optimal", "feasible", "time_limit"), r.status
    if r.bound is not None:
        assert r.bound <= opt + tol, (r.bound, opt)
    if r.status == "optimal":
        assert r.objective == pytest.approx(opt, abs=tol)
