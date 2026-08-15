"""Adversarial regression suite for the soundness / crash / deadline fixes merged
in the weeks of 2026-06-18..24 and 2026-08-08..15.

Three layers:

1. **Real triggering instances** (``_INSTANCES``) — the exact MINLPLib models that
   each bug was found on, so they provably exercise the fixed code path, checked
   against their BARON-confirmed optima (minlplib.solu). These are the gold
   standard: before the fix each returned an *unsound* result (false-feasible,
   false-infeasible, false-unbounded, or false-optimal); the suite asserts the
   sound outcome.

2. **Synthetic path-targeted problems** — constructed to hit a specific fixed code
   path that no small vendored instance covers (the OA maximize loop; the dense
   Jacobian XLA-compile guard on a > 1e6-entry model).

3. **Oracle-driven sweeps** (2026-08-08..15) — for the LP churn and the new
   derivative-free selectors there is no single triggering instance to vendor, so
   these tests carry their own oracle: an independent LP implementation
   (scipy/HiGHS) plus self-contained feasibility arithmetic, and, for the
   randomized MINLPs, box sampling. Every one of them counts its executed
   assertions and fails if that count is zero — a sweep that silently checks
   nothing reads exactly like a passing sweep (CLAUDE.md §6).

Soundness invariants (sense-aware), asserted everywhere:
  * not false-infeasible / not false-unbounded
  * dual bound on the correct side of the true optimum (a valid bound never
    crosses it)            -> catches false certificates (#277, #306)
  * incumbent never beats the true optimum   -> catches false-feasible (#310)
  * a gap=0 "optimal" sits at the true optimum -> catches false-optimal (#301)
  * the process survives (no native crash)     -> #313

Marked ``slow``; run with ``-m slow``.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import ObjectiveSense  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")

# (instance, true optimum, sense, time_limit, fix, what the bug did)
_INSTANCES = [
    (
        "ex1252a",
        128893.741,
        "min",
        40,
        "#310",
        "false-feasible 92117 (obj-only ints typed continuous)",
    ),
    (
        "hda",
        -5964.534084,
        "min",
        40,
        "#307",
        "false-infeasible (inverted fractional-power aux box)",
    ),
    (
        "ex8_5_4",
        -0.0004251471,
        "min",
        25,
        "#270",
        "false-infeasible (free var into undefined log domain)",
    ),
    (
        "carton7",
        191.7295481,
        "min",
        40,
        "#288/#289",
        "false-unbounded (dropped nonlinear bound in projection)",
    ),
    (
        "st_ph10",
        -10.5,
        "min",
        25,
        "#306",
        "false-optimal -28.06 (incumbent below its own dual bound)",
    ),
    (
        "nvs22",
        6.05822,
        "min",
        25,
        "#277",
        "false certificate (ill-conditioned OBBT pruned the optimum)",
    ),
    ("nvs12", -481.2, "min", 40, "#293", "unbounded hang (simplex MILP engine ignored time_limit)"),
]

_REL = 5e-3  # relative tolerance band for "beats the optimum" / "is the optimum"
_BND = 1e-2  # absolute slack for dual-bound-side checks (numerical)


def _band(opt: float) -> float:
    return max(_REL * abs(opt), 1e-4)


def _assert_sound(name, r, *, sense, opt, may_lack_incumbent=True):
    """Sense-aware soundness assertions against the oracle optimum ``opt``."""
    obj, bnd, status = r.objective, r.bound, r.status
    tol = _band(opt)

    assert status != "infeasible", f"{name}: FALSE-INFEASIBLE (instance is feasible)"
    assert status != "unbounded", f"{name}: FALSE-UNBOUNDED (instance is bounded)"

    if not may_lack_incumbent:
        assert obj is not None, f"{name}: no incumbent returned"

    if obj is not None:
        # No false-feasible: the incumbent must be a real feasible point, never
        # strictly better than the proven global optimum.
        if sense == "min":
            assert obj >= opt - tol, f"{name}: FALSE-FEASIBLE — {obj:.6g} < opt {opt:.6g}"
        else:
            assert obj <= opt + tol, f"{name}: FALSE-FEASIBLE — {obj:.6g} > opt {opt:.6g}"
        # No false-optimal: a certified optimum must sit at the true optimum.
        if status == "optimal":
            assert abs(obj - opt) <= tol, f"{name}: FALSE-OPTIMAL — {obj:.6g} != opt {opt:.6g}"

    # Sound dual bound: a valid lower (min) / upper (max) bound never crosses the
    # true optimum, and never crosses the incumbent.
    if bnd is not None:
        if sense == "min":
            assert bnd <= opt + tol + _BND, f"{name}: INVALID BOUND {bnd:.6g} > opt {opt:.6g}"
            if obj is not None:
                assert bnd <= obj + tol + _BND, (
                    f"{name}: UNSOUND CERT bound {bnd:.6g} > inc {obj:.6g}"
                )
        else:
            assert bnd >= opt - tol - _BND, f"{name}: INVALID BOUND {bnd:.6g} < opt {opt:.6g}"
            if obj is not None:
                assert bnd >= obj - tol - _BND, (
                    f"{name}: UNSOUND CERT bound {bnd:.6g} < inc {obj:.6g}"
                )


@pytest.mark.slow
@pytest.mark.parametrize("name,opt,sense,tl,fix,bug", _INSTANCES, ids=[i[0] for i in _INSTANCES])
def test_triggering_instance_is_sound(name, opt, sense, tl, fix, bug):
    """Each instance that triggered a recent bug must now return a sound result
    within the time limit (and the process must not crash)."""
    path = os.path.join(_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")
    t = time.perf_counter()
    r = dm.from_nl(path).solve(time_limit=tl, gap_tolerance=1e-4)
    wall = time.perf_counter() - t
    # Deadline honored (generous margin for one uninterruptible compile/solve that
    # straddles the deadline — the known diffuse residual, not a hang). #293/#311/#314.
    assert wall < tl + 60, f"{name}: ran {wall:.0f}s on a {tl}s limit (hang?) — {fix}"
    _assert_sound(name, r, sense=sense, opt=opt)


@pytest.mark.slow
def test_oa_maximize_is_sound():
    """OA on a MAXIMIZE convex MINLP must return the true maximum, not a negated /
    wrong point reported as 'optimal' (#301). Real trigger: syn05m (=max= 837.7324)."""
    from discopt.solvers.oa import solve_oa

    path = os.path.join(_DATA, "syn05m.nl")
    if not os.path.exists(path):
        pytest.skip("syn05m.nl not vendored")
    m = dm.from_nl(path)
    assert m._objective.sense == ObjectiveSense.MAXIMIZE
    r = solve_oa(m, time_limit=40, gap_tolerance=1e-4)
    _assert_sound("syn05m/OA", r, sense="max", opt=837.7324009)


@pytest.mark.slow
def test_oa_maximize_synthetic_concave():
    """Synthetic convex-MINLP MAXIMIZE that directly drives the OA loop (#301):
    max -(x-3)**2 - (y-2.5)**2, x in [0,5], y in {0..5}, x+y<=10  ->  -0.25 at y in {2,3}."""
    from discopt.solvers.oa import solve_oa

    m = dm.Model("oa_max")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(-((x - 3.0) ** 2) - (y - 2.5) ** 2)
    m.subject_to(x + y <= 10.0)
    r = solve_oa(m, time_limit=30, gap_tolerance=1e-5)
    _assert_sound("oa-synthetic", r, sense="max", opt=-0.25, may_lack_incumbent=False)


@pytest.mark.slow
def test_large_dense_jacobian_no_crash():
    """A sparse MINLP whose dense Jacobian (n_vars * n_constraints ~ 1.21e6) exceeds
    the 1e6 cap that crashed XLA's dense jacfwd compile (#313), with ~110 binaries
    (probing #313) and factorable equalities (#314). Must survive (no SIGBUS/SIGILL),
    honor the deadline within a margin, and return a sound certificate."""
    n = 1100
    m = dm.Model("big")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    bs = [m.binary(f"b{i}") for i in range(0, n, 10)]
    for i in range(n):
        m.subject_to(xs[i] * xs[i] + xs[(i + 1) % n] <= 10.0)
    for k, b in enumerate(bs):
        m.subject_to(xs[k] + 2.0 * b <= 6.0)
    m.minimize(dm.sum([xs[i] for i in range(n)]) - dm.sum(bs))
    budget = 8.0
    t = time.perf_counter()
    r = m.solve(time_limit=budget, gap_tolerance=1e-4)  # must not crash the process
    wall = time.perf_counter() - t
    assert wall < budget + 40, f"large model: deadline overrun wall={wall:.0f}s"
    assert r.status != "infeasible", "large model: FALSE-INFEASIBLE"
    # Sound certificate: a finite lower bound never exceeds the incumbent.
    if r.objective is not None and r.bound is not None:
        assert r.bound <= r.objective + 1e-3, "large model: UNSOUND CERT (bound > incumbent)"


# ---------------------------------------------------------------------------
# Week of 2026-08-08..15
# ---------------------------------------------------------------------------

_BIG = 1e20  # the LP layer's INF sentinel is 1e20, not f64::INFINITY


def _make_lp(rng, kind):
    """One adversarial LP shape: ``(c, A, b, lb, ub)`` for ``min c'x, Ax=b, lb<=x<=ub``.

    The shapes target what the LP layer took this week: degenerate vertices
    (#1023's unstable-pivot recovery), rank-deficient rows (#1025's threshold
    Markowitz), wild column scaling, a Farkas-certified empty system (#1019's
    margin) and an explicit primal ray (#1022's ray certification before
    claiming Unbounded).
    """
    import numpy as np

    m = int(rng.integers(2, 7))
    n = int(rng.integers(m + 1, m + 8))
    A = rng.normal(size=(m, n))
    c = rng.normal(size=n)
    lb = np.zeros(n)
    ub = np.full(n, 10.0)

    if kind == "degenerate":
        A = np.round(A)
        A[A == 0] = 1.0
        x0 = np.zeros(n)
        x0[: max(1, n // 3)] = 1.0
        b = A @ x0
    elif kind == "rank_deficient":
        A[-1] = A[0] + A[1] if m >= 2 else A[0]
        b = A @ rng.uniform(0, 3, size=n)
    elif kind == "badly_scaled":
        scale = 10.0 ** rng.integers(-6, 7, size=n)
        A, c, ub = A * scale, c * scale, ub / scale
        b = A @ (rng.uniform(0, 1, size=n) * ub)
    elif kind == "infeasible":
        # y'A = 0 with y'b > 0 is a Farkas certificate: no solution exists, box
        # or no box. The expected verdict is not an opinion.
        y = rng.normal(size=m)
        A = A - np.outer(y, y @ A) / (y @ y)
        b = y * 5.0
    elif kind == "unbounded":
        # A free column with strictly improving cost and no row activity: a ray.
        A = np.hstack([A, np.zeros((m, 1))])
        c = np.append(c, -1.0)
        lb = np.append(lb, 0.0)
        ub = np.append(ub, _BIG)
        b = A[:, :n] @ rng.uniform(0, 3, size=n)
    else:  # "generic"
        b = A @ rng.uniform(0, 5, size=n)

    return c, np.ascontiguousarray(A), b, lb, ub


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind",
    ["generic", "degenerate", "rank_deficient", "badly_scaled", "infeasible", "unbounded"],
)
def test_lp_layer_differential_vs_highs(kind):
    """Every per-node dual bound comes off the Rust simplex, so a wrong verdict
    there is a false certificate at the top. Eight of the 2026-08-08..15 PRs land
    in ``crates/discopt-core/src/lp/`` (#996, #1012, #1018, #1019, #1021, #1022,
    #1023, #1024, #1025), and no vendored instance isolates them.

    Two independent oracles per LP:

    * self-contained — a reported ``optimal`` point must satisfy ``Ax = b`` and
      the box, and ``c'x`` must equal the reported objective. No second solver is
      involved, so a violation is a defect outright.
    * differential — statuses must agree with scipy's HiGHS. The single genuine
      ambiguity is an empty feasible set with an unbounded recession cone, which
      either solver may legitimately report either way; everything else is one of
      the two being wrong.
    """
    import discopt._rust as R
    import numpy as np
    from scipy.optimize import linprog

    rng = np.random.default_rng(20260815)
    checks = 0
    for i in range(20):
        c, A, b, lb, ub = _make_lp(rng, kind)
        st, x, obj, _iters = R.solve_lp_py(c, A, b, lb, ub, 1e-9, 20000)

        if st == "optimal":
            x = np.asarray(x, dtype=float)
            checks += 3
            resid = float(np.abs(A @ x - b).max())
            assert resid <= 1e-6 * max(1.0, float(np.abs(b).max())), (
                f"{kind} lp {i}: 'optimal' point violates Ax=b by {resid:.3e}"
            )
            box = float(max((lb - x).max(), (x - ub).max()))
            assert box <= 1e-6, f"{kind} lp {i}: 'optimal' point is outside its box by {box:.3e}"
            cx = float(c @ x)
            assert abs(cx - obj) <= 1e-6 * max(1.0, abs(cx)), (
                f"{kind} lp {i}: reported objective {obj:.10g} != c'x {cx:.10g}"
            )

        bounds = [(float(lo), None if hi >= _BIG else float(hi)) for lo, hi in zip(lb, ub)]
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
        sst = {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(res.status)
        if sst is None or st not in ("optimal", "infeasible", "unbounded"):
            continue

        checks += 1
        if {st, sst} != {"infeasible", "unbounded"}:
            assert st == sst, f"{kind} lp {i}: discopt says {st}, HiGHS says {sst}"
            if st == "optimal":
                checks += 1
                assert abs(obj - float(res.fun)) <= 1e-6 * max(1.0, abs(res.fun)), (
                    f"{kind} lp {i}: discopt {obj:.10g} vs HiGHS {float(res.fun):.10g}"
                )

    # A shape whose LPs all landed on a status the comparison skips would make
    # this test a silent no-op (CLAUDE.md §6).
    assert checks > 0, f"{kind}: no assertion executed — the sweep checked nothing"


@pytest.mark.slow
@pytest.mark.parametrize("selector", ["direct", "surrogate"])
def test_dfo_never_fabricates_a_certificate(selector):
    """The derivative-free selectors added in #1006 have no dual argument —
    DIRECT is a sampling search, the surrogate path optimizes a *fitted* model —
    so whatever they return must not read as a proof. A finite ``bound``, a
    ``gap_certified=True`` or a ``status="optimal"`` off either path is a false
    certificate even when the point it names happens to be the true optimum,
    because downstream gates count the *claim*.

    The returned point must still be honest: inside the declared box, satisfying
    the declared rows, with the reported objective equal to the objective there.
    """
    import numpy as np
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    m = dm.Model(f"dfo_{selector}")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=3.0)
    m.minimize(100 * (y - x * x) ** 2 + (1 - x) ** 2)
    m.subject_to(x + y <= 2.5, name="r0")

    r = m.solve(solver=selector, time_limit=15)

    assert r.bound is None or not np.isfinite(r.bound), (
        f"{selector}: reported dual bound {r.bound!r} with no dual argument behind it"
    )
    assert not r.gap_certified, f"{selector}: gap_certified=True off a derivative-free method"
    assert r.status != "optimal", f"{selector}: status='optimal' with no certificate behind it"

    if r.x is None:
        return
    xs = np.asarray([float(np.ravel(np.asarray(r.x[n]))[0]) for n in ("x", "y")])
    lo = np.asarray([-2.0, -1.0])
    hi = np.asarray([2.0, 3.0])
    out = float(max((lo - xs).max(), (xs - hi).max()))
    assert out <= 1e-6, f"{selector}: returned point is outside the declared box by {out:.3e}"

    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    con = np.asarray(ev.evaluate_constraints(xs), dtype=float)
    viol = float(max(np.maximum(cl - con, 0.0).max(), np.maximum(con - cu, 0.0).max()))
    assert viol <= 1e-5, f"{selector}: returned point violates a declared row by {viol:.3e}"

    if r.objective is not None:
        # NLPEvaluator reports the internal *minimization* objective; this model
        # minimizes, so no sign flip is needed — assert that rather than assume it.
        assert not ev._negate
        here = float(ev.evaluate_objective(xs))
        assert abs(here - r.objective) <= 1e-5 * max(1.0, abs(here)), (
            f"{selector}: reported {r.objective:.10g}, objective at the point is {here:.10g}"
        )


def _fuzz_model(rng, idx):
    """A small random MINLP mixing the operators and shapes the week touched."""

    m = dm.Model(f"fuzz{idx}")
    scalars, meta = [], []
    for i in range(rng.integers(1, 4)):
        lo = round(float(rng.uniform(-5, 0)), 3)
        hi = round(lo + float(rng.uniform(0.5, 8)), 3)
        scalars.append(m.continuous(f"c{i}", lb=lo, ub=hi))
        meta.append((f"c{i}", lo, hi, False))
    for i in range(rng.integers(0, 2)):
        lo = int(rng.integers(-3, 0))
        hi = lo + int(rng.integers(1, 5))
        scalars.append(m.integer(f"i{i}", lb=lo, ub=hi))
        meta.append((f"i{i}", float(lo), float(hi), True))

    def term():
        kind = ["lin", "bilin", "sq", "exp", "log", "sqrt", "div"][int(rng.integers(0, 7))]
        a = scalars[int(rng.integers(0, len(scalars)))]
        b = scalars[int(rng.integers(0, len(scalars)))]
        k = round(float(rng.uniform(-3, 3)), 3)
        if kind == "lin":
            return k * a
        if kind == "bilin":
            return k * a * b
        if kind == "sq":
            return k * a * a
        if kind == "exp":
            return k * dm.exp(0.3 * a)
        if kind == "log":
            return k * dm.log(dm.exp(0.2 * a) + 1.5)  # argument provably positive
        if kind == "sqrt":
            return k * dm.sqrt(a * a + 1.0)
        return k * a / (a * a + 2.0)  # denominator provably positive

    obj = dm.sum([term() for _ in range(int(rng.integers(1, 4)))])
    (m.minimize if idx % 2 else m.maximize)(obj)
    sense = "min" if m._objective.sense == ObjectiveSense.MINIMIZE else "max"

    for ci in range(int(rng.integers(1, 4))):
        body = dm.sum([term() for _ in range(int(rng.integers(1, 3)))])
        rhs = round(float(rng.uniform(-6, 6)), 3)
        # Inequalities only. An equality can carve out a measure-zero feasible
        # set, which no sampled point can ever land on — the sampling arm below
        # would then be vacuous while still reporting no counterexample.
        if rng.random() < 0.5:
            m.subject_to(body <= rhs, name=f"r{ci}")
        else:
            m.subject_to(body >= rhs, name=f"r{ci}")
    return m, meta, sense


@pytest.mark.slow
def test_randomized_minlp_certificate_invariants():
    """Randomized (seeded) MINLPs, checked against invariants that hold for every
    model regardless of what the answer is.

    This is the broad net under the week's certificate work — the graduated flags
    (#1002, #996), the reformulation and relaxation changes (#982, #983, #984,
    #988, #1007, #1014, #1015) and the feral bump (#1025) all move numbers on
    arbitrary models, and no fixed instance list covers the shapes they touch.

    Per model: a reported incumbent must be inside its box, integral where the
    model says integral, feasible when re-evaluated from the DAG, and equal to
    the reported objective; the dual bound must not cross it; ``gap_certified``
    must carry a finite bound unless the status is ``infeasible`` (where the
    certificate is of the infeasibility, not of a gap — ``core.py``'s
    ``_NON_GAP_CERTIFICATE_STATUSES``); and box sampling must never turn up a
    point that refutes an ``infeasible`` verdict or beats a certified optimum.
    """
    import numpy as np
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    rng = np.random.default_rng(20260815)
    checks = 0
    sampled_feasible = 0  # proves the sampling arm was not vacuous

    for idx in range(10):
        model, meta, sense = _fuzz_model(rng, idx)
        r = model.solve(time_limit=5, gap_tolerance=1e-4)

        ev = NLPEvaluator(model)
        cl, cu = _infer_constraint_bounds(ev)
        cl = np.asarray(cl, dtype=float)
        cu = np.asarray(cu, dtype=float)
        sign = -1.0 if ev._negate else 1.0

        def value(xs, _ev=ev, _sign=sign, _cl=cl, _cu=cu):
            con = np.asarray(_ev.evaluate_constraints(xs), dtype=float)
            viol = (
                0.0
                if con.size == 0
                else float(max(np.maximum(_cl - con, 0.0).max(), np.maximum(con - _cu, 0.0).max()))
            )
            return viol, _sign * float(_ev.evaluate_objective(xs))

        checks += 1
        assert not (
            r.gap_certified
            and r.status != "infeasible"
            and (r.bound is None or not np.isfinite(r.bound))
        ), f"model {idx}: gap_certified with bound={r.bound}"

        if r.objective is not None and r.x is not None:
            xs = np.asarray([float(np.ravel(np.asarray(r.x[n]))[0]) for n, _, _, _ in meta])
            checks += 4
            lo = np.asarray([b for _, b, _, _ in meta])
            hi = np.asarray([b for _, _, b, _ in meta])
            box = float(max((lo - xs).max(), (xs - hi).max()))
            assert box <= 1e-6, f"model {idx}: incumbent outside its box by {box:.3e}"
            for (nm, _, _, is_int), v in zip(meta, xs):
                assert not is_int or abs(v - round(v)) <= 1e-5, (
                    f"model {idx}: integer {nm} took the value {v!r}"
                )
            viol, here = value(xs)
            assert viol <= 1e-5, f"model {idx}: incumbent violates a row by {viol:.3e}"
            assert abs(here - r.objective) <= 1e-5 * max(1.0, abs(here)), (
                f"model {idx}: reported {r.objective:.10g}, re-evaluated {here:.10g}"
            )
            if r.bound is not None and np.isfinite(r.bound):
                checks += 1
                if sense == "min":
                    assert r.bound <= r.objective + 1e-4, (
                        f"model {idx}: bound {r.bound:.10g} above incumbent {r.objective:.10g}"
                    )
                else:
                    assert r.bound >= r.objective - 1e-4, (
                        f"model {idx}: bound {r.bound:.10g} below incumbent {r.objective:.10g}"
                    )

        # Sampling refutation: the box is small and the constraints are cheap, so
        # a feasible sample is a witness the solver has to be consistent with.
        lo = np.asarray([b for _, b, _, _ in meta])
        hi = np.asarray([b for _, _, b, _ in meta])
        is_int = np.asarray([b for _, _, _, b in meta])
        pts = rng.uniform(lo, hi, size=(500, len(meta)))
        pts[:, is_int] = np.round(pts[:, is_int])
        best = None
        for p in pts:
            viol, val = value(p)
            if viol <= 1e-8 and (best is None or (val < best) == (sense == "min")):
                best = val
        if best is None:
            continue
        sampled_feasible += 1
        checks += 1
        assert r.status != "infeasible", (
            f"model {idx}: FALSE-INFEASIBLE — sampling found a feasible point"
        )
        if r.status == "optimal" and r.objective is not None:
            checks += 1
            slack = 1e-4 * max(1.0, abs(r.objective))
            if sense == "min":
                assert best >= r.objective - slack, (
                    f"model {idx}: FALSE-OPTIMAL — sample {best:.10g} beats certified "
                    f"{r.objective:.10g}"
                )
            else:
                assert best <= r.objective + slack, (
                    f"model {idx}: FALSE-OPTIMAL — sample {best:.10g} beats certified "
                    f"{r.objective:.10g}"
                )

    assert checks > 0, "no assertion executed — the sweep checked nothing"
    assert sampled_feasible > 0, (
        "no model produced a feasible sample — the refutation arm never fired and "
        "its silence means nothing"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "slow", "-s"]))
