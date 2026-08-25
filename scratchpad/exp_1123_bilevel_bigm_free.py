#!/usr/bin/env python -u
"""Entry experiment for issue #1123: does a Big-M-free complementarity path solve
bilevel KKT systems that the big-M encodings refuse?

HYPOTHESIS
----------
The KKT ``gdp``/``sos1`` big-M encodings refuse when follower multipliers are
unbounded -- which ``python/tests/test_bilevel_phase3.py`` documents as "the
common case".  #1123 proposes Big-M-free continuous reformulations of exactly
these complementarity relations.  If that direction is worth 3-5 months, the
minimum it must do is recover the known optimum on an instance the big-M path
refuses outright, without any user-asserted multiplier bound.

KILL CRITERION
--------------
Arm C (Scholtes, no ``multiplier_ub``) fails to recover the reference optimistic
optimum to 1e-3, or leaves a source complementarity residual above 1e-6.  Either
one falsifies the premise and the RFC's local-continuous mode is speculative.

ARMS
----
A  default KKT big-M, no multiplier_ub          -> expected: NotImplementedError
B  KKT big-M + user-asserted multiplier_ub      -> reference certified optimum
C  KKT + Scholtes homotopy, NO multiplier_ub    -> the hypothesis under test

Instances are the in-repo Bard LP (reused verbatim from test_bilevel_phase3) and
two further linear bilevel programs with independently derived optima, so a pass
is not a single-instance artifact (CLAUDE.md section 2).

MEASUREMENT DISCIPLINE (CLAUDE.md sections 6-11)
------------------------------------------------
* every check increments CHECKS; the script exits non-zero if CHECKS == 0
* no bare ``except``: probe failures propagate
* module provenance (``__file__``) asserted before any measurement
* per-instance progress printed unbuffered
"""

from __future__ import annotations

import sys

import discopt
import numpy as np
import scipy.optimize as scipy_opt
from discopt.bilevel import BilevelProblem
from discopt.modeling.core import Model
from discopt.mpec import solve_mpec

# ── section 8: prove which code is loaded, before measuring anything ──────────
print(f"==> discopt      {discopt.__file__}")
print(f"==> version      {discopt.__version__}")
import discopt._rust as _rust  # noqa: E402

print(f"==> rust ext     {_rust.__file__}")
from discopt.solvers.nlp_backend import available_backends  # noqa: E402

_BACKENDS = available_backends()
print(f"==> nlp backends {_BACKENDS}")
if not _BACKENDS:
    sys.exit("FATAL: no NLP backend; arm C cannot run and a 'pass' would be vacuous")

# ── section 6: executed-assertion counter ────────────────────────────────────
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


# ─────────────────────────────── instances ───────────────────────────────────
#
# Each builder returns (model, x, y, kwargs_for_BilevelProblem, reference_optimum,
# follower_argmin_callable).  The follower oracle is an independent scipy linprog
# solve, so "did we land on the true follower response" is checked outside discopt.


def bard_lp():
    """Verbatim from python/tests/test_bilevel_phase3.py::_bard_lp.

    Leader:   min x - 4y
    Follower: min y  s.t. x + y >= 3, y <= 2x, y in [0,10], x in [0,10]
    Known optimistic optimum: x=1, y=2, obj=-7.
    """
    m = Model("bard")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    kw = dict(
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
        lower_sense="min",
    )

    def follower(xv: float) -> float:
        res = scipy_opt.linprog(
            c=[1.0],
            A_ub=[[-1.0], [1.0]],
            b_ub=[xv - 3.0, 2.0 * xv],
            bounds=[(0.0, 10.0)],
            method="highs",
        )
        assert res.success, res.message
        return float(res.x[0])

    return m, x, y, kw, -7.0, follower


def follower_pushes_up():
    """Follower MAXIMIZES y; leader wants y small. Opposed objectives.

    Leader:   min x + y
    Follower: max y  s.t. y <= x, y <= 4 - x, y in [0,10], x in [0,4]
    Follower picks y = min(x, 4-x). Leader obj = x + min(x, 4-x):
      x <= 2 -> 2x (min 0 at x=0);  x > 2 -> 4. So optimum x=0, y=0, obj=0.
    """
    m = Model("push_up")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    kw = dict(
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[y <= x, y <= 4 - x],
        lower_sense="max",
    )

    def follower(xv: float) -> float:
        res = scipy_opt.linprog(
            c=[-1.0],
            A_ub=[[1.0], [1.0]],
            b_ub=[xv, 4.0 - xv],
            bounds=[(0.0, 10.0)],
            method="highs",
        )
        assert res.success, res.message
        return float(res.x[0])

    return m, x, y, kw, 0.0, follower


def two_row_active():
    """Both follower rows active at the optimum -> a biactive complementarity point,
    the degenerate case MPCC theory warns about.

    Leader:   min x - 2y
    Follower: min -y  s.t. y <= 1 + x, y <= 3 - x, y in [0,10], x in [0,3]
    Follower maximizes y: y = min(1+x, 3-x); rows coincide at x=1 (y=2).
    Leader obj = x - 2*min(1+x, 3-x):
      x <= 1 -> x - 2 - 2x = -x - 2 (min -3 at x=1)
      x > 1  -> x - 6 + 2x = 3x - 6 (min -3 at x=1)
    Optimum x=1, y=2, obj=-3, with BOTH follower rows tight.
    """
    m = Model("biactive")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 2 * y)
    kw = dict(
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=-y,
        lower_constraints=[y <= 1 + x, y <= 3 - x],
        lower_sense="min",
    )

    def follower(xv: float) -> float:
        res = scipy_opt.linprog(
            c=[-1.0],
            A_ub=[[1.0], [1.0]],
            b_ub=[1.0 + xv, 3.0 - xv],
            bounds=[(0.0, 10.0)],
            method="highs",
        )
        assert res.success, res.message
        return float(res.x[0])

    return m, x, y, kw, -3.0, follower


INSTANCES = [
    ("bard_lp", bard_lp, 50.0),
    ("follower_pushes_up", follower_pushes_up, 50.0),
    ("two_row_active", two_row_active, 50.0),
]


# ─────────────────────────────── the arms ────────────────────────────────────


def arm_a_bigm_refuses(build) -> tuple[bool, str]:
    """Default big-M KKT with no multiplier_ub. Documents the hole."""
    _m, _x, _y, kw, _ref, _fol = build()
    bl = BilevelProblem(_m, **kw)
    try:
        bl.formulate(method="kkt", mpec_method="gdp")
    except NotImplementedError as e:
        return True, str(e).split(".")[0][:110]
    return False, "big-M did NOT refuse (gate may have changed)"


def arm_b_bigm_with_bound(build, mu_ub: float) -> tuple[float | None, str]:
    """User-asserted multiplier_ub -> certified reference optimum."""
    m, _x, _y, kw, _ref, _fol = build()
    bl = BilevelProblem(m, multiplier_ub=mu_ub, **kw)
    res = bl.solve()
    if res.objective is None:
        return None, f"status={res.status}"
    return float(res.objective), f"status={res.status} gap_certified={res.gap_certified}"


def _flatten(model: Model, xvals: dict) -> np.ndarray:
    """Pack a {name: value} dict into x_flat in model._variables order."""
    flat = []
    for v in model._variables:
        val = xvals.get(v.name)
        if val is None:
            raise KeyError(f"no value for variable {v.name!r}; x_flat would be silently wrong")
        flat.append(np.asarray(val, dtype=np.float64).flatten())
    return np.concatenate(flat) if flat else np.zeros(0)


def _eval(expr, model: Model, x_flat: np.ndarray) -> np.ndarray:
    """Evaluate a model Expression at x_flat via the canonical DAG compiler."""
    from discopt._relax.dag_compiler import compile_expression

    fn = compile_expression(expr, model)
    return np.atleast_1d(np.asarray(fn(x_flat), dtype=np.float64))


def _source_complementarity_residual(model: Model, pairs, x_flat: np.ndarray) -> float:
    """r_mpcc = max_i min(f_i, g_i) on the SOURCE pairs, plus a nonnegativity penalty.

    Deliberately NOT the NLP solver's own barrier/KKT complementarity — #1123 is
    explicit that those are different quantities and must be separate columns.
    """
    worst = 0.0
    for p in pairs:
        f = _eval(p.f, model, x_flat)
        g = _eval(p.g, model, x_flat)
        # nonnegativity violation counts against us
        worst = max(worst, float(-np.min(np.minimum(f, 0.0))), float(-np.min(np.minimum(g, 0.0))))
        # orthogonality: elementwise min(|f|,|g|), worst over elements
        worst = max(worst, float(np.max(np.minimum(np.abs(f), np.abs(g)))))
    return worst


def _assert_big_m_free(model: Model, label: str) -> int:
    """PROVE the arm is Big-M-free rather than assuming it.

    A big-M / SOS1 lowering of a complementarity pair always introduces a selector
    BINARY. If the built model carries zero discrete variables, no big-M encoding
    ran. Returns the count of continuous variables checked so a vacuous scan
    (empty model) cannot pass silently.
    """
    discrete = [v for v in model._variables if getattr(v, "vtype", "continuous") != "continuous"]
    if discrete:
        raise AssertionError(
            f"{label}: model has {len(discrete)} discrete var(s) "
            f"{[v.name for v in discrete][:5]} — a big-M/SOS1 lowering ran, so this "
            f"arm is NOT Big-M-free and the whole comparison is void"
        )
    if not model._variables:
        raise AssertionError(f"{label}: model has no variables — scan was vacuous")
    return len(model._variables)


def _start_at_fraction(model: Model, frac: float) -> np.ndarray:
    """A deterministic start point at `frac` across the box of the CURRENT model.

    Must be called AFTER build_kkt_system(), which appends multiplier variables —
    sizing against the pre-KKT model produced a shape-(2) x0 for a 6-variable
    problem and numpy caught it. Sentinel bounds are clipped so an unbounded
    multiplier does not place the start at 1e20.
    """
    lb = np.concatenate([np.asarray(v.lb).flatten() for v in model._variables])
    ub = np.concatenate([np.asarray(v.ub).flatten() for v in model._variables])
    lo, hi = np.clip(lb, -1e3, 1e3), np.clip(ub, -1e3, 1e3)
    return lo + frac * (hi - lo)


def arm_c_scholtes(build, start_frac=None) -> tuple[float | None, float | None, dict, str]:
    """THE HYPOTHESIS: KKT + Scholtes homotopy, no multiplier_ub, no big-M.

    Per BilevelProblem.formulate's own docstring, the local Scholtes homotopy is
    driven at solve time via discopt.mpec.solve_mpec on the formulated pairs.
    """
    m, _x, _y, kw, _ref, _fol = build()
    bl = BilevelProblem(m, **kw)
    # build_kkt_system emits stationarity + primal feasibility and returns the
    # complementarity pairs WITHOUT lowering them -- so no big-M is ever built.
    kkt = bl.build_kkt_system()
    if not kkt.comp_pairs:
        return None, None, {}, "no complementarity pairs (nothing to test)"
    n_cont = _assert_big_m_free(m, "arm C (pre-solve)")
    # sized against the KKT-AUGMENTED model (build_kkt_system appended multipliers)
    x0 = None if start_frac is None else _start_at_fraction(m, start_frac)
    if x0 is not None and x0.shape[0] != len(
        _flatten(m, {v.name: np.asarray(v.lb) for v in m._variables})
    ):
        raise AssertionError(f"x0 shape {x0.shape} does not match the model's flat width")
    res = solve_mpec(m, kkt.comp_pairs, method="scholtes", t0=1.0, sigma=0.1, t_min=1e-9, x0=x0)
    # Scholtes only adds f*g <= t; re-check AFTER the reformulation, when a big-M
    # would actually have been emitted.
    _assert_big_m_free(m, "arm C (post-reformulation)")
    if res is None or res.x is None:
        return None, None, {}, f"scholtes returned no point (status={getattr(res, 'status', '?')})"
    x_flat = np.asarray(res.x, dtype=np.float64)
    xvals = {}
    off = 0
    for v in m._variables:
        n = int(np.asarray(v.lb).size)
        xvals[v.name] = x_flat[off : off + n].reshape(np.asarray(v.lb).shape)
        off += n
    # round-trip guard: the unpacking above must reproduce x_flat exactly, or the
    # residual below is computed against a different point than the solver returned.
    assert np.array_equal(_flatten(m, xvals), x_flat), "x_flat unpack/repack mismatch"
    resid = _source_complementarity_residual(m, kkt.comp_pairs, x_flat)
    return (
        float(res.objective),
        resid,
        xvals,
        f"status={res.status} iters={res.iterations} n_cont={n_cont} binaries=0",
    )


def arm_d_multistart(build, ref: float, n_starts: int = 9) -> tuple[int, int, list]:
    """RFC risk #4: initialization. One midpoint start on 3 instances is weak evidence.

    Sweep deterministic starts across the box and count how many recover the
    reference optimum. Deterministic (linspace, no RNG) so the result is
    reproducible — #1123 explicitly asks for reproducible branch provenance.
    """
    hits, objs = 0, []
    for k, frac in enumerate(np.linspace(0.05, 0.95, n_starts)):
        obj, resid, _xv, _d = arm_c_scholtes(build, start_frac=float(frac))
        ok = (
            obj is not None
            and abs(obj - ref) <= TOL_OBJ
            and resid is not None
            and resid <= TOL_COMP
        )
        hits += bool(ok)
        objs.append(None if obj is None else round(obj, 4))
        print(
            f"      start {k + 1}/{n_starts} frac={frac:.2f} "
            f"-> obj={objs[-1]} {'ok' if ok else 'MISS'}"
        )
    return hits, n_starts, objs


# ─────────────────────────────── driver ──────────────────────────────────────

TOL_OBJ = 1e-3
TOL_COMP = 1e-6

print("\n" + "=" * 78)
print("ENTRY EXPERIMENT — issue #1123: Big-M-free complementarity on bilevel KKT")
print("=" * 78)

verdicts = []

for name, build, mu_ub in INSTANCES:
    print(f"\n--- {name} " + "-" * (72 - len(name)))
    _, _, _, _, ref, follower = build()
    print(f"  reference optimistic optimum (derived by hand): {ref:+.6g}")

    # Arm A ------------------------------------------------------------------
    refused, detail = arm_a_bigm_refuses(build)
    check(f"{name}/A big-M refuses without multiplier_ub", refused, detail)

    # Arm B ------------------------------------------------------------------
    obj_b, detail_b = arm_b_bigm_with_bound(build, mu_ub)
    ok_b = obj_b is not None and abs(obj_b - ref) <= TOL_OBJ
    check(
        f"{name}/B big-M + multiplier_ub={mu_ub:g} recovers reference",
        ok_b,
        f"obj={obj_b if obj_b is None else round(obj_b, 6)} ref={ref} {detail_b}",
    )

    # Arm C — the hypothesis -------------------------------------------------
    obj_c, resid_c, xvals_c, detail_c = arm_c_scholtes(build)
    ok_c_obj = obj_c is not None and abs(obj_c - ref) <= TOL_OBJ
    check(
        f"{name}/C Scholtes (NO multiplier_ub, NO big-M) recovers reference",
        ok_c_obj,
        f"obj={obj_c if obj_c is None else round(obj_c, 6)} ref={ref} {detail_c}",
    )
    ok_c_res = resid_c is not None and resid_c <= TOL_COMP
    check(
        f"{name}/C source complementarity residual <= {TOL_COMP:g}",
        ok_c_res,
        f"r_mpcc={resid_c if resid_c is None else f'{resid_c:.3e}'}",
    )

    # Independent follower check: is the returned y the follower's true argmin?
    ok_c_fol = False
    fol_detail = "no point"
    if xvals_c:
        xv = float(np.ravel(xvals_c["x"])[0])
        yv = float(np.ravel(xvals_c["y"])[0])
        y_true = follower(xv)
        ok_c_fol = abs(yv - y_true) <= 1e-3
        fol_detail = f"x={xv:.6g} y={yv:.6g} scipy_argmin={y_true:.6g}"
    check(f"{name}/C returned y is the follower's true argmin (scipy)", ok_c_fol, fol_detail)

    # Arm D — initialization sensitivity (RFC risk #4) -----------------------
    print(f"    ... {name}/D multistart sensitivity")
    hits, n_starts, objs = arm_d_multistart(build, ref)
    check(
        f"{name}/D recovers reference from a majority of starts",
        hits > n_starts // 2,
        f"{hits}/{n_starts} starts | objs={objs}",
    )

    verdicts.append((name, refused, ok_b, ok_c_obj and ok_c_res and ok_c_fol, hits, n_starts))

# ─────────────────────────────── verdict ─────────────────────────────────────

print("\n" + "=" * 78)
print(f"EXECUTED CHECKS: {CHECKS}")
if CHECKS == 0:
    sys.exit("FATAL: probe executed zero checks — result is meaningless (CLAUDE.md §6)")

print(
    f"{'instance':<22}{'A refuses':>12}{'B certified':>14}{'C big-M-free':>15}{'D multistart':>15}"
)
for name, a, b, c, hits, n_starts in verdicts:
    print(f"{name:<22}{str(a):>12}{str(b):>14}{str(c):>15}{f'{hits}/{n_starts}':>15}")

n_hole = sum(1 for _, a, _, _, _, _ in verdicts if a)
n_c = sum(1 for _, _, _, c, _, _ in verdicts if c)
tot_hits = sum(h for *_, h, _ in verdicts)
tot_starts = sum(n for *_, n in verdicts)
print(f"multistart: {tot_hits}/{tot_starts} starts recovered the reference optimum")
print(
    f"\nbig-M refused on {n_hole}/{len(verdicts)}; "
    f"Big-M-free path succeeded on {n_c}/{len(verdicts)}"
)

if n_c == len(verdicts) and n_hole > 0:
    print("\nVERDICT: HYPOTHESIS SURVIVES — the Big-M-free path answers instances the")
    print("         big-M encodings refuse outright. #1123's local-continuous mode")
    print("         has measured justification on real in-repo bilevel models.")
    sys.exit(0)
elif n_c == 0:
    print("\nVERDICT: FALSIFIED — the Big-M-free path recovered nothing. Per the kill")
    print("         criterion, #1123's local-continuous mode is speculative; defer.")
    sys.exit(2)
else:
    print("\nVERDICT: PARTIAL — succeeded on some instances, not all. Report honestly;")
    print("         the failures are the interesting cases and must be diagnosed")
    print("         before committing to the full sequence.")
    sys.exit(3)
