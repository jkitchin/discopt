"""EXPERIMENT 5 -- the characteristic-polynomial (CP) formulation.

Galvan & Lindh (JCTC 2023) and Wang/Truhlar-style diabatic surrogates avoid
the non-differentiability of the adiabatic surfaces at a conical intersection
by fitting *smooth* surfaces instead.  Richings & Habershon (JPCL 2023,
"Machine Learning Seams of Conical Intersection: A Characteristic Polynomial
Approach") make the cleanest choice: fit the coefficients of the
characteristic polynomial of the potential matrix.  For two states those are
just the two symmetric functions of the adiabatic energies,

    T(x) = E1 + E2 = W11 + W22            (trace)
    D(x) = E1 * E2 = W11*W22 - W12^2      (determinant)

both of which are smooth *through* the intersection, unlike E1 and E2
individually.  Degeneracy is then a single polynomial condition -- the
discriminant vanishing:

    T(x)^2 - 4 D(x) = (E1-E2)^2 = 0

and at degeneracy E1 = E2 = T/2, so the MECI problem becomes

    min  T(x)/2      s.t.   T(x)^2 - 4 D(x) = 0                     (CP form)

This is representation-free (no diabatization needed), smooth, and
factorable -- exactly discopt's expression class.

But it has a property that should destroy a local NLP solver: since
T^2 - 4D = (E1-E2)^2 >= 0 identically, its gradient
   grad(T^2-4D) = 2(E1-E2) grad(E1-E2)
vanishes at *every* feasible point.  LICQ fails on the whole feasible set, so
the KKT multipliers are unbounded and Newton-type methods lose their
convergence theory.  A relaxation-based global solver does not need LICQ --
it needs valid bounds.

Hypothesis H5: discopt solves the CP form; local NLP solvers on the same
formulation either fail to converge or converge to points off the seam.

Kill criterion: if discopt cannot solve the CP form either, then the CP
formulation is not the bridge and the diabatic form is the only option.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402
import mecp_models as M  # noqa: E402

CHECKS = 0
NOTES: list[str] = []


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        NOTES.append(msg)
        print(f"   !! {msg}")
    return cond


def show(tag, r):
    print(
        f"   {tag:<30s} status={r.status:<12s} obj={_n(r.objective)} bound={_n(r.bound)} "
        f"gap={_g(r.gap)} cert={r.gap_certified} nodes={r.node_count:<6d} t={r.wall_time:.2f}s"
    )


def _n(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.1e}"


print("=" * 84)
print("EXPERIMENT 5 -- characteristic-polynomial MECI formulation")
print("=" * 84)

lp = M.LVCParams()
print(f"analytic MECI: qt={lp.meci_qt} qc=0 E={lp.meci_energy}")

# --------------------------------------------------------------------------
# 5A: CP form in discopt
# --------------------------------------------------------------------------
print("\n[5A] discopt, CP form:  min T/2  s.t.  T^2 - 4D = 0")


def cp_surfaces(qt, qc):
    """(T, D) built from the LVC diabatic matrix -- both smooth everywhere."""
    w11, w22, w12 = lp.diabats(qt, qc)
    return w11 + w22, w11 * w22 - w12**2


m = dm.Model("lvc_meci_charpoly")
qt = m.continuous("qt", lb=lp.box[0], ub=lp.box[1])
qc = m.continuous("qc", lb=lp.box[0], ub=lp.box[1])
T, D = cp_surfaces(qt, qc)
m.minimize(0.5 * T)
m.subject_to(T**2 - 4.0 * D == 0)
r = m.solve(time_limit=300.0)
show("CP / equality", r)
check(
    r.objective is not None and abs(r.objective - lp.meci_energy) < 1e-3,
    f"CP form objective {r.objective} != analytic {lp.meci_energy}",
)
if r.x is not None:
    x = np.array([float(np.asarray(r.x[v]).ravel()[0]) for v in ("qt", "qc")])
    lo, hi = lp.adiabats(*x)
    print(f"      x={np.round(x, 6)}  E_lo={lo:.8f} E_hi={hi:.8f} gap={hi - lo:.3e}")
    check(abs(hi - lo) < 1e-3, f"CP solution not degenerate: gap={hi - lo:.3e}")
if r.bound is not None and r.gap_certified:
    check(
        r.bound <= lp.meci_energy + 1e-6,
        f"SOUNDNESS: CP certified bound {r.bound} > analytic optimum {lp.meci_energy}",
    )

# --------------------------------------------------------------------------
# 5B: same in inequality form (the discriminant is nonneg, so <= 0 suffices)
# --------------------------------------------------------------------------
print("\n[5B] discopt, CP inequality:  min T/2  s.t.  T^2 - 4D <= 0")
print("     (valid because T^2-4D = (E1-E2)^2 >= 0 identically)")
m = dm.Model("lvc_meci_charpoly_ineq")
qt = m.continuous("qt", lb=lp.box[0], ub=lp.box[1])
qc = m.continuous("qc", lb=lp.box[0], ub=lp.box[1])
T, D = cp_surfaces(qt, qc)
m.minimize(0.5 * T)
m.subject_to(T**2 - 4.0 * D <= 0)
r_ineq = m.solve(time_limit=300.0)
show("CP / inequality", r_ineq)
check(
    r_ineq.objective is not None and abs(r_ineq.objective - lp.meci_energy) < 1e-3,
    f"CP inequality objective {r_ineq.objective} != analytic {lp.meci_energy}",
)

# --------------------------------------------------------------------------
# 5C: local NLP solvers on the CP form -- LICQ fails on the feasible set
# --------------------------------------------------------------------------
print("\n[5C] local NLP (SLSQP / trust-constr) on the SAME CP formulation")
print("     LICQ fails at every feasible point, so this should degrade")


def T_np(x):
    return float(cp_surfaces(x[0], x[1])[0])


def disc_np(x):
    t, d = cp_surfaces(x[0], x[1])
    return float(t * t - 4.0 * d)


bnds = [lp.box, lp.box]
rng = np.random.default_rng(4)
for method, cons_kind in (("SLSQP", "eq"), ("trust-constr", "eq")):
    n_ok = n_seam = 0
    errs = []
    for k in range(60):
        x0 = rng.uniform(-10, 10, size=2)
        try:
            res = minimize(
                lambda z: 0.5 * T_np(z),
                x0,
                method=method,
                bounds=bnds,
                constraints=(
                    [{"type": cons_kind, "fun": disc_np}]
                    if method == "SLSQP"
                    else [NonlinearConstraint(disc_np, 0.0, 0.0)]
                ),
                options={"maxiter": 800},
            )
        except Exception as exc:
            NOTES.append(f"{method}: raised {type(exc).__name__}: {exc}")
            continue
        n_ok += 1
        lo, hi = lp.adiabats(res.x[0], res.x[1])
        gap = float(hi - lo)
        if gap < 1e-4:
            n_seam += 1
            errs.append(abs(0.5 * T_np(res.x) - lp.meci_energy))
    CHECKS += 1
    med = float(np.median(errs)) if errs else float("nan")
    print(
        f"   {method:<14s} ran {n_ok:3d}/60   reached the seam (gap<1e-4): {n_seam:3d}/{n_ok}"
        f"   median |E-E*| on those: {med:.3e}"
    )

# --------------------------------------------------------------------------
# 5D: CP form on a model with more than one crossing basin
# --------------------------------------------------------------------------
print("\n[5D] CP form on the two-basin spin-crossing model (n=2)")
print("     for a spin crossing W12 == 0, so T = W1+W2, D = W1*W2")
tw = M.TwoWellParams(n=2)
e_true, x_true, _, basins = M.seam_oracle_2d(tw, n_grid=1601)
print(f"     oracle global E={e_true:.6f}; second basin E={basins[1][0]:.6f}")

m = dm.Model("twowell_charpoly")
bb = tw.bounds()
q = [m.continuous(f"q{i}", lb=bb[i][0], ub=bb[i][1]) for i in range(2)]
w1, w2 = tw.states(q, exp=dm.exp)
Tt = w1 + w2
Dd = w1 * w2
m.minimize(0.5 * Tt)
m.subject_to(Tt**2 - 4.0 * Dd <= 0)
t0 = time.time()
r_tw = m.solve(time_limit=300.0)
show("twowell CP / ineq", r_tw)
if r_tw.x is not None:
    xv = np.array([float(np.asarray(r_tw.x[f"q{i}"]).ravel()[0]) for i in range(2)])
    a, b = tw.states(list(xv), exp=np.exp)
    print(f"      x={np.round(xv, 5)}  W1={float(a):.6f}  gap={abs(float(a - b)):.2e}")
    check(
        abs(float(a) - e_true) < 5e-3,
        f"CP form on two-basin model gave E={float(a):.6f}, oracle global={e_true:.6f}",
    )

print("\n" + "=" * 84)
print(f"EXECUTED CHECKS: {CHECKS}")
print(f"NOTES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
if CHECKS == 0:
    sys.exit(2)
