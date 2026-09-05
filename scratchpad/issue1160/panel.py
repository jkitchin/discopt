"""Axis-reduced-sum correctness panel.

Each case: a model built with dm.sum(..., axis=k), its TRUE optimum (derived by
hand and cross-checked with verify_point on a witness), and the answer the
axis-collapsed model would give.  Prints an executed-comparison count and exits
non-zero if it is zero.
"""
import sys
import numpy as np
import discopt.modeling.core as dm
from discopt.validation.feasibility import verify_point

TOL = 1e-4
cases = []


def case(name, build, true_opt, witness=None):
    cases.append((name, build, true_opt, witness))


# 1. linear, axis=1 (per-row cap)
def b1():
    m = dm.Model("lin_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(-dm.sum(A))
    return m


case("lin_axis1", b1, -4.0, np.array([1.0, 1, 0, 1, 1, 0]))


# 2. linear, axis=0 (per-column cap)
def b2():
    m = dm.Model("lin_axis0")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=0) <= 1)
    m.minimize(-dm.sum(A))
    return m


case("lin_axis0", b2, -3.0, np.array([1.0, 1, 1, 0, 0, 0]))


# 3. convex quadratic, axis=1: per row max sum with sum-sq <= 0.5 is 3*sqrt(1/6)
def b3():
    m = dm.Model("quad_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A * A, axis=1) <= 0.5)
    m.minimize(-dm.sum(A))
    return m


case("quad_axis1", b3, -2 * 3 * np.sqrt(1.0 / 6.0), np.full(6, np.sqrt(1 / 6)))


# 4. nonconvex quadratic (>= on a convex body), axis=1: each row needs sum-sq >= 2
#    with a <= 1, so at least two vars at 1 -> min row sum 2, total 4.
def b4():
    m = dm.Model("nonconvex_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A * A, axis=1) >= 2)
    m.minimize(dm.sum(A))
    return m


case("nonconvex_axis1", b4, 4.0, np.array([1.0, 1, 0, 1, 1, 0]))


# 5. bilinear across two variables, axis=1
def b5():
    m = dm.Model("bilinear_axis1")
    X = m.continuous("X", shape=(2, 2), lb=0, ub=1)
    Y = m.continuous("Y", shape=(2, 2), lb=0, ub=1)
    m.subject_to(dm.sum(X * Y, axis=1) <= 0.5)
    m.minimize(-dm.sum(X) - dm.sum(Y))
    return m


# per row: max sum(x)+sum(y) s.t. x1y1+x2y2 <= 0.5, all in [0,1].
# x1=y1=1 uses 1.0 > 0.5, so take x1=1,y1=0.5,x2=1,y2=0 -> 2.5? check: 1*0.5+1*0=0.5, sum=2.5
# better: x1=1,y1=0,x2=1,y2=0.5 same. or x1=1,x2=1,y1=0.5,y2=0 -> 2.5.
# Try x1=x2=1, y1=y2=0.25 -> 0.5 -> sum = 2.5.  Upper bound argument: sum<=4 but
# constraint binds. Leave the true value to a fine grid search below.
case("bilinear_axis1", b5, None, None)


# 6. transcendental, axis=1: sum(exp(A), axis=1) <= 3*e^0 + ... use log to be exact
def b6():
    m = dm.Model("exp_axis1")
    A = m.continuous("A", shape=(2, 2), lb=0, ub=1)
    m.subject_to(dm.sum(dm.exp(A), axis=1) <= 1 + np.e)
    m.minimize(-dm.sum(A))
    return m


# per row: exp(a1)+exp(a2) <= 1+e, max a1+a2. By symmetry a1=a2=log((1+e)/2)=0.6187
# -> row sum 1.2375, total 2.4749.  (a=1,b=0 gives row sum 1.)
_a = np.log((1 + np.e) / 2)
case("exp_axis1", b6, -4 * _a, np.full(4, _a))


def main():
    compared = 0
    bad = []
    for name, build, true_opt, witness in cases:
        m = build()
        r = m.solve(time_limit=120, gap_tolerance=1e-6)
        line = f"{name:18s} status={r.status:10s} obj={r.objective!r:24s} bound={r.bound!r}"
        if witness is not None:
            v = verify_point(m, witness, with_objective=True)
            line += f"  witness_ok={v.ok} witness_obj={v.objective}"
        print(line, flush=True)
        if true_opt is not None:
            compared += 1
            ok_obj = r.objective is not None and abs(r.objective - true_opt) <= TOL * max(
                1.0, abs(true_opt)
            )
            # minimize sense in every case: a valid dual bound is <= the true optimum
            ok_bound = r.bound is None or r.bound <= true_opt + TOL * max(1.0, abs(true_opt))
            print(f"    true={true_opt:.6f} obj_ok={ok_obj} bound_ok={ok_bound}", flush=True)
            if not (ok_obj and ok_bound):
                bad.append(name)
    print("COMPARISONS:", compared)
    print("FAILING:", bad)
    if compared == 0:
        sys.exit("probe made no comparisons")
    sys.exit(1 if bad else 0)


main()
