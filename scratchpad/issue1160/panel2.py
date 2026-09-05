"""More axis-sum cases: integer/MILP, QP objective, least-squares, matmul-free."""
import sys
import numpy as np
import discopt.modeling.core as dm
from discopt.validation.feasibility import verify_point

TOL = 1e-4
cases = []


def case(name, build, true_opt, witness=None):
    cases.append((name, build, true_opt, witness))


def b_milp():
    m = dm.Model("milp_axis1")
    B = m.binary("B", shape=(2, 3))
    m.subject_to(dm.sum(B, axis=1) <= 2)
    m.minimize(-dm.sum(B))
    return m


case("milp_axis1", b_milp, -4.0, np.array([1.0, 1, 0, 1, 1, 0]))


def b_miqp():
    m = dm.Model("miqp_axis1")
    B = m.binary("B", shape=(2, 3))
    m.subject_to(dm.sum(B, axis=1) <= 2)
    m.minimize(dm.sum(B * B) - 4 * dm.sum(B))
    return m


# each b in {0,1}: b*b - 4b = -3b, so minimize -3*sum(B) with per-row cap 2 -> -12
case("miqp_axis1", b_miqp, -12.0, np.array([1.0, 1, 0, 1, 1, 0]))


def b_qp():
    m = dm.Model("qp_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(dm.sum(A * A) - 4 * dm.sum(A))
    return m


# with the row sum pinned at its cap 2, min sum(a^2) - 4*sum(a) = min sum(a^2) - 8
# spreads the row equally: 3*(2/3)^2 - 8 = -6.667 per row -> -13.333
case("qp_axis1", b_qp, -40.0 / 3.0, np.array([1.0, 1, 0, 1, 1, 0]))


def b_ls():
    m = dm.Model("ls_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) <= 2)
    m.minimize(dm.sum((A - 1) * (A - 1)))
    return m


# min sum (a-1)^2 with row sum <= 2 spreads: a = 2/3 -> 3*(1/9) = 1/3 per row -> 2/3
case("ls_axis1", b_ls, 2.0 / 3.0, np.array([1.0, 1, 0, 1, 1, 0]))


def b_eq():
    m = dm.Model("lin_eq_axis1")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=1) == 2)
    m.minimize(-dm.sum(A * A))
    return m


# per row sum exactly 2 -> maximize sum of squares -> two at 1 -> 2 per row -> -4
case("lin_eq_axis1", b_eq, -4.0, np.array([1.0, 1, 0, 1, 1, 0]))


def b_3d():
    m = dm.Model("lin_axis_3d")
    A = m.continuous("A", shape=(2, 2, 2), lb=0, ub=1)
    m.subject_to(dm.sum(A, axis=2) <= 1)
    m.minimize(-dm.sum(A))
    return m


# 4 rows of 2, each capped at 1 -> total 4
case("lin_axis_3d", b_3d, -4.0, np.array([1.0, 0, 1, 0, 1, 0, 1, 0]))


def b_1d():
    m = dm.Model("lin_axis0_1d")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    m.subject_to(dm.sum(x, axis=0) <= 2)
    m.minimize(-dm.sum(x))
    return m


# axis=0 on a 1-D operand IS a full reduction -> -2
case("lin_axis0_1d", b_1d, -2.0, np.array([1.0, 1, 0, 0]))


def main():
    compared = 0
    bad = []
    for name, build, true_opt, witness in cases:
        m = build()
        r = m.solve(time_limit=120, gap_tolerance=1e-6)
        line = f"{name:16s} status={r.status:10s} obj={r.objective!r:24s} bound={r.bound!r}"
        if witness is not None:
            v = verify_point(m, witness, with_objective=True)
            line += f"  witness_ok={v.ok} obj={v.objective}"
        print(line, flush=True)
        compared += 1
        ok_obj = r.objective is not None and abs(r.objective - true_opt) <= TOL * max(1, abs(true_opt))
        ok_bound = r.bound is None or r.bound <= true_opt + TOL * max(1, abs(true_opt))
        print(f"    true={true_opt:.6f} obj_ok={ok_obj} bound_ok={ok_bound}", flush=True)
        if not (ok_obj and ok_bound):
            bad.append(name)
    print("COMPARISONS:", compared)
    print("FAILING:", bad)
    if compared == 0:
        sys.exit("no comparisons")
    sys.exit(1 if bad else 0)


main()
