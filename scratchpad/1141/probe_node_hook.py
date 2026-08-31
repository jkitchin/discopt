"""Anti-vacuity + soundness probe for the fractional-node hook (#1141).

Ball MILP: max x+y over integer (x,y) in [-3,3]^2 with x^2+y^2 <= r^2, solved as a
single-tree LP master + separators. The nonlinear row lives ONLY in the callbacks,
so the answer is right only if the separation actually happened.

Prints an executed-assertion count and exits non-zero if it is zero (CLAUDE.md §6).
"""
import sys
import numpy as np
from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts

R2 = 5.0
LO, HI = -3.0, 3.0


def cut_at(x):
    """Gradient cut of g(x)=x0^2+x1^2-R2 <= 0 at xbar: 2*xbar.x <= R2 + |xbar|^2."""
    g = float(x[0] ** 2 + x[1] ** 2 - R2)
    if g <= 1e-9:
        return []
    return [(np.array([2 * x[0], 2 * x[1]]), R2 + float(x[0] ** 2 + x[1] ** 2))]


def run(with_node):
    stats = {}
    c = np.array([-1.0, -1.0])
    bounds = [(LO, HI), (LO, HI)]
    integrality = np.array([1, 1])
    kw = dict(
        c=c,
        A_ub=np.zeros((1, 2)),
        b_ub=np.array([1.0]),
        bounds=bounds,
        integrality=integrality,
        lazy_callback=cut_at,
        time_limit=30.0,
        gap_tolerance=1e-9,
    )
    if with_node:
        kw.update(node_callback=cut_at, node_hook_rounds=4, node_hook_cut_cap=200)
    r = solve_milp_with_lazy_cuts(**kw)
    stats = dict(r.callback_stats or {})
    return r, stats


checks = 0
best = None
for with_node in (False, True):
    r, st = run(with_node)
    print(f"node_hook={with_node}: status={r.status} x={r.x} obj={r.objective} "
          f"nodes={r.node_count} mipsol={st.get('mipsol_calls')} "
          f"mipnode={st.get('mipnode_calls')} node_cuts={st.get('node_cuts')} "
          f"driver_node_cuts={st.get('driver_node_cuts')}")
    assert r.x is not None, "no solution"
    checks += 1
    assert float(r.x[0] ** 2 + r.x[1] ** 2) <= R2 + 1e-6, f"INFEASIBLE point {r.x}"
    checks += 1
    assert abs(float(r.objective) - (-3.0)) < 1e-6, f"wrong optimum {r.objective}"
    checks += 1
    if with_node:
        assert st.get("mipnode_calls", 0) > 0, "node separator NEVER FIRED"
        checks += 1
        assert st.get("driver_node_cuts", 0) > 0, "node separator added no rows"
        checks += 1
    else:
        assert st.get("mipnode_calls", 0) == 0, "node separator fired without a hook"
        checks += 1

print(f"\nEXECUTED ASSERTIONS: {checks}")
if checks == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
