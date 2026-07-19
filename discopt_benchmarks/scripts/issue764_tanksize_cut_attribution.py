"""#764 entry experiment — is the cut engine the lever for tanksize?

Two decisive, cheap measurements (2026-07-19):

1. Cut REACHABILITY on the spatial path: instrument every discopt cut separator
   and count invocations during a default tanksize solve. Result: 0 (the spatial
   McCormick B&B has no cut seam; the integer-cut machinery is in _solve_milp_bb).

2. Cut ATTRIBUTION via SCIP: solve tanksize's root (limits/nodes=1) with SCIP's
   cut loop on vs separation off, and compare to discopt's root. Result: SCIP's
   cuts close only 2.91% of the root-to-opt gap; discopt root (0.8402) already
   matches SCIP's cut-loaded root (0.8508).

Verdict: cut line NO-GO for tanksize (real instance). The lever is per-node
throughput (C1 native kernel), not cuts (C3). See issue-764-scip-comparison.md.
"""
from __future__ import annotations
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

NL = "python/tests/data/minlplib_nl/tanksize.nl"
OPT = 1.2686437540


def reachability():
    os.environ["DISCOPT_CMIR_AGGREGATION"] = "1"
    from discopt.modeling.core import from_nl
    import discopt.solver as S
    from discopt._jax import cmir_cuts, cutting_planes as CP

    counts: dict[str, int] = {}

    def wrap(mod, name):
        orig = getattr(mod, name)

        def f(*a, **k):
            counts[name] = counts.get(name, 0) + 1
            return orig(*a, **k)

        setattr(mod, name, f)

    for nm in ("_separate_gomory_cuts", "_separate_mir_cuts",
               "_separate_aggregation_mir_cuts", "_root_cover_cut_loop"):
        if hasattr(S, nm):
            wrap(S, nm)
    if hasattr(cmir_cuts, "separate_cmir"):
        wrap(cmir_cuts, "separate_cmir")
    for nm in ("generate_cuts_at_node", "separate_rlt_cuts"):
        if hasattr(CP, nm):
            wrap(CP, nm)

    S.GOMORY_CUTS_ENABLED = True
    m = from_nl(NL)
    r = m.solve(time_limit=30.0, max_nodes=50)
    print(f"[reachability] root_bound={getattr(r, 'root_bound', None)} "
          f"nodes={r.node_count} separator_calls={counts}")


def attribution():
    from pyscipopt import Model as SM

    def scip_root(sepa_off):
        m = SM()
        m.hideOutput(True)
        m.readProblem(NL)
        m.setParam("limits/nodes", 1)
        if sepa_off:
            m.setParam("separating/maxrounds", 0)
            m.setParam("separating/maxroundsroot", 0)
        m.optimize()
        return m.getDualbound()

    withcuts, nocuts = scip_root(False), scip_root(True)
    d = OPT - nocuts
    print(f"[attribution] SCIP root cuts_on={withcuts:.6f} cuts_off={nocuts:.6f} "
          f"discopt_root=0.840197 opt={OPT:.6f}")
    print(f"[attribution] SCIP cuts close {(withcuts - nocuts) / d * 100:.2f}% of root gap; "
          f"cut-root vs discopt-root delta={withcuts - 0.840197:+.4f}")


if __name__ == "__main__":
    reachability()
    attribution()
