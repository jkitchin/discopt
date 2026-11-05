"""#2142 dual-proof entry experiment: GDPlib cstr through the native kernel, deterministic,
with the kernel's fixed rows permuted by PERM_SEED (seed 0 = unpermuted), as in
docs/dev/data/2142-scip-expert/py/d3_arm.py. The flag is DISCOPT_DUAL_PROOF_PROPAGATION.

  PERM_SEED=k DISCOPT_DUAL_PROOF_PROPAGATION=0|1 python -u e1_arm.py MAX_NODES [big-m|hull] [TIME_LIMIT]

Prints bound / nodes / wall / the kernel's dual-proof counters and an executed-assertion count
(CLAUDE.md §6); exits non-zero when nothing was asserted. Asserts the loaded extension is the
#2142 build (CLAUDE.md §8: it must accept the dual_proof_propagation kwarg)."""

import os
import sys
import time

import numpy as np

import discopt
import discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert "dual_proof_propagation" in (R.solve_spatial_tree_py.__text_signature__ or ""), "stale _rust"
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

import gdplib.cstr as cstr  # noqa: E402

max_nodes = int(sys.argv[1])
method = sys.argv[2] if len(sys.argv) > 2 else "big-m"
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0
seed = int(os.environ.get("PERM_SEED", "0"))
flag = os.environ.get("DISCOPT_DUAL_PROOF_PROPAGATION", "0")
n_assert = 0
calls = []
_orig = R.solve_spatial_tree_py


def _w(*a, **k):
    global n_assert
    assert not a
    want = flag.strip() in ("1", "true", "yes", "on")
    if int(k.get("max_nodes", 100_000)) > 1:  # 1-node root probes cannot use a pool
        assert bool(k.get("dual_proof_propagation", False)) == want, "flag did not reach the kernel"
        n_assert += 1
    if seed:
        ptr = np.asarray(k["fixed_row_ptr"])
        cols = np.asarray(k["fixed_cols"])
        co = np.asarray(k["fixed_coeffs"])
        rhs = np.asarray(k["fixed_rhs"])
        perm = np.random.default_rng(seed).permutation(len(rhs))
        nc, ncf, np_ = [], [], [0]
        for q in perm:
            nc.append(cols[ptr[q] : ptr[q + 1]])
            ncf.append(co[ptr[q] : ptr[q + 1]])
            np_.append(np_[-1] + ptr[q + 1] - ptr[q])
        k["fixed_row_ptr"] = np.array(np_, dtype=ptr.dtype)
        k["fixed_cols"] = np.concatenate(nc).astype(cols.dtype)
        k["fixed_coeffs"] = np.concatenate(ncf)
        k["fixed_rhs"] = rhs[perm].copy()
    t = time.perf_counter()
    r = _orig(*a, **k)
    assert "global_lps" in r["dual_proof"], "stale _rust (no #2142 global-LP counter)"
    calls.append(
        dict(
            max_nodes=k.get("max_nodes"),
            bound=r["bound"],
            nodes=r["node_count"],
            undecided=r.get("n_undecided"),
            status=r["status"],
            dual_proof=dict(r["dual_proof"]),
            wall=round(time.perf_counter() - t, 1),
        )
    )
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method=method, time_limit=tl, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(
    f"METHOD={method} PERM_SEED={seed} FLAG={flag} MAX_NODES={max_nodes} status={res.status} "
    f"obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
    f"gap_certified={getattr(res, 'gap_certified', None)} wall={wall:.1f}s",
    flush=True,
)
for c in calls:
    print("  kernel:", c, flush=True)
assert calls, "the kernel never ran"
n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
