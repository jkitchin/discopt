"""#2202 probe: MINLPLib nous1 through Model.solve with the native kernel's fixed rows
permuted by PERM_SEED (0 = unpermuted), as docs/dev/data/2142-dual-proof/e1_arm.py does
for cstr. A row permutation changes nothing but the order of floating-point work, which
is the class of difference that separates macOS/arm64 from Linux in #2202.

    PERM_SEED=k DISCOPT_DUAL_PROOF_PROPAGATION=0|1 DISCOPT_NODE_LP_SLACK_RETRY=0|1 \\
        TL=120 [MAX_NODES=N] python -u perm_probe.py

With MAX_NODES the solve is ``deterministic=True`` under that node budget (TL is then
only a backstop), so a run is a function of the model and the flags.

Prints status / certification / bound / nodes / wall and the kernel's undecided-node and
dual-proof counters, then an executed-assertion count (CLAUDE.md §6); exits non-zero if
the kernel never ran. Asserts the loaded extension is this tree's build (§8).
"""

import os
import sys
import time
from pathlib import Path

import discopt
import discopt._rust as R
import discopt.modeling as dm
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
assert Path(discopt.__file__).resolve().is_relative_to(_REPO / "python"), discopt.__file__
if os.environ.get("DISCOPT_NODE_LP_SLACK_RETRY") is not None:
    _sig = R.solve_spatial_tree_py.__text_signature__ or ""
    assert "node_lp_slack_retry" in _sig, "stale _rust: no #2202 retry"

seed = int(os.environ.get("PERM_SEED", "0"))
orig = R.solve_spatial_tree_py
caps = []
n_assert = 0


def wrap(*a, **k):
    if seed and int(k.get("max_nodes", 10**9)) > 1:
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
    res = orig(*a, **k)
    caps.append(res)
    return res


R.solve_spatial_tree_py = wrap
t = time.time()
_kw = {}
if os.environ.get("MAX_NODES"):
    _kw = {"deterministic": True, "max_nodes": int(os.environ["MAX_NODES"])}
r = dm.from_nl(str(_REPO / "python/tests/data/issue_1871/nous1.nl")).solve(
    time_limit=float(os.environ.get("TL", "120")), gap_tolerance=1e-4, **_kw
)
big = [c for c in caps if c.get("node_count", 0) > 1]
if not big:
    print("kernel never ran: 0 executed assertions", flush=True)
    sys.exit(1)
c = big[-1]
dp = c["dual_proof"]
n_assert += 1
print(
    f"seed={seed} dp={os.environ.get('DISCOPT_DUAL_PROOF_PROPAGATION')} "
    f"retry={os.environ.get('DISCOPT_NODE_LP_SLACK_RETRY')} {r.status} cert={r.gap_certified} "
    f"bound={r.bound:.6f} nodes={r.node_count} wall={time.time() - t:.1f} "
    f"lp={c['n_lp_solves']} undec={c['n_undecided']} rays={dp['rays']} "
    f"learned={dp['learned']} tight={dp['tightenings']}",
    flush=True,
)
print(f"executed assertions: {n_assert}", flush=True)
