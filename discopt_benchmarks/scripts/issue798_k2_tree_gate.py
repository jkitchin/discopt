#!/usr/bin/env python
"""Issue #798 / K2d — panel gate for the Rust convex LP-OA branch-and-cut tree.

Runs the native `solve_convex_tree_py` on the convex panel and checks:
  1. CORRECTNESS (non-negotiable): status == optimal; the certified incumbent
     equals the oracle optimum to tolerance; the dual bound is a valid upper bound
     (>= optimum) — never a false optimal or a bound below the oracle.
  2. NODES-TO-CERTIFY within ~2x of the Python LP-OA prototype (67/60/46/55 for
     rsyn0805m/rsyn0810m/rsyn0815m/syn40m).

Methodology matches the prototype (issue786_lpoa_bandc_prototype): the incumbent
is SEEDED with the oracle optimum so the measurement isolates the DUAL side —
nodes-to-certify, which in-tree cutting drives — from the primal (finding
incumbents is K3's job: rounding/diving/NLP). With the optimum seeded,
nodes-to-certify = how many nodes the tree needs to PROVE optimality.

Also reports the ablation `max_sep_rounds=0` (no in-tree cutting) to show the
cutting's node-count contribution (the #786/#797 15-18x lever).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discopt._rust as _rust  # noqa: E402
from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402
from issue798_k1_bytecheck import build_convex_arrays  # noqa: E402

# Python LP-OA prototype nodes-to-certify (issue #798 §2 / #797), PANEL order.
PROTOTYPE_NODES = {
    "rsyn0805m": 67,
    "rsyn0810m": 60,
    "rsyn0815m": 46,
    "syn40m": 55,
}
NODE_FACTOR = 2.0
GAP_TOL = 1e-4
TIME_LIMIT_S = 180.0


def run_tree(
    rm: RootModel,
    opt: float,
    max_sep_rounds: int,
    max_nodes: int = 20000,
    time_limit_s: float = TIME_LIMIT_S,
) -> dict:
    arrays = build_convex_arrays(rm, rm.lb, rm.ub)  # global box; tree does FBBT
    return _rust.solve_convex_tree_py(
        **arrays,
        max_nodes=max_nodes,
        gap_tol=GAP_TOL,
        int_tol=1e-5,
        oa_tol=1e-6,
        max_oa_rounds=60,
        max_sep_rounds=max_sep_rounds,
        fbbt_rounds=20,
        initial_incumbent=float(opt),  # seed → isolate nodes-to-certify (dual side)
        time_limit_s=time_limit_s,
    )


def main() -> bool:
    all_ok = True
    print(
        f"{'instance':12s} {'status':10s} {'nodes':>6s} {'proto':>6s} {'bound':>12s} "
        f"{'opt':>12s} {'noSep':>7s}  verdict"
    )
    for name in PANEL:
        opt = PANEL[name]["opt"]
        rm = RootModel(name)
        print(f"  [{name}] solving (sep)…", flush=True)
        r = run_tree(rm, opt, max_sep_rounds=12)
        # ablation (illustrative, node/time-capped so it can't dominate the gate):
        print(f"  [{name}] solving (no-sep ablation, capped)…", flush=True)
        r0 = run_tree(rm, opt, max_sep_rounds=0, max_nodes=600, time_limit_s=30.0)
        proto = PROTOTYPE_NODES[name]
        nodes = r["node_count"]
        bound = r["bound"]
        status = r["status"]

        # CORRECTNESS gates (hard).
        cert = status == "optimal"
        # dual bound is a valid UPPER bound on the max: bound >= opt - tol.
        sound_bound = bound >= opt - 1e-3 * max(1.0, abs(opt))
        # bound closed onto the oracle optimum (no false optimal below/above).
        bound_correct = abs(bound - opt) <= 1e-2 * max(1.0, abs(opt))
        # NODES gate.
        node_ok = nodes <= NODE_FACTOR * proto

        ok = cert and sound_bound and bound_correct and node_ok
        flags = []
        if not cert:
            flags.append(f"NOT-OPTIMAL({status})")
        if not sound_bound:
            flags.append("UNSOUND-BOUND")
        if not bound_correct:
            flags.append("BOUND-OFF")
        if not node_ok:
            flags.append(f"NODES>{NODE_FACTOR}x")
        verdict = "OK" if ok else "FAIL " + ",".join(flags)
        print(
            f"{name:12s} {status:10s} {nodes:6d} {proto:6d} {bound:12.4f} "
            f"{opt:12.4f} {r0['node_count']:7d}  {verdict}",
            flush=True,
        )
        all_ok = all_ok and ok
    print(
        f"\nK2 GATE (nodes-to-certify <= {NODE_FACTOR}x prototype, cert-clean): "
        f"{'PASS' if all_ok else 'FAIL'}"
    )
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
