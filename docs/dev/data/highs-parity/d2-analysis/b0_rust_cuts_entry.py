"""Phase B entry B0 (Rust side): the direct MILP driver with c-MIR / node cuts on or off.

Arms (each a fresh subprocess, A14 options as in ``d2_direct_driver_panel.py``):
``base`` (cmir off, node cuts off, the shipped route's settings), ``cmir``, ``node``,
``cmir_node``. Per arm two solves: ``max_nodes=2`` (the root bound after the cut loop)
and the full ``--time-limit`` (status/bound/incumbent). Prints an executed count and
exits non-zero when nothing ran or a certificate is wrong (CLAUDE.md §6).

Usage: python b0_rust_cuts_entry.py --instances a,b --time-limit 20 --out X.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SCRIPTS = REPO / "discopt_benchmarks" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import p42_rens_panel as R  # noqa: E402, N812

CHILD = r"""
import json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import master_parity_panel as MP
from discopt._rust import solve_milp_csc_py
from discopt.solvers.milp_simplex import _marshal_std_form
inst = L.read_instance(sys.argv[1])
TL, MAXN = float(sys.argv[3]), int(sys.argv[4])
cmir, node = sys.argv[5] == "1", sys.argv[6] == "1"
sgn = -1.0 if inst.maximize else 1.0
A_ub, b_ub, A_eq, b_eq = MP._ranged_to_ub_eq(inst.A, inst.row_lo, inst.row_hi)
s = _marshal_std_form(sgn * np.asarray(inst.c), A_ub, b_ub, A_eq, b_eq,
                      MP._bounds(inst.col_lo, inst.col_hi), inst.is_int.astype(int))
n = inst.n
t = time.perf_counter()
st, x, obj, bnd, nodes, it = solve_milp_csc_py(
    s.c, s.m, n + s.m, s.col_ptr, s.row_idx, s.vals, s.b, s.lb, s.ub, s.int_cols, n,
    sgn * float(inst.offset), MAXN, 1e-4, time_limit_s=TL, root_cuts=500,
    cut_rounds=50, cut_select=True, gmi_cuts=True, root_cut_prune=True,
    cmir_cuts=cmir, node_cuts=node, root_cut_time_s=max(0.5, 0.5 * TL))
print("RESULT " + json.dumps({"status": st, "obj": sgn * obj if np.isfinite(obj) else None,
      "bound": sgn * bnd if np.isfinite(bnd) else None, "nodes": int(nodes), "iters": int(it),
      "wall": time.perf_counter() - t}))
"""
ARMS = {"base": ("0", "0"), "cmir": ("1", "0"), "node": ("0", "1"), "cmir_node": ("1", "1")}


def run(path: str, tl: float, maxn: int, arm: str) -> dict:
    c, nd = ARMS[arm]
    p = subprocess.run([sys.executable, "-c", CHILD, path, str(SCRIPTS), str(tl), str(maxn), c, nd],
                       capture_output=True, text=True, timeout=4 * tl + 120)  # fmt: skip
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path} {arm}: rc={p.returncode} {p.stderr[-1500:]}")
    return json.loads(lines[0][len("RESULT ") :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = a.instances.split(",")
    pop = {m["name"]: m for m in R.population()}
    executed, wrong = 0, []
    with open(a.out, "w") as fh:
        for nm in keep:
            m = pop[nm]
            path = str(REPO / m["path"])
            for arm in ARMS:
                root = run(path, max(5.0, a.time_limit), 2, arm)
                full = run(path, a.time_limit, 100_000_000, arm)
                opt = m["oracle_objective"]
                if full["status"] == "optimal" and (
                    full["obj"] is None or abs(full["obj"] - opt) > 1e-4 * (1 + abs(opt))
                ):
                    wrong.append((nm, arm, full["obj"], opt))
                row = {"name": nm, "arm": arm, "opt": opt, "root": root, "full": full}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                executed += 1
                print(f"{nm:22s} {arm:9s} root_bnd={root['bound']} full={full['status']} "
                      f"bnd={full['bound']} obj={full['obj']} n={full['nodes']}",
                      flush=True)  # fmt: skip
    print(f"b0 rust: executed={executed} wrong={wrong}", flush=True)
    return 0 if executed and not wrong else 1


if __name__ == "__main__":
    sys.exit(main())
