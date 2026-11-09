"""D2: the Rust MILP *driver* called directly vs the shipped *route* (``Model.solve``).

The September A14 panel (``scratchpad/a14/rens_panel.py``, 21/38 certified) called the
driver directly -- ``solve_milp_py`` with the whole budget (``time_limit_s=TL``) and
``root_cuts=500, cut_rounds=50, cut_select=True, root_cut_time_s=0.5*TL`` -- while P0.3 /
P4.9 / P6.6 (14-16/38) measured the shipped route, whose Python wrapper
(``solver._solve_milp_simplex``) gives the driver a stall slice of ``min(0.5*rem, 10 s)``
and then re-enters or defers. This probe runs today's driver the September way on
MIPLIB-38 so the two entry points can be told apart at one build.

Per instance: MPS -> ``lp_milp_loader`` -> ``A_ub/A_eq`` -> ``milp_simplex._marshal_std_form``
-> ``solve_milp_csc_py`` with the A14 options, one fresh subprocess per solve. A result is
counted certified only when the status is ``optimal`` AND its objective matches the
published optimum within ``1e-4 (1+|opt|)`` (a mismatch is an incorrect certificate and
fails the run). Prints executed count; exits non-zero on 0 executed or any incorrect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import miplib_panel as M  # noqa: E402, N812

CHILD = r"""
import json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import master_parity_panel as MP
from discopt._rust import solve_milp_csc_py
from discopt.solvers.milp_simplex import _marshal_std_form
inst = L.read_instance(sys.argv[1])
TL = float(sys.argv[3])
sgn = -1.0 if inst.maximize else 1.0
A_ub, b_ub, A_eq, b_eq = MP._ranged_to_ub_eq(inst.A, inst.row_lo, inst.row_hi)
bounds = MP._bounds(inst.col_lo, inst.col_hi)
s = _marshal_std_form(sgn * np.asarray(inst.c), A_ub, b_ub, A_eq, b_eq, bounds,
                      inst.is_int.astype(int))
n = inst.n
t = time.perf_counter()
st, x, obj, bnd, nodes, it = solve_milp_csc_py(
    s.c, s.m, n + s.m, s.col_ptr, s.row_idx, s.vals, s.b, s.lb, s.ub, s.int_cols, n,
    sgn * float(inst.offset), 10_000_000, 1e-4, time_limit_s=TL, root_cuts=500,
    cut_rounds=50, cut_select=True, gmi_cuts=True, root_cut_prune=True,
    root_cut_time_s=max(0.5, 0.5 * TL))
w = time.perf_counter() - t
print("RESULT " + json.dumps({"status": st, "obj": sgn * obj if np.isfinite(obj) else None,
      "bound": sgn * bnd if np.isfinite(bnd) else None, "nodes": int(nodes), "wall": w}))
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mem = M._load_panel(M.PANEL_JSON)["members"]
    if a.instances:
        keep = set(a.instances.split(","))
        mem = [m for m in mem if m["name"] in keep]
    rows, executed, incorrect = [], 0, []
    for m in mem:
        p = subprocess.run([sys.executable, "-c", CHILD, str(REPO / m["path"]), str(HERE.parent),
                            str(a.time_limit)], capture_output=True, text=True,
                           timeout=4 * a.time_limit + 120)  # fmt: skip
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        if p.returncode != 0 or len(lines) != 1:
            raise RuntimeError(f"{m['name']}: rc={p.returncode} {p.stderr[-1500:]}")
        r = json.loads(lines[0][len("RESULT ") :])
        opt = m["oracle_objective"]
        cert = r["status"] == "optimal"
        if cert and (r["obj"] is None or abs(r["obj"] - opt) > 1e-4 * (1 + abs(opt))):
            incorrect.append((m["name"], r["obj"], opt))
        executed += 1
        rows.append({"name": m["name"], "opt": opt, "certified": cert, **r})
        print(f"{m['name']:22s} {r['status']:10s} nodes={r['nodes']:8d} wall={r['wall']:6.2f} "
              f"obj={r['obj']} opt={opt}", flush=True)  # fmt: skip
        Path(a.out).write_text(json.dumps({"rows": rows, "incorrect": incorrect}, indent=1))
    cert = sum(r["certified"] for r in rows)
    print(f"d2 direct: executed={executed} certified={cert} incorrect={len(incorrect)}", flush=True)
    return 0 if executed and not incorrect else 1


if __name__ == "__main__":
    sys.exit(main())
