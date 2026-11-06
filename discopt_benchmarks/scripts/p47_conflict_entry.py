"""highs-parity P4.7 entry: how small are the infeasibility proofs of pruned nodes?

Each MIPLIB-38 instance is solved on the Rust MILP route (``DISCOPT_LP_MILP_BACKEND=rust``)
at ``--time-limit`` in a fresh subprocess with ``DISCOPT_PROFILE=1``. The driver's
bound-neutral counters ``ConflictInfeasNodes`` / ``ConflictSmall`` /
``ConflictLocalBoundsSum`` (``bnb/milp_driver.rs``, the verified-Farkas arm) are summed
over every driver run of the solve: per verified-infeasible node, the number of LOCAL
bound changes (node bound tighter than the root's on the side the ray uses) its proof
rests on, the smaller over the two ray orientations. Hypothesis (plan P4.7): >= 30 % of
infeasible nodes need <= 10. Prints executed count; exits non-zero on 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_panel as P  # noqa: E402, N812
import miplib_panel as M  # noqa: E402, N812

_CTR = re.compile(r"^\s+(Conflict\w+)\s+(\d+)$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P._assert_discopt_under_test()
    mem = M._load_panel(M.PANEL_JSON)["members"]
    if a.instances:
        keep = set(a.instances.split(","))
        mem = [m for m in mem if m["name"] in keep]
    rows, executed = [], 0
    for m in mem:
        env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1",
                   PYTHONUNBUFFERED="1")  # fmt: skip
        cmd = [sys.executable, str(HERE.parent / "lp_milp_panel.py"), "worker",
               str(REPO / m["path"]), "R", str(a.time_limit), str(P.GAP), "100000000"]  # fmt: skip
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=4 * a.time_limit + 180)  # fmt: skip
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        if p.returncode != 0 or len(lines) != 1:
            raise RuntimeError(f"{m['name']}: rc={p.returncode} {p.stderr[-1500:]}")
        res = json.loads(lines[0][len("RESULT ") :])
        ctr: dict[str, int] = {}
        for ln in p.stderr.splitlines():
            mm = _CTR.match(ln)
            if mm:
                ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
        executed += 1
        n = ctr.get("ConflictInfeasNodes", 0)
        row = {"name": m["name"], "status": res["status"], "nodes": res["nodes"],
               "route": res.get("route"), "counters": ctr,
               "small_frac": (ctr.get("ConflictSmall", 0) / n) if n else None,
               "mean_k": (ctr.get("ConflictLocalBoundsSum", 0) / n) if n else None}  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
        Path(a.out).write_text(json.dumps({"header": P._header(), "rows": rows}, indent=1))
    tot = sum(r["counters"].get("ConflictInfeasNodes", 0) for r in rows)
    small = sum(r["counters"].get("ConflictSmall", 0) for r in rows)
    print(f"p47 entry: executed={executed} infeasible_nodes={tot} small={small} "
          f"frac={(small / tot) if tot else None}", flush=True)  # fmt: skip
    return 0 if executed and tot else 1


if __name__ == "__main__":
    sys.exit(main())
