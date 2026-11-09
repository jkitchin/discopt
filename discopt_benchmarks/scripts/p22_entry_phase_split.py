"""highs-parity P2.2 entry: phase-1 share of the Rust LP route's cold pivots, per L instance.

A crash basis can only remove phase-1 work (it changes the starting basis, not phase 2's
pricing), so the hypothesis "a crash cuts cold iterations >= 30 %" needs phase 1 to be well
over 30 % of the pivots first. Each instance is solved on the Rust route
(``DISCOPT_LP_MILP_BACKEND=rust``) in a fresh process with ``DISCOPT_PROFILE=1``; the
``Phase1Pivots`` / ``Phase2Pivots`` counters it dumps are summed (every LP of the solve).
Raw HiGHS's ``simplex_iteration_count`` on the same LP is recorded beside them. Prints the
number of instances measured and exits non-zero on 0.
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
_CTR = re.compile(
    r"^\s+(Phase1Pivots|Phase2Pivots|DegeneratePivots|BoundFlips|CrashTriangularColumns)"
    r"\s+(\d+)$"
)
_CHILD = r"""
import sys, os, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
r = m.solve(time_limit=float(sys.argv[3]))
print("RESULT", str(r.status), flush=True)
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", default="discopt_benchmarks/data/lp_milp/manifest.json")
    p.add_argument("--panel", default="L")
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import highspy

    members = [m for m in json.loads(Path(a.manifest).read_text())["members"]
               if m["panel"] == a.panel]  # fmt: skip
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    rows = []
    for i, mem in enumerate(members, 1):
        path = str(REPO / mem["path"])
        pr = subprocess.run([sys.executable, "-c", _CHILD, path, str(HERE.parent),
                             str(a.time_limit)], env=env, capture_output=True, text=True,
                            timeout=4 * a.time_limit + 120)  # fmt: skip
        status = [ln for ln in pr.stdout.splitlines() if ln.startswith("RESULT")]
        if pr.returncode != 0 or len(status) != 1:
            raise RuntimeError(f"{mem['name']}: rc={pr.returncode}\n{pr.stderr[-1500:]}")
        ctr: dict[str, int] = {}
        for ln in pr.stderr.splitlines():
            mm = _CTR.match(ln)
            if mm:
                ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(path)
        h.run()
        p1, p2 = ctr.get("Phase1Pivots", 0), ctr.get("Phase2Pivots", 0)
        row = {"name": mem["name"], "status": status[0].split()[1], "phase1": p1, "phase2": p2,
               "degenerate": ctr.get("DegeneratePivots", 0), "flips": ctr.get("BoundFlips", 0),
               "highs_iters": int(h.getInfo().simplex_iteration_count), "m": mem["m"]}  # fmt: skip
        rows.append(row)
        tot = p1 + p2
        print(
            f"[{i}/{len(members)}] {mem['name']:10s} phase1 {p1:6d} phase2 {p2:6d} "
            f"share {p1 / tot if tot else 0:.2f} highs {row['highs_iters']}",
            flush=True,
        )
    p1 = sum(r["phase1"] for r in rows)
    tot = p1 + sum(r["phase2"] for r in rows)
    shares = sorted(r["phase1"] / (r["phase1"] + r["phase2"]) for r in rows
                    if r["phase1"] + r["phase2"])  # fmt: skip
    summ = {
        "instances": len(rows),
        "phase1_total": p1,
        "pivots_total": tot,
        "phase1_share_pooled": p1 / tot if tot else None,
        "phase1_share_median": shares[len(shares) // 2] if shares else None,
        "rust_pivots_over_highs_iters": tot / max(1, sum(r["highs_iters"] for r in rows)),
    }
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1) + "\n")
    print("summary", json.dumps(summ), flush=True)
    print(f"measured={len(rows)}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
