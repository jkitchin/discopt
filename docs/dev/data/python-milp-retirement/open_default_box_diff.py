"""Differential for DISCOPT_MILP_RUST_OPEN_DEFAULT_BOX (Python-MILP-tree retirement).

Every in-repo corpus instance (python/tests/data/minlplib_nl/*.nl) is solved twice in
fresh subprocesses, flag OFF and ON back to back with the order alternating per
instance (CLAUDE.md §9), both on the Rust MILP route with DISCOPT_MILP_NO_STALL_TREE=1
(the configuration the two flags ship in together). Checked against
ref/minlplib/minlplib.solu (=opt= / =best=): no certified objective off the optimum, no
bound past it (minimize: bound <= opt; maximize: bound >= opt), no certification lost.
Prints per-instance rows and the executed-check count; exits 1 on any violation or when
nothing ran.

    python -u docs/dev/data/python-milp-retirement/open_default_box_diff.py OUT.jsonl [TL]
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CORPUS = sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl"))
OUT = Path(sys.argv[1])
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
SOLU = {}
for ln in (REPO / "ref/minlplib/minlplib.solu").read_text().splitlines():
    m = re.match(r"=(opt|best)=\s+(\S+)\s+(\S+)", ln)
    if m:
        SOLU[m.group(2)] = (m.group(1), float(m.group(3)))

PROBE = """
import json, sys, time, warnings
warnings.simplefilter("ignore")
import discopt.modeling as dm
from discopt.modeling.core import ObjectiveSense
m = dm.from_nl(sys.argv[1])
t = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]))
print("ROW " + json.dumps({"status": r.status, "cert": bool(r.gap_certified),
    "obj": r.objective, "bound": r.bound, "wall": time.perf_counter() - t,
    "max": m._objective is not None and m._objective.sense == ObjectiveSense.MAXIMIZE,
    "route": (r.algorithm_route or "")[:80]}))
"""


def solve(path, flag):
    env = {**os.environ, "DISCOPT_LP_MILP_BACKEND": "rust", "DISCOPT_MILP_NO_STALL_TREE": "1",
           "DISCOPT_MILP_RUST_OPEN_DEFAULT_BOX": flag}  # fmt: skip
    p = subprocess.run([sys.executable, "-c", PROBE, str(path), str(TL)], env=env,
                       capture_output=True, text=True, timeout=10 * TL + 300)  # fmt: skip
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
    return json.loads(rows[0][4:]) if rows else {"error": p.stderr[-800:]}


def problems(r, name):
    out = []
    if "error" in r or name not in SOLU:
        return out
    kind, opt = SOLU[name]
    tol = 1e-4 * max(1.0, abs(opt))
    if kind == "opt" and r["cert"] and r["obj"] is not None and abs(r["obj"] - opt) > tol:
        out.append(f"certified {r['obj']} != opt {opt}")
    if kind == "opt" and r["bound"] is not None:
        past = r["bound"] < opt - tol if r["max"] else r["bound"] > opt + tol
        if past:
            out.append(f"bound {r['bound']} past opt {opt}")
    return out


checks, bad = 0, []
with OUT.open("w") as fh:
    for i, path in enumerate(CORPUS):
        order = ("0", "1") if i % 2 == 0 else ("1", "0")
        arms = {f: solve(path, f) for f in order}
        row = {"instance": path.stem, "off": arms["0"], "on": arms["1"]}
        for arm in ("off", "on"):
            for pb in problems(row[arm], path.stem):
                bad.append((path.stem, arm, pb))
            checks += "error" not in row[arm]
        if row["off"].get("cert") and not row["on"].get("cert"):
            bad.append((path.stem, "on", "certification lost"))
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        o, n = row["off"], row["on"]
        diff = "" if (o.get("status"), o.get("cert"), o.get("obj")) == (n.get("status"), n.get("cert"), n.get("obj")) else "  <-- differs"
        print(f"{path.stem:24s} OFF {o.get('status')}/{o.get('cert')} {o.get('obj')} | ON {n.get('status')}/{n.get('cert')} {n.get('obj')}{diff}", flush=True)
print(f"executed checks {checks}; violations {len(bad)}", flush=True)
for b in bad:
    print("  !!!", b, flush=True)
sys.exit(0 if checks and not bad else 1)
