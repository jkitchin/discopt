"""Phase B: which cut families does SCIP apply on the rows HiGHS closes in-tree?

One SCIP run (pyscipopt, defaults, threads 1) per instance from the MPS. The run's
statistics are written to a file, and the "Separators" and "Constraint" cut tables are
parsed for cuts applied per separator. Prints an executed count; exits non-zero on 0.
Usage: python b1_scip_separators.py --instances a,b --time-limit 60 --out X.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
import p42_rens_panel as R  # noqa: E402, N812


def parse_stats(text: str) -> dict:
    out: dict = {"separators": {}, "constraints": {}}
    sec = None
    for ln in text.splitlines():
        if ln.startswith("Separators         :"):
            sec = "separators"
            continue
        if ln.startswith("Constraints        :"):
            sec = "constraints"
            continue
        if sec and ln and not ln.startswith("  "):
            sec = None
        if sec and ln.startswith("  "):
            name, _, rest = ln.partition(":")
            nums = rest.split()
            if not nums or not re.match(r"^-?[\d.]+$", nums[0]):
                continue
            out[sec][name.strip()] = nums
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pyscipopt

    pop = {m["name"]: m for m in R.population()}
    executed = 0
    with open(a.out, "w") as fh:
        for nm in a.instances.split(","):
            m = pyscipopt.Model()
            m.hideOutput()
            m.readProblem(str(REPO / pop[nm]["path"]))
            m.setParam("limits/time", a.time_limit)
            m.setParam("limits/gap", 1e-4)
            m.optimize()
            fd, path = tempfile.mkstemp(suffix=".stats")
            os.close(fd)
            m.writeStatistics(path)
            text = Path(path).read_text()
            os.unlink(path)
            hdr = next((ln for ln in text.splitlines() if ln.startswith("Separators")), "")
            row = {"name": nm, "status": m.getStatus(), "obj": m.getObjVal() if m.getNSols() else None,
                   "bound": m.getDualbound(), "nodes": m.getNNodes(), "time": m.getSolvingTime(),
                   "sep_header": hdr, **parse_stats(text)}  # fmt: skip
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            executed += 1
            print(f"{nm:22s} {row['status']:10s} nodes={row['nodes']} t={row['time']:.1f} "
                  f"bound={row['bound']}", flush=True)  # fmt: skip
    print(f"scip separators: executed={executed}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
