"""highs-parity P2.1: where the Rust LP route spends its time, per L-panel instance.

For each instance (``--instances``, else the top ``--top`` Rust/HiGHS wall ratios of a P0.3
run JSON) a child process solves it on the Rust route (``DISCOPT_LP_MILP_BACKEND=rust``)
with ``DISCOPT_PROFILE=1``: every ``solve_lp_warm_csc_py`` call is timed from Python, and
the Rust phase dump each call prints to stderr is parsed and summed. Raw HiGHS solves the
same LP for its wall and ``simplex_iteration_count``.

Buckets (share of the route's wall): ``python`` (wall outside the Rust LP calls: model
read-in, marshalling, the certificate ladder), ``pricing`` (PriceBtran + PriceSweep),
``ftran`` (AlphaFtran), ``update`` (FtUpdate), ``refactor`` (Refactorize + LuSymbolic +
LuNumeric), ``dual_other`` (DualPrepare + DualRecompute), ``rust_other`` (Rust LP time not
in a timed leaf phase). ``DualPivotLoop`` contains the leaf phases and is not added.
Each row is attributed to its largest bucket. Prints the number of instances profiled and
exits non-zero on 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

LEAVES = {
    "pricing": ("PriceBtran", "PriceSweep"),
    "ftran": ("AlphaFtran",),
    "update": ("FtUpdate",),
    "refactor": ("Refactorize", "LuSymbolic", "LuNumeric"),
    "dual_other": ("DualPrepare", "DualRecompute"),
}
_PHASE = re.compile(r"^\s+(\w+)\s+(\d+) calls\s+([\d.]+) ms")
_CTR = re.compile(r"^\s+(\w+)\s+(\d+)$")

_CHILD = r"""
import sys, os, json, time, tempfile
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import discopt._rust as R
calls = {"n": 0, "s": 0.0}
real = R.solve_lp_warm_csc_py
def timed(*a, **k):
    t = time.perf_counter(); out = real(*a, **k); calls["s"] += time.perf_counter() - t
    calls["n"] += 1; return out
R.solve_lp_warm_csc_py = timed
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
t = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[3]))
wall = time.perf_counter() - t
st = r.solver_stats or {}
print("RESULT " + json.dumps({"status": str(r.status), "wall": wall, "rust_calls": calls["n"],
      "rust_s": calls["s"], "iters": st.get("lp/iters")}), flush=True)
"""


def profile_rust(path: str, time_limit: float) -> dict:
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    cmd = [sys.executable, "-c", _CHILD, path, str(HERE.parent), str(time_limit)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=4 * time_limit + 120)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path}: rc={p.returncode}\n{p.stderr[-2000:]}")
    res = json.loads(lines[0][len("RESULT ") :])
    phases: dict[str, float] = {}
    ctrs: dict[str, int] = {}
    for ln in p.stderr.splitlines():
        m = _PHASE.match(ln)
        if m:
            phases[m.group(1)] = phases.get(m.group(1), 0.0) + float(m.group(3)) / 1e3
            continue
        m = _CTR.match(ln)
        if m:
            ctrs[m.group(1)] = ctrs.get(m.group(1), 0) + int(m.group(2))
    res["phases_s"] = phases
    res["counters"] = {k: v for k, v in ctrs.items() if v}
    return res


def highs_raw(path: str, time_limit: float) -> dict:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.readModel(path)
    t = time.perf_counter()
    h.run()
    return {"wall": time.perf_counter() - t, "iters": int(h.getInfo().simplex_iteration_count),
            "status": h.modelStatusToString(h.getModelStatus())}  # fmt: skip


def buckets(r: dict) -> dict:
    ph = r["phases_s"]
    out = {k: sum(ph.get(n, 0.0) for n in names) for k, names in LEAVES.items()}
    out["rust_other"] = max(0.0, r["rust_s"] - sum(out.values()))
    out["python"] = max(0.0, r["wall"] - r["rust_s"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--baseline", default="docs/dev/data/highs-parity/baseline-dcdac02/lmi.json")
    p.add_argument("--manifest", default="discopt_benchmarks/data/lp_milp/manifest.json")
    p.add_argument("--top", type=int, default=13)
    p.add_argument("--instances", default="")
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    man = {m["name"]: m for m in json.loads(Path(a.manifest).read_text())["members"]}
    if a.instances:
        names = a.instances.split(",")
    else:
        rows = [r for r in json.loads(Path(a.baseline).read_text())["rows"] if r["panel"] == "L"]
        rows.sort(key=lambda r: r["R"]["wall_med"] / r["H"]["wall_med"], reverse=True)
        names = [r["name"] for r in rows[: a.top]]
    table = []
    for i, name in enumerate(names, 1):
        path = str(REPO / man[name]["path"])
        rr = profile_rust(path, a.time_limit)
        hh = highs_raw(path, a.time_limit)
        b = buckets(rr)
        top = max(b, key=b.get)
        row = {"name": name, "rust_wall": rr["wall"], "rust_status": rr["status"],
               "rust_calls": rr["rust_calls"], "rust_iters": rr["iters"],
               "highs_wall": hh["wall"], "highs_iters": hh["iters"],
               "share": {k: v / rr["wall"] for k, v in b.items()}, "bucket": top,
               "counters": rr["counters"]}  # fmt: skip
        table.append(row)
        print(f"[{i}/{len(names)}] {name:10s} wall {rr['wall']:6.2f}s (HiGHS {hh['wall']:.3f}s) "
              f"iters {rr['iters']} vs {hh['iters']} calls {rr['rust_calls']} -> {top} "
              + " ".join(f"{k}={v:.0%}"
                         for k, v in sorted(row["share"].items(), key=lambda kv: -kv[1])),
              flush=True)  # fmt: skip
    Path(a.out).write_text(json.dumps(table, indent=1) + "\n")
    print(f"profiled={len(table)}", flush=True)
    return 0 if table else 1


if __name__ == "__main__":
    sys.exit(main())
