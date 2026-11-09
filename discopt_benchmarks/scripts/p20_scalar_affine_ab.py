"""highs-parity P2.0 (ii): interleaved A/B of the scalar-affine fast path, per route.

Arms: ``on`` (``scalar_linear_of`` as shipped) and ``off`` (patched to decline, so every
row goes through ``affine_of`` as before P2.0 (ii)). Each solve is a fresh subprocess on
the Rust (``DISCOPT_LP_MILP_BACKEND=rust``) or HiGHS route; arms alternate with rotated
order per rep. Checks per (instance, route): status, certificate and objective identical
across every ``on``/``off`` solve (bit-for-bit). Prints the comparison count, per-route
median wall ratio on/off with its spread, and exits non-zero on 0 comparisons or any
difference.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]

_CHILD = r"""
import sys, os, json, time, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import discopt._relax.affine_array as AA
assert hasattr(AA, "scalar_linear_of"), AA.__file__
if sys.argv[4] == "off":
    AA.scalar_linear_of = lambda *a, **k: None
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
t = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[3]))
print("RESULT " + json.dumps({"status": str(r.status), "cert": bool(r.gap_certified),
      "obj": r.objective, "nodes": int(r.node_count), "wall": time.perf_counter() - t}), flush=True)
"""

ROUTES = {"R": {"DISCOPT_LP_MILP_BACKEND": "rust"}, "H": {}}


def _solve(path: str, tl: float, arm: str, route: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_LP_MILP_BACKEND"}
    env.update(ROUTES[route])
    cmd = [sys.executable, "-c", _CHILD, path, str(HERE.parent), str(tl), arm]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=4 * tl + 120)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path} {arm} {route}: rc={p.returncode}\n{p.stderr[-1500:]}")
    return json.loads(lines[0][len("RESULT ") :])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--baseline", default="docs/dev/data/highs-parity/baseline-dcdac02/lmi.json")
    p.add_argument("--manifest", default="discopt_benchmarks/data/lp_milp/manifest.json")
    p.add_argument("--top", type=int, default=13)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    man = {m["name"]: m for m in json.loads(Path(a.manifest).read_text())["members"]}
    rows = [r for r in json.loads(Path(a.baseline).read_text())["rows"] if r["panel"] == "L"]
    rows.sort(key=lambda r: r["R"]["wall_med"] / r["H"]["wall_med"], reverse=True)
    names = [r["name"] for r in rows[: a.top]]
    load0 = os.getloadavg()
    out, comparisons, diffs = [], 0, []
    for i, name in enumerate(names, 1):
        path = str(REPO / man[name]["path"])
        for route in ROUTES:
            per: dict[str, list] = {"on": [], "off": []}
            for rep in range(a.reps):
                for arm in ("on", "off") if rep % 2 == 0 else ("off", "on"):
                    per[arm].append(_solve(path, a.time_limit, arm, route))
            ref = per["off"][0]
            for r in per["on"] + per["off"][1:]:
                comparisons += 1
                key = ("status", "cert", "obj")
                if any(r[k] != ref[k] for k in key) or (r["cert"] and r["nodes"] != ref["nodes"]):
                    diffs.append((name, route, ref, r))
            won = statistics.median(r["wall"] for r in per["on"])
            woff = statistics.median(r["wall"] for r in per["off"])
            out.append({"name": name, "route": route, "on": per["on"], "off": per["off"],
                        "wall_on": won, "wall_off": woff})  # fmt: skip
            print(f"[{i}/{len(names)}] {name:10s} {route} on {won:6.2f}s off {woff:6.2f}s "
                  f"ratio {won / woff:.3f}", flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"load_start": load0, "rows": out}, indent=1))
    summary = {}
    for route in ROUTES:
        rs = [r["wall_on"] / r["wall_off"] for r in out if r["route"] == route]
        mine = [r for r in out if r["route"] == route]
        summary[route] = {
            "median_ratio": statistics.median(rs),
            "sd_ratio": statistics.pstdev(rs),
            "sum_on": sum(r["wall_on"] for r in mine),
            "sum_off": sum(r["wall_off"] for r in mine),
        }
        print(f"{route}: median on/off {summary[route]['median_ratio']:.3f} "
              f"(sd {summary[route]['sd_ratio']:.3f}) total {summary[route]['sum_off']:.1f}s -> "
              f"{summary[route]['sum_on']:.1f}s", flush=True)  # fmt: skip
    doc = {"load_start": load0, "load_end": os.getloadavg(), "comparisons": comparisons,
           "diffs": diffs, "summary": summary, "rows": out}  # fmt: skip
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"comparisons={comparisons} diffs={len(diffs)} load {load0} -> {os.getloadavg()}")
    return 0 if comparisons and not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
