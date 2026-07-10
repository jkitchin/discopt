"""CUTS-1 verification — fixed-node-budget flag ON/OFF node-count check on
nvs17/19/24 (default solve path), current main.

Kill-criterion framing from the task mandate: oracle-injected / flag-enabled
aggregation c-MIR must reduce node counts >=5x on at least one of the three at a
fixed budget. Expectation from committed CUT-1 (2026-07-06): bit-identical ON vs
OFF (the separator self-disables; discopt's root is already ~99.9% closed).

Runs each (instance, flag) in a fresh subprocess so env + JAX state are clean.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

NL_DIR = Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL = ["nvs17", "nvs19", "nvs24"]
OPT = {"nvs17": -1100.4, "nvs19": -1098.4, "nvs24": -1033.2}
BUDGET_NODES = 1500
CAP_SECONDS = 120.0

CHILD = r"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
nl_path, flag, max_nodes, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
os.environ["DISCOPT_CMIR_AGGREGATION"] = flag
import discopt.modeling as dm
t0 = time.monotonic()
model = dm.from_nl(nl_path)
res = model.solve(time_limit=cap, max_nodes=max_nodes)
print(json.dumps({
    "status": str(res.status),
    "objective": res.objective,
    "bound": res.bound,
    "node_count": res.node_count,
    "wall": time.monotonic() - t0,
}))
"""


def run_one(name: str, flag: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "python")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(NL_DIR / f"{name}.nl"),
            flag,
            str(BUDGET_NODES),
            str(CAP_SECONDS),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=CAP_SECONDS + 120,
    )
    if out.returncode != 0:
        return {"error": out.stderr.strip()[-500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    print(
        f"CUTS-1 node-count check — DISCOPT_CMIR_AGGREGATION on/off, "
        f"budget={BUDGET_NODES} nodes, cap={CAP_SECONDS:.0f}s\n"
    )
    hdr = (
        f"{'instance':<8} {'opt':>10} | {'bound OFF':>12} {'nodes OFF':>9} | "
        f"{'bound ON':>12} {'nodes ON':>9} | identical?  incorrect?"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name in PANEL:
        off = run_one(name, "0")
        on = run_one(name, "1")
        row = {"instance": name, "opt": OPT[name], "off": off, "on": on}
        rows.append(row)
        if "error" in off or "error" in on:
            print(
                f"{name:<8} ERROR off={off.get('error', '')[:120]} on={on.get('error', '')[:120]}"
            )
            continue
        ident = off["bound"] == on["bound"] and off["node_count"] == on["node_count"]
        tol = 1e-4 * (1.0 + abs(OPT[name]))
        bad = any(
            r["objective"] is not None and r["objective"] < OPT[name] - tol for r in (off, on)
        )
        print(
            f"{name:<8} {OPT[name]:>10.1f} | {off['bound']:>12.4f} {off['node_count']:>9} | "
            f"{on['bound']:>12.4f} {on['node_count']:>9} | "
            f"{'YES (bit-identical)' if ident else 'NO'}  {'YES' if bad else 'no'}"
        )
    out = (
        Path(__file__).resolve().parent.parent
        / "results"
        / (
            "cuts1_cmir_flag_onoff_nodes_"
            + __import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S")
            + ".json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nPersisted -> {out}")


if __name__ == "__main__":
    main()
