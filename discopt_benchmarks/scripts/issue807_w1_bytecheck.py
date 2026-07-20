#!/usr/bin/env python
"""Issue #807 / W1 gate — persistent-LP tree integration (OA-only), rsyn* subset.

Runs the seeded best-bound tree OA-only (max_sep_rounds=0) with DISCOPT_CVX_NATIVELP
OFF (today's per-node cold solve_node_cut) vs ON (the shared persistent LP,
bounds-in-place dual-warm) on the scoped rsyn* family, and checks:
  1. CERT-CLEAN under the flag (the integration is sound): status honest, dual
     bound is a valid bound vs the oracle, never a false optimal.
  2. WALL: the ON path is faster at the same setting (W0 measured >=2x per node;
     it should show as less wall for a comparable tree).

Search-order regime (flavor ii): the warm path lands on different LP vertices, so
node_count may differ; the guard is cert-clean, not node parity. syn40m is OUT OF
SCOPE (#807 W0: OA-round-bound) — not run here.

Flag is process-global (read once), so ON/OFF run in separate subprocesses.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

RSYN = ["rsyn0805m", "rsyn0810m", "rsyn0815m"]  # config-B prototyped subset
MAX_NODES = 600  # representative of the cut-driven tree size (W2); OA-only bloats beyond
TIME_LIMIT = 120.0


def run_child() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import discopt._rust as _rust  # noqa: E402
    from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402
    from issue798_k1_bytecheck import build_convex_arrays  # noqa: E402

    flag = os.environ.get("DISCOPT_CVX_NATIVELP", "0")
    print(f"# DISCOPT_CVX_NATIVELP={flag}  (OA-only, seeded)")
    print(f"{'instance':12s} {'status':10s} {'nodes':>7s} {'wall_s':>8s} "
          f"{'bound':>12s} {'opt':>12s}  cert")
    tw = 0.0
    bad = 0
    for name in RSYN:
        opt = PANEL[name]["opt"]
        rm = RootModel(name)
        arrays = build_convex_arrays(rm, rm.lb, rm.ub)
        t = time.perf_counter()
        r = _rust.solve_convex_tree_py(
            **arrays, max_nodes=MAX_NODES, gap_tol=1e-4, int_tol=1e-5, oa_tol=1e-6,
            max_oa_rounds=60, max_sep_rounds=0, fbbt_rounds=20,
            initial_incumbent=float(opt), time_limit_s=TIME_LIMIT,
        )
        dt = time.perf_counter() - t
        tw += dt
        bound = r["bound"]
        status = r["status"]
        # Sound: dual bound never below the oracle optimum (upper bound for max).
        sound = bound >= opt - 1e-3 * max(1.0, abs(opt))
        # If it certified, the bound must have closed onto the oracle.
        closed = status != "optimal" or abs(bound - opt) <= 1e-2 * max(1.0, abs(opt))
        cert = sound and closed
        bad += 0 if cert else 1
        print(f"{name:12s} {status:10s} {r['node_count']:7d} {dt:8.2f} "
              f"{bound:12.4f} {opt:12.4f}  {cert}", flush=True)
    print(f"TOTAL wall={tw:.1f}s  cert-clean={'YES' if bad == 0 else 'NO(' + str(bad) + ')'}",
          flush=True)


def main() -> bool:
    if os.environ.get("_W1_CHILD") == "1":
        run_child()
        return True
    import subprocess

    outs = {}
    for flag in ("0", "1"):
        env = dict(os.environ, DISCOPT_CVX_NATIVELP=flag, _W1_CHILD="1")
        print(f"\n===== DISCOPT_CVX_NATIVELP={flag} =====", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                           capture_output=True, text=True)
        print(p.stdout, end="")
        if p.returncode != 0:
            print(p.stderr[-2000:])
        outs[flag] = p.stdout

    def parse(out):
        wall = None
        cert = None
        for line in out.splitlines():
            if line.startswith("TOTAL"):
                wall = float(line.split("wall=")[1].split("s")[0])
                cert = "cert-clean=YES" in line
        return wall, cert

    w_off, c_off = parse(outs["0"])
    w_on, c_on = parse(outs["1"])
    print("\n================ W1 GATE ================")
    speedup = (w_off / w_on) if (w_on and w_off) else float("nan")
    ok = bool(c_off) and bool(c_on) and (w_on is not None and w_off is not None and w_on <= w_off)
    print(f"cert-clean: OFF={c_off} ON={c_on}")
    print(f"wall: OFF={w_off}s  ON={w_on}s  speedup={speedup:.2f}x  (ON must be cert-clean AND <= OFF)")
    print(f"\nW1 {'PASS' if ok else 'REVIEW'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
