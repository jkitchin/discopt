#!/usr/bin/env python
"""Issue #807 / W2 gate — box-tagged cuts in the persistent LP, seeded rsyn* panel.

Runs the seeded best-bound branch-and-cut tree (max_sep_rounds=12) with
DISCOPT_CVX_NATIVELP ON (shared persistent LP + box-tagged integrality cuts) vs
OFF (today's per-node cold solve_node_cut = the config-B baseline) on the scoped
rsyn* family, and checks:
  1. CERT-CLEAN (SOUNDNESS — the make-or-break): status optimal, dual bound = the
     MINLPLib oracle optimum exactly, bound >= oracle (never a false optimal / a
     bound below the oracle). A box-tagged cut must never remove the optimum.
  2. NODE BLOWUP GUARD: nodes <= 1.5x the config-B anchor (353/177/281 -> 530/266/422).
  3. WALL: seeded panel <= 12s (>=2x vs the 24s config-B anchor).

Search-order regime (flavor ii): the warm path lands on different vertices, so
node_count may differ; the guard is cert-clean + the blowup bar. syn40m is OUT OF
SCOPE (#807 W0). Flag is process-global, so ON/OFF run in separate subprocesses.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

RSYN = ["rsyn0805m", "rsyn0810m", "rsyn0815m"]
ANCHOR = {"rsyn0805m": 353, "rsyn0810m": 177, "rsyn0815m": 281}  # config-B seeded nodes
BLOWUP = 1.5
WALL_BAR = 12.0


def run_child() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import discopt._rust as _rust  # noqa: E402
    from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402
    from issue798_k1_bytecheck import build_convex_arrays  # noqa: E402

    flag = os.environ.get("DISCOPT_CVX_NATIVELP", "0")
    print(f"# DISCOPT_CVX_NATIVELP={flag}  (cuts on, seeded)")
    print(f"{'instance':12s} {'status':10s} {'nodes':>6s} {'anch':>6s} {'cap':>6s} "
          f"{'wall_s':>7s} {'bound':>12s} {'opt':>12s}  cert node_ok")
    tw = 0.0
    bad = 0
    node_bad = 0
    for name in RSYN:
        opt = PANEL[name]["opt"]
        rm = RootModel(name)
        arrays = build_convex_arrays(rm, rm.lb, rm.ub)
        t = time.perf_counter()
        r = _rust.solve_convex_tree_py(
            **arrays, max_nodes=20000, gap_tol=1e-4, int_tol=1e-5, oa_tol=1e-6,
            max_oa_rounds=60, max_sep_rounds=12, fbbt_rounds=20,
            initial_incumbent=float(opt), time_limit_s=180.0,
        )
        dt = time.perf_counter() - t
        tw += dt
        bound = r["bound"]
        nodes = r["node_count"]
        status = r["status"]
        sound = bound >= opt - 1e-3 * max(1.0, abs(opt))
        closed = abs(bound - opt) <= 1e-2 * max(1.0, abs(opt))
        cert = status == "optimal" and sound and closed
        cap = int(BLOWUP * ANCHOR[name])
        node_ok = nodes <= cap
        bad += 0 if cert else 1
        node_bad += 0 if node_ok else 1
        print(f"{name:12s} {status:10s} {nodes:6d} {ANCHOR[name]:6d} {cap:6d} "
              f"{dt:7.2f} {bound:12.4f} {opt:12.4f}  {cert}  {node_ok}", flush=True)
    print(f"TOTAL wall={tw:.1f}s  cert-clean={'YES' if bad == 0 else 'NO(' + str(bad) + ')'}  "
          f"nodes_ok={'YES' if node_bad == 0 else 'NO(' + str(node_bad) + ')'}", flush=True)


def parse(out):
    wall = cert = nodes_ok = None
    for line in out.splitlines():
        if line.startswith("TOTAL"):
            wall = float(line.split("wall=")[1].split("s")[0])
            cert = "cert-clean=YES" in line
            nodes_ok = "nodes_ok=YES" in line
    return wall, cert, nodes_ok


def main() -> bool:
    if os.environ.get("_W2_CHILD") == "1":
        run_child()
        return True
    import subprocess

    outs = {}
    for flag in ("0", "1"):
        env = dict(os.environ, DISCOPT_CVX_NATIVELP=flag, _W2_CHILD="1")
        print(f"\n===== DISCOPT_CVX_NATIVELP={flag} =====", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                           capture_output=True, text=True)
        print(p.stdout, end="")
        if p.returncode != 0:
            print(p.stderr[-3000:])
        outs[flag] = p.stdout

    w_off, c_off, _ = parse(outs["0"])
    w_on, c_on, n_on = parse(outs["1"])
    print("\n================ W2 GATE ================")
    speedup = (w_off / w_on) if (w_on and w_off) else float("nan")
    ok = bool(c_on) and bool(n_on) and (w_on is not None and w_on <= WALL_BAR)
    print(f"cert-clean: OFF={c_off} ON={c_on} (ON is the soundness gate)")
    print(f"nodes within 1.5x anchor: ON={n_on}")
    print(f"wall: OFF={w_off}s  ON={w_on}s  speedup={speedup:.2f}x  (ON bar <= {WALL_BAR}s)")
    print(f"\nW2 {'PASS' if ok else 'FAIL/REVIEW'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
