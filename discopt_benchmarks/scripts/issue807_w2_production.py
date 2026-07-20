#!/usr/bin/env python
"""Issue #807 / W2 — production (UNSEEDED) panel net, the real #807 target metric.

Runs the full scoped rsyn* production panel UNSEEDED (no oracle incumbent — the
kernel must find its own) with cuts on, DISCOPT_CVX_NATIVELP ON (persistent tangent
pool + node-local cuts) vs OFF (today's per-node cold solve_node_cut), and reports
wall + cert-clean. This is the config-A target the seeded W2 gate only proxied.

Cert-clean: dual bound >= oracle optimum (valid upper bound for max), and if
optimal, closed onto it. Flag is process-global → ON/OFF in separate subprocesses.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

RSYN = ["rsyn0805m", "rsyn0810m", "rsyn0815m", "rsyn0820m", "rsyn0830m"]
SNAP = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")


def load_oracle():
    o = {}
    for line in open(SOLU):
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            try:
                o[p[1]] = float(p[2])
            except ValueError:
                pass
    return o


def run_child() -> None:
    import discopt.modeling as dm  # noqa: E402
    from discopt.solvers._convex_kernel import build_convex_spec, solve_convex_tree  # noqa: E402

    oracle = load_oracle()
    flag = os.environ.get("DISCOPT_CVX_NATIVELP", "0")
    print(f"# DISCOPT_CVX_NATIVELP={flag}  (production / unseeded, cuts on)")
    print(f"{'instance':12s} {'status':10s} {'nodes':>7s} {'wall_s':>8s} "
          f"{'bound':>12s} {'opt':>12s}  cert")
    tw = 0.0
    bad = 0
    for name in RSYN:
        f = SNAP + name + ".nl"
        m = dm.from_nl(f)
        spec = build_convex_spec(m)
        if spec is None:
            print(f"{name:12s} declined")
            continue
        opt = oracle.get(name)
        t = time.perf_counter()
        r = solve_convex_tree(spec, time_limit_s=180.0, initial_incumbent=None)
        dt = time.perf_counter() - t
        tw += dt
        bound = float(r["bound"])
        status = r["status"]
        sound = opt is None or bound >= opt - 1e-3 * max(1.0, abs(opt))
        closed = status != "optimal" or opt is None or abs(bound - opt) <= 1e-2 * max(1.0, abs(opt))
        cert = sound and closed
        bad += 0 if cert else 1
        os_ = "None" if opt is None else f"{opt:12.4f}"
        print(f"{name:12s} {status:10s} {r['node_count']:7d} {dt:8.2f} "
              f"{bound:12.4f} {os_:>12s}  {cert}", flush=True)
    print(f"TOTAL wall={tw:.1f}s  cert-clean={'YES' if bad == 0 else 'NO(' + str(bad) + ')'}",
          flush=True)


def parse(out):
    w = c = None
    for line in out.splitlines():
        if line.startswith("TOTAL"):
            w = float(line.split("wall=")[1].split("s")[0])
            c = "cert-clean=YES" in line
    return w, c


def main() -> bool:
    if os.environ.get("_W2P_CHILD") == "1":
        run_child()
        return True
    import subprocess

    outs = {}
    for flag in ("0", "1"):
        env = dict(os.environ, DISCOPT_CVX_NATIVELP=flag, _W2P_CHILD="1")
        print(f"\n===== DISCOPT_CVX_NATIVELP={flag} =====", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                           capture_output=True, text=True)
        print(p.stdout, end="")
        if p.returncode != 0:
            print(p.stderr[-3000:])
        outs[flag] = p.stdout

    w_off, c_off = parse(outs["0"])
    w_on, c_on = parse(outs["1"])
    print("\n================ W2 PRODUCTION NET ================")
    su = (w_off / w_on) if (w_on and w_off) else float("nan")
    print(f"cert-clean: OFF={c_off} ON={c_on}")
    print(f"wall: OFF={w_off}s  ON={w_on}s  speedup={su:.2f}x")
    print(f"\nNet: ON is {'FASTER' if (w_on and w_off and w_on < w_off) else 'NOT faster'} "
          f"and {'cert-clean' if c_on else 'NOT cert-clean'}")
    return bool(c_on)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
