#!/usr/bin/env python
"""Flag-graduation verdicts 2&3 setup step 2: verify each flag is LIVE in THIS build.

For each (flag, probe) pair, solve OFF then ON (flag alone) in isolated
subprocesses and report node/bound/status deltas. A flag that changes nothing on
its engaging instance => wrong build/panel => STOP.
"""
import json, os, subprocess, sys, time

SP = os.path.dirname(os.path.abspath(__file__))
PY = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a045b99cdfea826ff/.venv-grad/bin/python"

PROBES = [
    ("DISCOPT_LU_DENSITY_ROUTE", "nvs21"),
    ("DISCOPT_OBJ_BRANCH_PRIORITY", "nvs01"),
    ("DISCOPT_OBJ_BRANCH_PRIORITY", "nvs13"),
    ("DISCOPT_LIFT_LOOSE_PRODUCTS", "nvs09"),
]
TL = 40.0


def run(inst, env_extra):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.update(env_extra)
    try:
        cp = subprocess.run(
            [PY, os.path.join(SP, "solve_one.py"), inst, str(TL)],
            capture_output=True, text=True, timeout=TL + 120, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"instance": inst, "status": "OUTER_TIMEOUT"}
    for ln in cp.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    return {"instance": inst, "status": "NO_RESULT", "stderr": cp.stderr[-300:]}


def main():
    cache_off = {}
    rows = []
    for flag, inst in PROBES:
        if inst not in cache_off:
            t0 = time.time()
            cache_off[inst] = run(inst, {})
            r = cache_off[inst]
            print(f"[{time.time()-t0:6.1f}s] OFF {inst:12} {r.get('status'):>10} "
                  f"obj={r.get('objective')} bnd={r.get('bound')} n={r.get('node_count')}", flush=True)
        t0 = time.time()
        ron = run(inst, {flag: "1"})
        roff = cache_off[inst]
        changed = (ron.get("node_count") != roff.get("node_count")
                   or ron.get("bound") != roff.get("bound")
                   or ron.get("status") != roff.get("status"))
        print(f"[{time.time()-t0:6.1f}s] ON  {inst:12} {ron.get('status'):>10} "
              f"obj={ron.get('objective')} bnd={ron.get('bound')} n={ron.get('node_count')} "
              f"flag={flag} LIVE={changed}", flush=True)
        rows.append({"flag": flag, "instance": inst, "off": roff, "on": ron, "changed": changed})
    with open(os.path.join(SP, "liveness.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("WROTE liveness.json")


if __name__ == "__main__":
    main()
