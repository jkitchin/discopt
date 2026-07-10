#!/usr/bin/env python
"""Flag-graduation verdicts 2 & 3 driver (follow-on to BR-3 #602).

Same house pattern as BR-3's gate.py: shared OFF baseline + per-flag ON arms,
each ON arm composes the flag with DISCOPT_LU_DENSITY_ROUTE=1 (the #591 dense
retry). Sequential solves, isolated subprocess per solve (fresh interpreter
reads env flags at import). Resumable via jsonl.

Usage: gate.py <verdict:v2|v3> [arm1,arm2,...]
"""
import json, os, subprocess, sys, time

SP = os.path.dirname(os.path.abspath(__file__))
PY = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a045b99cdfea826ff/.venv-grad/bin/python"

# Verdict 2: BR-3 panel, DIFFERENT draw. Blockers kept (nvs21/st_e36/nvs09/alkyl).
# TL=40s.
PANEL_V2 = [
    # blockers
    "nvs21", "st_e36", "nvs09", "alkyl", "tls2",
    # integer NLP (fresh)
    "nvs03", "nvs05", "nvs08", "nvs10", "nvs11", "nvs12", "nvs16", "nvs18",
    # QP / MIQP (fresh)
    "st_miqp1", "st_miqp4", "st_miqp5", "st_test2", "st_testgr1",
    # spatial / bilinear (fresh)
    "ex5_2_2_case2", "ex5_2_4", "ex7_2_2", "ex4_1_2", "ex4_1_8", "ex8_1_1", "st_bpaf1a",
    # tree-opening (fresh)
    "st_e29", "st_e31", "gbd",
]

# Verdict 3: LARGER (35) and LONGER (TL=60s). Blockers kept.
PANEL_V3 = [
    # blockers
    "nvs21", "st_e36", "nvs09", "alkyl", "tls2",
    # integer NLP
    "nvs14", "nvs20", "nvs22", "nvs01", "nvs04", "nvs06", "nvs13", "nvs15", "nvs23",
    # QP / MIQP
    "st_miqp2", "st_miqp3", "st_miqp5", "st_test2", "meanvarx", "alan",
    # spatial / bilinear
    "ex5_2_2_case1", "ex5_2_4", "ex7_2_2", "ex7_2_3", "ex4_1_3", "ex4_1_8",
    "ex4_1_9", "ex8_1_1", "ex8_1_6", "st_bpaf1a", "st_bpaf1b",
    # tree / misc
    "st_e29", "st_e31", "gbd", "gkocis",
]

PANELS = {"v2": (PANEL_V2, 40.0), "v3": (PANEL_V3, 60.0)}

ARMS = {
    "off": {},
    "lu_density_route": {"DISCOPT_LU_DENSITY_ROUTE": "1"},
    "obj_branch_priority": {"DISCOPT_OBJ_BRANCH_PRIORITY": "1", "DISCOPT_LU_DENSITY_ROUTE": "1"},
    "lift_loose_products": {"DISCOPT_LIFT_LOOSE_PRODUCTS": "1", "DISCOPT_LU_DENSITY_ROUTE": "1"},
}


def run(inst, env_extra, tl):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.update(env_extra)
    try:
        cp = subprocess.run(
            [PY, os.path.join(SP, "solve_one.py"), inst, str(tl)],
            capture_output=True, text=True, timeout=tl + 180, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"instance": inst, "status": "OUTER_TIMEOUT"}
    for ln in cp.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    return {"instance": inst, "status": "NO_RESULT", "stderr": cp.stderr[-300:]}


def main():
    verdict = sys.argv[1]
    panel, tl = PANELS[verdict]
    jsonl = os.path.join(SP, f"{verdict}_results.jsonl")
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ARMS)

    done = {}
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                done[(d["arm"], d["result"]["instance"])] = d["result"]

    for arm in arms:
        env_extra = ARMS[arm]
        out = {}
        for inst in panel:
            if (arm, inst) in done:
                out[inst] = done[(arm, inst)]
                print(f"SKIP {arm} {inst} (cached)", flush=True)
                continue
            t0 = time.time()
            r = run(inst, env_extra, tl)
            out[inst] = r
            with open(jsonl, "a") as f:
                f.write(json.dumps({"arm": arm, "result": r}) + "\n")
            print(f"[{time.time()-t0:6.1f}s] {verdict} {arm:20} {inst:16} "
                  f"{str(r.get('status')):>12} obj={r.get('objective')} "
                  f"bnd={r.get('bound')} n={r.get('node_count')} w={r.get('wall')}", flush=True)
        with open(os.path.join(SP, f"{verdict}_{arm}.json"), "w") as f:
            json.dump({"arm": arm, "env": env_extra, "tl": tl, "results": out}, f, indent=2)
        print(f"WROTE {verdict}_{arm}.json", flush=True)
    print(f"{verdict} ALL_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
