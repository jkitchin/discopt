"""Step 2.3 A/B: DISCOPT_HIGHS_SEARCH_OBJ_PROP off vs on at fixed node budgets.

Usage: python step23_obj_prop.py <ref_dir> <out.jsonl> <jobs> <name:max_nodes:reps> ...

Runs every (instance, budget, arm, rep) as a ``lp_milp_panel.py worker`` subprocess (arm R,
600 s backstop, gap 1e-4), arms interleaved per rep, and appends one JSON row per run with the
summed ``HighsObjProp*`` / ``HighsMipSearchNodes`` profile counters (the probe that the
mechanism fired). Prints the executed-run count and exits non-zero when it is zero.
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PANEL = REPO / "discopt_benchmarks" / "scripts" / "lp_milp_panel.py"
CTR = re.compile(r"^\s*(HighsObjProp\w+|HighsMipSearchNodes|HighsMipSearchRuns)\s+(\d+)\s*$")


def run(ref: str, name: str, nodes: int, arm: str, rep: int) -> dict:
    env = dict(
        os.environ,
        DISCOPT_PANEL_DETERMINISTIC="1",
        RAYON_NUM_THREADS="1",
        DISCOPT_LP_MILP_BACKEND="rust",
        DISCOPT_PROFILE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(REPO / "python"),
    )
    env.pop("DISCOPT_HIGHS_SEARCH_OBJ_PROP", None)
    if arm == "obj":
        env["DISCOPT_HIGHS_SEARCH_OBJ_PROP"] = "1"
    path = f"{ref}/miplib2017/{name}.mps.gz"
    p = subprocess.run(
        [sys.executable, str(PANEL), "worker", path, "R", "600", "1e-4", str(nodes)],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{name} {arm} rc={p.returncode}\n{p.stderr[-2000:]}")
    res = json.loads(lines[0][len("RESULT ") :])
    ctr: dict[str, int] = {}
    for ln in (p.stdout + "\n" + p.stderr).splitlines():
        m = CTR.match(ln)
        if m:
            ctr[m.group(1)] = ctr.get(m.group(1), 0) + int(m.group(2))
    keep = ("status", "objective", "bound", "gap_certified", "nodes", "wall", "route")
    return {
        "name": name,
        "max_nodes": nodes,
        "arm": arm,
        "rep": rep,
        **{k: res.get(k) for k in keep},
        "ctr": ctr,
        "load": os.getloadavg()[0],
    }


def main() -> int:
    ref, out, jobs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tasks = []
    for spec in sys.argv[4:]:
        name, nodes, reps = spec.split(":")
        for rep in range(int(reps)):
            arms = ["off", "obj"] if rep % 2 == 0 else ["obj", "off"]
            tasks += [(name, int(nodes), arm, rep) for arm in arms]
    done = 0
    with open(out, "a") as f, ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(run, ref, *t) for t in tasks]
        for fu in futs:
            row = fu.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            done += 1
            print(json.dumps(row), flush=True)
    print(f"executed runs: {done}", flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
