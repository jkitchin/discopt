"""Parallel ``root_cuts_engine_panel.py screen`` (3 workers; the screen is not timed).

Usage: python screen_par.py <discopt_benchmarks/scripts> <out population.json> <time limit>
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import root_cuts_engine_panel as P  # noqa: E402

out = Path(sys.argv[2])
tl = float(sys.argv[3])
paths = P._corpus()
jobs = [(cfg, env, name, p) for cfg, env in P.CONFIGS.items() for name, p in paths.items()]


def run(job):
    cfg, env, name, p = job
    r = P._child(p, tl, {**env, "DISCOPT_NLPBB_ROOT_CUTS_ENGINE": "highs"})
    print(
        f"[screen {cfg}] {name:24s} stage_calls={r.get('stage_calls')} {r.get('error', '')}",
        flush=True,
    )
    return cfg, name, r


found, ran = [], 0
with ThreadPoolExecutor(3) as ex:
    for cfg, name, r in ex.map(run, jobs):
        ran += "error" not in r
        if r.get("stage_calls", 0) > 0:
            found.append({"instance": name, "config": cfg, "stage_calls": r["stage_calls"]})
out.write_text(json.dumps({"population": found, "screened": ran, "jobs": len(jobs)}, indent=1))
print(f"screen: jobs={len(jobs)} screened={ran} population={len(found)}", flush=True)
sys.exit(0 if ran else 1)
