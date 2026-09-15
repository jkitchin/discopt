"""H4 MINLP guard: DISCOPT_LP_MILP_BACKEND must be exactly neutral on MINLP solves.

Arms per instance, each a fresh subprocess: R (rust), H (highs), R2 (rust again, the
noise control). On every instance compare status / objective / bound / node_count /
gap_certified / LP-MILP route key. A R-vs-H difference counts as drift only when R and
R2 agree (otherwise the instance is timing-nondeterministic and is listed, not hidden).
Exits non-zero when nothing was compared or anything drifted (CLAUDE.md §5, §6).
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "python/tests/data/minlplib_nl"
TL, MAX_NODES = 20.0, 300
KEYS = ("status", "objective", "bound", "nodes", "gap_certified", "route", "error")

WORKER = r"""
import json, os, sys, discopt
# The checkout under test, not some other installed discopt (CLAUDE.md §8).
assert discopt.__file__.startswith(os.environ["GUARD_DISCOPT_SRC"]), discopt.__file__
from discopt.modeling.core import from_nl
path, tl, mn = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
try:
    m = from_nl(path)
except Exception as e:  # recorded per arm and compared, not swallowed
    print("RESULT " + json.dumps({"error": f"load {type(e).__name__}: {e}"[:300]}))
    sys.exit(0)
r = m.solve(time_limit=tl, max_nodes=mn)
s = r.solver_stats or {}
print("RESULT " + json.dumps({"status": r.status, "objective": r.objective, "bound": r.bound,
    "nodes": int(r.node_count or 0), "gap_certified": bool(r.gap_certified),
    "route": s.get("route/lp_milp_backend"), "error": None}))
"""


def run(path, backend):
    env = dict(
        os.environ,
        DISCOPT_LP_MILP_BACKEND=backend,
        PYTHONUNBUFFERED="1",
        GUARD_DISCOPT_SRC=str(REPO / "python"),
    )
    p = subprocess.run(
        [sys.executable, "-c", WORKER, str(path), str(TL), str(MAX_NODES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10 * TL + 300,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"status": None, "error": f"rc={p.returncode} {p.stderr[-300:]}"}
    return json.loads(lines[0][7:])


def one(path):
    res = {arm: run(path, b) for arm, b in (("R", "rust"), ("H", "highs"), ("R2", "rust"))}
    return path.name, res


def key(d):
    return {k: d.get(k) for k in KEYS}


files = sorted(CORPUS.glob("*.nl"))
compared = drift = noisy = 0
out = {}
with ThreadPoolExecutor(max_workers=int(os.environ.get("GUARD_WORKERS", "3"))) as ex:
    for i, (name, res) in enumerate(ex.map(one, files), 1):
        out[name] = res
        r, h, r2 = key(res["R"]), key(res["H"]), key(res["R2"])
        compared += 1
        tag = "ok"
        if r != h:
            if r != r2:
                noisy += 1
                tag = "NOISY (R != R2)"
            else:
                drift += 1
                tag = "DRIFT"
        print(
            f"[{i}/{len(files)}] {name:28s} {tag} R={r['status']},{r['nodes']},{r['objective']}"
            f" H={h['status']},{h['nodes']},{h['objective']} route={r['route']}/{h['route']}",
            flush=True,
        )
        if tag != "ok":
            print("   R ", r, "\n   H ", h, "\n   R2", r2, flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(f"compared={compared} drift={drift} noisy={noisy}")
sys.exit(1 if compared == 0 or drift else 0)
