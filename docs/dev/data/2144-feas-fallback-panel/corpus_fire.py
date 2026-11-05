"""#2144 fire phase on the in-repo .nl corpus: where does the fallback fire?

Parent: python corpus_fire.py OUT.jsonl SHARD NSHARDS TL  (one subprocess per instance)
Child:  python corpus_fire.py --child PATH TL
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CORPUS = Path("/home/user/discopt/python/tests/data/minlplib_nl")

if sys.argv[1] == "--child":
    import warnings

    import discopt

    assert discopt.__file__.startswith("/home/user/discopt/python/"), discopt.__file__
    from discopt._relax import primal_heuristics as ph
    from discopt.modeling.core import from_nl

    warnings.simplefilter("ignore")
    m = from_nl(sys.argv[2])
    t = time.perf_counter()
    r = m.solve(time_limit=float(sys.argv[3]))
    print(
        "@@R "
        + json.dumps(
            {
                "status": str(r.status),
                "objective": r.objective,
                "bound": r.bound,
                "nodes": getattr(r, "node_count", None),
                "wall": time.perf_counter() - t,
                "fb": dict(ph.FEAS_FALLBACK_COUNTS),
            }
        ),
        flush=True,
    )
    os._exit(0)

out, shard, nsh, tl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
files = sorted(CORPUS.glob("*.nl"))[shard::nsh]
env = dict(os.environ, DISCOPT_GDP_CONFIG_FEAS_FALLBACK="1")
n = 0
for f in files:
    try:
        p = subprocess.run(
            [sys.executable, "-u", __file__, "--child", str(f), tl],
            capture_output=True,
            text=True,
            timeout=float(tl) + 240,
            env=env,
        )
        ln = [x for x in p.stdout.splitlines() if x.startswith("@@R ")]
        rec = json.loads(ln[-1][4:]) if ln else {"error": (p.stderr or p.stdout)[-300:]}
    except subprocess.TimeoutExpired:
        rec = {"error": "killed"}
    rec["instance"] = f.stem
    n += 1
    with open(out, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f.stem, rec.get("fb"), rec.get("status"), rec.get("error", "")[:100], flush=True)
print("executed", n)
assert n > 0
