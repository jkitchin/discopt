"""#1013 panel driver: run every captured LP under one or more env arms.

Arms are given as `NAME:VAR=VAL,VAR=VAL` on the command line; each LP is run
through every arm, interleaved within a rep (CLAUDE.md §9), in child processes.
Per-LP progress is printed as it happens (§10); a solved count is printed at the
end and the script exits non-zero if it is zero (§6).

    python -u scratchpad/i1013/panel.py OUT.jsonl TL REPS base: harris:DISCOPT_LP_DUAL_HARRIS=1
"""

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LPS = sorted(glob.glob(os.path.join(ROOT, "scratchpad/i1013/lps/*.npz")))
out_path, tl, reps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
arms = []
for spec in sys.argv[4:]:
    name, _, kvs = spec.partition(":")
    env = {}
    for kv in filter(None, kvs.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    arms.append((name, env))
only = os.environ.get("I1013_ONLY")
if only:
    keep = set(only.split(","))
    LPS = [p for p in LPS if os.path.basename(p)[:-4] in keep]
print(f"LPs: {len(LPS)}  arms: {[a[0] for a in arms]}  tl={tl}s reps={reps}", flush=True)
print(
    "uptime:", subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), flush=True
)

solved = 0
skipped = 0
with open(out_path, "w") as fh:
    for path in LPS:
        skip = False
        for rep in range(reps):
            if skip:
                break
            for name, env in arms:
                e = dict(os.environ)
                e.update(env)
                e["DISCOPT_PROFILE"] = "1"
                t0 = time.perf_counter()
                p = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        os.path.join(ROOT, "scratchpad/i1013/lprun.py"),
                        path,
                        str(tl),
                    ],
                    capture_output=True,
                    text=True,
                    env=e,
                )
                assert p.returncode == 0, f"{path} {name}: rc={p.returncode}\n{p.stderr[-2000:]}"
                if any(ln.startswith("SKIP ") for ln in p.stdout.splitlines()):
                    print(f"{os.path.basename(path)[:-4]:32s} SKIP (no dual start)", flush=True)
                    skip = True
                    skipped += 1
                    break
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RES ")]
                assert len(lines) == 1, f"{path} {name}: {len(lines)} RES lines"
                rec = json.loads(lines[0][4:])
                rec["arm"] = name
                rec["rep"] = rep
                rec["proc_wall"] = time.perf_counter() - t0
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                solved += 1
                print(
                    f"{rec['lp']:32s} {name:8s} rep{rep} {rec['status']:10s} "
                    f"it={rec['iters']:6d} wall={rec['wall']:7.3f} "
                    f"degen={rec.get('DualDegeneratePivots', 0)}",
                    flush=True,
                )
print(f"solved cells: {solved}  skipped LPs: {skipped}", flush=True)
if solved == 0:
    sys.exit(1)
