"""#1013 entry experiment: is the dual pivot path marginally stable?

The issue's signature is that a *rounding-level* perturbation of the pivot
sequence — one that changes no tolerance, guard or bound formula — multiplies the
iteration count several-fold and can turn `optimal` into `iter_limit`. The #1008
levers that produced it (refactorization cadence, LU pivot threshold, symbolic
ordering reuse) were all deleted with those falsified flags, so this reproduces
the same perturbation FAMILY with a lever that needs no flag: a random
permutation of the LP's columns. The permuted LP is the same mathematical
program (same optimum, same feasible set) — only the order in which the ratio
test and the LU see the columns changes, i.e. the tie-breaks and the rounding.

Reports, per LP, the iteration count of the identity ordering and of `K` random
permutations. Iterations (not wall) are the metric: they are deterministic and
load-independent (CLAUDE.md §9).

Prints a solved-cell count and exits non-zero at zero (§6). Nothing is caught (§7).
"""

import json
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LPDIR = os.path.join(ROOT, "scratchpad/i1013/lps")
TMP = os.environ.get("I1013_TMP", "/tmp/i1013_perm")
os.makedirs(TMP, exist_ok=True)

out_path = sys.argv[1]
tl = sys.argv[2]
k = int(sys.argv[3])
names = sys.argv[4:]
arms = [a for a in os.environ.get("I1013_ARMS", "0,1").split(",")]


def permuted(path, seed, dst):
    z = np.load(path)
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
    if seed >= 0:
        p = np.random.default_rng(seed).permutation(ncol)
        A, c, lo, hi = A[:, p], c[p], lo[p], hi[p]
    A = sp.csc_matrix(A)
    np.savez(
        dst,
        c=c,
        indptr=A.indptr,
        indices=A.indices,
        data=A.data,
        shape=np.array(A.shape),
        b=b,
        lo=lo,
        hi=hi,
    )


cells = 0
with open(out_path, "w") as fh:
    for nm in names:
        src = os.path.join(LPDIR, nm + ".npz")
        for seed in range(-1, k):
            dst = os.path.join(TMP, f"{nm}_p{seed}.npz")
            permuted(src, seed, dst)
            for arm in arms:
                e = dict(os.environ)
                e["DISCOPT_PROFILE"] = "1"
                e["DISCOPT_LP_DUAL_STALL_HARRIS"] = arm
                p = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        os.path.join(ROOT, "scratchpad/i1013/lprun.py"),
                        dst,
                        tl,
                    ],
                    capture_output=True,
                    text=True,
                    env=e,
                )
                assert p.returncode == 0, f"{nm} seed={seed} arm={arm}: {p.stderr[-2000:]}"
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RES ")]
                if not lines:
                    print(f"{nm} seed={seed}: SKIP (no dual start)", flush=True)
                    break
                rec = json.loads(lines[0][4:])
                rec.update(lp=nm, seed=seed, arm=arm)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cells += 1
                print(
                    f"{nm:26s} seed={seed:3d} harris={arm} {rec['status']:10s} "
                    f"it={rec['iters']:7d} wall={rec['wall']:7.3f} "
                    f"degen={rec.get('DualDegeneratePivots', 0):7d} "
                    f"arms={rec.get('DualDegenerateRunArms', 0)} "
                    f"maxrun={rec.get('DualDegenerateRunMax', 0)} "
                    f"repiv={rec.get('DualStabilityRepivots', 0)} obj={rec['obj']!r}",
                    flush=True,
                )
            os.remove(dst)
print("cells:", cells, flush=True)
if cells == 0:
    sys.exit(1)
