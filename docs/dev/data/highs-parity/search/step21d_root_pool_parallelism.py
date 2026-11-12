"""Step 2.1d: parallelism of the first root separation round's pool, HiGHS vs discopt.

Usage: python -I step21d_root_pool_parallelism.py <hs_pool.dump> <our_pool.txt>

* ``hs_pool.dump``: the patched HiGHS build's pool dump (``HS_DUMP_POOL=<path>``) of
  ``highs --model_file mik-250-20-75-5.mps --read_solution_file <opt>.sol --presolve off``.
  Cuts are the ``PA``/``inds``/``vals``/``PA_RET`` records before the first ``PS``.
* ``our_pool.txt``: the ported pool's state at its first ``CutPool::separate`` call, written
  by a temporary instrument (not in the tree) at the top of the selection loop of
  ``crates/discopt-core/src/lp/highs_cutgen/pool.rs`` ``separate``: per live row
  ``C <index> <rhs>`` then a line of indices and a line of values, then ``SOL``/``LB``/``UB``.
  Run: the panel worker on mik-250-20-75-5 with ``DISCOPT_MILP_SEED_POINT`` (HiGHS's optimum),
  ``DISCOPT_MILP_PRESOLVE=0 DISCOPT_MILP_SUBMIP=0``.

Prints, for each pool, how many of the cut pairs are at most 0.1 parallel (HiGHS's
selection admits a cut only below that against every cut already selected), the median
parallelism and the mean cut length. Exits non-zero when no pair was compared (CLAUDE.md §6).
"""

import itertools
import math
import sys


def hx(t):
    return float.fromhex(t)


def highs_cuts(path):
    cuts, pend = [], None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "PA":
            pend = {"rhs": hx(t[2])}
        elif t[0] == "inds" and pend is not None:
            pend["inds"] = list(map(int, t[2:]))
        elif t[0] == "vals" and pend is not None:
            pend["vals"] = list(map(hx, t[2:]))
        elif t[0] == "PA_RET":
            if int(t[1]) >= 0:
                cuts.append(pend)
            pend = None
        elif t[0] == "PS":
            break
    return cuts


def our_cuts(path):
    lines = open(path).read().split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("C "):
            _, _, r = lines[i].split()
            out.append(
                {
                    "rhs": float(r),
                    "inds": list(map(int, lines[i + 1].split())),
                    "vals": list(map(float, lines[i + 2].split())),
                }
            )
            i += 3
        else:
            i += 1
    return out


def par(a, b):
    da = dict(zip(a["inds"], a["vals"]))
    d = sum(v * da.get(j, 0.0) for j, v in zip(b["inds"], b["vals"]))
    na = math.sqrt(sum(v * v for v in a["vals"]))
    nb = math.sqrt(sum(v * v for v in b["vals"]))
    return d / na / nb


compared = 0
for name, cs in (("HiGHS", highs_cuts(sys.argv[1])), ("discopt", our_cuts(sys.argv[2]))):
    ps = sorted(par(a, b) for a, b in itertools.combinations(cs, 2))
    compared += len(ps)
    print(
        f"{name}: cuts {len(cs)} pairs {len(ps)} pairs<=0.1 {sum(1 for p in ps if p <= 0.1)} "
        f"median {ps[len(ps) // 2]:.3g} mean_len {sum(len(c['inds']) for c in cs) / len(cs):.1f}"
    )
print(f"pairs compared: {compared}")
sys.exit(0 if compared > 0 else 1)
