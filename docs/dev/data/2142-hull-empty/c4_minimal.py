"""#2142 §114 step 1d: a minimal set of box bounds SCIP needs to call captured box B empty
(no cutoff). Greedy deletion: start from the box's tightened columns, reset each column to its
global bounds if SCIP still calls the box empty without it.  python -u c4_minimal.py BOXES.npz B"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation"))
import specio, scip_prop  # noqa: E402

d, kw = specio.load(sys.argv[1]); b = int(sys.argv[2])
gl, gh = kw["global_lo"].copy(), kw["global_hi"].copy()
pl, ph = d["rec_pre_lo"][b], d["rec_pre_hi"][b]
sp = scip_prop.ScipProp(kw)
n = [0]
def empty(S):
    lo, hi = gl.copy(), gh.copy()
    for j in S:
        lo[j], hi[j] = pl[j], ph[j]
    n[0] += 1
    return sp.propagate(lo, hi, None)[1] is None
S = [j for j in range(len(gl)) if pl[j] != gl[j] or ph[j] != gh[j]]
assert empty(S), "box not empty without cutoff"
print("tightened columns", len(S))
for j in list(S):
    T = [k for k in S if k != j]
    if empty(T):
        S = T
print("minimal set", len(S))
for j in S:
    print(f"  col {j} int={bool(kw['integrality'][j])} global [{gl[j]:.12g}, {gh[j]:.12g}] box [{pl[j]:.12g}, {ph[j]:.12g}]")
print(f"executed propagations: {n[0]}")
