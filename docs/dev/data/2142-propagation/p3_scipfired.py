"""Probe-fired check: does SCIP (harness of scip_prop.py) tighten the pre-propagation box at all,
and how many columns vs the kernel? Prints counts (§6)."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WT = os.environ["DISCOPT_TREE"]  # the tree under test (CLAUDE.md §8)
import discopt._rust as R
assert R.__file__.startswith(WT) and hasattr(R, "propagate_spec_py")
import scip_prop
d = dict(np.load(sys.argv[1])); params = {}
for kv in sys.argv[2:]:
    k, v = kv.split("="); params[k] = float(v) if "." in v or "e" in v else int(v)
KEYS = [k for k in d if not k.startswith("rec_") and k != "prop_rounds"]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in KEYS}
sp = scip_prop.ScipProp(kw, params)
tot_s = tot_k = n = 0
for b in range(len(d["rec_id"])):
    plo, phi, cut = d["rec_pre_lo"][b], d["rec_pre_hi"][b], float(d["rec_cutoff"][b])
    live, klo, khi = R.propagate_spec_py(**kw, lo=plo, hi=phi, cutoff=cut, rounds=100000)
    st, slo, shi = sp.propagate(plo, phi, cut)
    if slo is None or not live: continue
    pl = np.where(plo <= -1e19, -np.inf, plo); ph = np.where(phi >= 1e19, np.inf, phi)
    ts = ((slo > pl + 1e-6 * (1 + abs(pl))) | (shi < ph - 1e-6 * (1 + abs(ph)))).sum()
    tk = ((klo > pl + 1e-6 * (1 + abs(pl))) | (khi < ph - 1e-6 * (1 + abs(ph)))).sum()
    tot_s += ts; tot_k += tk; n += 1
print("params", params, "boxes compared", n, "cols tightened vs pre box: SCIP", tot_s, "kernel", tot_k)
print("executed comparisons:", n); sys.exit(0 if n else 1)
