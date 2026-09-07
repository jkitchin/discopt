"""Structural census for #1193 P2: dense-arm vs sparse-arm callback sizes.

Deterministic (counts, not timings), so it is valid under load. Exits non-zero
if it measured nothing (CLAUDE.md §6).
"""
import glob, os, sys
import numpy as np

import discopt.solver as S
from discopt._relax.cutting_planes import CutPool, LinearCut
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import from_nl

WT = "/Users/jkitchin/projects/discopt/wt-p2/"
assert S.__file__.startswith(WT), S.__file__
assert hasattr(S._AugmentedEvaluator, "has_sparse_structure"), "unpatched solver.py loaded"

ROOT = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
names = sys.argv[1:]
compared = 0
print(f"{'instance':<28} {'n':>6} {'m_aug':>6} {'jac dense':>12} {'jac sparse':>11} "
      f"{'hess dense':>12} {'hess sparse':>12}")
for nm in names:
    p = os.path.join(ROOT, nm + ".nl")
    if not os.path.exists(p):
        print(f"{nm}: MISSING", flush=True)
        continue
    m = from_nl(p)
    ev = NLPEvaluator(m)
    n = ev.n_variables
    rng = np.random.default_rng(0)
    pool = CutPool()
    for _ in range(10):                     # a plausible root cut round
        c = np.zeros(n); c[rng.integers(0, n, size=min(5, n))] = 1.0
        pool.add(LinearCut(coeffs=c, rhs=1.0, sense="<="))
    aug = S._AugmentedEvaluator(ev, pool)
    assert aug.has_sparse_structure(), f"{nm}: wrapped evaluator is not sparse"
    m_aug = aug.n_constraints
    jr, jc = aug.jacobian_structure()
    hr, hc = aug.hessian_structure()
    print(f"{nm:<28} {n:>6} {m_aug:>6} {m_aug*n:>12,} {len(jr):>11,} "
          f"{n*(n+1)//2:>12,} {len(hr):>12,}", flush=True)
    compared += 1

print(f"instances compared: {compared}")
if compared == 0:
    sys.exit(6)
