"""Attribute a profile's solve_model time to solve_model's direct callees.

Exclusive by construction: each child's time is what it accrued when called *from*
solve_model (pstats callers[...][3]), so the rows sum to solve_model's cumtime
minus its own tottime. Prints the count of rows attributed (CLAUDE.md §6).
"""

import pstats
import sys

p = pstats.Stats(sys.argv[1])
st = p.stats
root = next(k for k in st if k[2] == "solve_model" and k[0].endswith("discopt/solver.py"))
total = st[root][3]
kids = []
for func, (cc, nc, tt, ct, callers) in st.items():
    if root in callers:
        c = callers[root]
        kids.append((c[3], c[1], func))
kids.sort(reverse=True)
acc = 0.0
for ct, nc, (f, l, fn) in kids[: int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    acc += ct
    f = f.split("/discopt/")[-1]
    print(f"{ct:8.3f} {100 * ct / total:5.1f}% {nc:6d}  {f}:{l}({fn})")
print(f"solve_model {total:.3f}s  self {st[root][2]:.3f}s  shown {acc:.3f}s  children {len(kids)}")
assert kids, "no callees attributed"
