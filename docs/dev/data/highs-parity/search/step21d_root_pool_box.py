"""Step 2.1d: the first root separation round's point and box, HiGHS vs discopt.

Usage: python -I step21d_root_pool_box.py <hs_pool.dump> <our_pool.txt>
(the inputs of ``step21d_root_pool_parallelism.py``). Prints, for the LP point and the
column bounds each pool's first ``separate`` call received, how many columns differ and the
first few differences. Exits non-zero when nothing was compared (CLAUDE.md §6).
"""

import sys


def hx(t):
    return float.fromhex(t)


hs = {}
for line in open(sys.argv[1]):
    t = line.split()
    if t and t[0] in ("sol", "lb", "ub") and t[0] not in hs:
        hs[t[0]] = list(map(hx, t[2:]))
    if len(hs) == 3:
        break
ours = {}
for line in open(sys.argv[2]):
    t = line.split()
    if t and t[0] in ("SOL", "LB", "UB"):
        ours[t[0].lower()] = list(map(float, t[1:]))
compared = 0
for key in ("sol", "lb", "ub"):
    a, b = hs[key], ours[key]
    d = [(j, x, y) for j, (x, y) in enumerate(zip(a, b)) if abs(x - y) > 1e-7 * (1 + abs(x))]
    compared += len(a)
    print(f"{key}: columns {len(a)} vs {len(b)} differing {len(d)} first {d[:4]}")
print(f"values compared: {compared}")
sys.exit(0 if compared > 0 else 1)
