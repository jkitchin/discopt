"""Attribute LP wall to phases across every LP the R1 panel has finished.

Shares, not absolute times: the two panel arms ran concurrently, so absolute ms
are contention-inflated (CLAUDE.md §9). Contention inflates all phases roughly
alike, so share-of-wall is the defensible quantity here; no absolute-speed claim
is made. Prints an executed-parse count and exits non-zero if it is zero (§6).
"""
import re, sys, json

LOG = "scratchpad/i1008/r1_arm0.log"
PHASES = ["LuNumeric", "LuSymbolic", "Refactorize", "FtUpdate", "PriceBtran",
          "PriceSweep", "AlphaFtran", "DualPivotLoop"]

lines = open(LOG).read().splitlines()
ph = re.compile(r"^\s{2}(\w+)\s+(\d+) calls\s+([\d.]+) ms")
ctr = re.compile(r"^\s{2}(\w+)\s+(\d+)\s*$")
res = re.compile(r"^(\S+)\s+(optimal|infeasible|unbounded|iteration_limit|error)\s.*\(([\d.]+)s\)")

cur_ph, cur_ct, rows = {}, {}, []
for ln in lines:
    m = ph.match(ln)
    if m:
        cur_ph[m.group(1)] = (int(m.group(2)), float(m.group(3)))
        continue
    m = ctr.match(ln)
    if m:
        cur_ct[m.group(1)] = int(m.group(2))
        continue
    m = res.match(ln)
    if m:
        rows.append((m.group(1), m.group(2), float(m.group(3)), cur_ph, cur_ct))
        cur_ph, cur_ct = {}, {}

assert rows, "parsed no LP result lines"
print(f"{'tag':<22}{'wall_s':>9}{'piv':>7}{'facs':>6}{'fill':>7}  " +
      "".join(f"{p[:9]:>10}" for p in ("LuNumeric", "LuSymbol", "FtUpdate", "Price*")))
tot_wall = 0.0
agg = {p: 0.0 for p in PHASES}
n = 0
for tag, st, wall, p, c in rows:
    if not p:
        continue
    n += 1
    piv = c.get("Phase1Pivots", 0) + c.get("Phase2Pivots", 0)
    facs = c.get("LuSparseFactorizations", 0)
    bn, fn = c.get("LuBasisNnz", 0), c.get("LuFactorNnz", 0)
    fill = fn / bn if bn else float("nan")
    price = sum(p.get(k, (0, 0.0))[1] for k in ("PriceBtran", "PriceSweep", "AlphaFtran"))
    g = lambda k: p.get(k, (0, 0.0))[1] / 1000.0 / wall * 100
    print(f"{tag:<22}{wall:>9.1f}{piv:>7}{facs:>6}{fill:>7.1f}  "
          f"{g('LuNumeric'):>9.1f}%{g('LuSymbolic'):>9.1f}%{g('FtUpdate'):>9.1f}%"
          f"{price/1000/wall*100:>9.1f}%")
    tot_wall += wall
    for k in PHASES:
        agg[k] += p.get(k, (0, 0.0))[1] / 1000.0

print(f"\nLPs with a phase profile: {n} (of {len(rows)} results parsed)")
print(f"total wall {tot_wall:.1f}s")
for k in PHASES:
    print(f"  {k:<16}{agg[k]:>9.1f}s  {agg[k]/tot_wall*100:>6.1f}% of wall")
print(f"\nexecuted: results_parsed={len(rows)} profiles_attributed={n}")
sys.exit(0 if n else 1)
