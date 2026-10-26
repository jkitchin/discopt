"""#1975 step 4: prolog's outer piece T = x7 + x8 >= M, rescaled by s = 1/T.

x_k = u_k / s for the open columns (x0, x1, x6, x7, x8, x18, x19), u7 + u8 = 1.
Every row is divided by T (prolog is degree 1 in the open columns), the bounded block
(x2..x5, x9..x17) is unchanged. min f/T over s in [0, 1/M] > 0 proves f > 0 = opt on
the whole outer piece. The model is BOUNDED (all u finite after FBBT): no recession.
"""
import os, sys, time
import discopt.modeling as dm

M = float(sys.argv[1])
tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
m = dm.Model("prolog_outer")
s = m.continuous("s", lb=0.0, ub=1.0 / M if M > 0 else 0.0)
import os
box = os.environ.get("U01_BOX")  # "lo0,hi0,lo1,hi1": a neighbourhood of the optimum
u = {k: m.continuous(f"u{k}", lb=0.0) for k in (0, 1, 6, 7, 8, 18, 19)}
if box:
    lo0, hi0, lo1, hi1 = map(float, box.split(","))
    c0_ = m.subject_to
    c0_([u[0] >= lo0, u[0] <= hi0, u[1] >= lo1, u[1] <= hi1])
x = {k: m.continuous(f"x{k}", lb=0.0) for k in [2, 3, 4, 5] + list(range(9, 18))}
c = m.subject_to
c(u[7] + u[8] == 1)
for k in (0, 1, 6):  # x_k >= 0.2  ->  u_k >= 0.2 s
    c(u[k] >= 0.2 * s)
c(x[2] * u[0] + x[4] * u[1] - u[18] <= 0)
c(x[3] * u[0] + x[5] * u[1] - u[19] <= 0)
S1 = x[9] + x[10] + x[11]; S2 = x[12] + x[13] + x[14]; S3 = x[15] + x[16] + x[17]
c(x[2] + x[3] - 0.94 * S1 + 0.244 * S3 <= 0)
c(0.064 * S1 - 0.58 * S2 + 0.172 * S3 <= 0)
c(x[4] + x[5] + 0.048 * S1 + 0.247 * S2 - 0.916 * S3 <= 0)
xs = [x[k] for k in range(9, 18)]
c(sum(a * v for a, v in zip([1, 1.2, .8, 2, 1.8, 2.4, 3, 2.7, 3.2], xs)) <= 3712)
c(sum(a * v for a, v in zip([2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7], xs)) <= 5000)
# C7..C10 divided by T: the bounded column and the constant pick up a factor s
c(356.474947137148 * u[0] + 53.7083537310174 * u[1] + s * x[2] - 0.564264890180399 * u[18] <= 352 * s)
c(339.983422262764 * u[0] + 43.5418249774113 * u[1] + s * x[3] - 0.405939876920766 * u[19] <= 430 * s)
c(106.946746119538 * u[0] + 145.018955433089 * u[1] + s * x[4] - 0.507117039797071 * u[18] <= 222 * s)
c(173.929713444361 * u[0] + 203.031384299627 * u[1] + s * x[5] - 0.578889145413521 * u[19] <= 292 * s)
c(u[18] - 3340.8 * u[7] - 500 * u[8] <= 0)
c(u[19] - 371.2 * u[7] - 4500 * u[8] <= 0)
for c0, c1, c6, rr in [(.94, -.048, -.064, [(1, 2), (1.2, 1.8), (.8, 2.2)]),
                       (0.0, -.247, .58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
                       (-.244, .916, -.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)])]:
    for c7, c8 in rr:
        c(c0 * u[0] + c1 * u[1] + c6 * u[6] - c7 * u[7] - c8 * u[8] <= 0)
m.minimize(-(x[2] * u[0] + x[3] * u[0] + x[4] * u[1] + x[5] * u[1]) + 3712 * u[7] + 5000 * u[8])
t = time.time()
kw = {"max_nodes": int(sys.argv[3])} if len(sys.argv) > 3 else {}
r = m.solve(time_limit=tl, gap_tolerance=float(os.environ.get("GAP", "1e-4")), abs_gap_tolerance=1e-4, **kw)
print(f"M={M:g}: status={r.status} obj={r.objective} bound={r.bound} "
      f"certified={getattr(r, 'gap_certified', None)} nodes={r.node_count} wall={time.time()-t:.1f}s",
      flush=True)
