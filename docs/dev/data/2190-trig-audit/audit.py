"""#2190 audit: every sin/cos/tan enclosure and curvature verdict vs mpmath (60 digits).

Run from the repo root after ``maturin develop --release`` (needs ``mpmath``):

    python -u docs/dev/data/2190-trig-audit/audit.py | tee docs/dev/data/2190-trig-audit/audit.summary.log

Prints ``checks``/``misses`` per site and exits non-zero if no check executed.
A "miss" is an enclosure that does not contain the true range, a curvature
verdict contradicted by the true sign, a "no pole" claim with a pole inside,
or a periodic window shorter than one true period. Rust FBBT is probed
end-to-end through ``discopt.tightening.fbbt_box`` (forward via ``z == f(x)``,
backward via ``f(x) <= c`` / ``>= c``).
"""
import math, sys, warnings
import numpy as np, mpmath as mp
warnings.filterwarnings("ignore")
mp.mp.dps = 60
from discopt._relax.convexity import interval as iv
from discopt._relax import operator_relaxations as opr
from discopt._relax import uniform_relax as ur
from discopt.ro.formulations import _common as roc
import discopt.modeling as dm
from discopt.tightening import fbbt_box

rng = np.random.default_rng(7)
PI = mp.pi
def crit(f):
    return {"sin": (PI/2, PI), "cos": (mp.mpf(0), PI), "tan_pole": (PI/2, PI)}[f]
def true_range(f, lo, hi):
    L, H = mp.mpf(lo), mp.mpf(hi); F = getattr(mp, f)
    if f == "tan":
        c0, p = crit("tan_pole"); m = mp.ceil((L - c0)/p)
        if c0 + m*p <= H: return None  # pole inside -> unbounded
        return F(L), F(H)
    c0, p = crit(f); vals = [F(L), F(H)]; m = mp.ceil((L - c0)/p)
    while c0 + m*p <= H and len(vals) < 5: vals.append(F(c0 + m*p)); m += 1
    return min(vals), max(vals)

cases = []
for k in range(-600, 601):
    c = k * (np.pi/2)
    for lo, hi in ((c, c), (c, np.nextafter(c, np.inf)), (np.nextafter(c, -np.inf), c), (c, c + 0.3), (c - 0.3, c)):
        cases.append((lo, hi))
for big in (1e6, 1e9, 1e12):
    for _ in range(300):
        k = round(rng.uniform(-big, big) / (np.pi/2)); c = k * (np.pi/2)
        d = rng.uniform(0, 1e-3)
        cases.append((c - d, c + rng.uniform(0, 1e-3))); cases.append((c, c + d)); cases.append((c - d, c))
for _ in range(3000):
    mag = 10.0 ** rng.uniform(-2, 13); lo = rng.uniform(-mag, mag)
    cases.append((lo, lo + 10.0 ** rng.uniform(-9, 0.8)))
print("cases", len(cases), flush=True)

stats = {}
def rec(site, ok, info=None):
    s = stats.setdefault(site, [0, 0, None]); s[0] += 1
    if not ok:
        s[1] += 1
        if s[2] is None: s[2] = info

def contains(enc, tr):
    return enc is None or (mp.mpf(enc[0]) <= tr[0] and tr[1] <= mp.mpf(enc[1]))

for lo, hi in cases:
    for f in ("sin", "cos", "tan"):
        tr = true_range(f, lo, hi)
        # interval module (sin/cos/tan)
        out = getattr(iv, f)(iv.Interval(np.asarray(lo), np.asarray(hi)))
        e = (float(out.lo), float(out.hi))
        rec(f"interval.{f}", tr is None and e == (-math.inf, math.inf) or tr is not None and contains(e, tr), (lo, hi, e, tr))
        # AMP trig_range (None = abstain)
        r = opr.trig_range(f, lo, hi)
        rec(f"operator_relaxations.trig_range({f})", (r is None) or (tr is not None and contains(r, tr)), (lo, hi, r, tr))
        if f in ("sin", "cos"):
            r = (roc._sin_interval if f == "sin" else roc._cos_interval)(np.asarray(lo), np.asarray(hi))
            rec(f"ro._{f}_interval", contains((float(r[0]), float(r[1])), tr), (lo, hi, r, tr))
        # curvature verdicts: f'' = -f for sin/cos; for tan sign(f'') = sign(tan)
        cv = {"sin": ur._curv_sin, "cos": ur._curv_cos, "tan": ur._curv_tan}[f](lo, hi)
        if cv is not None:
            if tr is None: rec(f"uniform_relax._curv_{f}", False, (lo, hi, cv, "pole"))
            else:
                ok = (tr[1] <= 0) if (cv == "convex") == (f != "tan") else (tr[0] >= 0)
                rec(f"uniform_relax._curv_{f}", ok, (lo, hi, cv, tr))
        else:
            rec(f"uniform_relax._curv_{f}", True)
    # tan branch-safety claim
    if ur._tan_branch_safe(lo, hi):
        rec("uniform_relax._tan_branch_safe", true_range("tan", lo, hi) is not None, (lo, hi))
    else:
        rec("uniform_relax._tan_branch_safe", True)

# Rust FBBT, forward + backward, on a subsample
sub = [cases[i] for i in rng.choice(len(cases), 1500, replace=False)]
for lo, hi in sub:
    for f in ("sin", "cos"):
        F = getattr(dm, f); tr = true_range(f, lo, hi)
        m = dm.Model("p"); x = m.continuous("x", lb=lo, ub=hi); z = m.continuous("z", lb=-10, ub=10)
        m.subject_to(z == F(x)); m.minimize(z)
        bt = fbbt_box(m)
        zl, zh = float(bt.lb[1]), float(bt.ub[1])
        rec(f"rust fbbt forward {f}", (not bt.infeasible) and contains((zl, zh), tr) and bt.lb[0] <= lo and bt.ub[0] >= hi, (lo, hi, zl, zh, tr, bt.lb[0], bt.ub[0]))
        # backward: f(x) <= c with c inside the true range -> feasible extremes
        c = float((tr[0] + tr[1]) / 2)
        for sense in ("le", "ge"):
            m = dm.Model("q"); x = m.continuous("x", lb=lo, ub=hi)
            m.subject_to(F(x) <= c if sense == "le" else F(x) >= c); m.minimize(x)
            bt = fbbt_box(m)
            Fm = getattr(mp, f); L, H = mp.mpf(lo), mp.mpf(hi)
            feas = (lambda v: Fm(v) <= c) if sense == "le" else (lambda v: Fm(v) >= c)
            # true feasible min/max: endpoints if feasible, else bisect to root
            def root(a, b):
                fa = feas(a)
                for _ in range(200):
                    mid = (a + b) / 2
                    if feas(mid) == fa: a = mid
                    else: b = mid
                return a, b
            if hi - lo == 0:
                tmin = tmax = L if feas(L) else None
            else:
                tmin = L if feas(L) else root(L, H)[1]
                tmax = H if feas(H) else root(H, L)[1]
            if tmin is None or bt.infeasible:
                rec(f"rust fbbt backward {f} {sense}", not (tmin is not None and bt.infeasible), (lo, hi, c, "infeasible"))
                continue
            ok = mp.mpf(float(bt.lb[0])) <= tmin and tmax <= mp.mpf(float(bt.ub[0]))
            rec(f"rust fbbt backward {f} {sense}", ok, (lo, hi, c, float(bt.lb[0]), float(bt.ub[0]), float(tmin), float(tmax)))
# NBT periodic window: must span >= one true period
from discopt._relax import nonlinear_bound_tightening as nbt
assert hasattr(nbt, "_period_end_up")
rec("nbt free window", 2 * mp.mpf(nbt._PI_UP) >= 2 * PI)
for lo in [0.0, 1.0, 1e3, 1e6, 1e9, 1e12, -1e12, 1e14] + list(rng.uniform(-1e12, 1e12, 2000)) + list(rng.uniform(-10, 10, 2000)):
    lo = float(lo); hi = nbt._period_end_up(lo); lo2 = nbt._period_start_down(lo)
    rec("nbt periodic window end_up", mp.mpf(hi) - mp.mpf(lo) >= 2 * PI and mp.mpf(hi) - mp.mpf(lo) - 2*PI < 1e-12 * max(1, abs(lo)), (lo, float(mp.mpf(hi) - mp.mpf(lo) - 2*PI)))
    rec("nbt periodic window start_down", mp.mpf(lo) - mp.mpf(lo2) >= 2 * PI, (lo,))
tot = 0
for k, (n, miss, ex) in sorted(stats.items()):
    tot += n; print(f"{k:45s} checks={n:6d} misses={miss:5d}" + (f"  e.g. {ex}" if miss else ""))
print("total checks", tot); sys.exit(0 if tot else 1)
