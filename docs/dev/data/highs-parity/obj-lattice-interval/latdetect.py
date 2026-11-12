"""Static census: does the objective become lattice-valued via implied-lattice columns?
Primal: an equation row whose other columns are lattice fixes the column on a lattice.
Dual (b-ball's x12): every row of the column has only lattice others, sides and bounds on
a common lattice -> some optimum has it on that lattice. Dual-detected columns are not
reused (as the soundness argument requires)."""
import gzip, sys, json, math
from fractions import Fraction as F
INF = 1e19
def frac(v):
    f = F(v).limit_denominator(1000)
    return f if abs(float(f) - v) <= 1e-9 * max(1, abs(v)) else None
def fgcd(a, b):
    if a == 0: return abs(b)
    if b == 0: return abs(a)
    return F(math.gcd(a.numerator * b.denominator, b.numerator * a.denominator), a.denominator * b.denominator)
def parse(path):
    rows, rtype, cols, rhs, ints, lb, ub = {}, {}, {}, {}, set(), {}, {}
    obj = None; sec = None; inint = False; rng = {}
    for l in gzip.open(path, "rt", errors="replace"):
        if not l.strip() or l.startswith("*"): continue
        if not l[0].isspace():
            sec = l.split()[0]; continue
        t = l.split()
        if sec == "ROWS":
            rtype[t[1]] = t[0]
            if t[0] == "N" and obj is None: obj = t[1]
        elif sec == "COLUMNS":
            if len(t) >= 3 and t[1] == "'MARKER'":
                inint = "INTORG" in t[2]; continue
            c = t[0]; cols.setdefault(c, {})
            if inint: ints.add(c)
            for i in range(1, len(t) - 1, 2): cols[c][t[i]] = float(t[i + 1])
        elif sec == "RHS":
            for i in range(1, len(t) - 1, 2): rhs[t[-2 if len(t)%2==0 else i]] = 0
            k = 1 if len(t) % 2 == 1 else 0
            for i in range(k, len(t) - 1, 2): rhs[t[i]] = float(t[i + 1])
        elif sec == "RANGES":
            k = 1 if len(t) % 2 == 1 else 0
            for i in range(k, len(t) - 1, 2): rng[t[i]] = float(t[i + 1])
        elif sec == "BOUNDS":
            bt = t[0]; c = t[2] if len(t) >= 3 else t[1]; v = float(t[3]) if len(t) >= 4 else None
            if bt in ("UP",): ub[c] = v
            elif bt in ("LO",): lb[c] = v
            elif bt == "FX": lb[c] = ub[c] = v
            elif bt == "FR": lb[c] = -INF
            elif bt == "MI": lb[c] = -INF
            elif bt == "PL": ub[c] = INF
            elif bt == "BV": ints.add(c); lb[c] = 0; ub[c] = 1
            elif bt in ("LI", "UI"): ints.add(c); (lb if bt == "LI" else ub)[c] = v
    return rtype, cols, rhs, ints, lb, ub, obj, rng
def census(path):
    rtype, cols, rhs, ints, lb, ub, obj, rng = parse(path)
    rows = {}
    for c, d in cols.items():
        for r, v in d.items():
            if r != obj: rows.setdefault(r, []).append((c, v))
    cost = {c: d[obj] for c, d in cols.items() if obj in d and d[obj] != 0}
    contcost = [c for c in cost if c not in ints]
    if not contcost: return None
    step = {c: F(1) for c in ints}
    def sides(r):
        b = rhs.get(r, 0.0); t = rtype[r]
        if t == "E":
            if r in rng: return (b, b + abs(rng[r])) if rng[r] > 0 else (b - abs(rng[r]), b)
            return (b, b)
        if t == "L": return (-INF if r not in rng else b - abs(rng[r]), b)
        if t == "G": return (b, INF if r not in rng else b + abs(rng[r]))
        return None
    # primal fixpoint
    changed = True; prim = 0
    while changed:
        changed = False
        for r, ent in rows.items():
            s_ = sides(r)
            if not s_ or s_[0] != s_[1]: continue
            un = [(c, v) for c, v in ent if c not in step]
            if len(un) != 1: continue
            s, a = un[0]; g = F(0); ok = True
            for c, v in ent:
                if c == s: continue
                fv = frac(v)
                if fv is None: ok = False; break
                g = fgcd(g, abs(fv) * step[c])
            fa = frac(a); fb = frac(s_[0])
            if not ok or fa is None or fb is None: continue
            L = g / abs(fa) if g else None
            off = fb / fa
            if L is None:  # fixed value
                continue
            if (off / L).denominator != 1: L = fgcd(L, abs(off))
            if L == 0: continue
            step[s] = L; prim += 1; changed = True
    base = dict(step); dual = 0
    for s in contcost:
        if s in base: continue
        L = F(0); ok = True
        for b in (lb.get(s, 0.0), ub.get(s, INF)):
            if abs(b) < INF:
                fb = frac(b)
                if fb is None: ok = False; break
                L = fgcd(L, abs(fb))
        for r, d in ((r, d) for r, d in cols[s].items() if r != obj):
            ent = rows[r]; a = d; fa = frac(a)
            if fa is None: ok = False; break
            g = F(0)
            for c, v in ent:
                if c == s: continue
                if c not in base: ok = False; break
                fv = frac(v)
                if fv is None: ok = False; break
                g = fgcd(g, abs(fv) * base[c])
            if not ok: break
            L = fgcd(L, g / abs(fa))
            for sd in sides(r):
                if abs(sd) < INF:
                    fs = frac(sd)
                    if fs is None: ok = False; break
                    L = fgcd(L, abs(fs / fa))
            if not ok: break
        if ok and L > 0:
            step[s] = L; dual += 1
    lat = all(c in step for c in cost)
    lat_primal_only = all(c in base for c in cost)
    og = F(0)
    if lat:
        for c, v in cost.items():
            fv = frac(v)
            if fv is None: lat = False; break
            og = fgcd(og, abs(fv) * step[c])
    return {"contcost": len(contcost), "prim": prim, "dual": dual, "lattice": lat, "primal_only": lat_primal_only, "step": str(og) if lat else None}
if __name__ == "__main__":
    for p in sys.argv[1:]:
        try: r = census(p)
        except Exception as e: r = {"err": repr(e)[:80]}
        print(json.dumps({"name": p.split("/")[-1].replace(".mps.gz", ""), **({"no_cont_cost": True} if r is None else r)}), flush=True)
