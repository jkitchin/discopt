"""Per-instance gap attribution for D2 from checked-in data (read-only).

Sources:
  R (current default): docs/dev/data/highs-parity/d2-analysis/tol_robust_panel_v2_r3.json.gz, arm 'on'
     (TOL_ROBUST graduated default ON; every other graduated D2 lever is ON in both arms).
  H: docs/dev/data/highs-parity/p6.6/miplib.json + lmi.json (panel M), arm 'H'.
Prints an executed-row count and exits 2 if zero.
"""
import gzip, json, statistics as st, sys

ROOT = "/home/user/discopt/docs/dev/data/highs-parity/"
panel = json.load(gzip.open(ROOT + "d2-analysis/tol_robust_panel_v2_r3.json.gz"))
print("panel header:", panel["header"], "args:", {k: panel["args"][k] for k in ("flag", "time_limit", "gap", "reps")})
rows = panel["rows"]

h = {}
for f, want in (("p6.6/miplib.json", None), ("p6.6/lmi.json", "M")):
    d = json.load(open(ROOT + f))
    for r in d["rows"]:
        if want and r["panel"] != want:
            continue
        h[(r["name"], r["panel"])] = r

by = {}
for r in rows:
    by.setdefault((r["name"], r["panel"]), []).append(r)

def gap(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(1.0, abs(b))

out = []
n_exec = 0
for key, reps in sorted(by.items()):
    name, pan = key
    opt = reps[0]["opt"]
    on = [r["on"] for r in reps]
    off = [r["off"] for r in reps]
    n_exec += len(on)
    cert_on = sum(1 for a in on if a["gap_certified"])
    cert_off = sum(1 for a in off if a["gap_certified"])
    pg = [gap(a["objective"], opt) for a in on]
    dg = [gap(a["bound"], opt) for a in on]
    inc = sum(1 for a in on if a["objective"] is not None)
    nodes = st.median([a["nodes"] for a in on])
    wall = st.median([a["wall"] for a in on])
    hr = h.get(key)
    hc = hr["H"]["gap_certified"] if hr else None
    hw = hr["H"]["wall_med"] if hr else None
    hn = hr["H"]["nodes"] if hr else None
    rc = hr["R"]["gap_certified"] if hr else None
    pgm = st.median([x for x in pg if x is not None]) if any(x is not None for x in pg) else None
    dgm = st.median([x for x in dg if x is not None]) if any(x is not None for x in dg) else None
    # cause class for rows not certified 3/3
    cls = ""
    if cert_on == 3:
        cls = "cert"
    elif inc == 0:
        cls = "G2 no-incumbent" + (" (LP-stall)" if nodes < 100 else "")
    elif nodes < 100:
        cls = "G5 LP-throughput"
    elif dgm is not None and dgm <= 2e-4 and pgm is not None and pgm > 1e-4:
        cls = "G3 primal-only (bound at opt)"
    elif pgm is not None and pgm <= 1e-4 and dgm is not None and dgm > 1e-4:
        cls = "G4 dual-only (incumbent opt)"
    elif pgm is not None and dgm is not None and pgm <= 0.01 and dgm <= 0.01:
        cls = "G1 near (both <=1%)"
    else:
        cls = "G3/G4 both"
    out.append((pan, name, cert_on, cert_off, inc, pgm, dgm, int(nodes), wall, hc, hn, hw, rc, cls))

print(f"\nexecuted ON solves: {n_exec}; instances: {len(out)}")
if n_exec == 0:
    sys.exit(2)

def f(x, w=8, p=4):
    return f"{'—':>{w}}" if x is None else f"{x:{w}.{p}f}"

print(f"\n{'panel':6} {'instance':24} {'ON':>3} {'OFF':>3} {'inc':>3} {'pgap':>8} {'dgap':>8} {'nodes':>7} {'wall':>6} {'H':>2} {'Hnodes':>7} {'Hwall':>6} {'Rp66':>4}  class")
tot_on = tot_off = 0
for pan, name, c_on, c_off, inc, pgm, dgm, nodes, wall, hc, hn, hw, rc, cls in sorted(out, key=lambda t: (t[-1] != 'cert', t[0], t[1])):
    tot_on += c_on; tot_off += c_off
    print(f"{pan:6} {name:24} {c_on:>3} {c_off:>3} {inc:>3} {f(pgm)} {f(dgm)} {nodes:>7} {wall:6.1f} {('Y' if hc else ('N' if hc is not None else '?')):>2} {str(hn) if hn is not None else '—':>7} {f(hw,6,1)} {('Y' if rc else ('N' if rc is not None else '?')):>4}  {cls}")
print(f"\ncertified solves ON {tot_on} / OFF {tot_off} of {n_exec}")

# class summary over rows not 3/3
print("\nclass summary (instances not certified 3/3 in ON):")
summ = {}
for t in out:
    if t[-1] != "cert":
        summ.setdefault(t[-1], []).append(t[1] + (f"({t[2]}/3)" if t[2] else ""))
for k, v in sorted(summ.items()):
    print(f"  {k:32} {len(v):2}  {', '.join(v)}")

# Which of these does H certify?
print("\nnot-3/3 rows that H certifies:", sum(1 for t in out if t[-1] != 'cert' and t[9]))
print("not-3/3 rows that H fails too:", [t[1] for t in out if t[-1] != 'cert' and t[9] is False])
# MIPLIB-only count
mip = [t for t in out if t[0] == "MIPLIB"]
print(f"\nMIPLIB-38: ON certified-all-reps {sum(1 for t in mip if t[2]==3)}, any-rep {sum(1 for t in mip if t[2]>0)}, mean/rep {sum(t[2] for t in mip)/3:.1f}; H certified {sum(1 for t in mip if t[9])}")
mp = [t for t in out if t[0] == "M"]
print(f"M-24:      ON certified-all-reps {sum(1 for t in mp if t[2]==3)}, any-rep {sum(1 for t in mp if t[2]>0)}, mean/rep {sum(t[2] for t in mp)/3:.1f}; H certified {sum(1 for t in mp if t[9])}")
# wall totals (indicative; different hosts/dates)
print(f"ON total median wall MIPLIB {sum(t[8] for t in mip):.0f} s vs H p6.6 {sum(t[11] for t in mip if t[11]):.0f} s; M {sum(t[8] for t in mp):.0f} vs {sum(t[11] for t in mp if t[11]):.0f}")
