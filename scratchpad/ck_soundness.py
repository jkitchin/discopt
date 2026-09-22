"""Entry experiment (#1422): is a DECLINED convex-kernel attempt's dual bound sound?

Kill criterion: any instance whose discarded bound lies ABOVE its reference
optimum (min sense) by more than tolerance falsifies the whole mechanism.
"""
import json, math, re, sys, pathlib

SOLU = pathlib.Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib.solu"
rows = json.load(open(sys.argv[1]))

# Parse the oracle. Keep the marker: =opt= is proven, =best= is only an incumbent.
opt, best = {}, {}
for line in SOLU.read_text().splitlines():
    p = line.split()
    if len(p) < 3:
        continue
    tag, name, val = p[0], p[1], p[2]
    try:
        v = float(val)
    except ValueError:
        continue
    if tag == "=opt=":
        opt[name] = v
    elif tag == "=best=":
        best[name] = v
assert len(opt) > 100, f"oracle parse produced only {len(opt)} =opt= rows"
print(f"oracle: {len(opt)} =opt=, {len(best)} =best=")

# Sense: every instance here is a minimization in .nl terms; verify via the
# certifiers, whose incumbent must be >= bound and ~= the reference optimum.
TOL_REL, TOL_ABS = 1e-6, 1e-6
checked = violations = 0
print("\n--- DECLINED attempts: is the discarded bound <= true optimum? ---")
for r in sorted(rows, key=lambda r: r["name"]):
    if r["status"] == "optimal":
        continue
    b = r["bound"]
    if b is None or not math.isfinite(b) or abs(b) >= 1e19:
        continue
    ref = opt.get(r["name"])
    kind = "opt"
    if ref is None:
        ref = best.get(r["name"]); kind = "best"
    if ref is None:
        print(f"  {r['name']:26s} bound={b:>14.6g}   NO ORACLE ENTRY -- skipped")
        continue
    checked += 1
    slack = ref - b                      # must be >= 0 for a valid min-sense bound
    tol = TOL_ABS + TOL_REL * max(1.0, abs(ref))
    bad = slack < -tol
    violations += bad
    print(f"  {r['name']:26s} bound={b:>14.6g}  {kind}={ref:>14.6g}  "
          f"slack={slack:>13.6g}  {'*** VIOLATION ***' if bad else 'ok'}")

# Cross-check the CERTIFIERS too: their reported optimum must match the oracle.
print("\n--- CERTIFIED attempts: does the certified optimum match the oracle? ---")
cert_checked = cert_bad = 0
for r in sorted(rows, key=lambda r: r["name"]):
    if r["status"] != "optimal" or r["incumbent"] is None:
        continue
    ref = opt.get(r["name"], best.get(r["name"]))
    if ref is None:
        continue
    cert_checked += 1
    rel = abs(r["incumbent"] - ref) / max(1.0, abs(ref))
    bad = rel > 1e-4
    cert_bad += bad
    if bad:
        print(f"  {r['name']:26s} got={r['incumbent']:.8g} ref={ref:.8g} rel={rel:.3g} *** MISMATCH ***")
print(f"  {cert_checked} certifiers checked, {cert_bad} mismatched")

print(f"\nEXECUTED COMPARISONS: {checked + cert_checked}")
print(f"BOUND VIOLATIONS    : {violations}")
print(f"CERT MISMATCHES     : {cert_bad}")
if checked + cert_checked == 0:
    print("PROBE MEASURED NOTHING"); sys.exit(1)
sys.exit(2 if (violations or cert_bad) else 0)
