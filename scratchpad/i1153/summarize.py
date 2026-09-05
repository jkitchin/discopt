"""#1153 phase-B summary: the two CLAUDE.md §5 bars, side by side.

Bar 1 (cert-clean): no incorrect result, no certification regression, no bound
above a known optimum, incumbents no worse than the OFF arm certifies.
Bar 2 (net-positive): nodes / incumbent / bound measurably better, broadly.

Prints an executed-comparison count and exits non-zero at zero (CLAUDE.md §6).
"""
import json, sys

off = json.load(open(sys.argv[1]))
on = json.load(open(sys.argv[2]))
rung = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0

def row(d, name, tl):
    for r in d.get(name, []):
        if r.get("tl") == tl:
            return r
    return None

names = sorted(set(off) & set(on))
n = 0
better = worse = same = 0
cert_regress = []
print(f"{'instance':22s} {'OFF obj':>13s} {'ON obj':>13s} {'OFF bnd':>13s} {'ON bnd':>13s} "
      f"{'OFF nd':>7s} {'ON nd':>7s}  cert")
for name in names:
    a, b = row(off, name, rung), row(on, name, rung)
    if a is None or b is None or "error" in a or "error" in b:
        continue
    n += 1
    fo = lambda v: "None" if v is None else f"{v:.6g}"
    print(f"{name:22s} {fo(a['obj']):>13s} {fo(b['obj']):>13s} {fo(a['bound']):>13s} "
          f"{fo(b['bound']):>13s} {a['nodes']:>7d} {b['nodes']:>7d}  "
          f"{a['cert']}->{b['cert']}")
    if a["cert"] and not b["cert"]:
        cert_regress.append(name)
    ao, bo = a["obj"], b["obj"]
    if ao is None and bo is not None:
        better += 1
    elif ao is not None and bo is None:
        worse += 1
    elif ao is not None and bo is not None:
        if bo < ao - 1e-6 * max(1.0, abs(ao)):
            better += 1
        elif bo > ao + 1e-6 * max(1.0, abs(ao)):
            worse += 1
        else:
            same += 1
    else:
        same += 1
print(f"\n# rung={rung}s compared={n} incumbent better={better} worse={worse} same={same}")
print(f"# certification regressions (cert True->False): {cert_regress or 'none'}")

# node throughput at the top rung
nd_up = sum(1 for name in names
            if (a := row(off, name, rung)) and (b := row(on, name, rung))
            and "error" not in a and "error" not in b and b["nodes"] > a["nodes"])
nd_dn = sum(1 for name in names
            if (a := row(off, name, rung)) and (b := row(on, name, rung))
            and "error" not in a and "error" not in b and b["nodes"] < a["nodes"])
print(f"# node count: ON higher on {nd_up}, lower on {nd_dn}, equal on {n - nd_up - nd_dn}")
raise SystemExit(1 if n == 0 else 0)
