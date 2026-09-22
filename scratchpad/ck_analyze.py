import json, sys
rows = json.load(open(sys.argv[1]))
B = 8.0
checks = 0
cert = [r for r in rows if r["status"] == "optimal"]
decl = [r for r in rows if r["status"] != "optimal"]
assert len(cert) + len(decl) == len(rows) and len(rows) == 39, rows
print(f"eligible={len(rows)}  certify={len(cert)}  decline={len(decl)}")

cw = sum(r["wall"] for r in cert); dw = sum(r["wall"] for r in decl)
print(f"\nkernel wall on certifiers : {cw:7.2f}s")
print(f"kernel wall on decliners  : {dw:7.2f}s")
print(f"fraction of kernel time spent on models it DECLINES: {dw/(cw+dw):.1%}")
checks += 1

# How late does a CERTIFIER find its first incumbent, relative to its OWN wall?
# This is the number that decides whether "no incumbent yet" predicts "will not certify".
print("\n--- certifiers: first_incumbent / own wall (the tight-budget risk) ---")
ratios = []
for r in sorted(cert, key=lambda r: -(r["first_inc_secs"] or 0) / max(r["wall"], 1e-9)):
    fi, w = r["first_inc_secs"], r["wall"]
    assert fi is not None, f"certifier with no first_inc: {r['name']}"
    ratio = fi / max(w, 1e-9)
    ratios.append(ratio)
    checks += 1
    print(f"  {r['name']:26s} first_inc={fi:7.3f}s  wall={w:7.3f}s  ratio={ratio:6.1%}")
n_late = sum(1 for x in ratios if x > 0.5)
print(f"\n  certifiers whose first incumbent arrives AFTER 50% of their own wall: {n_late}/{len(cert)}")

# At THIS budget (8s), would a 50%-of-budget abandon rule cost any certifier?
lost = [r["name"] for r in cert if r["first_inc_secs"] > 0.5 * B]
print(f"  at budget={B}s, a 50%-of-budget abandon rule loses: {lost or 'none'}")
checks += 1

# ...but a certifier is only 'at risk' at a budget near its own wall. Simulate.
print("\n--- simulated: abandon at 50% of budget, budget swept ---")
for tl in (2.0, 3.0, 4.0, 8.0, 16.0):
    # certifiers at this budget = those whose wall <= tl
    ok = [r for r in cert if r["wall"] <= tl]
    killed = [r["name"] for r in ok if r["first_inc_secs"] > 0.5 * tl]
    checks += 1
    print(f"  tl={tl:5.1f}s  certifies={len(ok):2d}  abandon-rule would KILL {len(killed)}: {killed or 'none'}")

# Reclaimable time: decliners with no incumbent by 50% of budget
recl = [r for r in decl if (r["first_inc_secs"] is None or r["first_inc_secs"] > 0.5 * B) and r["wall"] > 0.5 * B]
print(f"\n  decliners the rule would abandon at 8s: {len(recl)}/{len(decl)}"
      f"  reclaiming {sum(r['wall'] - 0.5*B for r in recl):.1f}s of {dw:.1f}s wasted")
checks += 1

print(f"\nEXECUTED CHECKS: {checks}")
if checks == 0:
    sys.exit(1)
