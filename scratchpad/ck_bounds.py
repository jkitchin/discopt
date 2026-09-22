import json, sys, math
rows = json.load(open(sys.argv[1]))
decl = [r for r in rows if r["status"] != "optimal"]
checks = 0
finite = []
for r in decl:
    b = r["bound"]
    checks += 1
    ok = b is not None and math.isfinite(b) and abs(b) < 1e19
    if ok:
        finite.append(r)
print(f"decliners: {len(decl)}")
print(f"  of which the kernel PROVED a finite dual bound that is then discarded: {len(finite)}")
for r in sorted(finite, key=lambda r: r["name"]):
    inc = r["incumbent"]
    print(f"    {r['name']:26s} bound={r['bound']:>18.6g}  inc={'None' if inc is None else f'{inc:.6g}'}"
          f"  nodes={r['nodes']:5d}  {r['nodes']/max(r['wall'],1e-9):7.1f} nodes/s")
    checks += 1
cert = [r for r in rows if r["status"] == "optimal"]
print(f"\nnode throughput, certifiers : min={min(r['nodes']/max(r['wall'],1e-9) for r in cert):.1f} n/s")
print(f"node throughput, decliners  : "
      f"{sorted(round(r['nodes']/max(r['wall'],1e-9),1) for r in decl if r['wall']>1)}")
checks += 2
print(f"\nEXECUTED CHECKS: {checks}")
sys.exit(0 if checks else 1)
