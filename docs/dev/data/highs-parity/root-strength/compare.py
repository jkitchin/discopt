import json, sys
H = {}
for l in open("docs/dev/data/highs-parity/gap-plan-p3/p3_gap_diagnostic.jsonl"):
    r = json.loads(l)
    if r["job"] == "H": H[r["name"]] = r
cert = {}
for l in open(sys.argv[2]):
    r = json.loads(l)
    if r["arm"] == "graduated": cert[r["name"]] = r["status"] == "optimal" and bool(r.get("gap_certified"))
def gap(b, o):
    return None if b is None or o is None else abs(o - b) / max(1.0, abs(o)) * 100
rows = []
for l in open(sys.argv[1]):
    d = json.loads(l); h = H.get(d["name"], {}); o = d["opt"]
    hr = gap(h.get("root_bound"), o); dr = gap(d["bound"], o)
    rows.append((d["name"], hr, dr, h.get("status"), cert.get(d["name"]), h.get("nodes")))
print(f"{'row':24} {'Hroot%':>8} {'Droot%':>8} {'Dcert':>5} {'Hstat':>8} Hnodes")
for n, hr, dr, hs, c, hn in sorted(rows, key=lambda r: -((r[2] or 0) - (r[1] or 0))):
    f = lambda v: "-" if v is None else f"{v:8.2f}"
    print(f"{n:24} {f(hr)} {f(dr)} {str(c):>5} {str(hs):>8} {hn}")
