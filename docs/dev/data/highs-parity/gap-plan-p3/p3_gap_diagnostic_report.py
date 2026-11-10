"""Report for diag.jsonl: per row, HiGHS vs discopt certification, root gap, final dual/primal gap, nodes.

Gaps are |oracle - value| / max(1, |oracle|) in %, so the sense does not matter. Each row HiGHS
certifies and discopt does not is put in one class, tested in this order:
  primal  discopt's final dual gap <= 0.01% (the bound is done; the incumbent is missing or worse)
  root    discopt's root gap > 2x HiGHS's final root gap and > 0.05% absolute
  tree    otherwise (comparable root; the search does not close it)
Prints the counts it compared (CLAUDE.md §6) and exits non-zero if none.
"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    by.setdefault(r["name"], {})[r["job"]] = r

def gap(opt, v):
    if opt is None or v is None or abs(v) == float("inf"):
        return None
    return abs(opt - v) / max(1.0, abs(opt)) * 100

def f(x, w=7):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.2f}"

out = []
for name, d in by.items():
    if not all(k in d for k in "HDR"):
        continue
    H, D, R = d["H"], d["D"], d["R"]
    opt = H["opt"]
    hc = H.get("status") == "Optimal"
    dc = D.get("status") == "optimal" and bool(D.get("gap_certified"))
    e = dict(name=name, hc=hc, dc=dc,
             h_root=gap(opt, H.get("root_bound")), h_root1=gap(opt, H.get("first_root_bound")),
             d_root=gap(opt, R.get("bound")), d_dual=gap(opt, D.get("bound")), d_primal=gap(opt, D.get("objective")),
             h_dual=gap(opt, H.get("bound")), h_nodes=H.get("nodes"), d_nodes=D.get("nodes"),
             restarts=H.get("restarts"), h_wall=H.get("wall"), d_wall=D.get("wall"), d_status=D.get("status"))
    cls = ""
    if hc and not dc:
        if e["d_dual"] is not None and e["d_dual"] <= 0.01:
            cls = "primal"
        elif e["d_root"] is not None and e["h_root"] is not None and e["d_root"] > 2 * e["h_root"] and e["d_root"] - e["h_root"] > 0.05:
            cls = "root"
        else:
            cls = "tree"
    elif dc and not hc:
        cls = "ours-only"
    e["cls"] = cls
    out.append(e)

order = {"root": 0, "tree": 1, "primal": 2, "ours-only": 3, "": 4}
out.sort(key=lambda e: (order[e["cls"]], e["name"]))
print(f"{'row':24} {'class':9} {'H':>2} {'D':>2} {'Hroot1%':>7} {'Hroot%':>7} {'Droot%':>7} {'Hdual%':>7} {'Ddual%':>7} {'Dprim%':>7} {'Hnodes':>7} {'Dnodes':>8} rst {'Hwall':>5} {'Dwall':>5}")
for e in out:
    print(f"{e['name'][:24]:24} {e['cls']:9} {'Y' if e['hc'] else '.':>2} {'Y' if e['dc'] else '.':>2} {f(e['h_root1'])} {f(e['h_root'])} {f(e['d_root'])} {f(e['h_dual'])} {f(e['d_dual'])} {f(e['d_primal'])} "
          f"{e['h_nodes'] if e['h_nodes'] is not None else '-':>7} {e['d_nodes'] if e['d_nodes'] is not None else '-':>8} {e['restarts'] or 0:>3} {e['h_wall'] or 0:5.1f} {e['d_wall'] or 0:5.1f}")
n = len(out)
print(f"\nrows compared: {n}; HiGHS certified {sum(e['hc'] for e in out)}; discopt certified {sum(e['dc'] for e in out)}")
for c in ("root", "tree", "primal", "ours-only"):
    print(f"  {c:9} {sum(e['cls'] == c for e in out)}")
both = [e for e in out if e["h_root"] is not None and e["d_root"] is not None]
worse = sum(e["d_root"] > 2 * e["h_root"] and e["d_root"] - e["h_root"] > 0.05 for e in both)
print(f"root gap compared on {len(both)} rows: discopt's root > 2x HiGHS's on {worse}")
sys.exit(0 if n else 1)
