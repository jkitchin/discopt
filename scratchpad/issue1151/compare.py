"""#1151 panel analysis: ON vs OFF over the vendored .nl corpus."""
import json
import sys
import tomllib

with open("scratchpad/issue1151/panel.json") as fh:
    P = json.load(fh)
assert P["marker"] > 0, "the #1151 sources were not loaded"
R = P["results"]
with open("python/tests/data/known_optima.toml", "rb") as fh:
    OPT = tomllib.load(fh)


def opt_of(n):
    e = OPT.get(n)
    return float(e["optimum"]) if isinstance(e, dict) and "optimum" in e else None


compared = 0
diffs, work_terminated, truncated, errors = [], [], [], []
oracle_bad = {"on": [], "off": []}
verify_calls = {"on": 0, "off": 0}
row_scale_calls = {"on": 0, "off": 0}

for n in sorted(R):
    a, b = R[n]["on"], R[n]["off"]
    for arm, rec in (("on", a), ("off", b)):
        verify_calls[arm] += rec.get("verify_calls", 0)
        row_scale_calls[arm] += rec.get("row_scale_calls", 0)
    if "error" in a or "error" in b:
        errors.append((n, a.get("error"), b.get("error")))
        continue
    compared += 1
    (truncated if "time_limit" in (a["status"], b["status"]) else work_terminated).append(n)
    row = []
    if a["status"] != b["status"]:
        row.append(f"status {b['status']} -> {a['status']}")
    for k in ("objective", "bound"):
        va, vb = a.get(k), b.get(k)
        if (va is None) != (vb is None):
            row.append(f"{k} {vb!r} -> {va!r}")
        elif va is not None and abs(va - vb) > 1e-9 * (1 + abs(vb)):
            row.append(f"{k} {vb!r} -> {va!r} (d={va - vb:+.3e})")
    if a.get("node_count") != b.get("node_count"):
        row.append(f"nodes {b.get('node_count')} -> {a.get('node_count')}")
    if row:
        diffs.append((n, "; ".join(row)))
    for arm, rec in (("on", a), ("off", b)):
        if rec.get("objective") is not None and rec.get("oracle_obj") is not None:
            d = rec["objective"] - rec["oracle_obj"]
            if abs(d) > 1e-6 * (1 + abs(rec["oracle_obj"])):
                oracle_bad[arm].append((n, f"{d:+.3e}"))

print(f"instances compared        : {compared}")
print(f"  terminated on work      : {len(work_terminated)}")
print(f"  truncated by time_limit : {len(truncated)}  (not a comparison — see panel.py)")
print(f"  errored in an arm       : {len(errors)}")
print(f"verify_point calls        : ON {verify_calls['on']}  OFF {verify_calls['off']}")
print(f"_row_scales calls (pass 2): ON {row_scale_calls['on']}  OFF {row_scale_calls['off']}")
print(f"rows where the two forms diverged: {P['stats']['divergent_rows']}")
print(f"\nRESULT DIFFERENCES (status / objective / bound / nodes): {len(diffs)}")
for n, d in diffs:
    print(f"    {n}: {d}")
print(f"\n#1151 ORACLE VIOLATIONS  ON: {len(oracle_bad['on'])}  OFF: {len(oracle_bad['off'])}")
for arm in ("on", "off"):
    for n, d in oracle_bad[arm]:
        print(f"    {arm} {n}: {d}")
print("\nDUAL BOUND ABOVE THE REFERENCE OPTIMUM:")
viol = 0
checked = 0
for n in sorted(R):
    o = opt_of(n)
    if o is None:
        continue
    for arm in ("on", "off"):
        bnd = R[n][arm].get("bound")
        if bnd is None:
            continue
        checked += 1
        if bnd > o + max(1e-4, 1e-4 * abs(o)):
            print(f"    {arm} {n}: bound {bnd:.9g} > optimum {o:.9g}")
            viol += 1
print(f"    checked {checked} (arm, instance) bounds against known_optima.toml; {viol} violations")
if errors:
    print("\nERRORS:")
    for n, ea, eb in errors:
        print(f"    {n}: on={ea} off={eb}")
print(f"\nexecuted comparisons: {compared}")
if compared == 0 or verify_calls["on"] == 0:
    sys.exit("ANALYSIS MEASURED NOTHING")
