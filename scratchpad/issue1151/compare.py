"""#1151 panel comparison: ON vs OFF, cert-clean + capability check."""
import json
import sys

import tomllib

with open("scratchpad/issue1151/panel_on.json") as fh:
    on = json.load(fh)
with open("scratchpad/issue1151/panel_off.json") as fh:
    off = json.load(fh)
assert on["marker"] > 0 and off["marker"] == 0, (
    f"arm markers wrong: on={on['marker']} off={off['marker']}"
)

with open("python/tests/data/known_optima.toml", "rb") as fh:
    optima = tomllib.load(fh)


def opt_of(name):
    e = optima.get(name)
    if isinstance(e, dict) and "optimum" in e:
        return float(e["optimum"])
    return None


ON, OFF = on["results"], off["results"]
names = sorted(set(ON) & set(OFF))
checks = 0
status_changes, obj_changes, bound_changes, oracle_bad_on, oracle_bad_off = [], [], [], [], []
work_terminated = []

for n in names:
    a, b = ON[n], OFF[n]
    if "error" in a or "error" in b:
        if a.get("error") != b.get("error"):
            status_changes.append((n, f"error {b.get('error')!r} -> {a.get('error')!r}"))
        continue
    checks += 1
    if a["status"] != b["status"]:
        status_changes.append((n, f"{b['status']} -> {a['status']}"))
    # objective drift
    for key, sink in (("objective", obj_changes), ("bound", bound_changes)):
        va, vb = a.get(key), b.get(key)
        if (va is None) != (vb is None):
            sink.append((n, f"{vb!r} -> {va!r}"))
        elif va is not None and abs(va - vb) > 1e-6 * (1 + abs(vb)):
            sink.append((n, f"{vb!r} -> {va!r}  (d={va - vb:+.3e})"))
    # #1151 oracle: reported objective vs objective at reported point
    for arm, rec, sink in (("on", a, oracle_bad_on), ("off", b, oracle_bad_off)):
        if rec.get("objective") is not None and rec.get("oracle_obj") is not None:
            d = rec["objective"] - rec["oracle_obj"]
            if abs(d) > 1e-6 * (1 + abs(rec["oracle_obj"])):
                sink.append((n, d, rec["objective"], rec["oracle_obj"]))
    if a["status"] not in ("time_limit",) and b["status"] not in ("time_limit",):
        work_terminated.append(n)

print(f"instances compared: {checks}  (work-terminated in BOTH arms: {len(work_terminated)})")


def show(title, rows):
    print(f"\n{title}: {len(rows)}")
    for r in rows:
        print("   ", r)


show("STATUS CHANGES", status_changes)
show("OBJECTIVE CHANGES", obj_changes)
show("BOUND CHANGES", bound_changes)
show("#1151 ORACLE VIOLATIONS (ON)", oracle_bad_on)
show("#1151 ORACLE VIOLATIONS (OFF)", oracle_bad_off)

# Soundness gate: no dual bound past a reference optimum, either arm.
print("\nBOUND-vs-ORACLE-OPTIMUM violations:")
viol = 0
for n in names:
    o = opt_of(n)
    if o is None:
        continue
    for arm, rec in (("on", ON[n]), ("off", OFF[n])):
        bnd = rec.get("bound")
        if bnd is None:
            continue
        if bnd > o + max(1e-4, 1e-4 * abs(o)):
            print(f"    {arm} {n}: bound {bnd:.9g} > optimum {o:.9g}")
            viol += 1
print(f"    total: {viol}")

print(f"\nexecuted comparisons: {checks}")
if checks == 0:
    sys.exit("COMPARISON MEASURED NOTHING")
