"""Before/after comparison for #1199 over the in-repo MINLPLib corpus."""

import json
import sys

before = {r["name"]: r for r in json.load(open("scratchpad/i1199/panel_1199_before.json"))}
after = {r["name"]: r for r in json.load(open("scratchpad/i1199/panel_1199_after.json"))}
names = sorted(set(before) & set(after))
assert names, "no instances in common — nothing was compared"


def close(a, b, rel=1e-9):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= rel * (1.0 + max(abs(a), abs(b)))


rows_status, rows_obj, rows_nodes, rows_ratio, rows_err = [], [], [], [], []
compared = 0
for n in names:
    b, a = before[n], after[n]
    compared += 1
    if b.get("error") or a.get("error"):
        rows_err.append((n, b.get("error"), a.get("error")))
        continue
    if b["status"] != a["status"]:
        rows_status.append((n, b["status"], a["status"]))
    if not close(b.get("obj"), a.get("obj"), 1e-9):
        rows_obj.append((n, b.get("obj"), a.get("obj")))
    if b.get("nodes") != a.get("nodes"):
        rows_nodes.append((n, b.get("nodes"), a.get("nodes")))
    if b.get("ratio") is not None and a.get("ratio") is not None:
        if not close(b["ratio"], a["ratio"], 1e-6):
            rows_ratio.append((n, b["ratio"], a["ratio"]))

print(f"instances compared: {compared}")


def dump(title, rows):
    print(f"\n{title}: {len(rows)}")
    for r in rows:
        print("   ", r)


dump("status changed (before -> after)", rows_status)
dump("objective changed", rows_obj)
dump("node_count changed", rows_nodes)
dump("acceptance ratio changed", rows_ratio)
dump("errors", rows_err)

bad_b = [r["name"] for r in before.values() if r.get("accepts") is False]
bad_a = [r["name"] for r in after.values() if r.get("accepts") is False]
print(f"\nincumbents FAILING the acceptance arbiter — before: {bad_b}  after: {bad_a}")

# certification: no instance may lose a certificate it had
lost = [n for n in names if before[n].get("cert") and not after[n].get("cert")]
gained = [n for n in names if after[n].get("cert") and not before[n].get("cert")]
print(f"certificates lost: {lost}   gained: {gained}")
if compared == 0:
    sys.exit(1)
