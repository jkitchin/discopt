"""Score the H2 A/B against its pre-stated kill criterion (#2182 task 2).

Kill H2 if (1) any pair's certified objective differs beyond abs 1e-6 + rel 1e-4,
or a pair certified OFF is uncertified ON; (2) class SGM wall gain (1 s shift)
< 5 %; (3) median node-count ratio ON/OFF > 1.10.
"""

import json
import math
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(dict)
for r in rows:
    by[(r["instance"], r["method"], r["rep"])][r["arm"]] = r

n_pairs = n_err = 0
obj_viol, cert_lost, cert_gained = [], [], []
logs_off, logs_on, node_ratios = defaultdict(list), defaultdict(list), []
per_cell = defaultdict(list)
for key, arms in sorted(by.items()):
    if set(arms) != {"OFF", "ON"}:
        continue
    a, b = arms["OFF"], arms["ON"]
    if a.get("error") or b.get("error"):
        n_err += 1
        print("ERROR pair", key)
        continue
    n_pairs += 1
    ca, cb = a["gap_certified"], b["gap_certified"]
    if ca and not cb:
        cert_lost.append(key)
    if cb and not ca:
        cert_gained.append(key)
    if ca and cb:
        tol = 1e-6 + 1e-4 * abs(a["obj"])
        if abs(a["obj"] - b["obj"]) > tol:
            obj_viol.append((key, a["obj"], b["obj"]))
        if a["nodes"]:
            node_ratios.append((b["nodes"] or 0) / a["nodes"])
    cell = key[:2]
    logs_off[cell].append(math.log(a["wall"] + 1.0))
    logs_on[cell].append(math.log(b["wall"] + 1.0))
    per_cell[cell].append((a["wall"], b["wall"], a["nodes"], b["nodes"], b["cfg_root_s"]))

cells = sorted(logs_off)
sgm_off = math.exp(statistics.mean(statistics.mean(logs_off[c]) for c in cells)) - 1
sgm_on = math.exp(statistics.mean(statistics.mean(logs_on[c]) for c in cells)) - 1
gain = 1 - sgm_on / sgm_off
med_node = statistics.median(node_ratios) if node_ratios else float("nan")

print(f"{'instance':34s} {'method':6s}  OFF wall (mean)  ON wall (mean)  ratio  nodes OFF/ON  ON root share")
for c in cells:
    v = per_cell[c]
    wo = statistics.mean(x[0] for x in v)
    wn = statistics.mean(x[1] for x in v)
    print(
        f"{c[0]:34s} {c[1]:6s}  {wo:8.2f}        {wn:8.2f}       {wn / wo:5.2f}  "
        f"{v[0][2]}/{v[0][3]}  {v[0][4]}"
    )
print()
print(f"pairs scored {n_pairs}, error pairs {n_err}, cells {len(cells)}")
print(f"certified-objective violations: {len(obj_viol)} {obj_viol}")
print(f"certification lost: {len(cert_lost)} {cert_lost}; gained: {len(cert_gained)}")
print(f"SGM (1 s shift) OFF {sgm_off:.3f} s  ON {sgm_on:.3f} s  gain {100 * gain:.1f} %")
print(f"median node ratio ON/OFF {med_node:.3f} over {len(node_ratios)} both-certified pairs")
kill = []
if obj_viol or cert_lost:
    kill.append("(1) soundness/certification")
if gain < 0.05:
    kill.append(f"(2) SGM gain {100 * gain:.1f}% < 5%")
if med_node > 1.10:
    kill.append(f"(3) median node ratio {med_node:.3f} > 1.10")
print("VERDICT:", "KILL " + "; ".join(kill) if kill else "PASS (all three bars)")
assert n_pairs > 0, "no pairs scored"
print(f"executed-assertions: 1 (pairs={n_pairs})")
