# ruff: noqa -- frozen #1731 panel record, kept as run.
"""Judge the #1731 panel against both CLAUDE.md §5 bars. argv: panel.jsonl"""

import collections
import json
import math
import statistics
import sys

recs = [json.loads(line) for line in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in recs:
    by[r["name"]][r["arm"]] = r
pairs = {k: v for k, v in by.items() if "on" in v and "off" in v}
print(f"records {len(recs)}  paired instances {len(pairs)}")

checks = 0
problems = []


def tol(x):
    return 1e-6 + 1e-4 * abs(x)


for name, p in sorted(pairs.items()):
    for arm in ("off", "on"):
        r = p[arm]
        if r.get("crash"):
            problems.append((name, arm, "crash", r.get("rc"), r.get("stderr", "")[-200:]))
            continue
        ref, s = r["ref"], r["sense"]
        checks += 1
        # bound must not cross the reference optimum
        if r["bound"] is not None and math.isfinite(r["bound"]):
            if (s == "min" and r["bound"] > ref + tol(ref)) or (
                s == "max" and r["bound"] < ref - tol(ref)
            ):
                problems.append((name, arm, "bound crosses ref", r["bound"], ref))
            if r["objective"] is not None:
                if (s == "min" and r["bound"] > r["objective"] + tol(ref)) or (
                    s == "max" and r["bound"] < r["objective"] - tol(ref)
                ):
                    problems.append(
                        (name, arm, "bound crosses incumbent", r["bound"], r["objective"])
                    )
        if r["objective"] is not None and r["feas"] is not True:
            problems.append((name, arm, "incumbent not verified", r["feas"], r["viol"]))
        if r["status"] == "optimal" and abs(r["objective"] - ref) > tol(ref):
            problems.append((name, arm, "false optimal", r["objective"], ref))
        if r["status"] == "infeasible":
            problems.append((name, arm, "false infeasible", None, ref))
    on, off = p["on"], p["off"]
    if not on.get("crash") and not off.get("crash"):
        if off["cert"] and not on["cert"]:
            problems.append((name, "on", "certification regression", off["status"], on["status"]))

print(f"executed soundness checks {checks}")
if checks == 0:
    sys.exit(1)
print("problems:")
for x in problems:
    print("  ", x)

fired = [n for n, p in pairs.items() if p["on"].get("lat_calls", 0) > 0]
changed = [n for n, p in pairs.items() if p["on"].get("lat_changed", 0) > 0]
off_fired = [n for n, p in pairs.items() if p["off"].get("lat_calls", 0) > 0]
print(
    f"\nlattice rounding called (ON) on {len(fired)}; changed a bound on {len(changed)}; "
    f"called with flag OFF on {len(off_fired)} (must be 0)"
)
routes = collections.Counter(p["off"].get("route", "")[:24] for p in pairs.values())
print("routes (OFF):", dict(routes))

print("\nper-instance where rounding fired:")
print(
    f"{'name':<14}{'off st':>9}{'on st':>9}{'off nodes':>10}{'on nodes':>10}"
    f"{'off wall':>9}{'on wall':>9}  off bnd / on bnd / ref"
)
rat = []
for n in sorted(fired):
    a, b = pairs[n]["off"], pairs[n]["on"]
    print(
        f"{n:<14}{a['status']:>9}{b['status']:>9}{a['nodes']:>10}{b['nodes']:>10}"
        f"{a['wall']:>9.1f}{b['wall']:>9.1f}  {a['bound']} / {b['bound']} / {a['ref']}"
    )
    rat.append(math.log(max(b["wall"], 0.1) / max(a["wall"], 0.1)))
if rat:
    print(
        f"\nwall ON/OFF geomean over fired: {math.exp(statistics.mean(rat)):.3f} "
        f"(log sd {statistics.pstdev(rat):.3f}, n={len(rat)})"
    )
for arm in ("off", "on"):
    nopt = sum(1 for p in pairs.values() if p[arm].get("cert"))
    print(f"{arm}: gap_certified {nopt}/{len(pairs)}")
neutral = [
    n
    for n, p in pairs.items()
    if n not in fired
    and not p["on"].get("crash")
    and not p["off"].get("crash")
    and (p["on"]["nodes"] != p["off"]["nodes"] or p["on"]["status"] != p["off"]["status"])
]
print(
    f"non-fired instances whose nodes/status differ ON vs OFF (timing noise expected only "
    f"on time-limited runs): {len(neutral)} {neutral[:10]}"
)
