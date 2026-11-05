"""#2142 §116: is the convexity-memo fix bound-neutral on the corpus? Compare this run's OFF
arm with §115's OFF arm (before the fix), deterministic 2,000-node cells where neither run hit
the wall backstop. Prints executed-comparison count (§6)."""
import json
def rows(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[r["instance"]] = r
    return out
old = rows("../panel/corpus.json.partial.jsonl"); new = rows("corpus.json.partial.jsonl")
k0 = next(iter(new.values())); print("row keys:", sorted(k0)[:30])
n = diff = 0
for name in sorted(set(old) & set(new)):
    o, w = old[name]["off"], new[name]["off"]
    if max(o.get("wall", 0), w.get("wall", 0)) >= 110: continue
    n += 1
    key = ("status", "objective", "bound", "nodes", "gap_certified")
    assert all(x in o and x in w for x in key), (name, sorted(o))
    if any(o.get(x) != w.get(x) for x in key):
        diff += 1; print("DIFF", name, {x: (o.get(x), w.get(x)) for x in key})
print(f"compared {n}, differing {diff}")
print("executed comparisons:", n)
