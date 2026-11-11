import json
import statistics
import sys

OPT = {
    "b-ball": -1.5,
    "mik-250-20-75-1": -49716.0,
    "mik-250-20-75-2": -50768.0,
    "mik-250-20-75-3": -52242.0,
    "mik-250-20-75-5": -51532.0,
    "neos17": 0.1500025774,
    "neos-911970": 54.76,
    "gsvm2rl3": 0.33652753,
}
rows = [json.loads(ln) for ln in open(sys.argv[1])]
groups = {}
checks = 0
for r in rows:
    opt = OPT[r["name"]]
    tol = 1e-6 * (1 + abs(opt))
    assert r["bound"] <= opt + tol, ("bound past oracle", r)
    checks += 1
    if r["gap_certified"]:
        assert r["objective"] <= opt + 1e-4 * (1 + abs(opt)), ("bad certified obj", r)
        checks += 1
    if r["arm"] == "obj":
        assert r["ctr"].get("HighsObjPropAttached", 0) > 0, ("did not fire", r)
        checks += 1
    groups.setdefault((r["name"], r["max_nodes"]), {}).setdefault(r["arm"], []).append(r)


def gap(r):
    opt = OPT[r["name"]]
    b = opt if r["gap_certified"] else r["bound"]
    return 100 * (opt - b) / max(abs(opt), 1e-9)


HEAD = (
    "| instance | nodes | off gap % (reps) | obj gap % (reps) | off nodes | obj nodes "
    "| off wall | obj wall | obj changes | obj-infeasible | verdict |"
)
print(HEAD)
print("|---|---|---|---|---|---|---|---|---|---|---|")


def fmt(xs):
    return "/".join(f"{x:.3f}" for x in xs)


def join(xs):
    return "/".join(str(x) for x in xs)


for (name, mn), arms in groups.items():
    off, ob = arms.get("off", []), arms.get("obj", [])
    og = [gap(r) for r in off]
    bg = [gap(r) for r in ob]
    opt = OPT[name]
    ob_b = [opt if r["gap_certified"] else r["bound"] for r in off]
    bb_b = [opt if r["gap_certified"] else r["bound"] for r in ob]
    spread = max(ob_b) - min(ob_b) if ob_b else 0
    thr = max(2 * spread, 1e-6 * (1 + abs(opt)))
    worse = bool(bb_b and ob_b and max(bb_b) < min(ob_b) - thr)
    better = bool(bb_b and ob_b and min(bb_b) > max(ob_b) + thr)
    v = "WORSE" if worse else ("better" if better else "within noise")
    cells = [
        name,
        str(mn),
        fmt(og),
        fmt(bg),
        join(r["nodes"] for r in off),
        join(r["nodes"] for r in ob),
        fmt([r["wall"] for r in off]),
        fmt([r["wall"] for r in ob]),
        join(r["ctr"].get("HighsObjPropChanges") for r in ob),
        join(r["ctr"].get("HighsObjPropInfeasible") for r in ob),
        v,
    ]
    print("| " + " | ".join(cells) + " |")
    if og and bg:
        mo, mb = statistics.median(og), statistics.median(bg)
        rel = 100 * (mb - mo) / max(mo, 1e-12)
        print(f"  median gap off {mo:.4f} obj {mb:.4f} change {rel:+.1f} %")
print("executed checks:", checks)
sys.exit(0 if checks else 1)
