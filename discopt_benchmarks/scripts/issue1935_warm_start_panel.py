"""Entry experiment for #1935: does a parent-basis warm start close the gap to HiGHS?

#1935 put HiGHS, warm-started from the parent node's basis, under the native
spatial kernel and measured the median per-node LP time fall 6.27 -> 0.81 ms. Its
baseline was the kernel's COLD in-house simplex, so the number does not say how
much is the engine and how much is the warm start. This panel measures the warm
start alone, same engine, three arms of ``DISCOPT_NATIVE_WARM_START``: ``0``
cold (the default), ``1`` parent-basis warm start
(``SpatialTreeConfig::warm_start_children``), ``2`` warm start plus the
bound-flip repair of a dual-infeasible inherited basis
(``SimplexOptions::warm_flip_boxed``). Interleaved per instance (the arm order
rotates), each solve in its own single-threaded subprocess.

Instances: the in-repo MINLPLib corpus (``python/tests/data/minlplib_nl``) and
the pooling generator (Haverly 1-3; seeded plants at the default size and at two
larger sizes, whose node LPs cost 7-32 ms cold) -- pooling is the class where
#1935 gained most. Only kernel-routed instances are scored.

Per solve: status, objective, bound, ``gap_certified``, nodes, wall, route, the
kernel's ``tree/*`` counters, and an independent ``verify_point`` on the
published incumbent. Reported per arm: false certificates and bounds (against
``docs/dev/data/cert-optima.json`` where it has the instance, and against the
other arms' verified incumbents everywhere), bad points, certificates lost/gained,
solved count, shifted geometric mean of wall (1 s shift, unsolved at the limit),
and the median per-instance node-LP ms and pivots per node.

Exits non-zero when it scored nothing or a warm arm never warm-started
(CLAUDE.md §6).

    python -u discopt_benchmarks/scripts/issue1935_warm_start_panel.py \\
        --time-limit 60 --jobs 3 --out panel.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS = os.path.join(REPO, "python", "tests", "data", "minlplib_nl")
OPTIMA = os.path.join(REPO, "docs", "dev", "data", "cert-optima.json")
FLAG = "DISCOPT_NATIVE_WARM_START"
ARMS = ("0", "1", "2")
TOL_ABS, TOL_REL = 1e-6, 1e-4


def _instances() -> list[str]:
    out = [f"nl:{os.path.basename(p)[:-3]}" for p in sorted(glob.glob(f"{CORPUS}/*.nl"))]
    for form in ("p", "q", "pq"):
        out += [f"haverly:{case}:{form}" for case in (1, 2, 3)]
        out += [f"plant:{seed}:{form}" for seed in (5, 6, 7)]
    # Larger plants (pools, products, qualities): the expensive node LPs.
    for size in ("3:4:2", "4:5:3"):
        for form in ("q", "pq"):
            out += [f"plant:{seed}:{form}:{size}" for seed in (5, 6, 7)]
    return out


def _build(name: str):
    kind, *rest = name.split(":")
    if kind == "nl":
        from discopt.modeling.core import from_nl

        return from_nl(os.path.join(CORPUS, rest[0] + ".nl"))
    sys.path.insert(0, HERE)
    from _pooling import pool18

    if kind == "haverly":
        d = pool18.haverly(int(rest[0]))
    else:
        size = dict(zip(("nL", "nJ", "K"), map(int, rest[2:]), strict=False))
        d = pool18.plant(seed=int(rest[0]), **size)
    return pool18.build_pool(d, rest[1])[0]


def _worker(name: str, arm: str, tl: float) -> None:
    """One solve in this process; prints one JSON line. Exceptions propagate."""
    os.environ[FLAG] = arm
    import discopt
    import discopt._rust as _rust
    import numpy as np
    from discopt.validation.feasibility import verify_point

    # CLAUDE.md §8: the build under test must be the one carrying both switches.
    sig = _rust.solve_spatial_tree_py.__text_signature__ or ""
    assert "warm_start_children" in sig and "warm_flip_boxed" in sig, sig
    model = _build(name)
    t0 = time.perf_counter()
    r = model.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    ok = None
    if r.x is not None and r.objective is not None:
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in model._variables]
        )
        ok = bool(verify_point(model, flat).ok)
    stats = {k: v for k, v in (r.solver_stats or {}).items() if str(k).startswith("tree/")}
    rec = {
        "name": name,
        "arm": arm,
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "certified": bool(r.status == "optimal" and r.gap_certified),
        "nodes": int(r.node_count or 0),
        "wall": wall,
        "route": r.algorithm_route,
        "point_ok": ok,
        "sense": str(model._objective.sense),
        "stats": stats,
        "discopt": discopt.__file__,
    }
    print("RESULT " + json.dumps(rec, default=float), flush=True)


def _run_one(name: str, arm: str, tl: float) -> dict:
    cmd = [sys.executable, "-u", __file__, "--worker", name, arm, str(tl)]
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    t0 = time.perf_counter()

    def failed(why: str) -> dict:
        return {"name": name, "arm": arm, "error": why, "wall": time.perf_counter() - t0}

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * tl + 120, env=env)
    except subprocess.TimeoutExpired:
        return failed("subprocess timeout")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return failed(" | ".join((p.stderr or "").strip().splitlines()[-3:])[:300])


def _group(args) -> list[dict]:
    i, name, tl = args
    k = i % len(ARMS)
    out = [_run_one(name, a, tl) for a in ARMS[k:] + ARMS[:k]]
    for rec in out:
        s = rec.get("stats", {})
        print(
            f"[{i}] {name:28s} arm={rec['arm']} status={rec.get('status', 'ERR'):11s} "
            f"cert={rec.get('certified')} nodes={rec.get('nodes')} wall={rec['wall']:.1f}s "
            f"warm={s.get('tree/warm_starts')} route={rec.get('route')} "
            f"{rec.get('error', '')}",
            flush=True,
        )
    return out


def _min_sense(rec, v):
    """Map a published value to minimize units (so `bound <= optimum` is the test)."""
    return v if "min" in rec["sense"].lower() else -v


def _tol(ref: float) -> float:
    return TOL_ABS + TOL_REL * max(1.0, abs(ref))


def _sgm(xs, shift=1.0):
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift


def _summarize(recs: list[dict], tl: float) -> int:
    with open(OPTIMA) as f:
        optima = json.load(f)
    by = {}
    for r in recs:
        by.setdefault(r["name"], {})[r["arm"]] = r
    scored, checks = [], 0
    false = {a: [] for a in ARMS}
    bad_points = {a: [] for a in ARMS}
    errors = [r for r in recs if "error" in r]
    for name, arms in sorted(by.items()):
        if set(arms) != set(ARMS) or any("error" in a for a in arms.values()):
            continue
        if not any(str(a.get("route") or "").startswith("native-spatial") for a in arms.values()):
            continue
        scored.append(name)
        base = name.split(":", 1)[1] if name.startswith("nl:") else None
        for arm, rec in arms.items():
            if rec["point_ok"] is False:
                bad_points[arm].append(name)
            refs = []
            if base in optima:
                refs.append(("cert-optima", _min_sense(rec, optima[base]), True))
            for oa, other in arms.items():
                if oa != arm and other["objective"] is not None and other["point_ok"]:
                    refs.append((f"arm {oa}", _min_sense(other, other["objective"]), False))
            for label, ref, exact in refs:
                if rec["bound"] is not None:
                    checks += 1
                    if _min_sense(rec, rec["bound"]) > ref + _tol(ref):
                        false[arm].append(f"{name}: bound above {label} {ref}")
                if exact and rec["certified"] and rec["objective"] is not None:
                    checks += 1
                    if abs(_min_sense(rec, rec["objective"]) - ref) > _tol(ref):
                        false[arm].append(f"{name}: certified {rec['objective']} vs {ref}")
    if not scored:
        print("NOTHING SCORED: no kernel-routed instance completed in both arms", flush=True)
        return 2

    def col(arm):
        rows = [by[n][arm] for n in scored]
        solved = [r for r in rows if r["certified"]]
        per_node_ms = [
            1e3 * r["stats"]["tree/node_lp_s"] / r["nodes"]
            for r in rows
            if r["nodes"] and "tree/node_lp_s" in r["stats"]
        ]
        piv = [r["stats"]["tree/node_pivots"] / r["nodes"] for r in rows if r["nodes"]]
        warm = sum(r["stats"].get("tree/warm_starts", 0) for r in rows)
        mism = sum(r["stats"].get("tree/warm_shape_mismatch", 0) for r in rows)
        return {
            "solved": len(solved),
            "sgm": _sgm([r["wall"] if r["certified"] else tl for r in rows]),
            "wall": sum(r["wall"] for r in rows),
            "nodes": sum(r["nodes"] for r in rows),
            "med_ms": statistics.median(per_node_ms) if per_node_ms else float("nan"),
            "med_piv": statistics.median(piv) if piv else float("nan"),
            "warm": warm,
            "mism": mism,
            "false": len(false[arm]),
            "bad": len(bad_points[arm]),
        }

    cols = {a: col(a) for a in ARMS}
    print(f"\nscored {len(scored)} kernel-routed instances, {checks} bound/objective checks")
    print(f"{'':22s}" + "".join(f"{'mode ' + a:>14s}" for a in ARMS))
    for k, label in [
        ("solved", "certified"),
        ("sgm", "SGM wall s"),
        ("wall", "total wall s"),
        ("nodes", "total nodes"),
        ("med_ms", "median ms/node LP"),
        ("med_piv", "median pivots/node"),
        ("warm", "warm starts"),
        ("mism", "shape mismatches"),
        ("false", "FALSE bound/cert"),
        ("bad", "bad points"),
    ]:
        vals = [cols[a][k] for a in ARMS]
        fmt = (lambda v: f"{v:14.3f}") if isinstance(vals[0], float) else (lambda v: f"{v:14d}")
        print(f"{label:22s}" + "".join(fmt(v) for v in vals))
    for a in ARMS[1:]:
        lost = [n for n in scored if by[n]["0"]["certified"] and not by[n][a]["certified"]]
        gained = [n for n in scored if by[n][a]["certified"] and not by[n]["0"]["certified"]]
        print(f"mode {a} vs 0: lost {lost}  gained {gained}")
    for arm in ARMS:
        for f in false[arm]:
            print(f"FALSE arm={arm}: {f}")
        for b in bad_points[arm]:
            print(f"BAD POINT arm={arm}: {b}")
    if errors:
        print(f"{len(errors)} solves errored:")
        for e in errors:
            print(f"  {e['name']} arm={e['arm']}: {e['error']}")

    def ms(r):
        return 1e3 * r["stats"].get("tree/node_lp_s", float("nan")) / (r["nodes"] or 1)

    print("\nper instance, modes " + " / ".join(ARMS) + ": nodes | wall s | ms/node LP | cert")
    for n in scored:
        rs = [by[n][a] for a in ARMS]
        print(
            f"  {n:28s} "
            + " / ".join(f"{r['nodes']}" for r in rs)
            + " | "
            + " / ".join(f"{r['wall']:.1f}" for r in rs)
            + " | "
            + " / ".join(f"{ms(r):.3f}" for r in rs)
            + " | "
            + "".join("C" if r["certified"] else "-" for r in rs)
        )
    if checks == 0 or any(cols[a]["warm"] == 0 for a in ARMS[1:]):
        print("PROBE DID NOT FIRE: zero checks or a warm arm with zero warm starts", flush=True)
        return 2
    return 1 if any(cols[a]["false"] or cols[a]["bad"] for a in ARMS) else 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--instances", default="", help="comma-separated subset of names")
    ap.add_argument("--out", default="", help="append raw records (JSON lines) here")
    ap.add_argument("--summarize", default="", help="only summarize this JSON-lines file")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        return _summarize(recs, a.time_limit)
    names = _instances()
    if a.instances:
        want = set(a.instances.split(","))
        names = [n for n in names if n in want or n.split(":", 1)[-1] in want]
    print(f"load: {os.getloadavg()}  instances: {len(names)}  tl={a.time_limit}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for pair in ex.map(_group, [(i, n, a.time_limit) for i, n in enumerate(names)]):
            recs += pair
            if a.out:
                with open(a.out, "a") as f:
                    for r in pair:
                        f.write(json.dumps(r, default=float) + "\n")
    return _summarize(recs, a.time_limit)


if __name__ == "__main__":
    sys.exit(main())
