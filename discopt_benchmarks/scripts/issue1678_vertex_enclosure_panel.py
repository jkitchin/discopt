# ruff: noqa: N806 -- process-synthesis variable names (F, P, S, CR, CS) follow the literature
"""Graduation panel for ``DISCOPT_CONVEX_LIFT_VERTEX_ENCLOSURE`` (#1678 II.7; CLAUDE.md §5).

The flag changes the aux-column enclosure of a composite convex/concave lift
that the #358 conditioning guard would otherwise decline (natural interval
enclosure non-finite or above ``1e7``). Population:

* the in-repo ``.nl`` corpus (``python/tests/data/minlplib_nl``) -- the general
  regression watch, oracle ``minlplib.solu``;
* MINLPLib ``*hfsg`` instances (``--hfsg``), the GDP-hull class the flag targets;
* a convex-GDP family built here and solved with ``gdp_method="hull"``:
  process synthesis (log yields, the #1617 witness generalised over seeds and
  sizes) and a quadratic-disjunct placement family. Oracle: the big-M solve
  (convex kernel, which never builds the spatial relaxation, so it is
  flag-independent).

Each (instance, arm) solve runs in its own subprocess with a hard timeout; the
arm order alternates per instance (interleaved A/B, CLAUDE.md §9). Checks:

* cert-clean -- no published bound crosses the oracle; when both arms certify,
  objectives agree; every published incumbent passes ``verify_point`` on a
  freshly built model; no instance certified OFF loses its certificate ON;
* net-positive -- certificates gained, root bound, nodes on instances both
  arms certify, final bound otherwise, wall.

The ON arm counts how many lifts the refined enclosure admitted (``fired``), so
an instance where the flag never acted is visible as such. Prints an
executed-check count and exits non-zero when nothing was compared (§6).

    python -u discopt_benchmarks/scripts/issue1678_vertex_enclosure_panel.py \
        --time-limit 30 [--hfsg] [--gdp] [--corpus] --out panel.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

FLAG = "DISCOPT_CONVEX_LIFT_VERTEX_ENCLOSURE"
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"
BENCH = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark"


# ── convex GDP families ─────────────────────────────────────────────────────


def synthesis(seed: int, n_r: int, n_s: int):
    import discopt.modeling as dm
    import numpy as np

    rng = np.random.default_rng(seed)
    m = dm.Model(f"synth_s{seed}_{n_r}x{n_s}")
    F = m.continuous("F", lb=0, ub=100)
    P = m.continuous("P", lb=0, ub=200)
    S = m.continuous("S", lb=0, ub=200)
    CR = m.continuous("CR", lb=0, ub=800)
    CS = m.continuous("CS", lb=0, ub=800)
    reactors = []
    for _ in range(n_r):
        a = float(rng.uniform(30, 80))
        c = float(rng.uniform(5, 50))
        f = float(rng.uniform(40, 200))
        v = float(rng.uniform(1.0, 3.0))
        reactors.append([a * dm.log(1 + F / c) >= P, f + v * F <= CR])
    seps = []
    for _ in range(n_s):
        b = float(rng.uniform(0.8, 0.99))
        f = float(rng.uniform(30, 100))
        v = float(rng.uniform(0.8, 1.6))
        seps.append([b * P >= S, f + v * P <= CS])
    m.either_or(reactors, name="reactor")
    m.either_or(seps, name="separator")
    m.minimize(-(12 * S - 2 * F - CR - CS))
    return m


def placement(seed: int, n_sites: int):
    """Place one point in one of ``n_sites`` discs (quadratic disjunct rows) to
    minimise a linear cost plus a site charge."""
    import discopt.modeling as dm
    import numpy as np

    rng = np.random.default_rng(seed)
    m = dm.Model(f"place_s{seed}_{n_sites}")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    t = m.continuous("t", lb=0, ub=100)
    arms = []
    for _ in range(n_sites):
        a, b = (float(v) for v in rng.uniform(-8, 8, size=2))
        r = float(rng.uniform(0.5, 2.0))
        charge = float(rng.uniform(0, 10))
        arms.append([(x - a) ** 2 + (y - b) ** 2 <= r**2, t >= charge])
    m.either_or(arms, name="site")
    cx, cy = (float(v) for v in rng.normal(size=2))
    m.minimize(cx * x + cy * y + t)
    return m


def gdp_specs():
    out = []
    for seed in range(6):
        for n_r, n_s in ((3, 2), (4, 2), (5, 3)):
            out.append(("gdp", f"synthesis:{seed}:{n_r}:{n_s}"))
    for seed in range(6):
        for n in (3, 5):
            out.append(("gdp", f"placement:{seed}:{n}"))
    return out


def build_gdp(key: str):
    kind, *args = key.split(":")
    ints = [int(a) for a in args]
    return synthesis(*ints) if kind == "synthesis" else placement(*ints)


# ── worker ──────────────────────────────────────────────────────────────────


def worker(kind: str, key: str, arm: str, tl: float, method: str) -> dict:
    sys.path.insert(0, str(REPO / "python"))
    import discopt
    import numpy as np

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    import discopt._relax.uniform_relax as ur
    from discopt.modeling.core import ObjectiveSense, from_nl
    from discopt.validation.feasibility import verify_point

    assert hasattr(ur, "_curvature_enclosure"), "worktree without the #1678 change"
    os.environ[FLAG] = "1" if arm == "1" else "0"  # explicit: independent of the default
    fired = [0]
    orig = ur._curvature_enclosure

    def counted(*a, **k):
        out = orig(*a, **k)
        if out is not None:
            fired[0] += 1
        return out

    ur._curvature_enclosure = counted
    build = (lambda: from_nl(key)) if kind == "nl" else (lambda: build_gdp(key))
    m = build()
    t0 = time.perf_counter()
    kw = {"time_limit": tl, "deterministic": True}
    if kind == "gdp":
        kw["gdp_method"] = method
    r = m.solve(**kw)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.status in ("optimal", "feasible"):
        fresh = build()
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    return {
        "kind": kind,
        "key": key,
        "arm": arm,
        "method": method,
        "status": r.status,
        "gap_certified": bool(r.gap_certified),
        "objective": f(r.objective),
        "bound": f(r.bound),
        "root_bound": f(r.root_bound),
        "nodes": int(r.node_count or 0),
        "wall": wall,
        "verify_bad": bad,
        "fired": fired[0],
        "maximize": m._objective.sense == ObjectiveSense.MAXIMIZE,
        "route": str(r.algorithm_route)[:60],
    }


def run_one(kind, key, arm, tl, method, hard):
    cmd = [sys.executable, "-u", __file__, "--worker", kind, key, arm, str(tl), method]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
    except subprocess.TimeoutExpired:
        return {
            "kind": kind,
            "key": key,
            "arm": arm,
            "status": "hard_timeout",
            "wall": time.perf_counter() - t0,
        }
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {
        "kind": kind,
        "key": key,
        "arm": arm,
        "status": "crash",
        "stderr": p.stderr[-2000:],
        "wall": time.perf_counter() - t0,
    }


def load_solu():
    best = {}
    p = BENCH / "minlplib.solu"
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("=opt=", "=best="):
                best[parts[1]] = float(parts[2])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=5)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--hfsg", action="store_true")
    ap.add_argument("--gdp", action="store_true")
    ap.add_argument("--out", required=False)
    ap.add_argument("--only", help="comma-separated instance names (re-runs)")
    a = ap.parse_args()
    if a.worker:
        kind, key, arm, tl, method = a.worker
        print("RESULT " + json.dumps(worker(kind, key, arm, float(tl), method)), flush=True)
        return 0

    specs = []
    if a.corpus:
        specs += [("nl", str(p)) for p in sorted(CORPUS.glob("*.nl"))]
    if a.hfsg:
        specs += [("nl", str(p)) for p in sorted((BENCH / "minlplib" / "nl").glob("*hfsg.nl"))]
    if a.gdp:
        specs += gdp_specs()
    if a.only:
        keep = set(a.only.split(","))
        specs = [(k, key) for k, key in specs if (Path(key).stem if k == "nl" else key) in keep]
    solu = load_solu()
    tl = a.time_limit
    hard = 4 * tl + 120
    out = open(a.out, "a") if a.out else None  # noqa: SIM115 -- appended per row
    checks = 0
    viol: list[str] = []
    errors: list[str] = []
    rows = []
    for i, (kind, key) in enumerate(specs):
        res = {}
        oracle = None
        if kind == "gdp":
            o = run_one(kind, key, "0", tl, "big-m", hard)
            if o.get("gap_certified"):
                oracle = o["objective"]
        else:
            oracle = solu.get(Path(key).stem)
        for arm in ("0", "1") if i % 2 == 0 else ("1", "0"):
            res[arm] = run_one(kind, key, arm, tl, "hull", hard)
        name = Path(key).stem if kind == "nl" else key
        r0, r1 = res["0"], res["1"]
        for arm, r in res.items():
            if "objective" not in r:
                errors.append(f"{name} arm={arm}: {r['status']} {r.get('stderr', '')[-300:]}")
                continue
            mx = r["maximize"]
            tol = 1e-6 * (1 + abs(oracle)) if oracle is not None else None
            for lab in ("bound", "root_bound"):
                b = r.get(lab)
                if b is not None and oracle is not None:
                    checks += 1
                    crossed = (b < oracle - tol) if mx else (b > oracle + tol)
                    if crossed:
                        viol.append(f"{name} arm={arm}: {lab} {b} crosses oracle {oracle}")
            if r.get("verify_bad") is not None:
                checks += 1
                if r["verify_bad"]:
                    viol.append(f"{name} arm={arm}: incumbent fails verify_point")
        if "objective" in r0 and "objective" in r1:
            if r0["gap_certified"] and r1["gap_certified"]:
                checks += 1
                o0, o1 = r0["objective"], r1["objective"]
                if abs(o0 - o1) > 1e-4 * max(1.0, abs(o0)) + 1e-6:
                    viol.append(f"{name}: certified objectives differ {o0} vs {o1}")
            checks += 1
            if r0["gap_certified"] and not r1["gap_certified"]:
                viol.append(f"{name}: certification regression ON")
        row = {"name": name, "oracle": oracle, "off": r0, "on": r1}
        rows.append(row)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(
            f"[{i + 1}/{len(specs)}] {name:28s} fired={r1.get('fired')} "
            f"OFF {r0.get('status')}/{r0.get('gap_certified')} n={r0.get('nodes')} "
            f"root={r0.get('root_bound')} bd={r0.get('bound')} w={r0.get('wall', 0):.1f} | "
            f"ON {r1.get('status')}/{r1.get('gap_certified')} n={r1.get('nodes')} "
            f"root={r1.get('root_bound')} bd={r1.get('bound')} w={r1.get('wall', 0):.1f} "
            f"oracle={oracle}",
            flush=True,
        )
    cert0 = sum(1 for r in rows if r["off"].get("gap_certified"))
    cert1 = sum(1 for r in rows if r["on"].get("gap_certified"))
    fired = [r["name"] for r in rows if (r["on"].get("fired") or 0) > 0]
    w0 = sum(r["off"].get("wall", 0) for r in rows)
    w1 = sum(r["on"].get("wall", 0) for r in rows)
    print(
        f"\ninstances={len(rows)} certified OFF={cert0} ON={cert1} "
        f"total wall OFF={w0:.1f}s ON={w1:.1f}s"
    )
    print(f"flag fired on {len(fired)}: {fired}")
    print(f"violations ({len(viol)}):")
    for v in viol:
        print("  " + v)
    print(f"errors / hard timeouts ({len(errors)}):")
    for e in errors:
        print("  " + e)
    print(f"executed checks: {checks}")
    return 1 if checks == 0 else (2 if viol else 0)


if __name__ == "__main__":
    sys.exit(main())
