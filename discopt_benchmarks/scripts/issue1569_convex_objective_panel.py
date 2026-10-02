"""#1569 panel: the convex-quadratic objective node bound on array-variable models.

Before the fix the bound never engaged on any model with an array variable (the
test that decides it raised and abstained). The class it changes is therefore
exactly: array variables + a convex quadratic objective + nonconvex constraints
(so the spatial B&B path runs). Panel: three array-variable families (bilinear
least squares, wide-range integer quadratics, points-outside-a-ring) over sizes
and seeds. DAE optimal-control models are NOT in it: their objective carries
quadrature weights (``0.25 * u**2``), which the term classifier files as
``general_nl``, so the convex bound abstains on them in both arms -- a separate
gate, tracked in its own issue.

Each instance runs in a FRESH process per arm (``--one``), so the two code trees
never share an interpreter. Arms are given as two ``PYTHONPATH`` roots; each child
asserts which tree it loaded via a marker string in ``solver.py`` (CLAUDE.md §8).

    python -u discopt_benchmarks/scripts/issue1569_convex_objective_panel.py \\
        --base <tree>/python --fix <tree>/python [--time-limit 30]

Exits non-zero when nothing was compared (§6).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MARKER = "#1569: one entry per SCALAR"


def _family(kind: str, n: int, seed: int):
    import discopt.modeling as dm
    import numpy as np

    rng = np.random.default_rng(seed)
    m = dm.Model(f"{kind}_{n}_s{seed}")
    if kind == "bilinear_ls":
        x = m.continuous("x", shape=(n,), lb=-3, ub=3)
        for i in range(0, n - 3, 2):  # no wrap-around: cyclic rows contradict each other
            m.subject_to(x[i] * x[i + 1] - x[i + 2] * x[i + 3] >= 1.0)
        c = rng.uniform(-1, 1, n)
        m.minimize(sum((x[i] - float(c[i])) ** 2 for i in range(n)))
    elif kind == "int_quad":
        x = m.integer("x", shape=(n,), lb=0, ub=200)
        for i in range(n - 1):
            m.subject_to(x[i] * x[i + 1] <= float(rng.uniform(3000, 9000)))
        m.subject_to(x[0] * x[n - 1] >= 300)
        c = rng.uniform(20, 180, n)
        m.minimize(sum((x[i] - float(c[i])) ** 2 for i in range(n)) + x[0] * x[1])
    else:  # ring: n points outside the unit circle, nearest to random targets
        x = m.continuous("x", shape=(n,), lb=-2, ub=2)
        y = m.continuous("y", shape=(n,), lb=-2, ub=2)
        for i in range(n):
            m.subject_to(x[i] ** 2 + y[i] ** 2 >= 1.0)
        cx, cy = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)
        m.minimize(sum((x[i] - float(cx[i])) ** 2 + (y[i] - float(cy[i])) ** 2 for i in range(n)))
    return m


def instances() -> list[str]:
    names = []
    for kind, sizes in (("bilinear_ls", (4, 6, 8)), ("int_quad", (4, 6)), ("ring", (2, 3, 4))):
        for n in sizes:
            for seed in (0, 1, 2):
                names.append(f"{kind}:{n}:{seed}")
    return names


def build(name: str):
    kind, n, seed = name.split(":")
    return _family(kind, int(n), int(seed))


def run_one(name: str, tl: float, expect_fixed: bool) -> None:
    import logging

    import discopt
    import discopt.solver as solver_mod

    with open(solver_mod.__file__) as fh:
        fixed = MARKER in fh.read()
    assert fixed == expect_fixed, (
        f"{discopt.__file__}: marker {'missing' if expect_fixed else 'present'}"
    )
    engaged = []

    class _H(logging.Handler):
        def emit(self, rec):
            if "convex-objective node bound enabled" in rec.getMessage():
                engaged.append(1)

    lg = logging.getLogger("discopt")
    lg.setLevel(logging.DEBUG)
    h = _H(level=logging.DEBUG)
    lg.addHandler(h)
    r = build(name).solve(time_limit=tl, deterministic=True)
    lg.removeHandler(h)
    print(
        json.dumps(
            {
                "status": r.status,
                "cert": bool(r.gap_certified),
                "obj": r.objective,
                "bound": r.bound,
                "nodes": r.node_count,
                "wall": r.wall_time,
                "engaged": bool(engaged),
                "file": discopt.__file__,
            }
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--fix")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--one")
    ap.add_argument("--expect-fixed", type=int)
    a = ap.parse_args()
    if a.one:
        run_one(a.one, a.time_limit, bool(a.expect_fixed))
        return 0

    compared = checks = 0
    viol, gained, lost = [], [], []
    nodes = {"base": 0, "fix": 0}
    wall = {"base": 0.0, "fix": 0.0}
    tighter = looser = 0
    for name in instances():
        out = {}
        for arm, root, fx in (("base", a.base, 0), ("fix", a.fix, 1)):
            env = dict(os.environ, PYTHONPATH=root)
            p = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--one",
                    name,
                    "--expect-fixed",
                    str(fx),
                    "--time-limit",
                    str(a.time_limit),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=a.time_limit * 6 + 120,
            )
            if p.returncode != 0:
                raise RuntimeError(f"{name} {arm}: child failed\n{p.stderr[-2000:]}")
            out[arm] = json.loads(p.stdout.strip().splitlines()[-1])
        compared += 1
        b, f = out["base"], out["fix"]
        incs = [x["obj"] for x in (b, f) if x["obj"] is not None]
        best = min(incs) if incs else None
        for arm, x in (("base", b), ("fix", f)):
            wall[arm] += x["wall"]
            if x["bound"] is not None and best is not None:
                checks += 1
                if x["bound"] > best + 1e-6 * (1 + abs(best)):
                    viol.append(f"{name} {arm}: bound {x['bound']} above incumbent {best}")
        if b["cert"] and f["cert"]:
            checks += 1
            if b["status"] != f["status"]:
                viol.append(f"{name}: certified statuses differ {b['status']} vs {f['status']}")
            elif b["obj"] is not None and f["obj"] is not None:
                if abs(b["obj"] - f["obj"]) > 1e-5 * (1 + abs(b["obj"])):
                    viol.append(f"{name}: certified objectives differ {b['obj']} vs {f['obj']}")
            elif (b["obj"] is None) != (f["obj"] is None):
                viol.append(f"{name}: one certified arm has no objective")
            nodes["base"] += b["nodes"]
            nodes["fix"] += f["nodes"]
        if b["cert"] and not f["cert"]:
            lost.append(name)
        if f["cert"] and not b["cert"]:
            gained.append(name)
        if not b["cert"] and not f["cert"] and b["bound"] is not None and f["bound"] is not None:
            tighter += f["bound"] > b["bound"] + 1e-9
            looser += f["bound"] < b["bound"] - 1e-9
        print(
            f"{name:22s} base {b['status']:9s} {'C' if b['cert'] else '-'} n={b['nodes']:<6d} "
            f"b={b['bound']!s:.10s} eng={int(b['engaged'])} | fix {f['status']:9s} "
            f"{'C' if f['cert'] else '-'} n={f['nodes']:<6d} b={f['bound']!s:.10s} "
            f"eng={int(f['engaged'])} wall {b['wall']:.1f}/{f['wall']:.1f} "
            f"load={os.getloadavg()[0]:.2f}",
            flush=True,
        )
    print(f"\nCOMPARED {compared}; executed checks {checks}; violations {len(viol)}")
    for v in viol:
        print("  VIOLATION", v)
    print(f"certificates gained {len(gained)} {gained}")
    print(f"certificates lost   {len(lost)} {lost}")
    print(f"nodes on both-certified: base {nodes['base']} fix {nodes['fix']}")
    print(f"uncertified both: fix bound tighter {tighter}, looser {looser}")
    print(f"total wall: base {wall['base']:.0f}s fix {wall['fix']:.0f}s")
    return 0 if compared and checks else 1


if __name__ == "__main__":
    sys.exit(main())
