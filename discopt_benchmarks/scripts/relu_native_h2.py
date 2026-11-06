"""H2 panel for #2167: the native complementarity branch vs ``relu_bigm``.

Hypothesis H2 (issue #2167): the native branch matches ``relu_bigm`` node counts
within 1.5x and wins where big-M is weak or infinite.
Kill criterion: retire it if median nodes or wall are worse by more than 1.5x and
no instance class improves.

Arms per instance, interleaved with the order rotated per instance:

* ``bigm``   -- ``relu_bigm`` (HiGHS MILP route for a pure network; spatial B&B
  when the model is nonlinear);
* ``native0`` -- ``relu_complementarity`` lowered with ``method="native"``,
  ``DISCOPT_COMPL_BRANCH=0`` (ordinary spatial branching on ``y*s <= 0``);
* ``native1`` -- the same with ``DISCOPT_COMPL_BRANCH=1`` (the dichotomy).

Classes: ``box`` (input box [-1, 1], the box the weights were drawn for),
``loose`` (box [-L, L], so interval big-M constants are weak), ``physics``
(the H1 ring-constrained nonlinear model).

Every certified result is checked against every other certified arm on the same
instance (objectives agree within the certified gaps) and a dual bound may never
exceed another arm's verified incumbent. The run ends with the number of
comparisons made and exits non-zero when it is zero or any check failed
(CLAUDE.md §6). Each arm runs in its own subprocess, so the flag is read fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PANEL = [
    # (n_in, width, depth, cls, L)
    (2, 8, 2, "box", 1.0),
    (3, 8, 2, "box", 1.0),
    (2, 12, 2, "box", 1.0),
    (3, 6, 3, "box", 1.0),
    (4, 10, 2, "box", 1.0),
    (2, 8, 2, "loose", 10.0),
    (3, 8, 2, "loose", 10.0),
    (2, 12, 2, "loose", 10.0),
    (3, 6, 3, "loose", 30.0),
    (2, 8, 2, "physics", 1.0),
    (3, 8, 2, "physics", 1.0),
    (3, 6, 3, "physics", 1.0),
]

ARM_SRC = r"""
import json, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import numpy as np
from discopt.ml.network import NetworkDefinition
from discopt_benchmarks.scripts.relu_local_h1 import make_net, embed
import discopt.mpec as mpec
assert hasattr(mpec, "native_pair_columns"), mpec.__file__
arm, n_in, width, depth, cls, L, seed, tl = sys.argv[1:9]
n_in, width, depth, seed = int(n_in), int(width), int(depth), int(seed)
L, tl = float(L), float(tl)
net = make_net(seed, n_in, width, depth)
net = NetworkDefinition(net.layers, input_bounds=(-L * np.ones(n_in), L * np.ones(n_in)))
physics = cls == "physics"
t0 = time.perf_counter()
if arm == "bigm":
    m, nn = embed(net, "relu_bigm", physics)
    r = m.solve(time_limit=tl)
else:
    m, nn = embed(net, "relu_complementarity", physics)
    r = mpec.solve_mpec(m, nn.relations, method="native", time_limit=tl)
wall = time.perf_counter() - t0
x_in = None if r.x is None else [float(v) for v in np.asarray(r.x["nn_input"]).ravel()]
print("RESULT " + json.dumps(dict(
    status=str(r.status), objective=r.objective, bound=r.bound,
    certified=bool(r.gap_certified), nodes=int(r.node_count or 0), wall=wall,
    compl_branches=(r.solver_stats or {}).get("tree/compl_branches"),
    x_in=x_in, mpec=mpec.__file__)), flush=True)
"""


def run_arm(arm, inst, seed, tl):
    n_in, width, depth, cls, half_width = inst
    env = dict(os.environ, DISCOPT_COMPL_BRANCH="1" if arm == "native1" else "0")
    p = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            ARM_SRC,
            arm,
            str(n_in),
            str(width),
            str(depth),
            cls,
            str(half_width),
            str(seed),
            str(tl),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=tl * 4 + 120,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"arm {arm} produced no result:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")


def true_objective(inst, seed, x_in):
    import numpy as np
    from discopt_benchmarks.scripts.relu_local_h1 import make_net

    n_in, width, depth, cls, half_width = inst
    net = make_net(seed, n_in, width, depth)
    x = np.asarray(x_in)
    f = float(net.forward(x)[0])
    if cls == "physics":
        f += 0.05 * float(np.sum((x - 0.3) ** 2))
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    arms = ["bigm", "native0", "native1"]
    idx = [int(i) for i in args.only.split(",") if i] or list(range(len(PANEL)))
    # one JSON line per instance, flushed as written; closed at interpreter exit
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    n_cmp = n_fail = 0
    k = 0
    for i in idx:
        inst = PANEL[i]
        for seed in [int(s) for s in args.seeds.split(",")]:
            order = arms[k % 3 :] + arms[: k % 3]
            k += 1
            res = {a: run_arm(a, inst, seed, args.time_limit) for a in order}
            fails = []
            for a, r in res.items():
                if r["x_in"] is not None:
                    f = true_objective(inst, seed, r["x_in"])
                    n_cmp += 1
                    if abs(f - r["objective"]) > 1e-5 * max(1.0, abs(f)):
                        fails.append(f"{a}: reported {r['objective']} but x gives {f}")
            for a, ra in res.items():
                for b, rb in res.items():
                    if a == b or ra["bound"] is None or rb["objective"] is None:
                        continue
                    n_cmp += 1
                    tol = 1e-6 * max(1.0, abs(rb["objective"]))
                    if ra["bound"] > rb["objective"] + tol:
                        fails.append(f"{a} bound {ra['bound']} > {b} incumbent {rb['objective']}")
            n_fail += len(fails)
            rec = {
                "n_in": inst[0],
                "width": inst[1],
                "depth": inst[2],
                "cls": inst[3],
                "L": inst[4],
                "seed": seed,
                "order": order,
                "arms": res,
                "fails": fails,
            }
            for r in rec["arms"].values():
                r.pop("x_in", None)
            line = json.dumps(rec)
            print(line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()
    print(f"# comparisons made: {n_cmp}; failed checks: {n_fail}", flush=True)
    return 0 if (n_cmp and not n_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
