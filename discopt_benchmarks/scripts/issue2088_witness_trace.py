"""Witness tracer for #2088: find the node where a known optimum leaves the tree.

#2088: with ``DISCOPT_HULL_CONIC_QUADRATIC=0`` (eps-perspective hull), Pyomo's
``constrained_layout_clay0303`` under ``gdp_method="hull"`` was measured certifying
28351.42 against the optimum 26669.11. A false optimum means some node whose box
held the optimum was discarded: its bound came out above ``f*``, it was declared
infeasible, or a reduction removed the optimum from its children's boxes. This
script names that node.

It uses the B&B debug checkpoints (``discopt.debug.fire``) without attaching a
debug session, so the route is unchanged:

* ``BEFORE_IMPORT``: every evaluated node whose exported box contains the witness
  ``x*`` is reported with its bound. A bound above ``f*`` (relative 1e-4) is BAD.
* ``AFTER_PROCESS``: for such a node, the boxes of the children the tree just
  created (``last_created`` / ``node_box``) are checked. A child holding ``x*``
  with an inherited bound above ``f*`` is BAD. If no child holds ``x*`` and the
  incumbent is still above ``f*``, then ``x*`` was LOST (fathomed or reduced away).

Two passes, in the tree's (reformulated) variable space:

1. ``record``: run an arm that certifies the true optimum (``DISCOPT_HULL_CONIC_QUADRATIC=1``
   certified 26669.11 on the panel) and save the tree's final incumbent as the
   witness. The two arms must lower the model to the same tree columns; the
   script checks the column count and refuses on a mismatch.
2. ``check``: run the failing arm (``DISCOPT_HULL_CONIC_QUADRATIC=0``) against
   that witness.

Usage (repo root)::

    PYTHONHASHSEED=0 DISCOPT_HULL_CONIC_QUADRATIC=1 \\
        python -u discopt_benchmarks/scripts/issue2088_witness_trace.py record \\
        --witness /tmp/clay0303_witness.npy
    PYTHONHASHSEED=0 DISCOPT_HULL_CONIC_QUADRATIC=0 \\
        python -u discopt_benchmarks/scripts/issue2088_witness_trace.py check \\
        --witness /tmp/clay0303_witness.npy

``check`` prints the number of nodes and child boxes examined, and exits 2 if it
examined none (the probe did not fire, CLAUDE.md section 6). It exits 1 if it
found any BAD or LOST event, and 0 otherwise.

Measured 2026-11-02 on Linux (4 cores, pounce NLP), at 54c7f944 and a2e3b46: the
OFF arm certified the TRUE optimum 26669.11 (849/871 nodes), so the bug did not
reproduce. On a2e3b46, ``record`` on the conic arm (219 nodes, 56 s) and then
``check`` on the OFF arm examined 849 nodes and 194 child boxes; 11 nodes held
``x*``, and there were 0 events. An earlier version, using the OFF arm's own
incumbent as the witness, also found 0 events with six hash seeds run
concurrently on 54c7f944 (each to 400 s, about 700 nodes). Run this script where
the bug reproduces.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # discopt_benchmarks/, for ``benchmarks``

import discopt  # noqa: E402
import discopt.debug as dbg  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

from benchmarks import pyomo_gdp_examples as px  # noqa: E402

#: SCIP (Pyomo gdp.bigm, 300 s) optimum of clay0303, bound equal (#2088).
CLAY0303_OPT = 26669.109572447


def _contains(lb, ub, x) -> bool:
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if lb.shape[0] != x.shape[0]:
        raise ValueError(
            f"node box has {lb.shape[0]} columns, witness has {x.shape[0]}: the arms "
            "lowered the model differently, so this witness cannot be traced here"
        )
    tol = 1e-6 * (1.0 + np.abs(x))
    return bool(np.all(lb - tol <= x) and np.all(x <= ub + tol))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("mode", choices=("record", "check"))
    ap.add_argument("--instance", default="pyomo.constrained_layout_clay0303")
    ap.add_argument("--witness", required=True, help=".npy path to write (record) / read (check)")
    ap.add_argument("--fstar", type=float, default=CLAY0303_OPT, help="verified optimum")
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    print("discopt:", discopt.__file__, flush=True)
    print("env:", {k: v for k, v in os.environ.items() if k.startswith("DISCOPT")}, flush=True)
    fstar = float(args.fstar)
    over = fstar + 1e-4 * abs(fstar)
    xstar = np.load(args.witness) if args.mode == "check" else None

    m = convert_pyomo_gdp(px.builder(args.instance)()).model
    orig_fire = dbg.fire
    st: dict = {"nodes": 0, "contain": 0, "child_checks": 0, "events": [], "pending": False}

    def fire(cp, **kw):
        name = str(cp)
        tree = kw.get("tree")
        if args.mode == "record" and name == "terminated" and tree is not None:
            inc = tree.incumbent()
            if inc is not None:
                np.save(args.witness, np.asarray(inc[0], dtype=np.float64))
                st["recorded"] = float(inc[1])
                print(f"witness saved: obj={inc[1]} ncols={len(inc[0])}", flush=True)
        elif args.mode == "check" and name == "before_import":
            lb = np.asarray(kw["batch_lb"])
            ub = np.asarray(kw["batch_ub"])
            rl = np.asarray(kw["result_lbs"])
            for i in range(lb.shape[0]):
                st["nodes"] += 1
                if not _contains(lb[i], ub[i], xstar):
                    continue
                st["contain"] += 1
                b = float(rl[i])
                nid = int(kw["batch_ids"][i])
                bad = b > over
                print(
                    f"it={kw['iteration']} node={nid} contains x*: "
                    f"lb={b:.6f} {'BAD' if bad else 'ok'}",
                    flush=True,
                )
                if bad:
                    st["events"].append(("node-bound", kw["iteration"], nid, b))
                st["pending"] = True
        elif args.mode == "check" and name == "after_process" and st["pending"]:
            st["pending"] = False
            kids = np.asarray(tree.last_created(), dtype=np.int64)
            hit = []
            for c in kids:
                st["child_checks"] += 1
                clb, cub = tree.node_box(int(c))
                if _contains(clb, cub, xstar):
                    hit.append(int(c))
            inc = tree.incumbent()
            inc_obj = float(inc[1]) if inc is not None else float("inf")
            if hit:
                lbs = np.asarray(tree.node_lower_bounds(np.asarray(hit, dtype=np.int64)))
                print(f"  children holding x*: {hit} lbs={lbs.tolist()}", flush=True)
                for c, b in zip(hit, lbs, strict=True):
                    if float(b) > over:
                        st["events"].append(("child-bound", kw["iteration"], c, float(b)))
            elif inc_obj > over:
                print(
                    f"  x* LOST at it={kw['iteration']}: children {kids.tolist()}, "
                    f"incumbent {inc_obj}",
                    flush=True,
                )
                st["events"].append(("lost", kw["iteration"], kids.tolist(), inc_obj))
            else:
                print(f"  x* region closed; incumbent {inc_obj} <= f*", flush=True)
        return orig_fire(cp, **kw)

    dbg.fire = fire
    t0 = time.time()
    try:
        r = m.solve(gdp_method="hull", time_limit=args.time_limit)
    finally:
        dbg.fire = orig_fire
    print(
        f"result: status={r.status} obj={r.objective} bound={r.bound} "
        f"certified={r.gap_certified} nodes={r.node_count} wall={time.time() - t0:.1f}s",
        flush=True,
    )

    if args.mode == "record":
        if "recorded" not in st:
            print("PROBE DID NOT FIRE: no incumbent at the terminated checkpoint", flush=True)
            return 2
        if st["recorded"] > over:
            print(f"WARNING: recorded incumbent {st['recorded']} is above f*={fstar}", flush=True)
        return 0

    print(
        f"checked nodes={st['nodes']} containing_x*={st['contain']} "
        f"child_boxes={st['child_checks']} events={st['events']}",
        flush=True,
    )
    if st["nodes"] == 0:
        print("PROBE DID NOT FIRE: no node was examined", flush=True)
        return 2
    return 1 if st["events"] else 0


if __name__ == "__main__":
    # os._exit: skip interpreter teardown of solver threads after the verdict prints.
    code = main()
    sys.stdout.flush()
    os._exit(code)
