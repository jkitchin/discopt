"""Panel for ``strategy="relu_icnn"``: E2's question asked of the library strategy.

E2 (``e2_icnn_lp.py``) measured a hand-built epigraph LP against the big-M MILP.
This panel asks the same question of what ships: ``NNFormulation(...,
strategy="relu_icnn")``, which includes the result check ``Model.solve`` runs
(repair onto the network's graph, re-verify the model). The idea is from J. Kitchin,
"MIP surrogates" (2026-10-28,
https://kitchingroup.cheme.cmu.edu/posts/mip-surrogates/). PWLNet (Sanyal, Demir,
Türkay, Aydın, arXiv:2611.25969) raised the question. ICNNs are from Amos, Xu &
Kolter (ICML 2017).

Two uses per network, each solved by ``relu_bigm`` (exact) and ``relu_icnn``,
interleaved with alternating order, ``--reps`` times:

* ``min``  -- minimise the output;
* ``cap``  -- minimise ``sum(x)`` subject to ``y <= min + 0.5``, where the LP may
  leave neurons above ``relu(zhat)`` on slack rows and the check must repair them.

Checks, counted (exit non-zero if none ran, CLAUDE.md §6): both arms certify, and
neither's objective is below the other's dual bound (relu_bigm stops at the default
1e-4 gap; the LP closes exactly).

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.icnn_strategy_panel \
        --reps 3 --out <scratch>/icnn_panel.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import discopt
import discopt.modeling as dm
from discopt.ml.formulations.base import NNFormulation

from ._common import loaded_code_banner
from .e2_icnn_lp import PANEL, is_icnn, train_icnn


def solve(net, strategy, use, cap, tl):
    m = discopt.Model(f"{strategy}_{use}")
    f = NNFormulation(m, net, strategy=strategy, prefix="nn")
    f.formulate()
    if use == "min":
        m.minimize(f.outputs[0])
    else:
        m.subject_to(f.outputs[0] <= cap)
        m.minimize(dm.sum(f.inputs))
    t = time.perf_counter()
    r = m.solve(time_limit=tl)
    return r, time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)
    n_checks = 0
    ratios: dict[str, list[float]] = {"min": [], "cap": []}
    with open(a.out, "a") as fh:
        for i, (d, hidden, seed) in enumerate(PANEL):
            net, _ = train_icnn(d, hidden, seed)
            assert is_icnn(net)
            tag = f"d{d}_{'x'.join(map(str, hidden))}_s{seed}"
            ref, _ = solve(net, "relu_bigm", "min", None, a.time_limit)
            cap = ref.objective + 0.5
            for use in ("min", "cap"):
                walls: dict[str, list[float]] = {"relu_bigm": [], "relu_icnn": []}
                for rep in range(a.reps):
                    arms = ["relu_bigm", "relu_icnn"]
                    if (i + rep) % 2:
                        arms.reverse()
                    got = {}
                    for s in arms:
                        r, w = solve(net, s, use, cap, a.time_limit)
                        got[s] = r
                        walls[s].append(w)
                        row = {
                            "net": tag,
                            "use": use,
                            "arm": s,
                            "rep": rep,
                            "status": r.status,
                            "objective": r.objective,
                            "wall": w,
                            "load": os.getloadavg()[0],
                        }
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                    b, c = got["relu_bigm"], got["relu_icnn"]
                    if not (b.status == c.status == "optimal"):
                        print(f"NOT BOTH OPTIMAL {tag} {use}: {b.status} / {c.status}")
                        return 2
                    # Certificate consistency, not equality: relu_bigm stops at the
                    # default gap_tolerance (1e-4 rel) while the LP closes exactly, so
                    # each objective must lie in the other's [bound, objective] window.
                    tol = 1e-6 * max(1.0, abs(b.objective))
                    for p, q in ((b, c), (c, b)):
                        if not (q.bound - tol <= p.objective):
                            print(f"BELOW BOUND {tag} {use}: {p.objective} < {q.bound}")
                            return 3
                    n_checks += 1
                mb, mi = (
                    statistics.median(walls["relu_bigm"]),
                    statistics.median(walls["relu_icnn"]),
                )
                ratios[use].append(mb / mi)
                print(
                    f"{tag} {use}: bigm {mb:.3f}s (sd {statistics.pstdev(walls['relu_bigm']):.3f}) "
                    f"icnn {mi:.3f}s (sd {statistics.pstdev(walls['relu_icnn']):.3f}) "
                    f"ratio {mb / mi:.1f} load={os.getloadavg()[0]:.2f}",
                    flush=True,
                )
    for use, rs in ratios.items():
        print(f"median ratio [{use}] = {statistics.median(rs):.1f} over {len(rs)} nets")
    print(f"executed checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
