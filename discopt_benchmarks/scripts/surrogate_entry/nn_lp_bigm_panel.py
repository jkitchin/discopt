"""§5 class panel for ``DISCOPT_NN_LP_BIGM``: ReLU MILP embeddings, flag ON vs OFF.

The flag (T-N3.2, ``discopt.ml.bounds.lp_tighten_bounds``) changes only the
big-M constants and variable boxes of ReLU embeddings; the in-repo MINLPLib
corpus contains none, so this class is the panel (as for #2187's ReLU panel).
Networks are E1's (``e1_lp_bigm_tightening.PANEL``: sklearn ReLU MLPs of the
Kitchin 2026 test function), each embedded as ``relu_bigm`` and as
``relu_extended`` and minimised over the input box. The question comes from
PWLNet (Sanyal, Demir, Türkay, Aydın, arXiv:2611.25969v1), which uses interval
big-M constants; the method is Tjeng, Xiao & Tedrake 2019 and Grimstad &
Andersson 2019.

Each (network, strategy, arm) solve runs in a fresh subprocess with the flag in
its environment, ``deterministic=True``; arms are interleaved with alternating
order. Checks, counted (the script exits non-zero if none ran, CLAUDE.md §6):

* each published incumbent, re-evaluated through ``forward``, reproduces the
  reported objective;
* no arm's dual bound exceeds the best verified objective of either arm;
* when both arms certify, their objectives agree.

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.nn_lp_bigm_panel \
        --time-limit 120 --out <scratch>/lpbigm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from ._common import loaded_code_banner
from .e1_lp_bigm_tightening import PANEL as NETS

FLAG = "DISCOPT_NN_LP_BIGM"
STRATEGIES = ("relu_bigm", "relu_extended")

CHILD = r"""
import json, sys
import numpy as np
import discopt
from discopt.ml.formulations.base import NNFormulation
from discopt_benchmarks.scripts.surrogate_entry._common import describe, to_network, train_mlp

d, hidden, seed, strategy, tl = json.loads(sys.argv[1])
mlp, rmse = train_mlp(d, tuple(hidden), "relu", seed)
net = to_network(mlp, d)
m = discopt.Model("lpbigm")
f = NNFormulation(m, net, strategy=strategy, prefix="nn")
f.formulate()
m.minimize(f.outputs[0])
r = m.solve(time_limit=tl, deterministic=True)
out = describe(r)
out["lp_stats"] = getattr(f._strategy, "lp_bound_stats", None)
x = r.x.get(f.inputs.name) if r.x else None
if x is not None and out["objective"] is not None:
    out["forward_obj"] = float(np.ravel(net.forward(np.asarray(x, float).reshape(1, d)))[0])
print("RESULT " + json.dumps(out))
"""


def run_arm(item, on: bool, tl: float) -> dict:
    env = dict(os.environ)
    if on:
        env[FLAG] = "1"
    else:
        env.pop(FLAG, None)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    p = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", CHILD, json.dumps([*item, tl])],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
        timeout=tl * 4 + 300,
    )
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"child produced no result (rc={p.returncode}):\n{p.stderr[-3000:]}")
    return json.loads(line[len("RESULT ") :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)
    items = [(d, list(h), s, strat) for d, h, s in NETS for strat in STRATEGIES]
    if a.only is not None:
        items = [items[a.only]]
    n_checks = 0
    with open(a.out, "a") as fh:
        for i, item in enumerate(items):
            tag = f"d{item[0]}_{'x'.join(map(str, item[1]))}_s{item[2]}_{item[3]}"
            rows = {}
            for on in (False, True) if i % 2 == 0 else (True, False):
                row = run_arm(item, on, a.time_limit)
                row.update(net=tag, arm="ON" if on else "OFF")
                fwd = row.get("forward_obj")
                if fwd is not None:
                    if abs(fwd - row["objective"]) > 1e-4 * max(1.0, abs(fwd)):
                        print(f"FORWARD MISMATCH {tag} {row['arm']}: {fwd} vs {row['objective']}")
                        return 4
                    n_checks += 1
                rows[row["arm"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            verified = [r["forward_obj"] for r in rows.values() if "forward_obj" in r]
            best = min(verified, default=None)
            for r in rows.values():
                if r["bound"] is not None and best is not None:
                    if r["bound"] > best + 1e-4 * max(1.0, abs(best)):
                        print(f"BOUND ABOVE INCUMBENT {tag} {r['arm']}: {r['bound']} > {best}")
                        return 3
                    n_checks += 1
            if rows["ON"]["gap_certified"] and rows["OFF"]["gap_certified"]:
                o1, o2 = rows["ON"]["objective"], rows["OFF"]["objective"]
                if abs(o1 - o2) > 1e-4 * max(1.0, abs(o1)):
                    print(f"CERTIFIED OBJECTIVES DISAGREE {tag}: {o1} vs {o2}")
                    return 5
                n_checks += 1
            off, on = rows["OFF"], rows["ON"]
            print(
                f"[{i + 1}/{len(items)}] {tag}: OFF {off['status']} cert={off['gap_certified']} "
                f"nodes={off['nodes']} wall={off['wall']:.1f} bound={off['bound']} | "
                f"ON {on['status']} cert={on['gap_certified']} nodes={on['nodes']} "
                f"wall={on['wall']:.1f} bound={on['bound']} "
                f"lps={(on['lp_stats'] or {}).get('lps')} "
                f"load={os.getloadavg()[0]:.2f}",
                flush=True,
            )
    print(f"executed checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
