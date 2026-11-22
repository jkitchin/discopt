"""Class panel for ``DISCOPT_DEFINED_VARS``: embedded smooth-activation networks.

The in-repo MINLPLib corpus has no array variables, so it can only show whether
the affine-output rule harms scalar models; it cannot reach the class the flag
was built for (docs/dev/nn-module-plan.md §6, E3 defect 1). This panel is that
class: tanh, sigmoid and softplus surrogates of the Kitchin (2026) test function,
embedded with ``discopt.ml`` full_space and minimised over the input box.

Each (network, arm) solve runs in a fresh subprocess with the flag set in its
environment, ``deterministic=True``. Arms are interleaved per network with the
order alternating.

Checks (counted; the script exits non-zero if none ran, CLAUDE.md §6):
  * every published incumbent, re-evaluated through the network's exact
    ``forward``, reproduces the reported objective;
  * no arm's dual bound exceeds the best verified objective of either arm;
  * when both arms certify, their objectives agree.

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.defined_vars_nn_panel \
        --time-limit 120 --out <scratch>/dv_nn.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from ._common import loaded_code_banner

FLAG = "DISCOPT_DEFINED_VARS"

# (inputs, hidden, activation, seed); softplus nets are the trained tanh weights
# with the activation swapped, a different but well-defined function.
PANEL = [
    (2, (8,), "tanh", 0),
    (2, (16,), "tanh", 0),
    (2, (16,), "tanh", 1),
    (2, (32,), "tanh", 0),
    (2, (8, 8), "tanh", 0),
    (3, (12,), "tanh", 0),
    (5, (16,), "tanh", 0),
    (2, (16,), "logistic", 0),
    (2, (8, 8), "logistic", 0),
    (3, (12,), "logistic", 0),
    (2, (16,), "softplus", 0),
    (2, (8, 8), "softplus", 0),
]

CHILD = r"""
import json, sys
import numpy as np
from discopt.ml.network import Activation
from discopt.ml.formulations.base import NNFormulation
import discopt
from discopt_benchmarks.scripts.surrogate_entry._common import describe, to_network, train_mlp

d, hidden, act, seed, tl = json.loads(sys.argv[1])
train_act = "tanh" if act == "softplus" else act
mlp, rmse = train_mlp(d, tuple(hidden), train_act, seed)
net = to_network(mlp, d)
if act == "softplus":
    for layer in net.layers:
        if layer.activation == Activation.TANH:
            layer.activation = Activation.SOFTPLUS
m = discopt.Model("dv")
f = NNFormulation(m, net, strategy="full_space", prefix="nn")
f.formulate()
m.minimize(f.outputs[0])
r = m.solve(time_limit=tl, deterministic=True)
out = describe(r)
out["rmse"] = rmse
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
    p = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", CHILD, json.dumps([*item, tl])],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        timeout=tl * 4 + 120,
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
    n_checks = 0
    items = PANEL if a.only is None else [PANEL[a.only]]
    with open(a.out, "a") as fh:
        for i, item in enumerate(items):
            tag = f"d{item[0]}_{'x'.join(map(str, item[1]))}_{item[2]}_s{item[3]}"
            order = (False, True) if i % 2 == 0 else (True, False)
            rows = {}
            for on in order:
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
            best = min((r["forward_obj"] for r in rows.values() if "forward_obj" in r), default=None)
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
                f"wall={on['wall']:.1f} bound={on['bound']} load={os.getloadavg()[0]:.2f}",
                flush=True,
            )
    print(f"executed checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
