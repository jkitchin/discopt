"""Spread panel for ``DISCOPT_NN_LP_BIGM`` on ``relu_extended``: both arms perturbed.

The class panel (``nn_lp_bigm_panel.py``) is deterministic, so it has no
run-to-run spread to judge §5 bar 2 ("not harmful beyond the spread") against.
HiGHS's search is the source of variance instead: widening every interval bound
by a sound, negligible 1e-9 relative moved a ``relu_extended`` solve from 19 to
321 nodes, and an incumbent from -1.62 to +1.35. This panel measures that spread
for BOTH arms: each (network, arm) is solved under the same set of negligible
widenings of whatever bounds the arm uses (interval OFF, LP-tightened ON), and
the arms' outcome distributions are compared.

A widening by ``rel * (1 + max(|lb|, |ub|))`` keeps every bound valid, so each
perturbed model is still an exact encoding of the network.

Checks, counted (exit non-zero if none ran, CLAUDE.md §6): every incumbent
reproduces ``forward``; no bound exceeds the best verified objective seen for
that network in any run.

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.nn_lp_bigm_perturb_panel \
        --time-limit 120 --out <scratch>/perturb.jsonl
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
RELS = (1e-9, 3e-9, 1e-8)

CHILD = r"""
import json, sys
import numpy as np
import discopt
import discopt.ml.formulations.relu_bigm as rb
from discopt.ml.bounds import LayerBounds
from discopt.ml.formulations.base import NNFormulation
from discopt_benchmarks.scripts.surrogate_entry._common import describe, to_network, train_mlp

d, hidden, seed, rel, tl = json.loads(sys.argv[1])
net = to_network(train_mlp(d, tuple(hidden), "relu", seed)[0], d)
fired = []

def widen(fn):
    def wrapped(network, **kw):
        out = []
        for L, layer in zip(fn(network, **kw), network.layers):
            p = rel * (1.0 + np.maximum(np.abs(L.pre_lb), np.abs(L.pre_ub)))
            lo, hi = L.pre_lb - p, L.pre_ub + p
            if layer.activation.value == "relu":
                out.append(LayerBounds(lo, hi, np.maximum(lo, 0.0), np.maximum(hi, 0.0)))
            else:
                out.append(LayerBounds(lo, hi, lo, hi))
        fired.append(1)
        return out
    return wrapped

rb.propagate_bounds = widen(rb.propagate_bounds)
rb.lp_tighten_bounds = widen(rb.lp_tighten_bounds)
m = discopt.Model("p")
f = NNFormulation(m, net, strategy="relu_extended", prefix="nn")
f.formulate()
m.minimize(f.outputs[0])
r = m.solve(time_limit=tl, deterministic=True)
out = describe(r)
out["widened"] = len(fired)
out["lp_stats"] = getattr(f._strategy, "lp_bound_stats", None)
x = r.x.get(f.inputs.name) if r.x else None
if x is not None and out["objective"] is not None:
    out["forward_obj"] = float(np.ravel(net.forward(np.asarray(x, float).reshape(1, d)))[0])
print("RESULT " + json.dumps(out))
"""


def run_arm(d, hidden, seed, rel, on: bool, tl: float) -> dict:
    env = dict(os.environ)
    if on:
        env[FLAG] = "1"
    else:
        env.pop(FLAG, None)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    p = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", CHILD, json.dumps([d, list(hidden), seed, rel, tl])],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
        timeout=tl * 4 + 300,
    )
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"child produced no result (rc={p.returncode}):\n{p.stderr[-3000:]}")
    row = json.loads(line[len("RESULT ") :])
    if row["widened"] != 1:
        raise RuntimeError(f"the widening did not fire exactly once ({row['widened']})")
    if on != bool(row["lp_stats"]):
        raise RuntimeError("the arm did not use the bounds its flag says")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)
    n_checks = 0
    with open(a.out, "a") as fh:
        for i, (d, hidden, seed) in enumerate(NETS):
            tag = f"d{d}_{'x'.join(map(str, hidden))}_s{seed}"
            rows = []
            for j, rel in enumerate(RELS):
                for on in (False, True) if (i + j) % 2 == 0 else (True, False):
                    row = run_arm(d, hidden, seed, rel, on, a.time_limit)
                    row.update(net=tag, arm="ON" if on else "OFF", rel=rel)
                    fwd = row.get("forward_obj")
                    if fwd is not None:
                        if abs(fwd - row["objective"]) > 1e-4 * max(1.0, abs(fwd)):
                            print(f"FORWARD MISMATCH {tag}: {fwd} vs {row['objective']}")
                            return 4
                        n_checks += 1
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        f"{tag} rel={rel:g} {row['arm']}: {row['status']} "
                        f"cert={row['gap_certified']} obj={row['objective']} "
                        f"bound={row['bound']} nodes={row['nodes']} wall={row['wall']:.1f} "
                        f"load={os.getloadavg()[0]:.2f}",
                        flush=True,
                    )
            best = min((r["forward_obj"] for r in rows if "forward_obj" in r), default=None)
            for r in rows:
                if r["bound"] is not None and best is not None:
                    if r["bound"] > best + 1e-4 * max(1.0, abs(best)):
                        print(f"BOUND ABOVE INCUMBENT {tag} {r['arm']}: {r['bound']} > {best}")
                        return 3
                    n_checks += 1
    print(f"executed checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
