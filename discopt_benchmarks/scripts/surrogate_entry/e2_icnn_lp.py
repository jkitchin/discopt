"""Entry experiment E2: an input-convex ReLU network minimised as an LP, no binaries.

Prompted by J. Kitchin, "MIP surrogates" (2026-10-28,
https://kitchingroup.cheme.cmu.edu/posts/mip-surrogates/). There, pycse's
``JAXICNNRegressor(formulation="convex")`` drops the ReLU binaries. For an
input-convex network (Amos, Xu & Kolter, ICML 2017) whose output is minimised,
``z >= 0, z >= zhat`` suffices. The post measured 0.47 s as a MILP with 122
binaries vs 0.06 s as an LP on a (64, 64) ICNN, with the same minimum.

Network class tested: a *sequential* ReLU network (discopt's
``NetworkDefinition`` has no skip connections) whose first layer is
unconstrained and whose later weights (hidden-to-hidden and output) are all
nonnegative. Each later unit is then a nonnegative combination of convex
functions passed through a convex nondecreasing ReLU, so the output is convex in
x. Minimising it, the epigraph rows give the exact minimum. This is the
one-sided property the post states.

Hypothesis
    On ICNNs, discopt's default big-M MILP route is materially slower than the
    binary-free LP, so a structural ``convex`` route is worth adding.

Kill criterion
    Median MILP/LP end-to-end wall ratio < 2 on the panel. The LP's root bound is
    already exact here (with q relaxed, the big-M rows admit z = max(0, zhat)),
    so a good MILP solver may close at the root and leave nothing to gain.

Soundness check
    Both arms must reach the same certified objective. The script counts these
    checks and exits non-zero if none ran (§6).

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.e2_icnn_lp \
        --reps 3 --out <scratch>/e2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import discopt
import discopt.modeling as dm
import numpy as np
from discopt.ml.bounds import propagate_bounds
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition

from ._common import BOX, convex_f, describe, loaded_code_banner, sample

PANEL = [
    (2, (32, 16), 0),
    (2, (64, 64), 0),
    (2, (64, 64, 64), 0),
    (5, (64, 64), 0),
    (5, (128, 128), 0),
    (10, (64, 64), 0),
    (2, (64, 64), 1),
    (5, (64, 64), 1),
]


def train_icnn(d, hidden, seed, iters=4000, lr=3e-3):
    """Full-batch Adam on MSE, projecting later-layer weights onto W >= 0."""
    rng = np.random.default_rng(seed)
    X, y = sample(d, 400 * d, seed, convex_f)
    Xt, yt = sample(d, 2000, seed + 1000, convex_f)
    sizes = [d, *hidden, 1]
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1]))
        if i > 0:
            W = np.abs(W) / sizes[i]
        Ws.append(W)
        bs.append(np.zeros(sizes[i + 1]))
    params = Ws + bs
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    L = len(Ws)
    for t in range(1, iters + 1):
        hs, zs = [X], []
        for i in range(L):
            z = hs[-1] @ Ws[i] + bs[i]
            zs.append(z)
            hs.append(np.maximum(z, 0) if i < L - 1 else z)
        g = 2 * (hs[-1][:, 0] - y)[:, None] / len(y)
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            if i < L - 1:
                g = g * (zs[i] > 0)
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            g = g @ Ws[i].T
        grads = gW + gb
        for j, (p, gr) in enumerate(zip(params, grads, strict=False)):
            m1[j] = 0.9 * m1[j] + 0.1 * gr
            m2[j] = 0.999 * m2[j] + 0.001 * gr**2
            p -= lr * (m1[j] / (1 - 0.9**t)) / (np.sqrt(m2[j] / (1 - 0.999**t)) + 1e-8)
        for i in range(1, L):
            np.maximum(Ws[i], 0.0, out=Ws[i])
    layers = [
        DenseLayer(Ws[i].copy(), bs[i].copy(), Activation.RELU if i < L - 1 else Activation.LINEAR)
        for i in range(L)
    ]
    net = NetworkDefinition(layers, input_bounds=(np.full(d, -BOX), np.full(d, BOX)))
    pred = net.forward(Xt)
    rmse = float(np.sqrt(np.mean((np.ravel(pred) - yt) ** 2)))
    return net, rmse


def is_icnn(net) -> bool:
    return all(np.all(layer.weights >= 0) for layer in net.layers[1:])


def solve_milp(net, time_limit):
    m = discopt.Model("e2_milp")
    _, outs = ReluBigMFormulation(m, net, "nn", None).build()
    m.minimize(outs[0])
    return m, m.solve(time_limit=time_limit)


def solve_lp(net, time_limit):
    """Epigraph LP: zhat = W^T z_prev + b, z >= 0, z >= zhat (no binaries)."""
    if not is_icnn(net):
        raise ValueError("LP route is exact only for an ICNN being minimised")
    lb = propagate_bounds(net)
    m = discopt.Model("e2_lp")
    d = net.input_size
    prev = m.continuous("x", shape=(d,), lb=-BOX, ub=BOX)
    for k, layer in enumerate(net.layers):
        W, b = layer.weights, layer.biases
        n = W.shape[1]
        zhat = m.continuous(f"zhat{k}", shape=(n,), lb=lb[k].pre_lb, ub=lb[k].pre_ub)
        m.subject_to(zhat == dm.sum(W.T * prev, axis=1) + b)
        if layer.activation == Activation.RELU:
            z = m.continuous(f"z{k}", shape=(n,), lb=lb[k].post_lb, ub=lb[k].post_ub)
            m.subject_to(z >= zhat)
            prev = z
        else:
            prev = zhat
    m.minimize(prev[0])
    return m, m.solve(time_limit=time_limit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)
    wm = discopt.Model("warm")
    wx = wm.continuous("x", lb=0, ub=1)
    wb = wm.binary("b")
    wm.subject_to(wx <= wb)
    wm.minimize(wx - wb)
    wm.solve(time_limit=10)

    n_checks = 0
    fh = open(a.out, "a")
    panel = PANEL if a.only is None else [PANEL[a.only]]
    for d, hidden, seed in panel:
        net, rmse = train_icnn(d, hidden, seed)
        assert is_icnn(net)
        tag = f"d{d}_{'x'.join(map(str, hidden))}_s{seed}"
        n_bin = sum(
            int(np.sum((lb.pre_lb < 0) & (lb.pre_ub > 0)))
            for lb, layer in zip(propagate_bounds(net), net.layers, strict=False)
            if layer.activation == Activation.RELU
        )
        print(json.dumps({"net": tag, "rmse": rmse, "binaries": n_bin}), flush=True)
        for rep in range(a.reps):
            arms = [("milp", solve_milp), ("lp", solve_lp)]
            if rep % 2:
                arms.reverse()
            got = {}
            for name, fn in arms:
                t = time.perf_counter()
                _, res = fn(net, a.time_limit)
                row = describe(res)
                row.update(
                    wall_outer=time.perf_counter() - t,
                    net=tag,
                    rmse=rmse,
                    binaries=n_bin,
                    arm=name,
                    rep=rep,
                )
                got[name] = row
                print(json.dumps(row), flush=True)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if got["milp"]["gap_certified"] and got["lp"]["status"] == "optimal":
                o1, o2 = got["milp"]["objective"], got["lp"]["objective"]
                if abs(o1 - o2) > 1e-4 * max(1.0, abs(o1)):
                    print(f"OBJECTIVE MISMATCH {o1} vs {o2}", flush=True)
                    return 3
                n_checks += 1
    print(f"executed objective checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
