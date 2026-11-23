"""Entry experiment E1: LP-based big-M tightening for embedded ReLU networks.

This is the entry experiment that ``docs/dev/nn-module-plan.md`` already names
for T-N3.2 ("LP-based big-M tightening"). It was prompted by PWLNet
(Sanyal, Demir, Türkay, Aydın, arXiv:2611.25969v1, 2026), whose big-M constants
come from interval propagation alone. They report the interval bounds as
"conservative after several layers", and in their Van de Vusse study (Sec.
5.3.3) the hidden-unit box of an unrolled RNN shrank from about 6018 to about
520 once propagation used only the reachable inputs. discopt's
``ReluBigMFormulation`` also uses interval bounds alone
(``discopt.ml.bounds.propagate_bounds``). The LP-tightening method itself is
Tjeng, Xiao & Tedrake (ICLR 2019) and Grimstad & Andersson (Comput. Chem. Eng.
131, 2019): solve min/max of each pre-activation over the LP relaxation of the
layers before it.

Hypothesis
    Per-neuron LP bounds shrink the big-M interval of the unstable ReLUs
    enough, and stabilise enough neurons (binaries removed), that the embedded
    MILP solves faster *including* the time spent computing the LP bounds.

Kill criterion (from the plan doc, T-N3.2)
    Median big-M width shrinkage over unstable neurons < 20%, OR the median
    end-to-end time (tightening + solve) does not improve.

Soundness checks (CLAUDE.md §5, bound-changing)
    * Feasible-point sampling: forward passes at random inputs in the box must
      land inside every LP-tightened pre-activation interval.
    * Certified objectives of the two arms must agree.
    The script counts both kinds of assertion and exits non-zero if none ran
    (§6).

Caveat: the LP values come from HiGHS in floating point. This probe pads each
bound by 1e-6 (1 + |v|). A production implementation would need a safe dual
bound (e.g. Neumaier-Shcherbina) instead.

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.e1_lp_bigm_tightening \
        --reps 2 --time-limit 120 --out <scratch>/e1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import discopt
import numpy as np
from discopt.ml.bounds import LayerBounds, propagate_bounds
from discopt.ml.formulations.relu_bigm import ReluBigMFormulation
from discopt.ml.network import Activation
from scipy.optimize import linprog

from ._common import BOX, describe, loaded_code_banner, to_network, train_mlp

PANEL = [
    # (input dim, hidden sizes, seed)
    (2, (32, 16), 0),
    (2, (32, 32), 0),
    (2, (64, 64), 0),
    (2, (32, 32, 32), 0),
    (5, (32, 32), 0),
    (5, (64, 64), 0),
    (5, (32, 32, 32), 0),
    (10, (50, 50), 0),
    (2, (32, 32), 1),
    (5, (32, 32), 1),
]

PAD_REL = 1e-6


# --------------------------------------------------------------------------
# LP bound tightening
# --------------------------------------------------------------------------


def lp_tighten(net, lb0, ub0) -> tuple[list[LayerBounds], int]:
    """Layer-by-layer LP bounds on every pre-activation; returns (bounds, n_lps).

    Variables are stacked as [x, zhat_0, z_0, zhat_1, z_1, ...]. For layer k the
    LP holds the exact affine rows of all layers up to k, and the triangle
    relaxation (the projection of the big-M rows with q relaxed to [0, 1]) of
    every unstable ReLU in layers < k, with those layers' already-tightened
    bounds. It then minimises and maximises each zhat_k[j].
    """
    d = net.input_size
    out: list[LayerBounds] = []
    n_lps = 0
    # Column bounds and equality/inequality rows accumulate as layers are added.
    col_lb = list(lb0)
    col_ub = list(ub0)
    A_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    prev_idx = list(range(d))  # columns of the previous layer's output
    prev_lb, prev_ub = np.asarray(lb0, float), np.asarray(ub0, float)

    def pad_rows(rows, n):
        return [np.concatenate([r, np.zeros(n - r.size)]) for r in rows]

    for k, layer in enumerate(net.layers):
        W = np.asarray(layer.weights, float)
        b = np.asarray(layer.biases, float)
        n = W.shape[1]
        # Interval start for this layer from the tightened previous box.
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        ilb = Wp.T @ prev_lb + Wn.T @ prev_ub + b
        iub = Wp.T @ prev_ub + Wn.T @ prev_lb + b
        zhat_idx = list(range(len(col_lb), len(col_lb) + n))
        col_lb += list(ilb)
        col_ub += list(iub)
        ncol = len(col_lb)
        for j in range(n):
            row = np.zeros(ncol)
            row[zhat_idx[j]] = 1.0
            row[prev_idx] = -W[:, j]
            A_eq.append(row)
            b_eq.append(b[j])
        Aeq = np.array(pad_rows(A_eq, ncol))
        Aub = np.array(pad_rows(A_ub, ncol)) if A_ub else None
        bnds = list(zip(col_lb, col_ub, strict=False))
        lo, hi = ilb.copy(), iub.copy()
        if k > 0:  # layer 0 interval bounds are already exact for a box
            for j in range(n):
                c = np.zeros(ncol)
                c[zhat_idx[j]] = 1.0
                for sgn in (1.0, -1.0):
                    r = linprog(
                        sgn * c,
                        A_ub=Aub,
                        b_ub=np.array(b_ub) if A_ub else None,
                        A_eq=Aeq,
                        b_eq=np.array(b_eq),
                        bounds=bnds,
                        method="highs",
                    )
                    n_lps += 1
                    if r.status != 0:
                        raise RuntimeError(f"LP for layer {k} neuron {j} failed: {r.message}")
                    v = sgn * r.fun
                    pad = PAD_REL * (1.0 + abs(v))
                    if sgn > 0:
                        lo[j] = max(lo[j], v - pad)
                    else:
                        hi[j] = min(hi[j], v + pad)
        for j in range(n):
            col_lb[zhat_idx[j]], col_ub[zhat_idx[j]] = lo[j], hi[j]
        if layer.activation == Activation.RELU:
            post_lb, post_ub = np.maximum(lo, 0), np.maximum(hi, 0)
            z_idx = list(range(len(col_lb), len(col_lb) + n))
            col_lb += list(post_lb)
            col_ub += list(post_ub)
            ncol = len(col_lb)
            for j in range(n):
                L, U = lo[j], hi[j]
                zi, hj = z_idx[j], zhat_idx[j]
                if L >= 0:
                    row = np.zeros(ncol)
                    row[zi], row[hj] = 1.0, -1.0
                    A_eq.append(row)
                    b_eq.append(0.0)
                elif U <= 0:
                    row = np.zeros(ncol)
                    row[zi] = 1.0
                    A_eq.append(row)
                    b_eq.append(0.0)
                else:
                    # z >= zhat  ->  zhat - z <= 0
                    row = np.zeros(ncol)
                    row[hj], row[zi] = 1.0, -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)
                    # z <= U (zhat - L) / (U - L)
                    s = U / (U - L)
                    row = np.zeros(ncol)
                    row[zi], row[hj] = 1.0, -s
                    A_ub.append(row)
                    b_ub.append(-s * L)
            prev_idx = z_idx
        elif layer.activation == Activation.LINEAR:
            post_lb, post_ub = lo.copy(), hi.copy()
            prev_idx = zhat_idx
        else:
            raise ValueError(f"E1 covers ReLU/linear nets only, got {layer.activation}")
        out.append(LayerBounds(lo, hi, post_lb, post_ub))
        prev_lb, prev_ub = post_lb, post_ub
    return out, n_lps


class _InjectedBounds(ReluBigMFormulation):
    """ReluBigMFormulation that uses supplied layer bounds instead of intervals."""

    injected: list[LayerBounds]

    def _propagate(self, *, input_bounds):  # noqa: D401
        return self.injected


def solve_arm(net, bounds, time_limit):
    m = discopt.Model("e1")
    f = _InjectedBounds(m, net, "nn", None)
    f.injected = bounds
    _, outs = f.build()
    m.minimize(outs[0])
    t = time.perf_counter()
    res = m.solve(time_limit=time_limit)
    wall = time.perf_counter() - t
    row = describe(res)
    row["wall_outer"] = wall
    return row


def unstable(bounds, net):
    n = 0
    for lb, layer in zip(bounds, net.layers, strict=False):
        if layer.activation == Activation.RELU:
            n += int(np.sum((lb.pre_lb < 0) & (lb.pre_ub > 0)))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--only", type=int, default=None, help="run one panel index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)

    # Warm-up solve so the first timed arm does not pay import/JIT costs.
    wm = discopt.Model("warm")
    wx = wm.continuous("x", lb=0, ub=1)
    wb = wm.binary("b")
    wm.subject_to(wx <= wb)
    wm.minimize(wx - wb)
    wm.solve(time_limit=10)

    n_sample_checks = 0
    n_obj_checks = 0
    fh = open(a.out, "a")
    panel = PANEL if a.only is None else [PANEL[a.only]]
    for d, hidden, seed in panel:
        mlp, rmse = train_mlp(d, hidden, "relu", seed)
        net = to_network(mlp, d)
        lb0, ub0 = np.full(d, -BOX), np.full(d, BOX)
        ib = propagate_bounds(net)
        t = time.perf_counter()
        lpb, n_lps = lp_tighten(net, lb0, ub0)
        t_lp = time.perf_counter() - t

        # Feasible-point sampling: no reachable pre-activation outside LP bounds.
        rng = np.random.default_rng(12345)
        X = rng.uniform(-BOX, BOX, size=(a.samples, d))
        h = X
        for lb, layer in zip(lpb, net.layers, strict=False):
            zhat = h @ layer.weights + layer.biases
            if np.any(zhat < lb.pre_lb - 1e-9) or np.any(zhat > lb.pre_ub + 1e-9):
                print(f"UNSOUND LP bound on {d}/{hidden}/{seed}", flush=True)
                return 2
            n_sample_checks += zhat.size
            h = np.maximum(zhat, 0) if layer.activation == Activation.RELU else zhat

        # Big-M width shrinkage over neurons that are unstable under intervals.
        # Layer 0 is excluded: interval bounds of one affine map over a box are
        # already exact, so its shrinkage is 0 by construction and would only
        # dilute the median.
        shrink = []
        for il, ll, layer in list(zip(ib, lpb, net.layers, strict=False))[1:]:
            if layer.activation != Activation.RELU:
                continue
            mask = (il.pre_lb < 0) & (il.pre_ub > 0)
            w_i = (il.pre_ub - il.pre_lb)[mask]
            w_l = (ll.pre_ub - ll.pre_lb)[mask]
            shrink += list(1.0 - w_l / w_i)
        info = {
            "net": f"d{d}_{'x'.join(map(str, hidden))}_s{seed}",
            "rmse": rmse,
            "unstable_interval": unstable(ib, net),
            "unstable_lp": unstable(lpb, net),
            "median_shrink_deep": float(np.median(shrink)) if shrink else 0.0,
            "lp_time": t_lp,
            "n_lps": n_lps,
        }
        print(json.dumps(info), flush=True)

        for rep in range(a.reps):
            arms = [("interval", ib), ("lp", lpb)]
            if rep % 2:
                arms.reverse()  # interleave the arm order (§9)
            got = {}
            for name, bnds in arms:
                row = solve_arm(net, bnds, a.time_limit)
                row.update(info, arm=name, rep=rep)
                got[name] = row
                print(json.dumps(row), flush=True)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            gi, gl = got["interval"], got["lp"]
            if gi["gap_certified"] and gl["gap_certified"]:
                tol = 1e-4 * max(1.0, abs(gi["objective"]))
                if abs(gi["objective"] - gl["objective"]) > tol:
                    print(f"OBJECTIVE MISMATCH {gi['objective']} vs {gl['objective']}")
                    return 3
                n_obj_checks += 1
    print(f"executed sample checks={n_sample_checks} objective checks={n_obj_checks}")
    if n_sample_checks == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
