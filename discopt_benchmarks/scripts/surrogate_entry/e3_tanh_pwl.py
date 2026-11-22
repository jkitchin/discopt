"""Entry experiment E3: piecewise-linear tanh vs the exact global solve.

Prompted by PWLNet (Sanyal, Demir, Türkay, Aydın, arXiv:2611.25969v1, 2026),
Sec. 4.4-4.5 and 5.2. They replace tanh after training with a 3-, 5- or 7-piece
chord interpolant on [-4, 4] and embed the result as a MILP. On Williams-Otto
this solved in ~0.4 s and reached 72.1-72.2 $/s against an exact optimum of
75.82. Their exact-tanh alternative is a local Ipopt solve.

discopt already has both relevant pieces. ``full_space`` solves the tanh network
to global optimality. ``discopt.modeling.nonlinear_to_pwl`` (#1482) does the
replacement in two modes: ``"approximate"`` is PWLNet's chord interpolant,
uncertified; ``"outer"`` uses error bands, so it is a rigorous relaxation that
refines until it certifies. The open question is whether either is worth wiring
into ``add_predictor``.

Hypothesis
    On embedded tanh surrogates, a PWL route (outer, or approximate) reaches the
    network's optimum, or a point within the surrogate's own test error of it,
    faster than the global ``full_space`` solve.

Kill criterion
    If the median end-to-end wall of ``outer`` is not below ``full_space``'s, the
    certified PWL route gives nothing. If ``approximate`` is also not faster, or
    its verified objective is worse than ``full_space`` by more than the
    network's test RMSE, the PWLNet-style route gives nothing either.

Soundness checks
    * Every certified arm's bound must be <= the best verified objective of
      every arm (min sense).
    * Every reported incumbent is re-evaluated through the network's
      ``forward`` and must match the reported objective.
    The script counts these checks and exits non-zero if none ran (§6).

Run::

    python -u -m discopt_benchmarks.scripts.surrogate_entry.e3_tanh_pwl \
        --reps 2 --time-limit 300 --out <scratch>/e3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import discopt
import discopt.modeling as dm
import numpy as np
from discopt.ml.formulations.base import NNFormulation

from ._common import describe, loaded_code_banner, to_network, train_mlp

PANEL = [
    (2, (8,), 0),
    (2, (16,), 0),
    (2, (32,), 0),
    (2, (16, 16), 0),
    (5, (16,), 0),
    (5, (16, 16), 0),
    (2, (16,), 1),
]


def build(net, scalar: bool):
    """The tanh network minimised over its input box.

    ``scalar=False`` uses the library ``full_space`` formulation, with one
    array-valued constraint per layer. ``scalar=True`` writes the same rows with
    one scalar constraint per neuron. ``nonlinear_to_pwl`` only replaces scalar
    terms, so if it skips the vectorised form, the scalar build tells us whether
    that is the only obstacle.
    """
    m = discopt.Model("e3")
    if not scalar:
        f = NNFormulation(m, net, strategy="full_space", prefix="nn")
        f.formulate()
        m.minimize(f.outputs[0])
        return m, f.inputs
    from discopt.ml.bounds import propagate_bounds

    lbs = propagate_bounds(net)
    lo, hi = net.input_bounds
    x = m.continuous("x", shape=(net.input_size,), lb=lo, ub=hi)
    prev = [x[i] for i in range(net.input_size)]
    for k, layer in enumerate(net.layers):
        W, b = layer.weights, layer.biases
        nxt = []
        for j in range(W.shape[1]):
            zh = m.continuous(f"zh{k}_{j}", lb=lbs[k].pre_lb[j], ub=lbs[k].pre_ub[j])
            m.subject_to(
                zh == sum(float(W[i, j]) * prev[i] for i in range(len(prev))) + float(b[j])
            )
            if layer.activation.value == "tanh":
                z = m.continuous(f"z{k}_{j}", lb=lbs[k].post_lb[j], ub=lbs[k].post_ub[j])
                m.subject_to(z == dm.tanh(zh))
                nxt.append(z)
            else:
                nxt.append(zh)
        prev = nxt
    m.minimize(prev[0])
    return m, x


def x_of(res, inputs, d):
    v = res.x.get(inputs.name) if res.x else None
    return None if v is None else np.asarray(v, float).reshape(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(loaded_code_banner(), flush=True)

    n_checks = 0
    fh = open(a.out, "a")
    panel = PANEL if a.only is None else [PANEL[a.only]]
    for d, hidden, seed in panel:
        mlp, rmse = train_mlp(d, hidden, "tanh", seed)
        net = to_network(mlp, d)
        tag = f"d{d}_{'x'.join(map(str, hidden))}_s{seed}"
        # Which build does the PWL transform see terms in?
        coverage = {}
        for scalar in (False, True):
            m, _ = build(net, scalar)
            t = dm.nonlinear_to_pwl(m, mode="outer")
            coverage[scalar] = (len(t.terms), len(t.skipped))
        pwl_scalar = coverage[False][0] == 0
        print(
            json.dumps(
                {
                    "net": tag,
                    "rmse": rmse,
                    "pwl_terms_vector": coverage[False],
                    "pwl_terms_scalar": coverage[True],
                }
            ),
            flush=True,
        )
        arms = [
            ("full_space", False, None),
            ("full_space_scalar", True, None),
            ("pwl_outer", pwl_scalar, ("outer", 8)),
            ("pwl_approx4", pwl_scalar, ("approximate", 4)),
            ("pwl_approx8", pwl_scalar, ("approximate", 8)),
        ]
        for rep in range(a.reps):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            rows = []
            for name, scalar, pwl in order:
                m, inputs = build(net, scalar)
                t0 = time.perf_counter()
                if pwl is None:
                    res = m.solve(time_limit=a.time_limit)
                else:
                    mode, seg = pwl
                    res = dm.nonlinear_to_pwl(m, mode=mode, segments=seg).solve(
                        time_limit=a.time_limit
                    )
                row = describe(res)
                row.update(
                    wall_outer=time.perf_counter() - t0, net=tag, rmse=rmse, arm=name, rep=rep
                )
                xv = x_of(res, inputs, d)
                if xv is not None and row["objective"] is not None:
                    fwd = float(np.ravel(net.forward(xv[None, :]))[0])
                    row["forward_obj"] = fwd
                    if abs(fwd - row["objective"]) > 1e-4 * max(1.0, abs(fwd)):
                        print(f"FORWARD MISMATCH {name}: {fwd} vs {row['objective']}", flush=True)
                        return 4
                    n_checks += 1
                rows.append(row)
                print(json.dumps(row), flush=True)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            best = min((r["forward_obj"] for r in rows if "forward_obj" in r), default=None)
            for r in rows:
                if r["gap_certified"] and r["bound"] is not None and best is not None:
                    if r["bound"] > best + 1e-4 * max(1.0, abs(best)):
                        print(f"BOUND ABOVE INCUMBENT {r['arm']}: {r['bound']} > {best}")
                        return 3
                    n_checks += 1
    print(f"executed checks={n_checks}")
    return 0 if n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
