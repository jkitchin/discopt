"""Issue #694 — synthetic proxy for the #654 class (sonet*/qap/graphpart).

The decisive #654-class instances (sonet23v4, super3t, qap, eg_all_s, sonet22v4)
live only in the big Dropbox corpus and are absent from CI / this environment, and
MINLPLib is network-blocked. To exercise the *slow-build* regime the #694
hypothesis targets, we synthesize models with the same structural signature: a
dense quadratic over binaries (QAP / max-clique / network-design-like), which lifts
O(n^2) bilinear products -> O(n^2) McCormick envelope rows, so the
``build_uniform_relaxation`` cost grows into the multi-second range that dominates
sonet23v4 (16.8s).

Two families:
  * ``clique`` — max-weight-clique-style dense quadratic over ``n`` binaries with a
    cardinality band; objective is the quadratic form (products carry the cost, so
    the interval floor is loose and the bound must come from the envelopes — the
    ``plain``-is-None signature of the #654 class).
  * ``qap`` — a small quadratic assignment (n facilities) with assignment
    constraints and a sum-of-products objective.

Run::

    python discopt_benchmarks/scripts/issue694_synthetic_proxy.py --family clique --n 60
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from issue694_anytime_build_probe import probe_model  # type: ignore


def _balanced_sum(terms: list):
    """Sum a list of expressions via pairwise reduction so the expression tree is
    O(log n) deep, not O(n) — a chained ``sum()`` builds a left-nested tree that
    overflows the canonicalizer's recursion on dense quadratics."""
    if not terms:
        return 0.0
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms) - 1, 2):
            nxt.append(terms[i] + terms[i + 1])
        if len(terms) % 2 == 1:
            nxt.append(terms[-1])
        terms = nxt
    return terms[0]


def _seeded_matrix(n: int, seed: int) -> np.ndarray:
    # Deterministic pseudo-random symmetric weights in [-1, 1] (no Math.random ban
    # here — plain numpy — but keep it seeded for reproducibility).
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    return (a + a.T) * 0.5


def build_clique(n: int, seed: int = 7):
    import discopt as do

    m = do.Model(f"synth_clique_{n}")
    x = [m.binary(f"x{i}") for i in range(n)]
    q = _seeded_matrix(n, seed)
    # dense quadratic objective over binaries: sum_{i<j} q_ij x_i x_j  (MINIMIZE)
    terms = [float(q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)]
    m.minimize(_balanced_sum(terms))
    # a cardinality band so the LP is non-trivial
    k = n // 3
    m.subject_to(_balanced_sum(list(x)) >= k)
    m.subject_to(_balanced_sum(list(x)) <= 2 * k)
    lb = np.zeros(n)
    ub = np.ones(n)
    return m, lb, ub


def build_qap(n: int, seed: int = 11):
    import discopt as do

    m = do.Model(f"synth_qap_{n}")
    f = _seeded_matrix(n, seed) + 1.0  # flows >= 0-ish
    d = _seeded_matrix(n, seed + 1) + 1.0  # distances
    x = {(i, j): m.binary(f"x_{i}_{j}") for i in range(n) for j in range(n)}
    # assignment constraints
    for i in range(n):
        m.subject_to(_balanced_sum([x[(i, j)] for j in range(n)]) == 1)
    for j in range(n):
        m.subject_to(_balanced_sum([x[(i, j)] for i in range(n)]) == 1)
    # quadratic cost sum_{i,k,j,l} f_ik d_jl x_ij x_kl  (thin it to keep it buildable)
    terms = []
    for i in range(n):
        for k in range(i + 1, n):
            for j in range(n):
                for ll in range(n):
                    if ll == j:
                        continue
                    c = float(f[i, k] * d[j, ll])
                    if abs(c) < 1e-9:
                        continue
                    terms.append(c * x[(i, j)] * x[(k, ll)])
    m.minimize(_balanced_sum(terms))
    lb = np.zeros(n * n)
    ub = np.ones(n * n)
    return m, lb, ub


def build_netdesign(n: int, seed: int = 17):
    """Network-design proxy (sonet* signature): a LINEAR objective (minimize design
    cost) coupled to bilinear *constraints* (capacity = design * flow). Here the
    interval floor on the linear objective is finite from row 0 (trivially), but the
    USEFUL bound comes from the bilinear constraint envelopes — so the anytime
    question is whether the bound *tightens* continuously as envelope rows accrue."""
    import discopt as do

    m = do.Model(f"synth_netdesign_{n}")
    rng = np.random.default_rng(seed)
    # continuous design vars y_i in [0,1], flow vars f_i in [0,1]
    y = [m.continuous(f"y{i}", lb=0.0, ub=1.0) for i in range(n)]
    fl = [m.continuous(f"f{i}", lb=0.0, ub=1.0) for i in range(n)]
    cost = [float(rng.uniform(0.5, 2.0)) for i in range(n)]
    m.minimize(_balanced_sum([cost[i] * y[i] for i in range(n)]))
    # bilinear coupling: sum over a random neighborhood of y_i*f_j >= demand
    dem = rng.uniform(0.1, 0.4, size=n)
    for i in range(n):
        nbrs = [(i + t) % n for t in range(1, max(2, n // 4))]
        terms = [y[i] * fl[j] for j in nbrs]
        m.subject_to(_balanced_sum(terms) >= float(dem[i]) * len(nbrs))
    lb = np.zeros(2 * n)
    ub = np.ones(2 * n)
    return m, lb, ub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["clique", "qap", "netdesign"], default="clique")
    ap.add_argument("--n", type=int, nargs="+", default=[40, 60, 80])
    ap.add_argument("--solve-tl", type=float, default=20.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    for n in args.n:
        name = f"synth_{args.family}_{n}"
        print(f"\n=== {name} ===", flush=True)
        if args.family == "clique":
            model, lb, ub = build_clique(n)
        elif args.family == "qap":
            model, lb, ub = build_qap(n)
        else:
            model, lb, ub = build_netdesign(n)
        r = probe_model(model, name, lb, ub, solve_tl=args.solve_tl)
        results.append(r)
        if r["status"] != "ok":
            print(f"  {r['status']}", flush=True)
            continue
        print(
            f"  vars={r['n_vars']} cons={r['n_cons']} rows={r['n_rows']} cols={r['n_cols']} "
            f"build_wall={r['build_wall']:.3f}s obj_floor={r['obj_floor']} "
            f"full_bound={r['full_bound']}",
            flush=True,
        )
        print("  build%  row%   elapsed   status     bound         finite", flush=True)
        for cp in r["checkpoints"]:
            bstr = "None" if cp["bound"] is None else f"{cp['bound']:.6g}"
            print(
                f"  {cp['build_frac']*100:5.1f}  {cp['frac_rows']*100:5.1f}  "
                f"{cp['build_elapsed']:7.3f}  {cp['status']:>14}  {bstr:>12}  {cp['finite']}",
                flush=True,
            )
        ff = r["first_finite_build_frac"]
        print(
            f"  --> first finite bound at build_frac="
            f"{'never' if ff is None else f'{ff*100:.1f}%'}",
            flush=True,
        )

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
