"""Pooling-problem builders from #1619 D-11 (the textbook's ``pool18`` helper).

``plant(seed)`` generates a two-pool, five-crude blending instance; ``haverly(case)``
the Haverly (1978) instances; ``build_pool(d, form)`` the p-, q- or pq-formulation;
``in_units(d, unit)`` the same instance with flows measured in ``unit`` barrels.
"""

import copy
import math
from functools import reduce

import discopt.modeling as dm
import numpy as np


def lsum(terms):
    terms = list(terms)
    return reduce(lambda a, b: a + b, terms) if terms else 0.0


def haverly(case=1):
    """Haverly (1978) instances 1-3: crudes A, B into one pool, C bypassing it."""
    return dict(
        inputs={
            "A": dict(cost=6.0, qual=[3.0]),
            "B": dict(cost=13.0 if case == 3 else 16.0, qual=[1.0]),
            "C": dict(cost=10.0, qual=[2.0]),
        },
        pools={"P": dict(feeds=["A", "B"])},
        products={
            "X": dict(price=9.0, demand=600.0 if case == 2 else 100.0, qmax=[2.5]),
            "Y": dict(price=15.0, demand=200.0, qmax=[1.5]),
        },
        pool_out=[("P", "X"), ("P", "Y")],
        bypass=[("C", "X"), ("C", "Y")],
    )


def plant(seed=5, nI=5, nL=2, nJ=3, K=2):
    rng = np.random.default_rng(seed)
    qual = np.round(rng.uniform(0.5, 4.5, (nI, K)), 1)
    cost = np.round(18 - 2.0 * qual.mean(1) + rng.uniform(-1, 1, nI), 1)
    crude = "ABCDE"
    inputs = {
        crude[i]: dict(cost=float(cost[i]), qual=[float(v) for v in qual[i]]) for i in range(nI)
    }
    pools = {}
    for l in range(nL):
        feeds = sorted(rng.choice(nI, size=3, replace=False))
        pools[f"P{l + 1}"] = dict(
            feeds=[crude[i] for i in feeds], cap=float(np.round(rng.uniform(80, 160)))
        )
    products = {
        f"X{j + 1}": dict(
            price=float(np.round(rng.uniform(12, 18), 1)),
            demand=float(np.round(rng.uniform(50, 150))),
            qmax=[float(v) for v in np.round(rng.uniform(1.5, 3.0, K), 1)],
        )
        for j in range(nJ)
    }
    bypass = [(crude[i], j) for i in range(nI) for j in products if rng.uniform() < 0.25]
    return dict(
        inputs=inputs,
        pools=pools,
        products=products,
        bypass=bypass,
        pool_out=[(l, j) for l in pools for j in products],
    )


def pooling_model(d, form, newvar):
    I, L, J = d["inputs"], d["pools"], d["products"]
    K = len(next(iter(I.values()))["qual"])
    out = {l: [j for (m, j) in d["pool_out"] if m == l] for l in L}
    cap = {l: L[l].get("cap", sum(J[j]["demand"] for j in out[l])) for l in L}
    y = {(l, j): newvar(f"y_{l}{j}", 0, min(cap[l], J[j]["demand"])) for (l, j) in d["pool_out"]}
    z = {(i, j): newvar(f"z_{i}{j}", 0, J[j]["demand"]) for (i, j) in d["bypass"]}
    sell = {
        j: lsum(
            [y[l, j] for l in L if j in out[l]] + [z[i, j] for (i, jj) in d["bypass"] if jj == j]
        )
        for j in J
    }
    bypassed = {i: [z[i, j] for (ii, j) in d["bypass"] if ii == i] for i in I}
    rows = []
    if form == "p":
        f = {(i, l): newvar(f"f_{i}{l}", 0, cap[l]) for l in L for i in L[l]["feeds"]}
        p = {
            (l, k): newvar(
                f"p_{l}{k + 1}",
                min(I[i]["qual"][k] for i in L[l]["feeds"]),
                max(I[i]["qual"][k] for i in L[l]["feeds"]),
            )
            for l in L
            for k in range(K)
        }
        py = {(l, k, j): p[l, k] * y[l, j] for l in L for k in range(K) for j in out[l]}
        for l in L:
            rows.append(
                (lsum(f[i, l] for i in L[l]["feeds"]) - lsum(y[l, j] for j in out[l]), 0, 0)
            )
            rows += [
                (
                    lsum(py[l, k, j] for j in out[l])
                    - lsum(I[i]["qual"][k] * f[i, l] for i in L[l]["feeds"]),
                    0,
                    0,
                )
                for k in range(K)
            ]
        into = lambda j, k: [py[l, k, j] for l in L if j in out[l]]
        bought = {i: lsum([f[i, l] for l in L if i in L[l]["feeds"]] + bypassed[i]) for i in I}
    else:
        q = {(i, l): newvar(f"q_{i}{l}", 0, 1) for l in L for i in L[l]["feeds"]}
        qy = {(i, l, j): q[i, l] * y[l, j] for l in L for i in L[l]["feeds"] for j in out[l]}
        rows += [(lsum(q[i, l] for i in L[l]["feeds"]), 1, 1) for l in L]
        into = lambda j, k: [
            I[i]["qual"][k] * qy[i, l, j] for l in L if j in out[l] for i in L[l]["feeds"]
        ]
        bought = {
            i: lsum([qy[i, l, j] for l in L if i in L[l]["feeds"] for j in out[l]] + bypassed[i])
            for i in I
        }
        if form == "pq":
            for l in L:
                rows += [(lsum(qy[i, l, j] for i in L[l]["feeds"]) - y[l, j], 0, 0) for j in out[l]]
                rows += [
                    (lsum(qy[i, l, j] for j in out[l]) - cap[l] * q[i, l], -np.inf, 0)
                    for i in L[l]["feeds"]
                ]
    rows += [(lsum(y[l, j] for j in out[l]), -np.inf, cap[l]) for l in L if "cap" in L[l]]
    for j in J:
        rows.append((sell[j], -np.inf, J[j]["demand"]))
        rows += [
            (
                lsum(
                    into(j, k) + [I[i]["qual"][k] * z[i, jj] for (i, jj) in d["bypass"] if jj == j]
                )
                - J[j]["qmax"][k] * sell[j],
                -np.inf,
                0,
            )
            for k in range(K)
        ]
    profit = lsum(J[j]["price"] * sell[j] for j in J) - lsum(I[i]["cost"] * bought[i] for i in I)
    return profit, rows


def build_pool(d, form, box=None):
    m = dm.Model(f"pool_{form}")
    vs = {}

    def newvar(name, lo, hi):
        lo, hi = box[name] if box is not None else (lo, hi)
        vs[name] = m.continuous(name, lb=float(lo), ub=float(hi))
        return vs[name]

    profit, rows = pooling_model(d, form, newvar)
    for g, lo, hi in rows:
        if lo == hi:
            m.subject_to(g == lo)
        else:
            if math.isfinite(hi):
                m.subject_to(g <= hi)
            if math.isfinite(lo):
                m.subject_to(g >= lo)
    m.maximize(profit)
    return m, vs


def in_units(d, unit):
    e = copy.deepcopy(d)
    for v in e["inputs"].values():
        v["cost"] *= unit
    for v in e["products"].values():
        v["price"] *= unit
        v["demand"] /= unit
    for v in e["pools"].values():
        if "cap" in v:
            v["cap"] /= unit
    return e
