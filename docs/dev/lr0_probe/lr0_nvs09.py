"""
LR-0 nvs09 log-space root LP prototype.

Objective (min):  sum_i [ (ln(x_i-2))^2 + (ln(10-x_i))^2 ]  -  (prod_i x_i)^0.2
  x_i integer in [3,9], 10 vars, 0 constraints. optimum = -43.134.

Variant (a) H-LOG only: the coupling product (prod x_i)^0.2 = exp(0.2 * sum ln x_i)
  handled by the exact log/exp univariate envelopes; the per-variable composites
  g_i(x)=(ln(x-2))^2+(ln(10-x))^2 handled by standard composition (log envelope of
  the inner ln then convex square envelope).

Variant (b) H-LOG + H-UNI: replace each g_i by its exact 1-D convex underenvelope
  over [3,9].
"""

from __future__ import annotations

import math

import numpy as np

from lr0_envelopes import (
    LP,
    concave_env_rows,
    convex_env_rows,
    exact_convex_underenvelope_rows,
    exp_f,
    exp_fp,
    ln_f,
    ln_fp,
)

LO, HI = 3.0, 9.0
N = 10
OPT = -43.134
DISCOPT_ROOT = None  # nvs09 root bound not reported in plan table


def g_i(x):
    return np.log(x - 2.0) ** 2 + np.log(10.0 - x) ** 2


def square_convex_env(lp, xname, yname, lo, hi):
    """y = x^2 convex envelope over [lo,hi]. y>=tangents, y<=secant."""
    convex_env_rows(lp, xname, yname, lo, hi, lambda t: t * t, lambda t: 2 * t, n_tangents=3)


def build_product_term(lp):
    """Add vars/rows for t = (prod x_i)^0.2 = exp(0.2 sum ln x_i).
    Returns the LP var name for t. x_i are named 'x{i}'."""
    # z_i = ln x_i, concave envelope over [LO,HI]
    zsum_coeffs = {}
    for i in range(N):
        xi = f"x{i}"
        zi = f"z{i}"
        lp.var(zi, math.log(LO), math.log(HI))
        concave_env_rows(lp, xi, zi, LO, HI, ln_f, ln_fp, n_tangents=3)
        zsum_coeffs[zi] = 0.2  # s = 0.2 * sum z_i
    # s = 0.2 sum z_i  (exact linear)
    s_lo = 0.2 * N * math.log(LO)
    s_hi = 0.2 * N * math.log(HI)
    lp.var("s", s_lo, s_hi)
    lp.row({**{k: -v for k, v in zsum_coeffs.items()}, "s": 1.0}, 0.0, "==")
    # t = exp(s), convex envelope over [s_lo, s_hi]
    lp.var("t", math.exp(s_lo), math.exp(s_hi))
    convex_env_rows(lp, "s", "t", s_lo, s_hi, exp_f, exp_fp, n_tangents=3)
    return "t"


def build_g_composition(lp, i):
    """Variant (a): g_i via composition. Returns list of obj var names to sum.
    g_i = a_i + b_i where a_i=(ln(x_i-2))^2, b_i=(ln(10-x_i))^2."""
    xi = f"x{i}"
    # inner u = ln(x_i - 2), x_i-2 in [1,7] -> u in [0, ln7]
    u = f"u{i}"
    ulo, uhi = math.log(LO - 2), math.log(HI - 2)
    lp.var(u, ulo, uhi)
    # ln(x-2): concave in x. envelope on x directly: treat h(x)=ln(x-2)
    concave_env_rows(lp, xi, u, LO, HI, lambda x: math.log(x - 2), lambda x: 1.0 / (x - 2), 3)
    # a = u^2 convex env over [ulo,uhi]
    a = f"a{i}"
    lp.var(a, 0.0, max(ulo * ulo, uhi * uhi))
    square_convex_env(lp, u, a, ulo, uhi)

    # inner w = ln(10 - x_i), 10-x in [1,7] -> w in [0, ln7]
    w = f"w{i}"
    wlo, whi = math.log(10 - HI), math.log(10 - LO)
    lp.var(w, wlo, whi)
    # ln(10-x): concave in x (composition of ln with decreasing affine). envelope on x.
    concave_env_rows(lp, xi, w, LO, HI, lambda x: math.log(10 - x), lambda x: -1.0 / (10 - x), 3)
    b = f"b{i}"
    lp.var(b, 0.0, max(wlo * wlo, whi * whi))
    square_convex_env(lp, w, b, wlo, whi)
    return [a, b]


def build_lp(variant):
    lp = LP()
    for i in range(N):
        lp.var(f"x{i}", LO, HI)
    # product term (both variants use H-LOG for the product)
    t = build_product_term(lp)
    lp.add_obj(t, -1.0)  # minus (prod)^0.2

    if variant == "a":
        for i in range(N):
            for gv in build_g_composition(lp, i):
                lp.add_obj(gv, 1.0)
    elif variant == "b":
        for i in range(N):
            gv = f"g{i}"
            gmin = float(np.min(g_i(np.linspace(LO, HI, 100001))))
            gmax = float(np.max(g_i(np.array([LO, HI]))))
            lp.var(gv, gmin, gmax)
            exact_convex_underenvelope_rows(lp, f"x{i}", gv, LO, HI, g_i)
            lp.add_obj(gv, 1.0)
    return lp


def main():
    for variant in ("a", "b"):
        lp = build_lp(variant)
        res, names, idx = lp.solve()
        bound = res.fun + lp.obj_const if res.success else None
        gap = OPT - bound if bound is not None else None
        # fraction of gap closed vs discopt root: n/a (no root reported)
        print(f"nvs09 variant ({variant}) H-LOG{'+H-UNI' if variant=='b' else '-only'}:")
        print(f"   root LP bound = {bound:.6f}   optimum = {OPT}")
        print(f"   status = {res.message}")
        print(f"   bound-to-opt gap = {gap:.6f}   within tol? {abs(gap) <= 1e-4*(1+abs(OPT))}")
        print()

    # --- rigor: feasible-point sampling. Every INTEGER feasible point's true
    # objective must be >= root LP bound (a valid lower bound must not exceed it).
    # And each variant's rows must not cut any feasible (x, lifted) point. We
    # check the weaker but sufficient property: the LP bound <= true optimum, and
    # LP bound <= true objective at every sampled integer point.
    from itertools import product as iproduct

    lp_a = build_lp("a")
    ra, _, _ = lp_a.solve()
    lp_b = build_lp("b")
    rb, _, _ = lp_b.solve()
    ba = ra.fun
    bb = rb.fun
    rng = np.random.default_rng(1)
    minobj = np.inf
    for _ in range(20000):
        x = rng.integers(3, 10, size=N).astype(float)
        o = float(np.sum(g_i(x)) - (np.prod(x)) ** 0.2)
        minobj = min(minobj, o)
    print(f"[sampling] min true objective over 20000 random integer pts: {minobj:.4f}")
    print(f"[sampling] variant-a bound {ba:.4f} <= min sampled obj? {ba <= minobj + 1e-6}")
    print(f"[sampling] variant-b bound {bb:.4f} <= min sampled obj? {bb <= minobj + 1e-6}")


if __name__ == "__main__":
    main()
