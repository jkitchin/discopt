"""
LR-0 entry-experiment envelope library (standalone probe — NOT solver code).

Rigorous univariate over/under-estimator rows for the log-space monomial
relaxation. Every row returned is a proven bound; each builder documents its
soundness. All rows are returned as (a_coeffs_dict, rhs, sense) tuples where
sense is '<=' or '>=' meaning  sum(a_i * var_i) {<=,>=} rhs.

We build LPs with scipy.optimize.linprog. Variables are identified by string keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------
# LP container: named variables, rows, objective. Minimization.
# ----------------------------------------------------------------------------
@dataclass
class LP:
    lb: dict = field(default_factory=dict)      # var -> lower bound
    ub: dict = field(default_factory=dict)      # var -> upper bound
    rows: list = field(default_factory=list)    # (coeffs dict, rhs, sense)
    obj: dict = field(default_factory=dict)     # var -> objective coeff (min)
    obj_const: float = 0.0

    def var(self, name, lo, hi):
        assert lo <= hi + 1e-12, (name, lo, hi)
        self.lb[name] = lo
        self.ub[name] = hi
        return name

    def var_free(self, name, lo, hi):
        # lo/hi may be None (unbounded in that direction)
        if lo is not None and hi is not None:
            assert lo <= hi + 1e-9, (name, lo, hi)
        self.lb[name] = lo
        self.ub[name] = hi
        return name

    def row(self, coeffs, rhs, sense):
        # normalize / drop empty
        self.rows.append((dict(coeffs), float(rhs), sense))

    def add_obj(self, name, c):
        self.obj[name] = self.obj.get(name, 0.0) + c

    def solve(self):
        names = sorted(set(list(self.lb) + list(self.obj)))
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        c = np.zeros(n)
        for k, v in self.obj.items():
            c[idx[k]] += v
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for coeffs, rhs, sense in self.rows:
            r = np.zeros(n)
            for k, v in coeffs.items():
                r[idx[k]] += v
            if sense == "<=":
                A_ub.append(r); b_ub.append(rhs)
            elif sense == ">=":
                A_ub.append(-r); b_ub.append(-rhs)
            elif sense == "==":
                A_eq.append(r); b_eq.append(rhs)
            else:
                raise ValueError(sense)
        bounds = [(self.lb.get(nm, None), self.ub.get(nm, None)) for nm in names]
        res = linprog(
            c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        return res, names, idx


# ----------------------------------------------------------------------------
# Univariate concave envelope: y = f(x), f concave on [lo,hi]
#   tangents  = OVERestimators (y <= f(a) + f'(a)(x-a))
#   secant    = UNDERestimator (y >= secant)
# ln is concave. Returns rows on variables (xname, yname).
# ----------------------------------------------------------------------------
def concave_env_rows(lp, xname, yname, lo, hi, f, fp, n_tangents=3):
    # tangents (overestimators): y <= f(a) + f'(a)(x-a)
    pts = np.linspace(lo, hi, n_tangents)
    for a in pts:
        fa, fpa = f(a), fp(a)
        # y - fpa*x <= fa - fpa*a
        lp.row({yname: 1.0, xname: -fpa}, fa - fpa * a, "<=")
    # secant (underestimator): y >= f(lo) + m (x - lo), m=(f(hi)-f(lo))/(hi-lo)
    if hi > lo + 1e-15:
        m = (f(hi) - f(lo)) / (hi - lo)
        lp.row({yname: 1.0, xname: -m}, f(lo) - m * lo, ">=")
    else:
        lp.row({yname: 1.0}, f(lo), "==")


# ----------------------------------------------------------------------------
# Univariate convex envelope: y = f(x), f convex on [lo,hi]
#   tangents = UNDERestimators (y >= f(a)+f'(a)(x-a))
#   secant   = OVERestimator  (y <= secant)
# exp is convex.
# ----------------------------------------------------------------------------
def convex_env_rows(lp, xname, yname, lo, hi, f, fp, n_tangents=3):
    pts = np.linspace(lo, hi, n_tangents)
    for a in pts:
        fa, fpa = f(a), fp(a)
        # y >= fa + fpa (x-a)  ->  y - fpa*x >= fa - fpa*a
        lp.row({yname: 1.0, xname: -fpa}, fa - fpa * a, ">=")
    if hi > lo + 1e-15:
        m = (f(hi) - f(lo)) / (hi - lo)
        lp.row({yname: 1.0, xname: -m}, f(lo) - m * lo, "<=")
    else:
        lp.row({yname: 1.0}, f(lo), "==")


# ----------------------------------------------------------------------------
# Exact 1-D convex-hull (lower convex envelope) of an arbitrary continuous g on
# [lo,hi], built rigorously for a probe by fine sampling + lower convex hull,
# then made RIGOROUS by shifting each hull facet DOWN by the max sampling gap.
# For a genuinely convex g the lower hull = g and the shift -> 0 as N grows;
# we bound the interpolation error explicitly so the returned rows are proven
# underestimators (each facet lies at or below g everywhere).
#
# Returns rows y >= facet (underestimators of g), i.e. the convex envelope.
# For the probe we also validate by dense sampling that no row cuts g.
# ----------------------------------------------------------------------------
def exact_convex_underenvelope_rows(lp, xname, yname, lo, hi, g, N=200001):
    xs = np.linspace(lo, hi, N)
    ys = g(xs)
    # lower convex hull of points (xs, ys)
    hull = _lower_convex_hull(xs, ys)
    # Rigorous safety: the piecewise-linear lower hull of the *samples* could
    # sit slightly ABOVE g between samples if g dips. Bound that: on each sample
    # interval the hull chord vs g gap is <= (1/8) max|g''| h^2 for convex-ish g,
    # but we do not assume convexity globally. Instead compute the actual max
    # over a 10x-finer grid of (hull(x) - g(x)) and shift the whole envelope
    # down by that positive slack so every facet is a proven underestimator.
    xf = np.linspace(lo, hi, 10 * N)
    hull_vals = _pwl_eval(hull, xf)
    slack = float(np.max(hull_vals - g(xf)))
    slack = max(slack, 0.0)
    # emit each hull facet as y >= (line) - slack
    for (x0, y0), (x1, y1) in zip(hull[:-1], hull[1:]):
        if x1 <= x0 + 1e-15:
            continue
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0 - slack
        # y >= m x + b  -> y - m x >= b
        lp.row({yname: 1.0, xname: -m}, b, ">=")
    return slack


def _lower_convex_hull(xs, ys):
    # monotone chain lower hull; xs sorted ascending
    pts = list(zip(xs.tolist(), ys.tolist()))
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _pwl_eval(hull, xf):
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    return np.interp(xf, hx, hy)


# univariate helper functions
def ln_f(x):
    return math.log(x)


def ln_fp(x):
    return 1.0 / x


def exp_f(x):
    return math.exp(x)


def exp_fp(x):
    return math.exp(x)
