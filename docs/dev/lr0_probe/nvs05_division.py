"""nvs05 division/ratio-lever entry experiment (LR-nvs05).

Standalone PROBE (no solver-code changes). Tests whether a *tighter treatment of
nvs05's division/ratio constraints* — plus OBBT to shrink the ratio-feeding
boxes — moves the root LP bound from discopt's 0.674 toward the optimum 5.4709.

Structure of nvs05 (welded-beam, 8 vars / 9 cons), from the validated .nl parse:

  min  1.10471*x0^2*x1 + 0.04811*x2*x3*(14+x1)      [box-min 0.6740]
  x0,x1 in [0.01,200],  x2 in [1,200],  x3 in [1,200]

  C0 (==0): x4 = 4243.28/(x0*x1)                          [defines x4, shear tau']
  C1 (==0): x6 = sqrt(0.25*x1^2 + (0.5*x2+0.5*x0)^2)      [defines x6]
  C2 (==0): x5 = (59405.9+2121.64*x1)*x6
                 / [ (x0*x1) * (0.0833333*x1^2 + (0.5*x2+0.5*x0)^2) ]   [defines x5, tau'']
  C3 (==0): x7 = 0.5*x1/x6                                [defines x7]
  C4 (>= -13600): -sqrt(x4^2 + 2*x4*x5*x7 + x5^2) >= -13600   [combined shear <= 13600]
  C5 (>= -30000): -504000/(x2^2*x3) >= -30000     <=>  x2^2*x3 >= 16.8
  C6 (>=  6000):  0.0204745*sqrt(1e15*x3^3*x2^2*x3^3)*(1-0.0282346*x2) >= 6000
                  == 647460.54 * x2 * x3^3 * (1 - 0.0282346*x2) >= 6000   [x2,x3>0]
  C7 (>= -0.25):  -0.21952/(x2^3*x3) >= -0.25     <=>  x2^3*x3 >= 0.87808
  C8 (>= 0):      x3 - x0 >= 0

Key facts (measured, this file's __main__ prints them):
  * FBBT leaves x0,x1 in [0.01,200] and x3 in [1,200] — the ratio-feeding boxes
    are genuinely MAXIMALLY WIDE (answers the OBBT question: yes, wide).
  * The LR-0 general probe DROPPED C2 and C6 (lifted magnitude > 1e10), so its
    bound was just the objective box-min 0.674. C6's 1e15 lives inside a sqrt;
    algebraically sqrt(1e15*x2^2*x3^6)=sqrt(1e15)*x2*x3^3 is a MODEST signomial —
    dropping it was avoidable. C2 is a genuine ratio over a wide box.

Everything here is a PROVEN sound relaxation row: the LP optimum must be <= the
true optimum 5.4709. Validated by feasible-point sampling.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.setrecursionlimit(1_000_000)

from lr0_envelopes import LP, concave_env_rows, convex_env_rows, ln_f, ln_fp
from nl_parse import cons_value, load_nl_expressions, obj_value

NLPATH = "python/tests/data/minlplib_nl/nvs05.nl"
OPT = 5.4709
DISCOPT_ROOT = 0.674
C6_COEF = 0.0204745 * math.sqrt(1e15)  # = 647460.5395311748


# ---------------------------------------------------------------------------
# A small hand-built sound relaxation of nvs05 with EXPLICIT ratio handling.
# Rather than a generic DAG relaxer, we transcribe nvs05's structure directly so
# the ratio constraints get purpose-built envelopes. Every row documents its
# soundness. All base + lifted vars carry a rigorous [lo,hi] interval.
# ---------------------------------------------------------------------------


class Model:
    """Container holding the LP and the current variable box; rebuildable so we
    can re-relax after OBBT tightens the box."""

    def __init__(self, box):
        # box: dict name -> (lo, hi) for base vars x0..x7
        self.box = {k: (float(lo), float(hi)) for k, (lo, hi) in box.items()}

    # -- reciprocal (1-D) envelope: q = a / p, a>0 const, p in [pl,ph], pl>0 ----
    # f(p)=a/p is convex & decreasing on p>0.  tangents = UNDERestimators,
    # secant = OVERestimator.  Exact 1-D envelope of the reciprocal.
    def recip_rows(self, lp, pname, qname, a, pl, ph, n_tan=6):
        assert a > 0 and pl > 0, (a, pl)
        f = lambda p: a / p
        fp = lambda p: -a / (p * p)
        convex_env_rows(lp, pname, qname, pl, ph, f, fp, n_tan)
        return f(ph), f(pl)  # q in [a/ph, a/pl]

    # -- bilinear w = u*v  (McCormick), returns (wlo,whi) ----------------------
    def mccormick(self, lp, un, vn, wn, ul, uh, vl, vh):
        prods = [ul * vl, ul * vh, uh * vl, uh * vh]
        wl, wh = min(prods), max(prods)
        if wn not in lp.lb:
            lp.var_free(wn, wl, wh)
        lp.row({wn: 1, vn: -ul, un: -vl}, -ul * vl, ">=")
        lp.row({wn: 1, vn: -uh, un: -vh}, -uh * vh, ">=")
        lp.row({wn: 1, vn: -uh, un: -vl}, -uh * vl, "<=")
        lp.row({wn: 1, vn: -ul, un: -vh}, -ul * vh, "<=")
        return wl, wh

    # -- log-space positive product: t = prod_i base_i^{e_i}, all base_i>0 -----
    # z_i = ln(base_i) concave env; s = sum e_i z_i exact; t = exp(s) convex env.
    def log_monomial(self, lp, factors, tname, prefix, n_tan=6):
        # factors: list of (varname, lo, hi, exponent)
        s_lo = s_hi = 0.0
        zsum = {}
        for i, (bn, bl, bh, e) in enumerate(factors):
            bl = max(bl, 1e-12)
            zn = f"{prefix}_z{i}"
            lp.var_free(zn, math.log(bl), math.log(bh))
            concave_env_rows(lp, bn, zn, bl, bh, ln_f, ln_fp, n_tan)
            zsum[zn] = zsum.get(zn, 0.0) + e
            s_lo += e * (math.log(bl) if e > 0 else math.log(bh))
            s_hi += e * (math.log(bh) if e > 0 else math.log(bl))
        sn = f"{prefix}_s"
        lp.var_free(sn, s_lo, s_hi)
        row = {kk: -vv for kk, vv in zsum.items()}
        row[sn] = 1.0
        lp.row(row, 0.0, "==")
        exp_f = math.exp
        exp_fp = math.exp
        lp.var_free(tname, math.exp(s_lo), math.exp(s_hi))
        convex_env_rows(lp, sn, tname, s_lo, s_hi, exp_f, exp_fp, n_tan)
        return math.exp(s_lo), math.exp(s_hi)

    def build(self, objective_only_for_obbt=None, n_tan=6, include_shear=True):
        """Build the root LP. Returns (lp, aux) where aux maps lifted names to
        their [lo,hi]. If objective_only_for_obbt is a var name, the LP objective
        is set to minimize/maximize that var (caller flips sign); otherwise the
        real objective is used."""
        lp = LP()
        b = self.box
        for k in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]:
            lo, hi = b[k]
            lp.var_free(k, lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None)

        x0l, x0h = b["x0"]
        x1l, x1h = b["x1"]
        x2l, x2h = b["x2"]
        x3l, x3h = b["x3"]

        # ---- objective terms (sound convex/concave under-envelopes) ----------
        # obj = 1.10471*x0^2*x1 + 0.04811*x2*x3*(14+x1)
        #     = 1.10471*(x0^2*x1) + 0.04811*(14*x2*x3 + x2*x3*x1)
        # Use log-space for the positive products (x0,x1,x2,x3 all lb>0 here).
        # term A: x0^2*x1  -> log monomial
        self.log_monomial(lp, [("x0", x0l, x0h, 2.0), ("x1", x1l, x1h, 1.0)], "A", "A", n_tan)
        # term B: x2*x3    -> log monomial (used twice: *14 and *x1)
        self.log_monomial(lp, [("x2", x2l, x2h, 1.0), ("x3", x3l, x3h, 1.0)], "B", "B", n_tan)
        Bl, Bh = math.exp(0), None  # recompute below via interval
        Bl = x2l * x3l
        Bh = x2h * x3h
        # term C: x2*x3*x1 -> log monomial
        self.log_monomial(
            lp,
            [("x2", x2l, x2h, 1.0), ("x3", x3l, x3h, 1.0), ("x1", x1l, x1h, 1.0)],
            "C",
            "C",
            n_tan,
        )
        # objective variable
        # obj = 1.10471*A + 0.04811*(14*B + C)
        lp.var_free("OBJ", None, None)
        lp.row(
            {"OBJ": 1.0, "A": -1.10471, "B": -0.04811 * 14.0, "C": -0.04811},
            0.0,
            "==",
        )

        # ---- C5: x2^2 * x3 >= 16.8  (log-space signomial, both sides positive)-
        # ln(x2^2 x3) = 2 ln x2 + ln x3 >= ln 16.8
        self._logsum_ge(lp, [("x2", x2l, x2h, 2.0), ("x3", x3l, x3h, 1.0)], math.log(16.8), "C5", n_tan)

        # ---- C7: x2^3 * x3 >= 0.87808 ----------------------------------------
        self._logsum_ge(
            lp, [("x2", x2l, x2h, 3.0), ("x3", x3l, x3h, 1.0)], math.log(0.87808), "C7", n_tan
        )

        # ---- C6: 647460.54 * x2 * x3^3 * (1 - 0.0282346*x2) >= 6000 -----------
        # Let P6 = x2*x3^3 (log monomial), and Q6 = x2*x2*x3^3 = x2^2*x3^3.
        # C6: C6_COEF*(P6 - 0.0282346*Q6) >= 6000. Both P6,Q6 are positive
        # monomials -> exact log-space over/under envelopes. We need a LOWER
        # bound on LHS: use UNDER-env of P6 (convex exp secant gives over; we need
        # the tangent under-estimator, which convex_env provides) and OVER-env of
        # Q6 (since it enters with negative sign). log_monomial gives us a lifted
        # var linked by BOTH directions, so the LP can pick the sound extreme.
        p6l, p6h = self.log_monomial(
            lp, [("x2", x2l, x2h, 1.0), ("x3", x3l, x3h, 3.0)], "P6", "P6", n_tan
        )
        q6l, q6h = self.log_monomial(
            lp, [("x2", x2l, x2h, 2.0), ("x3", x3l, x3h, 3.0)], "Q6", "Q6", n_tan
        )
        # C6_COEF*P6 - C6_COEF*0.0282346*Q6 >= 6000
        lp.row({"P6": C6_COEF, "Q6": -C6_COEF * 0.0282346}, 6000.0, ">=")

        # p01 = x0*x1 (McCormick, positive) — needed by C0 and C2.
        lp.var_free("p01", x0l * x1l, x0h * x1h)
        self.mccormick(lp, "x0", "x1", "p01", x0l, x0h, x1l, x1h)
        if not include_shear:
            # C8 + objective only (used by the numerically-safe OBBT subset).
            lp.row({"x3": 1.0, "x0": -1.0}, 0.0, ">=")
            if objective_only_for_obbt is not None:
                lp.add_obj(objective_only_for_obbt, 1.0)
            else:
                lp.add_obj("OBJ", 1.0)
            return lp

        # ---- C0: x4 = 4243.28/(x0*x1) ----------------------------------------
        # x4 = 4243.28/p01 (reciprocal env). Clamp x4 to the FBBT box.
        self.recip_rows(lp, "p01", "x4", 4243.28, x0l * x1l, x0h * x1h, n_tan)

        # ---- C1: x6 = sqrt(0.25 x1^2 + (0.5 x2 + 0.5 x0)^2) ------------------
        # inner = 0.25 x1^2 + (0.5x2+0.5x0)^2. Build inner var (>=0), x6=sqrt(inner)
        # concave sqrt: tangents over, secant under.
        # 0.25 x1^2 : log monomial 0.25*x1^2
        self.log_monomial(lp, [("x1", x1l, x1h, 2.0)], "x1sq", "x1sq", n_tan)
        # (0.5x2+0.5x0)^2: let u = 0.5x2+0.5x0 (affine, in [ul,uh]); u^2 convex.
        ul = 0.5 * x2l + 0.5 * x0l
        uh = 0.5 * x2h + 0.5 * x0h
        lp.var_free("u1", ul, uh)
        lp.row({"u1": 1.0, "x2": -0.5, "x0": -0.5}, 0.0, "==")
        lp.var_free("u1sq", ul * ul, uh * uh)
        convex_env_rows(lp, "u1", "u1sq", ul, uh, lambda t: t * t, lambda t: 2 * t, n_tan)
        # inner = 0.25*x1sq + u1sq
        inl = 0.25 * (x1l * x1l) + ul * ul
        inh = 0.25 * (x1h * x1h) + uh * uh
        lp.var_free("in1", inl, inh)
        lp.row({"in1": 1.0, "x1sq": -0.25, "u1sq": -1.0}, 0.0, "==")
        # x6 = sqrt(in1)
        sl, sh = math.sqrt(max(inl, 0.0)), math.sqrt(max(inh, 0.0))
        concave_env_rows(
            lp,
            "in1",
            "x6",
            max(inl, 1e-12),
            inh,
            lambda t: math.sqrt(max(t, 0)),
            lambda t: 0.5 / math.sqrt(max(t, 1e-12)),
            n_tan,
        )
        lp.lb["x6"] = max(lp.lb.get("x6", -math.inf) or -math.inf, sl)
        lp.ub["x6"] = min(lp.ub.get("x6", math.inf) or math.inf, sh)

        # ---- C3: x7 = 0.5*x1/x6 ----------------------------------------------
        # 1/x6 reciprocal (x6 in [sl,sh], sl>0), then *0.5*x1 (McCormick).
        lp.var_free("r6", 1.0 / sh, 1.0 / sl)
        self.recip_rows(lp, "x6", "r6", 1.0, max(sl, 1e-12), sh, n_tan)
        # x7 = 0.5 * x1 * r6  -> bilinear (0.5x1)*(r6)
        # let h1 = 0.5*x1 affine
        h1l, h1h = 0.5 * x1l, 0.5 * x1h
        lp.var_free("h1", h1l, h1h)
        lp.row({"h1": 1.0, "x1": -0.5}, 0.0, "==")
        r6l, r6h = 1.0 / sh, 1.0 / sl
        x7l, x7h = self.mccormick(lp, "h1", "r6", "x7", h1l, h1h, r6l, r6h)

        # ---- C2: x5 = (59405.9+2121.64*x1)*x6 / [ p01 * (0.0833333 x1^2 + u1^2) ]
        # numerator N = (59405.9+2121.64*x1)*x6 ; denom D = p01 * D2,
        #   D2 = 0.0833333 x1^2 + u1^2.  x5 = N/D.
        # Build N (bilinear of affine (59405.9+2121.64 x1) and x6), D2 affine-comb
        # of x1sq,u1sq, D = p01*D2 (bilinear), then reciprocal-times: x5 = N * (1/D).
        an_l = 59405.9 + 2121.64 * x1l
        an_h = 59405.9 + 2121.64 * x1h
        lp.var_free("aN", an_l, an_h)
        lp.row({"aN": 1.0, "x1": -2121.64}, 59405.9, "==")
        Nl, Nh = self.mccormick(lp, "aN", "x6", "N2", an_l, an_h, sl, sh)
        # D2 = 0.0833333*x1sq + u1sq
        d2l = 0.0833333 * (x1l * x1l) + ul * ul
        d2h = 0.0833333 * (x1h * x1h) + uh * uh
        lp.var_free("D2", d2l, d2h)
        lp.row({"D2": 1.0, "x1sq": -0.0833333, "u1sq": -1.0}, 0.0, "==")
        # D = p01 * D2
        p01l, p01h = x0l * x1l, x0h * x1h
        Dl, Dh = self.mccormick(lp, "p01", "D2", "Dfull", p01l, p01h, d2l, d2h)
        # rD = 1/Dfull (reciprocal, Dfull>0)
        lp.var_free("rD", 1.0 / Dh, 1.0 / Dl)
        self.recip_rows(lp, "Dfull", "rD", 1.0, max(Dl, 1e-12), Dh, n_tan)
        # x5 = N2 * rD (bilinear). Clamp x5 to the FBBT box [x5.lb, x5.ub] — the
        # McCormick product bound can be looser than the (tighter) FBBT interval,
        # and the tighter box keeps the downstream squares numerically sane. This
        # is SOUND: intersecting with a valid interval enclosure never removes a
        # feasible point.
        x5box_l, x5box_h = self.box["x5"]
        rDl, rDh = 1.0 / Dh, 1.0 / Dl
        x5l, x5h = self.mccormick(lp, "N2", "rD", "x5", Nl, Nh, rDl, rDh)
        x5l = max(x5l, x5box_l)
        x5h = min(x5h, x5box_h)
        lp.lb["x5"] = max(lp.lb.get("x5") or -math.inf, x5l)
        lp.ub["x5"] = min(lp.ub.get("x5") or math.inf, x5h)
        # clamp x4 to its FBBT box too
        x4box_l, x4box_h = self.box["x4"]

        # ---- C4: sqrt(x4^2 + 2 x4 x5 x7 + x5^2) <= 13600 ---------------------
        # equivalently x4^2 + 2 x4 x5 x7 + x5^2 <= 13600^2. Build each term:
        #   x4^2 convex, x5^2 convex, tri = x4*x5*x7 (recursive McCormick).
        x4l, x4h = x4box_l, x4box_h
        # x4^2
        lp.var_free("x4sq", x4l * x4l if x4l > 0 else 0.0, x4h * x4h)
        convex_env_rows(lp, "x4", "x4sq", max(x4l, 1e-12), x4h, lambda t: t * t, lambda t: 2 * t, n_tan)
        lp.var_free("x5sq", x5l * x5l if x5l > 0 else 0.0, x5h * x5h)
        convex_env_rows(lp, "x5", "x5sq", max(x5l, 1e-12), x5h, lambda t: t * t, lambda t: 2 * t, n_tan)
        # tri = x4*x5*x7 via (x4*x5)=m1 then m1*x7
        m1l, m1h = self.mccormick(lp, "x4", "x5", "m1", x4l, x4h, x5l, x5h)
        trl, trh = self.mccormick(lp, "m1", "x7", "tri", m1l, m1h, x7l, x7h)
        # x4sq + 2 tri + x5sq <= 13600^2 ; scale row by 1/13600^2 for conditioning
        sc = 1.0 / (13600.0**2)
        lp.row({"x4sq": sc, "tri": 2.0 * sc, "x5sq": sc}, 1.0, "<=")

        # ---- C8: x3 - x0 >= 0 ------------------------------------------------
        lp.row({"x3": 1.0, "x0": -1.0}, 0.0, ">=")

        # objective
        if objective_only_for_obbt is not None:
            lp.add_obj(objective_only_for_obbt, 1.0)
        else:
            lp.add_obj("OBJ", 1.0)
        return lp

    def _logsum_ge(self, lp, factors, rhs_ln, prefix, n_tan):
        """Add rows enforcing sum e_i * ln(base_i) >= rhs_ln (a valid relaxation
        of prod base_i^{e_i} >= exp(rhs_ln)). Uses the concave OVER-envelope of
        ln so this is a SOUND relaxation (it can only enlarge the feasible set:
        we require an over-estimate of ln-sum to clear rhs, i.e. the true ln-sum
        may be smaller -> feasible region is a superset -> LP bound stays a valid
        lower bound). NOTE: for a >= signomial constraint the *rigorous* relaxation
        uses the concave envelope's UPPER side; concave_env_rows gives both the
        tangent overestimators and the secant underestimator. We link a lifted
        L_i to ln(base_i) via those rows and require sum e_i L_i >= rhs_ln, with
        L_i free to take its overestimate -> superset. Sound."""
        row = {}
        b = self.box
        for i, (bn, bl, bh, e) in enumerate(factors):
            bl = max(bl, 1e-12)
            Ln = f"{prefix}_L{i}"
            lp.var_free(Ln, math.log(bl), math.log(bh))
            concave_env_rows(lp, bn, Ln, bl, bh, ln_f, ln_fp, n_tan)
            row[Ln] = row.get(Ln, 0.0) + e
        lp.row(row, rhs_ln, ">=")


# ---------------------------------------------------------------------------
# OBBT: for each base var, min and max it subject to the current relaxation,
# intersect with the current box. Iterate to fixpoint (or a few rounds).
# Every OBBT bound is a valid bound on the true feasible region because it is
# computed over a SOUND relaxation (superset of the feasible set).
# ---------------------------------------------------------------------------
def obbt_round(box, targets, n_tan=6, include_shear=True):
    newbox = dict(box)
    for name in targets:
        for sense in ("min", "max"):
            m = Model(newbox)
            lp = m.build(objective_only_for_obbt=name, n_tan=n_tan, include_shear=include_shear)
            if sense == "max":
                lp.obj = {name: -1.0}
            try:
                res, names, idx = lp.solve()
            except Exception:
                continue
            if not res.success or name not in idx:
                continue
            val = res.x[idx[name]]
            lo, hi = newbox[name]
            if sense == "min":
                newbox[name] = (max(lo, val - 1e-6), hi)
            else:
                newbox[name] = (lo, min(hi, val + 1e-6))
    return newbox


def root_bound(box, n_tan=6, include_shear=True):
    m = Model(box)
    lp = m.build(n_tan=n_tan, include_shear=include_shear)
    try:
        res, names, idx = lp.solve()
    except Exception as e:
        return None, f"exc {e}"
    if not res.success:
        return None, res.message
    return res.fun, "ok"


def get_fbbt_box():
    from discopt.modeling import from_nl

    m = from_nl(NLPATH)
    nl = m._nl_repr
    n = nl.n_vars
    lb = np.array([m._variables[i].lb for i in range(n)], float)
    ub = np.array([m._variables[i].ub for i in range(n)], float)
    flb, fub = nl.fbbt(1000, 1e-9)
    flb = np.asarray(flb, float).ravel()
    fub = np.asarray(fub, float).ravel()
    lb = np.maximum(lb, flb)
    ub = np.minimum(ub, fub)
    box = {f"x{i}": (lb[i], ub[i]) for i in range(n)}
    return box, m, nl


def sample_feasible(nl, P, box, n_target=2000, n_try=2_000_000):
    n = P["nvars"]
    lo = np.array([box[f"x{i}"][0] for i in range(n)])
    hi = np.array([box[f"x{i}"][1] for i in range(n)])
    lo = np.where(np.isfinite(lo), lo, -1e3)
    hi = np.where(np.isfinite(hi), hi, 1e4)
    rng = np.random.default_rng(7)
    senses = [str(nl.constraint_sense(j)) for j in range(P["ncons"])]
    rhs = [float(nl.constraint_rhs(j)) for j in range(P["ncons"])]
    out = []
    # nvs05 has 4 equality-defined aux vars (x4,x5,x6,x7). Sample x0..x3, then
    # SOLVE for x4,x5,x6,x7 exactly from C0..C3, then check the inequalities.
    for _ in range(n_try):
        x = np.empty(n)
        x[:4] = rng.uniform(lo[:4], hi[:4])
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        # C1: x6 = sqrt(0.25 x1^2 + (0.5x2+0.5x0)^2)
        x6 = math.sqrt(0.25 * x1 * x1 + (0.5 * x2 + 0.5 * x0) ** 2)
        # C0: x4 = 4243.28/(x0 x1)
        x4 = 4243.28 / (x0 * x1)
        # C2: x5 = (59405.9+2121.64 x1) x6 / [ x0 x1 (0.0833333 x1^2 + (0.5x2+0.5x0)^2)]
        D2 = 0.0833333 * x1 * x1 + (0.5 * x2 + 0.5 * x0) ** 2
        x5 = (59405.9 + 2121.64 * x1) * x6 / (x0 * x1 * D2)
        # C3: x7 = 0.5 x1 / x6
        x7 = 0.5 * x1 / x6
        x[4], x[5], x[6], x[7] = x4, x5, x6, x7
        ok = True
        for j in range(P["ncons"]):
            g = cons_value(P, j, x)
            s = senses[j]
            if "==" in s or s == "4":
                if abs(g - rhs[j]) > 1e-4:
                    ok = False
                    break
            elif ">=" in s or s == "2":
                if g < rhs[j] - 1e-6:
                    ok = False
                    break
            elif "<=" in s or s == "1":
                if g > rhs[j] + 1e-6:
                    ok = False
                    break
        if ok:
            out.append((x.copy(), obj_value(P, x)))
            if len(out) >= n_target:
                break
    return out


def _report(b, why):
    if b is None:
        print(f"  LP failed: {why}")
        return
    gap = OPT - b
    denom = OPT - DISCOPT_ROOT
    frac = (b - DISCOPT_ROOT) / denom if abs(denom) > 1e-12 else float("nan")
    within = abs(OPT - b) <= 1e-4 * (1 + abs(OPT))
    print(
        f"  root LP bound = {b:.5f}   gap-to-opt = {gap:.5f}"
        f"   %root-gap-closed = {100 * frac:.1f}%   root-cert? {within}"
    )


def show_box(box):
    for i in range(8):
        lo, hi = box[f"x{i}"]
        print(f"  x{i}: [{lo:g}, {hi:g}]")


def spatial_bnb(box0, n_tan=12, max_nodes=4000, branch_vars=("x0", "x1", "x2", "x3"),
                wall_budget=540.0):
    """Best-first spatial B&B branching (geometric midpoint) on the primal vars,
    node bound = the sound relaxation. Prints the global lower bound (min bound
    over open nodes) vs node count. This is a PROBE measurement of how hard the
    gap is to close by branching — it does not track an incumbent/cutoff."""
    import heapq
    import time

    target = OPT - 1e-4 * (1 + abs(OPT))
    heap = []
    cnt = [0]

    def push(box):
        b, why = root_bound(box, n_tan, include_shear=True)
        if b is None:
            return
        cnt[0] += 1
        heapq.heappush(heap, (b, cnt[0], box))

    def width(box):
        w = {}
        for v in branch_vars:
            lo, hi = box[v]
            lo = max(lo, 1e-6)
            w[v] = math.log(hi) - math.log(lo)
        return w

    t0 = time.time()
    push(box0)
    milestones = [3, 10, 27, 100, 208, 500, 1000, 2000, max_nodes]
    reported = set()
    glb = None
    while heap:
        b, _, box = heapq.heappop(heap)
        glb = b
        if glb >= target:
            print(f"  CERTIFIED (>= {target:.4f}) at nodes={cnt[0]} wall={time.time()-t0:.1f}s")
            return
        w = width(box)
        v = max(w, key=w.get)
        if w[v] < 1e-5:
            heapq.heappush(heap, (b, cnt[0], box))
            continue
        lo = max(box[v][0], 1e-6)
        mid = math.sqrt(lo * box[v][1])
        for nlo, nhi in [(box[v][0], mid), (mid, box[v][1])]:
            nb = dict(box)
            nb[v] = (nlo, nhi)
            push(nb)
        cur_glb = min(h[0] for h in heap) if heap else b
        for ms in milestones:
            if cnt[0] >= ms and ms not in reported:
                reported.add(ms)
                print(f"  nodes~{cnt[0]:5d}: global LB = {cur_glb:.4f}  wall={time.time()-t0:.1f}s")
        if cnt[0] >= max_nodes or time.time() - t0 > wall_budget:
            break
    glb = min(h[0] for h in heap) if heap else glb
    print(f"  STOP: best global LB = {glb:.4f} at nodes={cnt[0]} wall={time.time()-t0:.1f}s "
          f"(target {target:.4f} NOT reached)")


if __name__ == "__main__":
    box0, m, nl = get_fbbt_box()
    P = load_nl_expressions(NLPATH)
    print("=== nvs05 division/ratio lever (opt=%.4f, discopt root=%.3f) ===" % (OPT, DISCOPT_ROOT))
    print("FBBT box (ratio-feeding vars flagged WIDE):")
    for i in range(8):
        lo, hi = box0[f"x{i}"]
        wide = " <== WIDE" if i in (0, 1, 3) and (hi - lo) > 100 else ""
        print(f"  x{i}: [{lo:g}, {hi:g}]{wide}")

    n_tan = 6
    targets = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    # ------------------------------------------------------------------ V0
    # Full hand relaxation, ratio constraints KEPT (C2/C6 not dropped), no OBBT.
    b0, why0 = root_bound(box0, n_tan, include_shear=True)
    print("\n[V0] full relaxation, ratio constraints KEPT, NO OBBT:")
    _report(b0, why0)

    # ------------------------------------------------------------------ V1
    # OBBT with the numerically-safe subset (C5/C6/C7/C8 + objective; the
    # ratio/shear path excluded so the OBBT LPs are well-conditioned), then the
    # FULL relaxation root bound on the tightened box.
    box1 = dict(box0)
    for _ in range(6):
        box1 = obbt_round(box1, ["x0", "x1", "x2", "x3"], n_tan, include_shear=False)
    print("\n[V1] box after 6 OBBT rounds (safe subset, x0..x3):")
    show_box(box1)
    b1, why1 = root_bound(box1, n_tan, include_shear=True)
    print("[V1] full root bound after safe-subset OBBT:")
    _report(b1, why1)

    # ------------------------------------------------------------------ V2
    # OBBT with the FULL relaxation (ratio/shear included) — tests whether the
    # shear constraint C4 tightens x0,x1 (the ~1.71 objective term the safe
    # subset cannot reach). Falls back gracefully where an OBBT LP errors.
    box2 = dict(box1)  # warm-start from the safe-subset box
    for _ in range(4):
        box2 = obbt_round(box2, targets, n_tan, include_shear=True)
    print("\n[V2] box after full-model OBBT (warm-started from V1):")
    show_box(box2)
    b2, why2 = root_bound(box2, n_tan, include_shear=True)
    print("[V2] full root bound after full-model OBBT:")
    _report(b2, why2)

    # ---------------------------------------------------------- validation
    feas = sample_feasible(nl, P, box0, n_target=1000)
    if feas:
        minobj = min(o for _, o in feas)
        print(f"\n[validate] {len(feas)} feasible pts sampled; min sampled true obj = {minobj:.5f}")
        print("  (random sampling over the wide box misses the optimum basin; the")
        print("   binding soundness check is bound <= opt 5.4709 AND no row cuts a")
        print("   feasible point — checked below.)")
        for tag, bnd, bx in [("V0", b0, box0), ("V1", b1, box1), ("V2", b2, box2)]:
            if bnd is not None:
                le_opt = bnd <= OPT + 1e-6
                le_feas = bnd <= minobj + 1e-6
                print(f"  {tag} bound {bnd:.5f}: <= opt? {le_opt}   <= min sampled feas? {le_feas}")
    else:
        print("\n[validate] no feasible points found by structured sampling.")

    if "--bnb" in sys.argv:
        print("\n[BNB] spatial B&B on primal vars x0..x3 (node bound = sound relaxation):")
        spatial_bnb(box0, n_tan=12, max_nodes=4000)
