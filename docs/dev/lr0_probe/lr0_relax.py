"""
LR-0 general sound relaxation of an .nl DAG into a linear program.

For each DAG node we introduce an LP variable and add rigorous convex/concave
envelope rows relating it to its children's LP variables, plus an interval
[lo,hi] enclosure. Every row is a PROVEN over/under estimator on the node box.

Monomial specialisation (H-LOG, the campaign's hypothesis): a product of nodes
each having a strictly positive lower bound is relaxed in log space
   z_i = ln(child_i)  (concave envelope),  s = sum z_i  (exact),
   t = exp(s)         (convex envelope).
Toggle with use_log_monomial=True/False to compare against recursive McCormick.

The probe only needs SOUNDNESS. Where a construction would be unsound on the box
(e.g. sqrt of a possibly-negative argument, division), we fall back to the loose
interval bound (a box constant) — always valid, possibly weak. That is allowed
by the plan (be conservative).
"""

from __future__ import annotations

import math

import numpy as np

from lr0_envelopes import LP, concave_env_rows, convex_env_rows, exp_f, exp_fp, ln_f, ln_fp
from nl_parse import Node


class Relaxer:
    def __init__(self, lp: LP, var_lo, var_hi, use_log_monomial=True, n_tan=4):
        self.lp = lp
        self.use_log = use_log_monomial
        self.n_tan = n_tan
        self._k = 0
        self._cache = {}
        self._icache = {}
        # base variables x0..x{n-1}
        self.nvars = len(var_lo)
        self._xlo = list(var_lo)
        self._xhi = list(var_hi)
        for i in range(self.nvars):
            lo_i = var_lo[i] if np.isfinite(var_lo[i]) else None
            hi_i = var_hi[i] if np.isfinite(var_hi[i]) else None
            self.lp.var_free(f"x{i}", lo_i, hi_i)

    def fresh(self, lo, hi, prefix="w"):
        self._k += 1
        name = f"{prefix}{self._k}"
        # keep genuinely-unbounded directions FREE (None) rather than a huge
        # finite sentinel — a huge finite bound wrecks LP scaling and is no
        # tighter than free. free bounds are always sound (never cut the box).
        blo = lo if np.isfinite(lo) else None
        bhi = hi if np.isfinite(hi) else None
        self.lp.var_free(name, blo, bhi)
        return name, lo, hi

    # ---- interval arithmetic to get node [lo,hi] (rigorous enclosure) --------
    def interval(self, node):
        key = id(node)
        c = self._icache.get(key)
        if c is not None:
            return c
        r = self._interval(node)
        self._icache[key] = r
        return r

    def _interval(self, node):
        k = node.kind
        if k == "var":
            return self._xlo[node.a], self._xhi[node.a]
        if k == "const":
            return node.a, node.a
        if k == "neg":
            lo, hi = self.interval(node.a); return -hi, -lo
        if k == "+":
            l1, h1 = self.interval(node.a); l2, h2 = self.interval(node.b)
            return l1 + l2, h1 + h2
        if k == "-":
            l1, h1 = self.interval(node.a); l2, h2 = self.interval(node.b)
            return l1 - h2, h1 - l2
        if k == "*":
            l1, h1 = self.interval(node.a); l2, h2 = self.interval(node.b)
            prods = [l1 * l2, l1 * h2, h1 * l2, h1 * h2]
            return min(prods), max(prods)
        if k == "/":
            l1, h1 = self.interval(node.a); l2, h2 = self.interval(node.b)
            if l2 <= 0 <= h2:
                return -1e12, 1e12
            cands = [l1 / l2, l1 / h2, h1 / l2, h1 / h2]
            return min(cands), max(cands)
        if k == "^":
            l1, h1 = self.interval(node.a)
            p = node.b.a if node.b.kind == "const" else None
            if p is None:
                return -1e12, 1e12
            return self._pow_interval(l1, h1, p)
        if k == "sqrt":
            l1, h1 = self.interval(node.a)
            l1 = max(l1, 0.0)
            return math.sqrt(max(l1, 0.0)), math.sqrt(max(h1, 0.0))
        if k == "ln":
            l1, h1 = self.interval(node.a)
            if l1 <= 0:
                return -1e12, math.log(h1) if h1 > 0 else -1e12
            return math.log(l1), math.log(h1)
        if k == "exp":
            l1, h1 = self.interval(node.a); return math.exp(l1), math.exp(h1)
        if k == "sum":
            lo = hi = 0.0
            for c in node.children:
                l, h = self.interval(c); lo += l; hi += h
            return lo, hi
        if k == "prod":
            lo, hi = 1.0, 1.0
            for c in node.children:
                l, h = self.interval(c)
                cands = [lo * l, lo * h, hi * l, hi * h]
                lo, hi = min(cands), max(cands)
            return lo, hi
        raise ValueError(k)

    def _pow_interval(self, l, h, p):
        if p == int(p) and int(p) >= 0:
            ip = int(p)
            if ip % 2 == 0:
                if l <= 0 <= h:
                    return 0.0, max(l**ip, h**ip)
                return min(l**ip, h**ip), max(l**ip, h**ip)
            return l**ip, h**ip
        # fractional power: need l>0
        if l <= 0:
            l = max(l, 1e-12)
        return min(l**p, h**p), max(l**p, h**p)

    # ---- relax: return (lp_var_name, lo, hi) for node -----------------------
    def relax(self, node):
        key = id(node)
        if key in self._cache:
            return self._cache[key]
        res = self._relax(node)
        self._cache[key] = res
        return res

    def _affine_of(self, node):
        """If node is affine in base vars, return (coeffs dict, const). Else None."""
        k = node.kind
        if k == "const":
            return {}, node.a
        if k == "var":
            return {f"x{node.a}": 1.0}, 0.0
        if k == "neg":
            r = self._affine_of(node.a)
            if r is None:
                return None
            c, b = r
            return {kk: -vv for kk, vv in c.items()}, -b
        if k in ("+", "-"):
            ra = self._affine_of(node.a); rb = self._affine_of(node.b)
            if ra is None or rb is None:
                return None
            ca, ba = ra; cb, bb = rb
            out = dict(ca); sgn = 1.0 if k == "+" else -1.0
            for kk, vv in cb.items():
                out[kk] = out.get(kk, 0.0) + sgn * vv
            return out, ba + sgn * bb
        if k == "*":
            ra = self._affine_of(node.a); rb = self._affine_of(node.b)
            if ra is None or rb is None:
                return None
            ca, ba = ra; cb, bb = rb
            # affine*affine is affine only if one side is constant
            if not ca:
                return {kk: ba * vv for kk, vv in cb.items()}, ba * bb
            if not cb:
                return {kk: bb * vv for kk, vv in ca.items()}, bb * ba
            return None
        if k == "sum":
            out = {}; const = 0.0
            for c in node.children:
                r = self._affine_of(c)
                if r is None:
                    return None
                cc, bb = r
                for kk, vv in cc.items():
                    out[kk] = out.get(kk, 0.0) + vv
                const += bb
            return out, const
        return None

    # ---- monomial factor extraction ---------------------------------------
    def _as_monomial(self, node):
        """Return list of (factor_node, exponent) if node is a product/power of
        positive-lb subexpressions, else None. Constants are folded into `coef`.
        Returns (coef, [(child,exp)...]) or None."""
        k = node.kind
        if k == "*":
            ra = self._as_monomial(node.a); rb = self._as_monomial(node.b)
            if ra is None or rb is None:
                return None
            ca, fa = ra; cb, fb = rb
            return ca * cb, fa + fb
        if k == "prod":
            coef = 1.0; facs = []
            for c in node.children:
                r = self._as_monomial(c)
                if r is None:
                    return None
                coef *= r[0]; facs += r[1]
            return coef, facs
        if k == "^" and node.b.kind == "const":
            r = self._as_monomial(node.a)
            if r is None:
                return None
            c, f = r
            p = node.b.a
            # (coef * prod)^p = coef^p * prod^p ; only clean if coef==1 or single
            if abs(c - 1.0) < 1e-15:
                return 1.0, [(fn, e * p) for fn, e in f]
            if len(f) == 0:
                return c**p, []
            return None
        if k == "const":
            return node.a, []
        if k == "neg":
            r = self._as_monomial(node.a)
            if r is None:
                return None
            return -r[0], r[1]
        # a bare positive-lb subexpression is a degree-1 factor
        lo, hi = self.interval(node)
        return 1.0, [(node, 1.0)]

    def _relax(self, node):
        lo, hi = self.interval(node)
        k = node.kind

        # affine -> exact, no new var needed (represent as linear combo)
        aff = self._affine_of(node)
        if aff is not None:
            coeffs, const = aff
            name, nlo, nhi = self.fresh(lo, hi)
            row = dict(coeffs); row[name] = -1.0
            self.lp.row(row, -const, "==")
            return name, lo, hi

        # try positive monomial (H-LOG) --------------------------------------
        if self.use_log:
            mono = self._as_monomial(node)
            if mono is not None:
                coef, facs = mono
                # collapse repeated factors, check strict positivity of each
                ok = True
                agg = {}
                for fn, e in facs:
                    flo, fhi = self.interval(fn)
                    if flo <= 1e-9:
                        ok = False; break
                    agg[id(fn)] = (fn, agg.get(id(fn), (fn, 0.0))[1] + e)
                # Only apply H-LOG to a genuine product/power (>=2 distinct
                # factors, OR a single factor with non-unit exponent). A bare
                # degree-1 factor equal to `node` must fall through to its own
                # structural handler (else infinite self-recursion).
                nontrivial = (len(agg) >= 2) or any(
                    abs(e - 1.0) > 1e-12 for _, e in agg.values()
                )
                is_self = (len(agg) == 1 and next(iter(agg.values()))[0] is node)
                if ok and nontrivial and not is_self and any(e != 0 for _, e in agg.values()):
                    return self._relax_log_monomial(node, coef, list(agg.values()), lo, hi)

        # generic handlers ----------------------------------------------------
        if k == "neg":
            cn, clo, chi = self.relax(node.a)
            name, nlo, nhi = self.fresh(-chi, -clo)
            self.lp.row({name: 1.0, cn: 1.0}, 0.0, "==")
            return name, -chi, -clo
        if k in ("+", "-", "sum"):
            # affine handled above; if here, children nonlinear -> lift each
            terms = []
            const = 0.0
            if k == "sum":
                kids = [(1.0, c) for c in node.children]
            else:
                kids = [(1.0, node.a), (1.0 if k == "+" else -1.0, node.b)]
            name, nlo, nhi = self.fresh(lo, hi)
            row = {name: -1.0}
            for sgn, c in kids:
                if c.kind == "const":
                    const += sgn * c.a; continue
                cn, clo, chi = self.relax(c)
                row[cn] = row.get(cn, 0.0) + sgn
            self.lp.row(row, -const, "==")
            return name, lo, hi
        if k == "*":
            return self._relax_bilinear(node, lo, hi)
        if k == "^":
            return self._relax_power(node, lo, hi)
        if k == "sqrt":
            return self._relax_sqrt(node, lo, hi)
        if k == "ln":
            return self._relax_ln(node, lo, hi)
        if k == "exp":
            return self._relax_exp(node, lo, hi)
        if k == "prod":
            # recursive McCormick product (fallback when not positive-monomial)
            cur = None
            for c in node.children:
                cn, clo, chi = self.relax(c)
                if cur is None:
                    cur, curlo, curhi = cn, clo, chi
                else:
                    cur, curlo, curhi = self._mccormick(cur, curlo, curhi, cn, clo, chi)
            return cur, curlo, curhi
        # unknown: loose interval box
        name, nlo, nhi = self.fresh(lo, hi)
        return name, lo, hi

    # ---- H-LOG monomial ----------------------------------------------------
    def _relax_log_monomial(self, node, coef, agg_facs, lo, hi):
        # facs: list of (child_node, exponent). all children have positive lb.
        s_lo = 0.0; s_hi = 0.0
        zsum = {}
        for fn, e in agg_facs:
            cn, clo, chi = self.relax(fn)     # lp var for the factor
            clo = max(clo, 1e-12)
            zname, zlo, zhi = self.fresh(math.log(clo), math.log(chi), prefix="z")
            # z = ln(child): concave envelope in terms of the child LP var
            concave_env_rows(self.lp, cn, zname, clo, chi, ln_f, ln_fp, self.n_tan)
            zsum[zname] = zsum.get(zname, 0.0) + e
            s_lo += e * (math.log(clo) if e > 0 else math.log(chi))
            s_hi += e * (math.log(chi) if e > 0 else math.log(clo))
        sname, slo, shi = self.fresh(s_lo, s_hi, prefix="s")
        row = {kk: -vv for kk, vv in zsum.items()}; row[sname] = 1.0
        self.lp.row(row, 0.0, "==")
        # t = exp(s): convex envelope
        tname, tlo, thi = self.fresh(math.exp(s_lo), math.exp(s_hi), prefix="t")
        convex_env_rows(self.lp, sname, tname, s_lo, s_hi, exp_f, exp_fp, self.n_tan)
        # result = coef * t
        rname, rlo, rhi = self.fresh(*sorted([coef * math.exp(s_lo), coef * math.exp(s_hi)]))
        self.lp.row({rname: 1.0, tname: -coef}, 0.0, "==")
        return rname, rlo, rhi

    # ---- McCormick bilinear -------------------------------------------------
    def _relax_bilinear(self, node, lo, hi):
        an, alo, ahi = self.relax(node.a)
        bn, blo, bhi = self.relax(node.b)
        return self._mccormick(an, alo, ahi, bn, blo, bhi)

    def _mccormick(self, an, alo, ahi, bn, blo, bhi):
        prods = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
        wlo, whi = min(prods), max(prods)
        wn, _, _ = self.fresh(wlo, whi)
        # underestimators: w >= alo*b + blo*a - alo*blo ; w >= ahi*b + bhi*a - ahi*bhi
        self.lp.row({wn: 1, bn: -alo, an: -blo}, -alo * blo, ">=")
        self.lp.row({wn: 1, bn: -ahi, an: -bhi}, -ahi * bhi, ">=")
        # overestimators: w <= ahi*b + blo*a - ahi*blo ; w <= alo*b + bhi*a - alo*bhi
        self.lp.row({wn: 1, bn: -ahi, an: -blo}, -ahi * blo, "<=")
        self.lp.row({wn: 1, bn: -alo, an: -bhi}, -alo * bhi, "<=")
        return wn, wlo, whi

    # ---- power (integer square / cube / fractional) ------------------------
    def _relax_power(self, node, lo, hi):
        cn, clo, chi = self.relax(node.a)
        p = node.b.a
        if p == 2:
            wn, _, _ = self.fresh(max(0.0, lo), hi)
            # convex: tangents underest, secant overest
            convex_env_rows(self.lp, cn, wn, clo, chi, lambda t: t * t, lambda t: 2 * t, self.n_tan)
            return wn, lo, hi
        if p == 3 and clo >= 0:
            wn, _, _ = self.fresh(clo**3, chi**3)
            convex_env_rows(self.lp, cn, wn, clo, chi, lambda t: t**3, lambda t: 3 * t * t, self.n_tan)
            return wn, clo**3, chi**3
        if clo > 0:
            f = lambda t: t**p
            fp = lambda t: p * t ** (p - 1)
            wn, _, _ = self.fresh(min(clo**p, chi**p), max(clo**p, chi**p))
            # convex if p>1 or p<0; concave if 0<p<1
            if p > 1 or p < 0:
                convex_env_rows(self.lp, cn, wn, clo, chi, f, fp, self.n_tan)
            else:
                concave_env_rows(self.lp, cn, wn, clo, chi, f, fp, self.n_tan)
            return wn, min(clo**p, chi**p), max(clo**p, chi**p)
        # loose box
        name, _, _ = self.fresh(lo, hi)
        return name, lo, hi

    def _relax_sqrt(self, node, lo, hi):
        cn, clo, chi = self.relax(node.a)
        clo = max(clo, 0.0)
        if chi <= clo:
            chi = clo + 1e-9
        # sqrt concave: tangents overest, secant underest
        wn, _, _ = self.fresh(math.sqrt(clo), math.sqrt(chi))
        concave_env_rows(self.lp, cn, wn, clo, chi,
                         lambda t: math.sqrt(max(t, 0)),
                         lambda t: 0.5 / math.sqrt(max(t, 1e-12)), self.n_tan)
        return wn, math.sqrt(clo), math.sqrt(chi)

    def _relax_ln(self, node, lo, hi):
        cn, clo, chi = self.relax(node.a)
        if clo <= 0:
            name, _, _ = self.fresh(lo, hi); return name, lo, hi
        wn, _, _ = self.fresh(math.log(clo), math.log(chi))
        concave_env_rows(self.lp, cn, wn, clo, chi, ln_f, ln_fp, self.n_tan)
        return wn, math.log(clo), math.log(chi)

    def _relax_exp(self, node, lo, hi):
        cn, clo, chi = self.relax(node.a)
        wn, _, _ = self.fresh(math.exp(clo), math.exp(chi))
        convex_env_rows(self.lp, cn, wn, clo, chi, exp_f, exp_fp, self.n_tan)
        return wn, math.exp(clo), math.exp(chi)
