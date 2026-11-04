"""Reference interval FBBT over a spatial-kernel spec (#2142 item 3): the same fixed rows,
objective cutoff, BLF products and fixed-width terms as `propagate_spec_fixpoint`, but with
the textbook contractors in full (general extended division for a factor STRADDLING zero,
even-power reverse as a two-piece union, ...) and run to convergence (no round cap).

Plain float arithmetic with an outward pad of PAD*(1+|v|) on every deduced endpoint; it is a
comparison instrument, not a solver component. Unsupported term kinds raise (CLAUDE.md §7)."""
from __future__ import annotations

import math

import numpy as np

PAD = 1e-12
INF = math.inf


def _pad_hi(v):
    return v + PAD * (1.0 + abs(v)) if math.isfinite(v) else v


def _pad_lo(v):
    return v - PAD * (1.0 + abs(v)) if math.isfinite(v) else v


class Spec:
    def __init__(self, d):
        self.n = int(d["n_cols"])
        self.c = np.asarray(d["c"], float)
        self.integ = np.asarray(d["integrality"]).astype(bool)
        ptr, cols, co, rhs = (np.asarray(d[k]) for k in ("fixed_row_ptr", "fixed_cols", "fixed_coeffs", "fixed_rhs"))
        self.rows = [(cols[ptr[r]:ptr[r + 1]].astype(int), co[ptr[r]:ptr[r + 1]].astype(float), float(rhs[r]))
                     for r in range(len(rhs))]
        self.terms = []
        for k in range(len(d["term_kind"])):
            kind = int(d["term_kind"][k])
            if kind == 1:
                self.terms.append(("mono", int(d["term_i"][k]), int(d["term_out"][k]), int(d["term_p"][k])))
            elif kind == 0:
                self.terms.append(("bil", int(d["term_i"][k]), int(d["term_j"][k]), int(d["term_out"][k])))
            else:
                raise NotImplementedError(f"reference FBBT: term kind {kind} not implemented")
        self.blf = []
        ap, ac, aco, acst = (np.asarray(d[k]) for k in ("blf_a_ptr", "blf_a_cols", "blf_a_coeffs", "blf_a_const"))
        bp, bc, bco, bcst = (np.asarray(d[k]) for k in ("blf_b_ptr", "blf_b_cols", "blf_b_coeffs", "blf_b_const"))
        for t, w in enumerate(np.asarray(d["blf_w"])):
            self.blf.append(((ac[ap[t]:ap[t + 1]].astype(int), aco[ap[t]:ap[t + 1]].astype(float), float(acst[t])),
                             (bc[bp[t]:bp[t + 1]].astype(int), bco[bp[t]:bp[t + 1]].astype(float), float(bcst[t])),
                             int(w)))
        nz = np.flatnonzero(self.c)
        self.obj = (nz, self.c[nz])


class Empty(Exception):
    pass


class Box:
    def __init__(self, lo, hi, tol):
        self.lo = [(-INF if v <= -1e19 else float(v)) for v in lo]
        self.hi = [(INF if v >= 1e19 else float(v)) for v in hi]
        self.tol = tol
        self.changed = False

    def set_hi(self, j, v):
        v = _pad_hi(v)
        if v < self.hi[j] - self.tol * (1 + abs(self.hi[j])) or (self.hi[j] == INF and math.isfinite(v)):
            self.hi[j] = v
            self.changed = True
        if self.lo[j] > self.hi[j] + 1e-7 * (1 + abs(self.hi[j])):
            raise Empty

    def set_lo(self, j, v):
        v = _pad_lo(v)
        if v > self.lo[j] + self.tol * (1 + abs(self.lo[j])) or (self.lo[j] == -INF and math.isfinite(v)):
            self.lo[j] = v
            self.changed = True
        if self.lo[j] > self.hi[j] + 1e-7 * (1 + abs(self.hi[j])):
            raise Empty


def lin_le(B, cols, a, rhs):
    """sum a_k x_cols[k] <= rhs."""
    tot, ninf, kinf = 0.0, 0, -1
    for k, (j, c) in enumerate(zip(cols, a)):
        if c == 0.0:
            continue
        m = c * (B.lo[j] if c > 0 else B.hi[j])
        if math.isfinite(m):
            tot += m
        else:
            ninf, kinf = ninf + 1, k
    if ninf == 0 and tot > rhs + 1e-7 * (1 + abs(rhs)):
        raise Empty
    if ninf > 1:
        return
    for k, (j, c) in enumerate(zip(cols, a)):
        if abs(c) < 1e-12:
            continue
        if ninf == 1 and k != kinf:
            continue
        mk = 0.0 if ninf == 1 else c * (B.lo[j] if c > 0 else B.hi[j])
        resid = rhs - (tot - mk)
        if c > 0:
            B.set_hi(j, resid / c)
        else:
            B.set_lo(j, resid / c)


def form_iv(B, f):
    cols, a, cst = f
    lo = hi = cst
    for j, c in zip(cols, a):
        if c >= 0:
            lo += c * B.lo[j]
            hi += c * B.hi[j]
        else:
            lo += c * B.hi[j]
            hi += c * B.lo[j]
    return lo, hi


def form_to(B, f, tlo, thi):
    cols, a, cst = f
    if math.isfinite(thi):
        lin_le(B, cols, a, thi - cst)
    if math.isfinite(tlo):
        lin_le(B, cols, -a, -(tlo - cst))


def _mul(a, b):
    if a == 0.0 or b == 0.0:
        return 0.0
    return a * b


def iv_mul(al, ah, bl, bh):
    ps = [_mul(x, y) for x in (al, ah) for y in (bl, bh)]
    return min(ps), max(ps)


def ext_div(wl, wh, gl, gh, fl, fh):
    """Hull of {F in [fl,fh] : exists w in [wl,wh], g in [gl,gh], w = F*g}; None = no info."""
    if gl > 0 or gh < 0:
        qs = [w / g for w in (wl, wh) for g in (gl, gh) if math.isfinite(w / g) or True]
        qs = [q for q in qs if not math.isnan(q)]
        if not qs:
            return None
        return min(qs), max(qs)
    if gl == 0.0 and gh == 0.0:
        return None
    if wl <= 0.0 <= wh:
        return None  # 0 in W, 0 in G: F unconstrained
    pieces = []
    if wl > 0:
        if gh > 0:
            pieces.append((wl / gh, INF))
        if gl < 0:
            pieces.append((-INF, wl / gl))
    else:  # wh < 0
        if gh > 0:
            pieces.append((-INF, wh / gh))
        if gl < 0:
            pieces.append((wh / gl, INF))
    keep = [(max(a, fl), min(b, fh)) for a, b in pieces if max(a, fl) <= min(b, fh)]
    if not keep:
        return (INF, -INF)  # empty
    return min(k[0] for k in keep), max(k[1] for k in keep)


def product(B, fa, fb, w):
    al, ah = form_iv(B, fa)
    bl, bh = form_iv(B, fb)
    if all(math.isfinite(v) for v in (al, ah, bl, bh)):
        pl, ph = iv_mul(al, ah, bl, bh)
        B.set_lo(w, pl)
        B.set_hi(w, ph)
    for (ff, (fl, fh)), (gl, gh) in (((fa, (al, ah)), (bl, bh)), ((fb, (bl, bh)), (al, ah))):
        q = ext_div(B.lo[w], B.hi[w], gl, gh, fl, fh)
        if q is not None:
            if q[0] > q[1]:
                raise Empty
            form_to(B, ff, q[0], q[1])
        # re-read: a reverse step may have moved the other factor
        al, ah = form_iv(B, fa)
        bl, bh = form_iv(B, fb)


def mono(B, i, s, p):
    xl, xh = B.lo[i], B.hi[i]
    if p % 2 == 1:
        B.set_lo(s, xl ** p if math.isfinite(xl) else -INF)
        B.set_hi(s, xh ** p if math.isfinite(xh) else INF)
        sl, sh = B.lo[s], B.hi[s]
        B.set_lo(i, math.copysign(abs(sl) ** (1 / p), sl) if math.isfinite(sl) else -INF)
        B.set_hi(i, math.copysign(abs(sh) ** (1 / p), sh) if math.isfinite(sh) else INF)
        return
    if xl >= 0:
        lo_, hi_ = xl ** p, xh ** p
    elif xh <= 0:
        lo_, hi_ = xh ** p, xl ** p
    else:
        lo_, hi_ = 0.0, max(xl ** p, xh ** p)
    B.set_lo(s, lo_)
    B.set_hi(s, hi_)
    sl, sh = max(B.lo[s], 0.0), B.hi[s]
    if sh < -1e-7:
        raise Empty
    rh = max(sh, 0.0) ** (1 / p)
    rl = sl ** (1 / p)
    pieces = [(-rh, -rl), (rl, rh)]
    keep = [(max(a, xl), min(b, xh)) for a, b in pieces if max(a, xl) <= min(b, xh) + 1e-12 * (1 + abs(b))]
    if not keep:
        raise Empty
    B.set_lo(i, min(k[0] for k in keep))
    B.set_hi(i, max(k[1] for k in keep))


def fbbt(spec: Spec, lo, hi, cutoff=None, max_rounds=100000, tol=1e-12):
    """Returns (live, lo, hi, rounds)."""
    B = Box(lo, hi, tol)
    try:
        for r in range(1, max_rounds + 1):
            B.changed = False
            if cutoff is not None and math.isfinite(cutoff):
                lin_le(B, spec.obj[0], spec.obj[1], cutoff)
            for cols, a, rhs in spec.rows:
                lin_le(B, cols, a, rhs)
            for fa, fb, w in spec.blf:
                product(B, fa, fb, w)
            for t in spec.terms:
                if t[0] == "mono":
                    mono(B, t[1], t[2], t[3])
                else:
                    product(B, (np.array([t[1]]), np.array([1.0]), 0.0), (np.array([t[2]]), np.array([1.0]), 0.0), t[3])
            for j in np.flatnonzero(spec.integ):
                nl, nh = math.ceil(B.lo[j] - 1e-6), math.floor(B.hi[j] + 1e-6)
                if nl > B.lo[j] + 1e-9:
                    B.lo[j] = float(nl); B.changed = True
                if nh < B.hi[j] - 1e-9:
                    B.hi[j] = float(nh); B.changed = True
                if B.lo[j] > B.hi[j] + 1e-9:
                    raise Empty
            if not B.changed:
                break
    except Empty:
        return False, np.array(B.lo), np.array(B.hi), r
    return True, np.array(B.lo), np.array(B.hi), r
