"""Issue #1230 step 1 (entry experiment): is the margin-free Rust NS bound sound,
and how large a margin would make it sound?

`lp/simplex/refine.rs::ns_safe_bound_csc` evaluates the Neumaier-Shcherbina bound

    g(y) = bᵀy + Σ_j min_{x_j∈[l_j,u_j]} (c − Aᵀy)_j x_j   ≤   p*

with `bᵀy` and each `(Aᵀy)_j` in double-double, but then rounds the reduced cost to
f64 (`rc = c[j] - aty.to_f64()`), multiplies by the box end in f64, and subtracts
**no margin**.  The Python twin `milp_simplex._safe_lp_lower_bound_std` subtracts
`1e-9 * (1 + |bᵀy| + Σ_j |contrib_j|)`.  Issue #1230 step 1 asks whether the missing
margin can push `g` above `p*`, with the kill criterion: if it can, the consolidated
implementation must carry a margin *and* DD accumulation, not DD alone.

That matters more than when #1230 was filed: after the HiGHS LP/MILP routing (#1258)
this function certifies the **default** pure-LP route (`lp_milp_highs.ns_bound`) as
well as the MINLP node kernels (`bnb/spatial_kernel.rs`, `bnb/convex_kernel.rs`).

Two rigorous oracles, both against the production function:

  A. `lp_milp_highs.exact_ns_bound(y, sf)` — the same formula in exact rationals with
     outward rounding.  `g_exact(y) <= p*` by weak duality for every `y`, so
     `g_fp > g_exact` is fp excess over a provably valid bound.  (Here it is the plain
     exact evaluation at the same `y`: the synthetic LPs have all-finite boxes, so its
     project-and-shift dual correction never fires.)
  B. an exact rational upper bound on `p*`: HiGHS's optimal basis re-solved over the
     rationals with exact box feasibility checked.  `A x = b` then holds exactly, so
     `cᵀx >= p*`.  `g_fp > cᵀx` is an outright soundness violation.

`S = 1 + |bᵀy| + Σ_j |contrib_j|` is the base of the Python twin's margin; the probe
reports both errors as multiples of `S`, which is what sizes the fix.

Measured 2026-09-18, seed 20260918, 250 LPs per regime (m in [4,16), n in [m+2,m+24)):

    executed comparisons : 1000 vs the exact NS bound, 980 vs exact p*
    g_fp > g_exact(y)    : 566/1000   max (g_fp - g_exact)/S = 6.19e-16
    g_fp > p*  (UNSOUND) : 206/980    max (g_fp - p*)/S      = 4.40e-16

So the hypothesis is CONFIRMED and the kill criterion applies -- but the overshoot is
a few ulps of `S`, not the O(n·ulp·range) blow-up the issue feared.  The Python twin's
existing `1e-9 * S` margin covers it with ~6 orders of headroom.

Sampling limit, stated per the #727 lesson: the real-instance arm yields ZERO usable
comparisons.  Of the three netlib-class LPs shipped in `tests/data`, e226 is the one
that solves and the Rust bound abstains on it (`-inf`; that is the documented
`ns-exact-dual-correction` path), and forest6/klein1 are infeasible by construction.
The violation *existence* result does not need broader sampling; the *magnitude* bound
above is synthetic-only and should not be quoted as a corpus-wide envelope.

Run:  PYTHONPATH=python python -u scratchpad/ns_safe_bound_margin.py [trials_per_regime]

Prints executed-comparison counts and exits non-zero if any is zero (CLAUDE.md §6).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import discopt
import discopt.solvers.lp_milp_highs as H
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from discopt._rust import ns_safe_bound_csc_py

PSTAR_MAX_M = 90  # the exact rational basis solve is O(m^3) in growing rationals

_marker = ns_safe_bound_csc_py(
    np.array([0.0]), np.array([1.0]), 1, 1,
    np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64), np.array([1.0]),
    np.array([0.0]), np.array([-1e20]), np.array([1e20]),
)  # fmt: skip
assert _marker is None, f"marker failed: expected None on an open side, got {_marker}"
print("marker       : REFINE_INF=1e20 abstention confirmed")


def make_lp(rng, regime, m, n):
    if regime == "well":
        amag, bwidth = 0.0, 1.0
    elif regime == "moderate":
        amag, bwidth = 3.0, 1e2
    elif regime == "ill":
        amag, bwidth = 6.0, 1e5
    else:  # "cancel": huge coefficients and huge boxes -> huge box terms
        amag, bwidth = 7.0, 1e8
    dens = 0.35
    mask = rng.random((m, n)) < dens
    mask[np.arange(m), rng.integers(0, n, m)] = True
    A = np.zeros((m, n))
    sgn = rng.choice([-1.0, 1.0], size=(m, n))
    A[mask] = (sgn * 10.0 ** rng.uniform(-amag, amag, size=(m, n)))[mask]
    lo = -bwidth * rng.random(n)
    hi = bwidth * rng.random(n)
    x0 = lo + (hi - lo) * rng.random(n)
    b = A @ x0
    cmag = amag if regime != "cancel" else 7.0
    c = 10.0 ** rng.uniform(-cmag / 2, cmag / 2, size=n) * rng.choice([-1.0, 1.0], n)
    return H.StdForm.from_arrays(c, sp.csc_matrix(A), b, lo, hi)


def real_instances():
    d = np.load(Path(discopt.__file__).parent.parent / "tests" / "data"
                / "lp_highs_certificate_instances.npz")  # fmt: skip
    for nm in sorted({k.split("__")[0] for k in d.files}):
        shape = tuple(int(v) for v in d[f"{nm}__shape"])
        A = sp.csc_matrix((d[f"{nm}__data"], d[f"{nm}__indices"], d[f"{nm}__indptr"]), shape=shape)
        yield nm, H.StdForm.from_arrays(
            d[f"{nm}__c"], A, d[f"{nm}__b"], d[f"{nm}__xl"], d[f"{nm}__xu"],
            float(d[f"{nm}__obj_const"]),
        )  # fmt: skip


def solve_highs(sf, tight=True):
    highspy = H.require_highspy()
    opts = [("output_flag", False)]
    if tight:
        opts += [("primal_feasibility_tolerance", 1e-9), ("dual_feasibility_tolerance", 1e-9),
                 ("presolve", "off")]  # fmt: skip
    h = H._new_highs(highspy, opts)
    st, _ = H._pass_model(h, highspy, sf, integer=False)
    if st == highspy.HighsStatus.kError:
        return None
    h.run()
    if H._status_name(h) != "kOptimal":
        return None
    sol, basis = h.getSolution(), h.getBasis()
    cs = np.asarray(
        [int(s) == int(highspy.HighsBasisStatus.kBasic) for s in basis.col_status], dtype=bool
    )
    return np.asarray(sol.col_value, np.float64), np.asarray(sol.row_dual, np.float64), cs


def exact_pstar_upper(sf, x, basic):
    """Exact rational upper bound on p*.

    Take HiGHS's basic columns; if fewer than m, extend with the nonbasic columns
    that keep the block full rank (float QR with column pivoting decides *which*
    columns -- a float choice that can only make the oracle abstain, never make it
    wrong).  Fix the remaining columns at the exact float bound HiGHS put them on,
    solve `A_B x_B = b - A_N x_N` over the rationals, and check the box exactly.
    `A x = b` then holds exactly, so `cᵀx >= p*` rigorously.
    """
    m = sf.m
    if m > PSTAR_MAX_M:
        return None
    Ad = sf.A.toarray()
    cand = list(np.flatnonzero(basic))
    if len(cand) < m:
        rest = [j for j in range(sf.n) if not basic[j]]
        order = np.argsort(-np.abs(Ad[:, rest]).max(axis=0)) if rest else np.zeros(0, int)
        cand += [rest[i] for i in order]
    if len(cand) < m:
        return None
    sub = Ad[:, cand]
    _, R, piv = sla.qr(sub, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    if d.size < m or d[0] <= 0.0 or d[m - 1] <= 1e-11 * d[0]:
        return None
    B = sorted(cand[i] for i in piv[:m])
    inB = set(B)
    xl, xu = sf.xl, sf.xu
    xN = {
        j: Fraction(float(xl[j] if abs(x[j] - xl[j]) <= abs(x[j] - xu[j]) else xu[j]))
        for j in range(sf.n)
        if j not in inB
    }
    rhs = [Fraction(float(sf.b[i])) for i in range(m)]
    for j, v in xN.items():
        if v:
            col = Ad[:, j]
            for i in np.flatnonzero(col):
                rhs[i] -= Fraction(float(col[i])) * v
    # `_exact_solve` takes M row-major: row i is [A[i, B_0], ..., A[i, B_{m-1}]].
    M = [[Fraction(float(Ad[i, j])) for j in B] for i in range(m)]
    xB = H._exact_solve(M, rhs, None)
    if xB is None:
        return None
    total = Fraction(float(sf.obj_const))
    for k, j in enumerate(B):
        v = xB[k]
        if v < Fraction(float(xl[j])) or v > Fraction(float(xu[j])):
            return None
        total += Fraction(float(sf.c[j])) * v
    for j, v in xN.items():
        total += Fraction(float(sf.c[j])) * v
    return total


def exact_terms(y: np.ndarray, sf) -> tuple[Fraction, Fraction]:
    """Exact `(g_exact(y), S)` where `S = 1 + |bᵀy| + Σ_j |contrib_j|` is the base of
    the Python twin's margin.  Returns `(None, S)` if a term is `-inf`."""
    Y = [Fraction(float(v)) for v in y]
    bty = sum((Fraction(float(sf.b[i])) * Y[i] for i in range(sf.m)), Fraction(0))
    total = bty + Fraction(float(sf.obj_const))
    S = Fraction(1) + abs(bty)
    A = sf.A.tocsc()
    ok = True
    for j in range(sf.n):
        aty = Fraction(0)
        for p in range(A.indptr[j], A.indptr[j + 1]):
            v = Y[int(A.indices[p])]
            if v:
                aty += Fraction(float(A.data[p])) * v
        r = Fraction(float(sf.c[j])) - aty
        if r > 0:
            if sf.xl[j] <= -H.INF:
                ok = False
                continue
            t = r * Fraction(float(sf.xl[j]))
        elif r < 0:
            if sf.xu[j] >= H.INF:
                ok = False
                continue
            t = r * Fraction(float(sf.xu[j]))
        else:
            continue
        total += t
        S += abs(t)
    return (total if ok else None), S


def nvs05_node_lp():
    """The real MINLP node LP shipped at tests/data/nvs05_node171_decline_lp.npz,
    `A x <= b_ub` over `[node_lb, node_ub]`, converted to std form with slacks."""
    f = Path(discopt.__file__).parent.parent / "tests" / "data" / "nvs05_node171_decline_lp.npz"
    if not f.exists():
        return None
    d = np.load(f)
    shape = tuple(int(v) for v in d["A_shape"])
    # the stored arrays are CSR (indptr has shape[0]+1 entries), not CSC
    ctor = sp.csr_matrix if len(d["A_indptr"]) == shape[0] + 1 else sp.csc_matrix
    A = sp.csc_matrix(ctor((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape))
    m, n = A.shape
    # `bounds` is the full (n, 2) box; node_lb/node_ub cover only the branched vars.
    bnd = np.asarray(d["bounds"], np.float64)
    lo, hi = bnd[:, 0].copy(), bnd[:, 1].copy()
    b = np.asarray(d["b_ub"], np.float64)
    # A x + s = b, 0 <= s <= INF  ->  keep s finite so the NS box term exists.
    slack_hi = np.maximum(0.0, b - (A @ np.where(np.isfinite(lo), lo, 0.0)))
    slack_hi = np.where(np.isfinite(slack_hi), np.abs(slack_hi) + 1.0, 1e6)
    Astd = sp.hstack([A, sp.identity(m, format="csc")], format="csc")
    return H.StdForm.from_arrays(
        np.r_[np.asarray(d["c"], np.float64), np.zeros(m)], Astd, b,
        np.r_[lo, np.zeros(m)], np.r_[hi, slack_hi], float(np.ravel(d["obj_offset"])[0]),
    )  # fmt: skip


class Acc:
    def __init__(self):
        self.n_ex = self.n_p = self.abstain = 0
        self.v_ex = self.v_p = 0
        self.ex_over_S: list[float] = []  # (g_fp - g_exact)/S , signed
        self.p_over_S: list[float] = []  # (g_fp - p*)/S      , signed
        self.amp: list[float] = []  # S / max(1,|p*|)

    def report(self, label):
        print(f"\n[{label}]")
        print(f"  comparisons: vs exact NS = {self.n_ex}, vs exact p* = {self.n_p}, "
              f"abstained = {self.abstain}")  # fmt: skip
        print(f"  g_fp > g_exact(y) : {self.v_ex}/{self.n_ex}")
        print(f"  g_fp > p*         : {self.v_p}/{self.n_p}   (UNSOUND)")
        for nm, xs in (("(g_fp-g_exact)/S", self.ex_over_S), ("(g_fp-p*)/S", self.p_over_S)):
            if xs:
                print(f"    {nm}: max={max(xs):+.3e}  median={np.median(xs):+.3e}")
        if self.amp:
            print(f"    amplification S/max(1,|p*|): median={np.median(self.amp):.3e} "
                  f"max={max(self.amp):.3e}")  # fmt: skip


def measure(sf, acc, tag, tight=True):
    got = solve_highs(sf, tight)
    if got is None:
        print(f"    {tag}: HiGHS not kOptimal", flush=True)
        return
    x, y, basic = got
    if not np.all(np.isfinite(y)):
        return
    g_fp = H.ns_bound(y, sf)
    if g_fp is None:
        acc.abstain += 1
        print(f"    {tag}: Rust bound abstained (-inf)", flush=True)
        return
    g_ex, S = exact_terms(y, sf)
    note = ""
    if g_ex is not None:
        acc.n_ex += 1
        e = (Fraction(g_fp) - g_ex) / S
        acc.ex_over_S.append(float(e))
        if e > 0:
            acc.v_ex += 1
        note += f" (g_fp-g_ex)/S={float(e):+.2e}"
    ub = exact_pstar_upper(sf, x, basic)
    if ub is not None:
        acc.n_p += 1
        d = (Fraction(g_fp) - ub) / S
        acc.p_over_S.append(float(d))
        acc.amp.append(float(S / max(Fraction(1), abs(ub))))
        if Fraction(g_fp) > ub:
            acc.v_p += 1
            note += f" UNSOUND (g_fp-p*)/S={float(d):+.2e}"
    print(f"    {tag}: g_fp={g_fp:.10g}{note}", flush=True)


def main() -> int:
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    rng = np.random.default_rng(20260918)
    accs: dict[str, Acc] = {}

    print("\n--- real instances shipped in the repo ---", flush=True)
    accs["real"] = Acc()
    for nm, sf in real_instances():
        print(f"  {nm}: m={sf.m} n={sf.n}", flush=True)
        measure(sf, accs["real"], nm, tight=False)
    node = nvs05_node_lp()
    if node is not None:
        print(f"  nvs05_node171: m={node.m} n={node.n}", flush=True)
        measure(node, accs["real"], "nvs05_node171", tight=False)

    for regime in ("well", "moderate", "ill", "cancel"):
        print(f"\n--- synthetic: {regime} ---", flush=True)
        accs[regime] = Acc()
        for t in range(n_per):
            m = int(rng.integers(4, 16))
            n = int(rng.integers(m + 2, m + 24))
            measure(make_lp(rng, regime, m, n), accs[regime], f"{regime}/{t}")

    print("\n" + "=" * 78)
    print("ns_safe_bound_csc (Rust, margin-free) vs exact rational oracles")
    print("=" * 78)
    te = tp = ve = vp = 0
    for k, a in accs.items():
        a.report(k)
        te += a.n_ex
        tp += a.n_p
        ve += a.v_ex
        vp += a.v_p
    all_e = [v for a in accs.values() for v in a.ex_over_S]
    all_p = [v for a in accs.values() for v in a.p_over_S]
    print(f"\nTOTAL comparisons: exact-NS={te}  exact-p*={tp}")
    print(f"TOTAL g_fp > g_exact(y): {ve}   max (g_fp-g_ex)/S = {max(all_e):+.3e}")
    print(f"TOTAL g_fp > p*        : {vp}   max (g_fp-p*)/S   = {max(all_p):+.3e}")
    need = max(max(all_e), max(all_p))
    print(f"\nPython twin's margin is 1e-9*S; measured error needs {need:.1e}*S")
    if te == 0 or tp == 0:
        print("FAIL: a comparison count is zero -- the probe measured nothing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
