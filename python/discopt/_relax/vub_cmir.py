"""Path-aggregation c-MIR with variable-bound substitution (#1917, F1).

A root separator for the McCormick LP of the spatial B&B route. It targets the
GDP big-M / fixed-charge class: linear rows whose continuous columns carry
*variable* upper bounds ``x_j <= u_j y_k`` (``y_k`` binary). The textbook case is
a big-M mass balance ``f_out - f_in + M y <= M`` plus the switch ``f_out <= u y``:
under a fractional ``y`` the LP "fabricates" outflow with no inflow, and the
single-row MIR / Gomory / cover separators cannot see it, because the inequality
that removes it (``f_out <= f_in``) needs the continuous ``f_out`` to be written
through its VUB before rounding. Measured on GDPlib ``syngas`` under
``gdp.bigm`` (#1917): the root bound moves 2.06 -> 4339 (oracle 4669.02); the
existing ``cmir_cuts.separate_cmir`` and the #282 stage-2 VUB separator both
leave it at 2.06.

Procedure (Marchand & Wolsey 2001, "Aggregation and mixed integer rounding to
solve MIPs", with the bound-substitution step of SCIP's ``cuts.c``):

1. **Path aggregation.** Start from a row, and while the c-MIR of the aggregate
   is not violated, eliminate the continuous column farthest from its bounds by
   adding a nonnegative multiple of another row (one with small slack whose
   coefficient on that column has the opposite sign).
2. **Bound substitution.** Each continuous column is written as distance
   ``s_j >= 0`` from a simple bound (``lb``/``ub``) or a variable bound
   (``vlb``/``vub``). Two strategies are tried and the more efficacious cut kept:
   ``"closest"`` (nearest bound at ``x*``) and ``"vub_pos"`` (a column with a
   positive coefficient goes through its VUB). The second is the difference
   from the #282 stage-2 probe, which falsified VUB aggregation on rsyn/syn
   with nearest-bound substitution only.
3. **Complementation + MIR.** Integer columns are complemented at the midpoint
   of their (finite) box; the MIR function is applied for scalings ``delta`` in
   ``{|g_j|, |g_j|/2, |g_j|/4, |g_j|/8}``.

**Validity.** Each aggregate is a nonnegative combination of rows of ``A x <= b``;
every ``s_j`` is nonnegative at every feasible point (a finite simple bound, or a
variable-bound row that is itself a row of the system with ``y`` binary on
``[0, 1]``); every integer column has a finite box. MIR of such a row is valid for
the mixed-integer set. Columns the caller does not mark integer are treated as
continuous, which is always valid (it only weakens a cut). A safety margin
``1e-7 (1 + sum|coef|)`` is added to every right-hand side.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from math import floor
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

_EPS = 1e-9
_MINFRAC, _MAXFRAC = 0.05, 0.999
#: Largest |coef| ratio a cut may carry. Wider rows are dropped as numerically
#: unsafe (dropping a cut only weakens the bound).
_MAX_DYNAMISM = 1e8
_STRATEGIES = ("closest", "vub_pos")

VBounds = dict[int, list[tuple[int, float]]]


def vub_cmir_root_enabled() -> bool:
    """Whether the root path-aggregation c-MIR separator runs (**default OFF**).

    ``DISCOPT_VUB_CMIR_ROOT=1`` turns it on. Bound-changing (it adds valid
    inequalities to every node LP of the spatial McCormick route), so per
    CLAUDE.md §5 it stays OFF until a corpus-wide ON/OFF panel passes both bars
    (cert-clean and net-positive). Audit row: ``docs/dev/flag-retirement-audit.md``.
    """
    return os.environ.get("DISCOPT_VUB_CMIR_ROOT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def detect_variable_bounds(
    A: sp.csr_matrix,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    is_int: np.ndarray,
) -> tuple[VBounds, VBounds]:
    """Variable bounds read off two-nonzero rows ``a_p x_p + a_q y_q <= 0``.

    ``x_p`` continuous, ``y_q`` integer with box exactly ``[0, 1]``. ``a_p > 0``
    gives the VUB ``x_p <= (-a_q/a_p) y_q``; ``a_p < 0`` gives the VLB
    ``x_p >= (-a_q/a_p) y_q``. Returns ``(vub, vlb)`` mapping ``p`` to a list of
    ``(q, coef)``.
    """
    vub: VBounds = {}
    vlb: VBounds = {}
    indptr, indices, data = A.indptr, A.indices, A.data
    for i in range(A.shape[0]):
        lo, hi = indptr[i], indptr[i + 1]
        if abs(b[i]) > _EPS:
            continue
        cols = indices[lo:hi]
        vals = data[lo:hi]
        keep = np.abs(vals) > _EPS
        if int(keep.sum()) != 2:
            continue
        (c0, c1), (v0, v1) = cols[keep], vals[keep]
        for p, ap, q, aq in ((c0, v0, c1, v1), (c1, v1, c0, v0)):
            if is_int[p] or not is_int[q] or lb[q] != 0.0 or ub[q] != 1.0:
                continue
            (vub if ap > 0 else vlb).setdefault(int(p), []).append((int(q), float(-aq / ap)))
    return vub, vlb


def _mir(g: dict, zstar: dict, hs: np.ndarray, sstar: np.ndarray, beta: float, d: float):
    """MIR of ``sum g_j z_j + sum h_j s_j <= beta`` scaled by ``1/d``."""
    bt = beta / d
    f0 = bt - floor(bt)
    if f0 < _MINFRAC or f0 > _MAXFRAC:
        return None
    gam = {}
    lhs = 0.0
    for j, gj in g.items():
        gr = gj / d
        r = round(gr)
        if abs(gr - r) < 1e-9:
            gr = float(r)
        fj = gr - floor(gr)
        gam[j] = floor(gr) + max(0.0, fj - f0) / (1.0 - f0)
        lhs += gam[j] * zstar[j]
    hcut = np.minimum(hs, 0.0) / (d * (1.0 - f0))
    lhs += float(hcut @ sstar)
    rhs = float(floor(bt))
    viol = lhs - rhs
    if viol <= 1e-6:
        return None
    nrm2 = sum(v * v for v in gam.values()) + float(hcut @ hcut)
    return viol / max(nrm2, 1e-12) ** 0.5, gam, hcut, rhs


def _cmir_once(a, beta, x, lb, ub, is_int, vub, vlb, strat):
    """c-MIR of ``a . x <= beta`` under one substitution strategy.

    Returns ``(coef, rhs, efficacy)`` over the original columns, or ``None``.
    """
    n = a.shape[0]
    a = a.copy()
    beta = float(beta)
    cont_terms = []  # (j, h_j, kind, (q, coef) | None)
    for j in np.nonzero(np.abs(a) > _EPS)[0]:
        if is_int[j]:
            continue
        aj = float(a[j])
        cands = []
        if np.isfinite(lb[j]):
            cands.append((x[j] - lb[j], "lb", None))
        if np.isfinite(ub[j]):
            cands.append((ub[j] - x[j], "ub", None))
        for q, cf in vlb.get(j, ()):
            cands.append((x[j] - cf * x[q], "vlb", (q, cf)))
        for q, cf in vub.get(j, ()):
            cands.append((cf * x[q] - x[j], "vub", (q, cf)))
        if not cands:
            return None  # a free continuous column: no valid substitution
        if strat == "vub_pos" and aj > 0 and vub.get(j):
            cands = [c for c in cands if c[1] == "vub"]
        pref = [c for c in cands if c[1] in (("lb", "vlb") if aj > 0 else ("ub", "vub"))]
        best = min(pref, key=lambda c: c[0]) if pref else None
        if best is None or best[0] > 1e-6:
            best = min(cands, key=lambda c: (c[0], c[1] not in ("vlb", "vub")))
        _dist, kind, data = best
        a[j] = 0.0
        if kind == "lb":  # x = lb + s
            beta -= aj * lb[j]
            cont_terms.append((j, aj, kind, None))
        elif kind == "ub":  # x = ub - s
            beta -= aj * ub[j]
            cont_terms.append((j, -aj, kind, None))
        elif kind == "vlb":  # x = cf*y + s
            q, cf = data
            a[q] += aj * cf
            cont_terms.append((j, aj, kind, data))
        else:  # vub: x = cf*y - s
            q, cf = data
            a[q] += aj * cf
            cont_terms.append((j, -aj, kind, data))
    int_cols = [j for j in np.nonzero(np.abs(a) > _EPS)[0] if is_int[j]]
    if not int_cols:
        return None
    g: dict = {}
    zstar: dict = {}
    comp: dict = {}
    for j in int_cols:
        if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            return None
        if x[j] > 0.5 * (lb[j] + ub[j]):  # z = ub - x
            comp[j] = True
            beta -= a[j] * ub[j]
            g[j] = -a[j]
            zstar[j] = ub[j] - x[j]
        else:  # z = x - lb
            comp[j] = False
            beta -= a[j] * lb[j]
            g[j] = a[j]
            zstar[j] = x[j] - lb[j]
    sstar_l = []
    for j, _h, kind, data in cont_terms:
        if kind == "lb":
            sstar_l.append(x[j] - lb[j])
        elif kind == "ub":
            sstar_l.append(ub[j] - x[j])
        elif kind == "vlb":
            sstar_l.append(x[j] - data[1] * x[data[0]])
        else:
            sstar_l.append(data[1] * x[data[0]] - x[j])
    sstar = np.maximum(np.asarray(sstar_l, dtype=np.float64), 0.0)
    hs = np.asarray([t[1] for t in cont_terms], dtype=np.float64)

    deltas = sorted({abs(g[j]) for j in int_cols if zstar[j] > 1e-6 and abs(g[j]) > 1e-6})
    best_res = None
    tried: set = set()
    for d0 in deltas:
        for d in (d0, d0 / 2, d0 / 4, d0 / 8):
            key = round(d, 12)
            if key in tried or d < 1e-6:
                continue
            tried.add(key)
            res = _mir(g, zstar, hs, sstar, beta, d)
            if res is not None and (best_res is None or res[0] > best_res[0]):
                best_res = res
    if best_res is None:
        return None
    _eff, gam, hcut, rhs = best_res
    coef = np.zeros(n)
    rhs_x = rhs
    for j in int_cols:
        if comp[j]:
            coef[j] -= gam[j]
            rhs_x -= gam[j] * ub[j]
        else:
            coef[j] += gam[j]
            rhs_x += gam[j] * lb[j]
    for (j, _h, kind, data), hc in zip(cont_terms, hcut):
        if hc == 0.0:
            continue
        if kind == "lb":  # s = x - lb
            coef[j] += hc
            rhs_x += hc * lb[j]
        elif kind == "ub":  # s = ub - x
            coef[j] -= hc
            rhs_x -= hc * ub[j]
        elif kind == "vlb":  # s = x - cf*y
            coef[j] += hc
            coef[data[0]] -= hc * data[1]
        else:  # s = cf*y - x
            coef[j] -= hc
            coef[data[0]] += hc * data[1]
    if not np.all(np.isfinite(coef)) or not np.isfinite(rhs_x):
        return None
    cmax = float(np.abs(coef).max())
    if cmax <= _EPS:
        return None
    # Remove relatively tiny coefficients SOUNDLY (floating residue of the
    # aggregation): drop c_j x_j and relax the rhs by -min(c_j lb_j, c_j ub_j),
    # its smallest value over the box. A column without a finite box keeps it.
    for j in np.nonzero((coef != 0.0) & (np.abs(coef) < cmax / _MAX_DYNAMISM))[0]:
        if np.isfinite(lb[j]) and np.isfinite(ub[j]):
            rhs_x -= min(coef[j] * lb[j], coef[j] * ub[j])
            coef[j] = 0.0
    nz = np.abs(coef[coef != 0.0])
    if nz.max() / nz.min() > _MAX_DYNAMISM:
        return None
    viol = float(coef @ x - rhs_x)
    nrm = float(np.linalg.norm(coef))
    if viol <= 1e-6 * (1.0 + abs(rhs_x)):
        return None
    margin = 1e-7 * (1.0 + float(np.abs(coef).sum()))
    return coef, rhs_x + margin, viol / nrm


def cmir_of_aggregate(a, beta, x, lb, ub, is_int, vub, vlb):
    """The most efficacious c-MIR over the substitution strategies, or ``None``."""
    best = None
    for strat in _STRATEGIES:
        r = _cmir_once(a, beta, x, lb, ub, is_int, vub, vlb, strat)
        if r is not None and (best is None or r[2] > best[2]):
            best = r
    return best


def separate_path_cmir(
    A: sp.csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    is_int: np.ndarray,
    vub: VBounds,
    vlb: VBounds,
    *,
    max_aggr: int = 6,
    max_slack: float = 0.1,
    max_cuts: int = 100,
    deadline: Optional[float] = None,
) -> list[tuple[np.ndarray, float]]:
    """One separation round: path aggregation from every row, best cuts first.

    ``A`` is CSR over ``n`` columns with ``A x <= b``. Returns ``[(coef, rhs)]``
    with dense ``coef`` of length ``n``, sorted by efficacy, deduplicated.
    """
    A = sp.csr_matrix(A)
    m, n = A.shape
    Ac = A.tocsc()
    slack = b - A @ x
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()) + 1e-12
    rel = slack / nrm
    usable = rel <= max_slack
    bd = np.zeros(n)
    for j in range(n):
        if is_int[j]:
            continue
        lo = [x[j] - lb[j]] if np.isfinite(lb[j]) else []
        lo += [x[j] - cf * x[q] for q, cf in vlb.get(j, ())]
        hi = [ub[j] - x[j]] if np.isfinite(ub[j]) else []
        hi += [cf * x[q] - x[j] for q, cf in vub.get(j, ())]
        bd[j] = min(min(lo) if lo else np.inf, min(hi) if hi else np.inf)
    interesting = is_int | (bd > 1e-6)
    indptr, indices, data = A.indptr, A.indices, A.data
    starts = [
        int(i)
        for i in np.argsort(rel, kind="stable")
        if np.any(interesting[indices[indptr[i] : indptr[i + 1]]])
    ]
    cuts: list = []
    seen: set = set()
    for s in starts:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        agg = np.zeros(n)
        agg[indices[indptr[s] : indptr[s + 1]]] = data[indptr[s] : indptr[s + 1]]
        rhs = float(b[s])
        used = {s}
        for step in range(max_aggr + 1):
            res = cmir_of_aggregate(agg, rhs, x, lb, ub, is_int, vub, vlb)
            if res is not None:
                key = tuple(np.round(res[0] / max(np.abs(res[0]).max(), 1e-12), 6))
                if key not in seen:
                    seen.add(key)
                    cuts.append(res)
                break
            if step == max_aggr:
                break
            cand = [
                int(c) for c in np.nonzero(np.abs(agg) > _EPS)[0] if not is_int[c] and bd[c] > 1e-6
            ]
            cand.sort(key=lambda c: -bd[c])
            picked = None
            for jc in cand:
                lo_, hi_ = Ac.indptr[jc], Ac.indptr[jc + 1]
                rows = [
                    int(r)
                    for r, v in zip(Ac.indices[lo_:hi_], Ac.data[lo_:hi_])
                    if usable[r] and r not in used and abs(v) > _EPS and v * agg[jc] < 0
                ]
                if rows:
                    picked = (jc, min(rows, key=lambda r: rel[r]))
                    break
            if picked is None:
                break
            j, r = picked
            ra, rb = indptr[r], indptr[r + 1]
            rcols, rvals = indices[ra:rb], data[ra:rb]
            arj = float(rvals[rcols == j][0])
            lam = -agg[j] / arj
            agg[rcols] += lam * rvals
            agg[j] = 0.0
            rhs += lam * float(b[r])
            used.add(r)
            mx = float(np.abs(agg).max())
            if mx > 1e9 or mx < _EPS:
                break
    cuts.sort(key=lambda t: -t[2])
    return [(c, float(r)) for c, r, _e in cuts[:max_cuts]]


@dataclass
class RootVubCmirResult:
    """Cuts over the ORIGINAL model columns, plus the evidence they fired."""

    A: Optional[np.ndarray]  # (k, n_orig) or None
    b: Optional[np.ndarray]
    rounds: int = 0
    bound_trace: list = field(default_factory=list)
    n_vub: int = 0
    n_vlb: int = 0
    rejected_by_screen: int = 0
    stop_reason: str = ""


def root_vub_cmir_cuts(
    model,
    lb: np.ndarray,
    ub: np.ndarray,
    n_orig: int,
    *,
    deadline: float,
    max_rounds: int = 40,
    screen_points: Sequence[np.ndarray] = (),
) -> RootVubCmirResult:
    """Separate path-aggregation c-MIR rounds on the root MILP relaxation.

    The relaxation is :func:`build_milp_relaxation` over the box ``(lb, ub)``;
    its first ``n_orig`` columns are the model's flat variables. Only model
    integer columns (``< n_orig``) count as integer; auxiliary columns are
    treated as continuous (valid, possibly weaker). Only cuts whose support lies
    in the original columns are kept, so the rows can be appended to ANY lifted
    layout of the model by zero-padding.

    Every kept cut is checked against ``screen_points`` (verified-feasible
    original-column points); a cut a screen point violates is a separator defect
    and is dropped and counted, never installed.
    """
    from scipy.optimize import linprog

    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import build_milp_relaxation

    out = RootVubCmirResult(A=None, b=None)
    relax, _info = build_milp_relaxation(
        model,
        None,  # type: ignore[arg-type]  # ``terms`` is ignored by the uniform build
        DiscretizationState(),
        bound_override=(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)),
        build_deadline=deadline,
    )
    if relax._A_ub is None or relax._b_ub is None or relax._integrality is None:
        out.stop_reason = "no_rows"
        return out
    A0 = sp.csr_matrix(relax._A_ub, dtype=np.float64)
    b0 = np.asarray(relax._b_ub, dtype=np.float64).ravel()
    c = np.asarray(relax._c, dtype=np.float64).ravel()
    ncol = A0.shape[1]
    if ncol < n_orig:
        raise ValueError(f"relaxation has {ncol} columns < n_orig={n_orig}")
    bnds = np.array(
        [(-np.inf if lo is None else lo, np.inf if hi is None else hi) for lo, hi in relax._bounds],
        dtype=np.float64,
    )
    L, U = bnds[:, 0], bnds[:, 1]
    is_int = np.zeros(ncol, dtype=bool)
    is_int[:n_orig] = np.asarray(relax._integrality, dtype=bool)[:n_orig]
    if not is_int.any():
        out.stop_reason = "no_integers"
        return out
    vub, vlb = detect_variable_bounds(A0, b0, L, U, is_int)
    out.n_vub = sum(len(v) for v in vub.values())
    out.n_vlb = sum(len(v) for v in vlb.values())
    screen = [np.asarray(p, dtype=np.float64)[:n_orig] for p in screen_points]

    cut_a: list[np.ndarray] = []
    cut_b: list[float] = []
    seen: set = set()
    A, b = A0, b0
    out.stop_reason = "max_rounds"
    for _rnd in range(max_rounds):
        left = deadline - time.perf_counter()
        if left <= 0:
            out.stop_reason = "deadline"
            break
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=list(zip(L, U)),
            method="highs",
            options={"time_limit": max(left, 1e-3)},
        )
        if res.status != 0 or res.x is None:
            out.stop_reason = f"lp_status_{res.status}"
            break
        out.bound_trace.append(float(res.fun) + float(relax._obj_offset))
        new = []
        for coef, rhs in separate_path_cmir(A, b, res.x, L, U, is_int, vub, vlb, deadline=deadline):
            if np.any(coef[n_orig:] != 0.0):
                continue
            co = coef[:n_orig]
            key = tuple(np.round(co / np.abs(co).max(), 9)) + (round(rhs / np.abs(co).max(), 9),)
            if key in seen:
                continue
            bad = False
            for p in screen:
                terms = co * p
                if float(terms.sum()) - rhs > 1e-6 * (1.0 + abs(rhs) + float(np.abs(terms).sum())):
                    bad = True
                    break
            if bad:
                out.rejected_by_screen += 1
                continue
            seen.add(key)
            new.append((coef, rhs))
        if not new:
            out.stop_reason = "no_cuts"
            break
        out.rounds += 1
        for coef, rhs in new:
            cut_a.append(coef[:n_orig].copy())
            cut_b.append(rhs)
        A = sp.vstack([A, sp.csr_matrix(np.array([co for co, _ in new]))], format="csr")
        b = np.concatenate([b, [r for _, r in new]])
    if cut_a:
        out.A = np.vstack(cut_a)
        out.b = np.asarray(cut_b, dtype=np.float64)
    return out
