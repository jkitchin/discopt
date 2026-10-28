#!/usr/bin/env python3
"""#2021 section 3 entry experiment: a diagonal-perspective split of a dense convex
quadratic row, built OUTSIDE the library and injected into the OA master's cuts.

For a convex row ``0.5 x'Qx + c'x + k <= 0`` whose quadratic columns ``x_S`` are each
defined by one linear equality ``x_S = F w + f0`` over other columns ``w``, the
quadratic part is ``w'Mw + (linear)`` with ``M = 0.5 F'Q_SS F``. With ``D = t diag(M)``
on the ``w_i`` that are semicontinuous (``lb >= 0``, a row ``w_i <= U_i z_i`` with
``z_i`` binary) and ``M - D`` PSD, the tangent at ``x^ = F w^ + f0`` plus
``coeffs[z_i] -= d_i w^_i^2`` and ``rhs -= d_i w^_i^2`` is valid for every feasible
point: the tangent of the PSD remainder plus the perspective cut of each
``d_i w_i^2`` (``z_i = 1``: the shift cancels; ``z_i = 0``: ``w_i = 0``).

Injection: ``oa._append_master_cut`` is wrapped. A cut whose ``constraint_id`` is a
detected row is REPLACED by the cut above taken at its supporting point projected
through ``F`` (the tangent is recomputed at ``x^``, never reused), so the probe
emits only rows that the algebra proves valid. Arms ``off``/``on`` run in their own
subprocesses, interleaved. Prints per-solve progress, the master trees' nodes and
restarts, and an EXECUTED count (solves whose ON arm applied at least one cut);
exits non-zero when nothing ran or an ON arm applied none.

    python -u issue2021_perspective_row_probe.py --nl-dir <dir> \\
        --instances portfol_classical050_1,... --time-limit 30 --reps 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("PROBE_REPO", Path.cwd()))


def detect(model):
    """``[(row, S, F, f0, W, d, zcols, Q, c, k)]`` for every row the split applies to."""
    import numpy as np
    from discopt._relax.perspective import _semicontinuity_rows
    from discopt._relax.problem_classifier import (
        _EXTRACTION_DECLINES,
        _extract_linear_coefficients_sparse,
        _extract_quadratic_coefficients,
        dense_Q,
    )
    from discopt._relax.term_classifier import _compute_var_offset

    n = sum(v.size for v in model._variables)
    lb = np.empty(n)
    for v in model._variables:
        base = _compute_var_offset(v, model)
        lb[base : base + v.size] = np.ravel(np.broadcast_to(np.asarray(v.lb, float), v.shape))
    semi = _semicontinuity_rows(model)
    eq_rows = []
    for con in model._constraints:
        if con.sense != "==":
            continue
        try:
            terms, const = _extract_linear_coefficients_sparse(con.body, model, n)
        except _EXTRACTION_DECLINES:
            continue
        eq_rows.append((terms, const))
    out = []
    for r, con in enumerate(model._constraints):
        if con.sense != "<=":
            continue
        try:
            Q, c, k = _extract_quadratic_coefficients(con.body, model, n)
        except _EXTRACTION_DECLINES:
            continue
        Q = dense_Q(Q)
        S = np.flatnonzero(np.abs(Q).sum(axis=1) > 0)
        if S.size < 2:
            continue
        QSS = Q[np.ix_(S, S)]
        if np.linalg.eigvalsh(0.5 * (QSS + QSS.T))[0] < -1e-10 * max(1.0, abs(QSS).max()):
            continue  # not convex
        sset = set(int(j) for j in S)
        defs = {}
        for terms, const in eq_rows:
            hit = [j for j in terms if j in sset]
            if len(hit) != 1 or hit[0] in defs:
                continue
            j = hit[0]
            defs[j] = (terms, const)
        if len(defs) != S.size:
            continue
        wcols = sorted({i for j in defs for i in defs[j][0] if i not in sset})
        widx = {i: t for t, i in enumerate(wcols)}
        F = np.zeros((S.size, len(wcols)))
        f0 = np.zeros(S.size)
        for a, j in enumerate(S):
            terms, const = defs[int(j)]
            aj = terms[int(j)]
            for i, v in terms.items():
                if i != j:
                    F[a, widx[i]] = -v / aj
            f0[a] = -const / aj
        M = 0.5 * F.T @ QSS @ F
        M = 0.5 * (M + M.T)
        elig = np.array([i in semi and lb[i] >= 0.0 for i in wcols])
        if not elig.any():
            continue
        diag = np.where(elig, np.diag(M), 0.0)
        margin = 1e-9 * max(1.0, float(np.abs(np.diag(M)).max()))
        if np.linalg.eigvalsh(M)[0] < margin:
            # M itself must clear the margin before any of it can be split off.
            lo_ok = False
        else:
            lo_ok = True
        if not lo_ok:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if np.linalg.eigvalsh(M - np.diag(mid * diag))[0] >= margin:
                lo = mid
            else:
                hi = mid
        t = lo * 0.999
        d = t * diag
        np.linalg.cholesky(M - np.diag(d) - margin * np.eye(M.shape[0]))  # certificate
        zcols = []
        for i in wcols:
            if i in semi:
                ind, e, _u = semi[i]
                zcols.append(_compute_var_offset(ind, model) + e)
            else:
                zcols.append(-1)
        out.append(
            dict(
                row=r,
                S=S,
                F=F,
                f0=f0,
                W=np.array(wcols),
                d=d,
                z=np.array(zcols),
                Q=Q,
                c=np.asarray(c, float),
                k=float(k),
                t=t,
            )
        )
    return out


def _cut(rowinfo, p, n_vars):
    import numpy as np

    S, F, f0, W, d, z, Q, c, k = (
        rowinfo[key] for key in ("S", "F", "f0", "W", "d", "z", "Q", "c", "k")
    )
    wbar = np.asarray(p, float)[W]
    xhat = F @ wbar + f0
    QSS = Q[np.ix_(S, S)]
    coeffs = np.array(c[:n_vars], dtype=float)
    coeffs[S] += QSS @ xhat
    rhs = -k + 0.5 * float(xhat @ QSS @ xhat)
    applied = 0
    for di, wi, zi in zip(d, wbar, z):
        if di > 0 and zi >= 0 and wi > 0:
            sh = di * wi * wi
            coeffs[zi] -= sh
            rhs -= sh
            applied += 1
    return coeffs, rhs, applied


def _solve(nl: str, tl: float, arm: str) -> None:
    import discopt
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.solvers import milp_highs as mh
    from discopt.solvers import oa
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(REPO)), (discopt.__file__, REPO)
    model = from_nl(nl)
    rows = detect(model)
    by_row = {ri["row"]: ri for ri in rows}
    applied = [0, 0, 0, 0]  # cuts replaced, z shifts, id checks passed, id checks failed
    if arm == "on":
        orig = oa._append_master_cut

        def _wrapped(a_rows, b_rows, coeffs, rhs, *args, **kw):
            cid = kw.get("constraint_id")
            sp = kw.get("supporting_point")
            if cid in by_row and sp is not None and len(coeffs) >= len(sp) - 1:
                n_vars = len(coeffs)
                # The id must name the row: the library's own tangent at ``sp`` must
                # equal this row's tangent at ``sp`` (no shift, x^ = sp[S]).
                ri = by_row[cid]
                QSS = ri["Q"][np.ix_(ri["S"], ri["S"])]
                xs = np.asarray(sp, float)[ri["S"]]
                mine = np.array(ri["c"][:n_vars], dtype=float)
                mine[ri["S"]] += QSS @ xs
                scale = max(1.0, float(np.abs(mine).max()))
                if np.allclose(coeffs, mine, atol=1e-7 * scale):
                    applied[2] += 1
                else:
                    applied[3] += 1
                    return orig(a_rows, b_rows, coeffs, rhs, *args, **kw)
                new, new_rhs, k = _cut(by_row[cid], sp, n_vars)
                if k:
                    coeffs, rhs = new, new_rhs
                    applied[0] += 1
                    applied[1] += k
            return orig(a_rows, b_rows, coeffs, rhs, *args, **kw)

        oa._append_master_cut = _wrapped
    masters = []
    orig_lazy = mh.solve_milp_with_lazy_cuts

    def _lazy(*a, **k):
        res = orig_lazy(*a, **k)
        cs = res.callback_stats or {}
        masters.append(
            {
                "restarts": cs.get("restarts"),
                "nodes": res.node_count,
                "bound": res.bound,
                "status": res.status.value,
            }
        )
        return res

    mh.solve_milp_with_lazy_cuts = _lazy
    t0 = time.perf_counter()
    r = model.solve(time_limit=tl, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "gap_certified": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "nodes": r.node_count,
                "wall": wall,
                "route": getattr(r, "algorithm_route", None),
                "verify_failed": bad,
                "rows_detected": [
                    (ri["row"], int(ri["S"].size), int((ri["d"] > 0).sum()), ri["t"]) for ri in rows
                ],
                "cuts_replaced": applied[0],
                "z_shifts": applied[1],
                "id_ok": applied[2],
                "id_mismatch": applied[3],
                "masters": masters,
            },
            default=str,
        ),
        flush=True,
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        _solve(sys.argv[2], float(sys.argv[3]), sys.argv[4])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl-dir", type=Path, required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = [s for s in a.instances.split(",") if s]
    executed = 0
    on_unfired = 0
    k = 0
    with a.out.open("a") as fh:
        for rep in range(a.reps):
            for name in names:
                arms = ("off", "on") if k % 2 == 0 else ("on", "off")
                k += 1
                for arm in arms:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            __file__,
                            "--_solve",
                            str(a.nl_dir / f"{name}.nl"),
                            str(a.time_limit),
                            arm,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=a.time_limit + 240,
                        env=dict(os.environ, PROBE_REPO=str(REPO)),
                    )
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                    if not lines:
                        res = {"status": "CRASH", "rc": proc.returncode, "err": proc.stderr[-3000:]}
                    else:
                        res = json.loads(lines[-1][7:])
                    res.update(name=name, arm=arm, rep=rep)
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    if arm == "on":
                        if res.get("cuts_replaced"):
                            executed += 1
                        else:
                            on_unfired += 1
                    ms = res.get("masters") or []
                    print(
                        f"{name} rep={rep} arm={arm} {res['status']} cert={res.get('gap_certified')} "
                        f"obj={res.get('objective')} bound={res.get('bound')} "
                        f"wall={res.get('wall')} rows={res.get('rows_detected')} "
                        f"replaced={res.get('cuts_replaced')} shifts={res.get('z_shifts')} "
                        f"id_ok={res.get('id_ok')} id_bad={res.get('id_mismatch')} "
                        f"masters={[(m['restarts'], m['nodes'], m['bound']) for m in ms]} "
                        f"verify_failed={res.get('verify_failed')} "
                        f"err={res.get('err', '')[-400:]}",
                        flush=True,
                    )
    print(f"EXECUTED {executed}; ON solves that applied no cut: {on_unfired}")
    if executed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
