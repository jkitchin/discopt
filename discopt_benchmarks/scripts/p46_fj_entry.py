"""highs-parity P4.6 entry: does a standalone Feasibility Jump find a point where the
Rust MILP route found none?

Feasibility Jump (Luteberget & Sartor 2023) is an LP-free primal heuristic: rows
``lo <= a x <= hi`` are split into ``<=`` rows with weights ``w_i``; the search
minimises ``sum_i w_i max(0, a_i x - b_i)`` by moving one variable at a time to its
*jump value* -- the value in its domain (integral for an integer column) minimising the
weighted violation with every other variable held -- and, at a local minimum, bumps the
weights of the violated rows. This prototype uses the paper's "focused" selection: pick
up to 25 violated rows at random and move the best-scoring variable in them (plus a few
random columns); when no candidate improves, increase the weights of every violated row by 1.

It is a Python prototype, so it is budgeted in *moves* as well as wall time, and both are
recorded: the Rust implementation P4.6 would build runs the same moves orders of
magnitude faster, so "found within N moves" is the transferable quantity.

Every returned point is re-checked against the instance's original rows, column
bounds and integrality (``verify``) before it counts. Prints executed / verified counts
and exits non-zero when nothing ran (CLAUDE.md section 6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812
import miplib_panel as M  # noqa: E402, N812

INF = 1e20
FEAS_TOL = 1e-6
INT_TOL = 1e-6


def verify(inst, x: np.ndarray) -> tuple[bool, float]:
    """Max violation over rows, bounds and integrality; ``ok`` when <= tolerances."""
    ax = inst.A @ x
    lo = np.where(np.isfinite(inst.row_lo), inst.row_lo, -np.inf)
    hi = np.where(np.isfinite(inst.row_hi), inst.row_hi, np.inf)
    rv = np.maximum(np.maximum(lo - ax, ax - hi), 0.0)
    rv = rv / (1.0 + np.maximum(np.abs(np.nan_to_num(lo, posinf=0, neginf=0)),
                                np.abs(np.nan_to_num(hi, posinf=0, neginf=0))))  # fmt: skip
    bv = np.maximum(np.maximum(inst.col_lo - x, x - inst.col_hi), 0.0)
    iv = np.abs(x - np.round(x)) * inst.is_int
    worst = float(max(rv.max(initial=0), bv.max(initial=0)))
    return worst <= FEAS_TOL and float(iv.max(initial=0)) <= INT_TOL, worst


class FJ:
    def __init__(self, inst, seed: int):
        amat = inst.A.tocsr()
        rows, bs = [], []
        # Split ranged / equality rows into `a x <= b` rows.
        for i in range(amat.shape[0]):
            lo, hi = inst.row_lo[i], inst.row_hi[i]
            if np.isfinite(hi) and hi < INF:
                rows.append((i, 1.0))
                bs.append(hi)
            if np.isfinite(lo) and lo > -INF:
                rows.append((i, -1.0))
                bs.append(-lo)
        idx = np.array([r for r, _ in rows], dtype=np.int64)
        sgn = np.array([s for _, s in rows])
        self.G = sp.csr_matrix(sp.diags(sgn) @ amat[idx])
        self.Gc = self.G.tocsc()
        self.b = np.array(bs, dtype=np.float64)
        self.lb = np.where(np.isfinite(inst.col_lo), inst.col_lo, -INF)
        self.ub = np.where(np.isfinite(inst.col_hi), inst.col_hi, INF)
        self.is_int = inst.is_int.astype(bool)
        self.rng = np.random.default_rng(seed)
        # Start: the bound nearest 0 (0 when it is inside the box).
        x = np.clip(np.zeros(len(self.lb)), self.lb, self.ub)
        x[self.is_int] = np.ceil(x[self.is_int] - 1e-9)
        self.x = x
        self.act = self.G @ x
        self.w = np.ones(len(self.b))

    def _col(self, j):
        s, e = self.Gc.indptr[j], self.Gc.indptr[j + 1]
        return self.Gc.indices[s:e], self.Gc.data[s:e]

    def _f(self, j, t, ri, a):
        """Weighted violation over column j's rows with x_j = t."""
        v = self.act[ri] + a * (t - self.x[j]) - self.b[ri]
        return float(np.sum(self.w[ri] * np.maximum(v, 0.0)))

    def jump(self, j):
        ri, a = self._col(j)
        xj = self.x[j]
        cands = [xj]
        if len(ri):
            tight = xj + (self.b[ri] - self.act[ri]) / a
            cands.extend(tight.tolist())
        lo, hi = self.lb[j], self.ub[j]
        if lo > -INF:
            cands.append(lo)
        if hi < INF:
            cands.append(hi)
        best_t, best_f = xj, self._f(j, xj, ri, a) if len(ri) else 0.0
        f0 = best_f
        for t in cands:
            t = min(max(t, lo), hi)
            ts = (np.floor(t + 1e-9), np.ceil(t - 1e-9)) if self.is_int[j] else (t,)
            for tt in ts:
                tt = min(max(tt, lo), hi)
                if self.is_int[j] and abs(tt - round(tt)) > 1e-9:
                    continue
                if tt == xj:
                    continue
                f = self._f(j, tt, ri, a)
                if f < best_f - 1e-12:
                    best_t, best_f = tt, f
        return best_t, f0 - best_f

    def move(self, j, t):
        ri, a = self._col(j)
        self.act[ri] += a * (t - self.x[j])
        self.x[j] = t

    def run(self, max_moves: int, wall: float):
        t0 = time.perf_counter()
        moves = bumps = 0
        n = len(self.x)
        while moves < max_moves and time.perf_counter() - t0 < wall:
            viol = np.flatnonzero(self.act - self.b > FEAS_TOL * (1 + np.abs(self.b)))
            if viol.size == 0:
                return self.x.copy(), moves, bumps, time.perf_counter() - t0
            # Candidates from up to 25 violated rows (the paper samples its good set);
            # weights are bumped only when none of them has an improving move.
            pick = viol if viol.size <= 25 else self.rng.choice(viol, 25, replace=False)
            cand = set()
            for i in pick:
                s, e = self.G.indptr[i], self.G.indptr[i + 1]
                cand.update(self.G.indices[s:e].tolist())
            cand.update(self.rng.integers(0, n, size=min(5, n)).tolist())
            best = (0.0, None, None)
            for j in cand:
                t, sc = self.jump(j)
                if sc > best[0] + 1e-12:
                    best = (sc, j, t)
            if best[1] is None:
                self.w[viol] += 1.0
                bumps += 1
                continue
            self.move(best[1], best[2])
            moves += 1
        return None, moves, bumps, time.perf_counter() - t0


def rust_run(inst, seed: int, max_moves: int, wall: float):
    """The Rust implementation (``bnb/primal_fj.rs``) on the same instance."""
    from discopt._rust import fj_search_py

    a = sp.csr_matrix(inst.A, dtype=np.float64)
    a.sort_indices()
    t0 = time.perf_counter()
    x, moves, bumps = fj_search_py(
        a.shape[0], a.shape[1], a.indptr.astype(np.int64), a.indices.astype(np.int64),
        a.data.astype(np.float64), np.asarray(inst.row_lo, dtype=np.float64),
        np.asarray(inst.row_hi, dtype=np.float64), np.asarray(inst.col_lo, dtype=np.float64),
        np.asarray(inst.col_hi, dtype=np.float64), np.asarray(inst.is_int, dtype=bool),
        seed=seed, max_moves=max_moves, time_limit_s=wall,
    )  # fmt: skip
    return (None if x is None else np.asarray(x)), moves, bumps, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="neos-1425699,enlight_hard,neos-2624317-amur")
    ap.add_argument("--panel-json", default=str(M.PANEL_JSON))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-moves", type=int, default=200_000)
    ap.add_argument("--wall", type=float, default=60.0)
    ap.add_argument("--engine", choices=("python", "rust"), default="rust")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = {m["name"]: m for m in M._load_panel(Path(a.panel_json))["members"]}
    rows, executed, verified = [], 0, 0
    for name in a.instances.split(","):
        inst = L.read_instance(str(REPO / man[name]["path"]))
        for seed in range(a.seeds):
            if a.engine == "rust":
                x, moves, bumps, wall = rust_run(inst, seed, a.max_moves, a.wall)
            else:
                x, moves, bumps, wall = FJ(inst, seed).run(a.max_moves, a.wall)
            executed += 1
            ok, worst = (False, None) if x is None else verify(inst, x)
            verified += ok
            obj = None if x is None else float(inst.c @ x + inst.offset)
            row = {"name": name, "engine": a.engine, "seed": seed, "found": x is not None,
                   "verified": ok,
                   "worst_violation": worst, "objective": obj, "opt": man[name]["oracle_objective"],
                   "moves": moves, "weight_bumps": bumps, "wall": wall}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1) + "\n")
    print(f"p46 entry: executed={executed} verified={verified}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
