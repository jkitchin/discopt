"""Exact infeasibility oracle for block-structured covering models (IIS engine).

Some infeasible MILPs are a covering argument in disguise: ``B`` blocks of a few
integer columns each, and a set of *linking* rows ``sum_j a_j x_j >= b`` over
binaries, each binary switched on by one row of one block. ``no-ip-65059`` of the
MIPcc27 set is the type: 9 identical blocks of 5 general integers, 243 linking
rows, infeasible because 243 points need 10 of the blocks' inequalities to cover
them and only 9 exist. Branch and bound does not prove that in minutes (the
blocks are interchangeable, so every subtree recurs 9! times) and neither does
CDCL on the order encoding.

This module decides such a system exactly, by enumerating each block:

1. **Structure** (:class:`BlockMaster`, from the matrix alone). A *linking row* has
   at least two columns, all binary, all coefficients positive, a positive lower
   bound and no upper bound. Every other row is a *block row*. A *satellite* is a
   column of a linking row that lies in exactly one block row (one satellite per
   block row). The remaining columns of block rows are *core* columns; they must be
   integer with a finite box. A core column may itself lie in linking rows (a
   pigeonhole model has no satellites at all). Blocks are the connected components
   of core columns through block rows.
2. **Enumeration.** For one block and one subsystem, every assignment of its core
   columns is tested against the block's active rows without a satellite; each
   active row with a satellite gives that satellite an interval. An assignment is
   valid when every interval is non-empty; its *pattern* is the largest value each
   satellite can take (linking coefficients are positive, so larger is better),
   plus the assignment's own contribution to the linking rows its core columns lie
   in.
3. **Master.** Blocks with the same set of patterns are aggregated. The subsystem
   is feasible iff an integer choice of one pattern per block meets every active
   linking row: a small covering IP, solved with presolve on and off (an
   infeasibility claim stands only when both agree, the #1634 standard), and a
   feasible verdict is turned into a point of the original subsystem that is
   re-checked by arithmetic before it is returned.

All block and linking data must be integer; the enumeration is ``int64`` arithmetic
with exact floor/ceiling division, so the only floating-point step is the master.
A block outside the rule (a relaxed core column, an unbounded core box, a product
of domains above the cap, a satellite whose best value is negative) is not
enumerated: it enters the master *verbatim*, as its own rows and columns, for at
most ``max_verbatim`` blocks and never for all of them. Past that the oracle
raises :class:`Unsupported` and the caller has no verdict from it.

**Implied integers** (``implied_int``). A core column whose integrality the probe
relaxed is still enumerated over integers when every active row holding it has
coefficient +-1 on it and only held-integer columns besides: with the others fixed
at integers the column's feasible set is an interval with integer endpoints, so an
integer value exists whenever a real one does. Its domain is then its active bounds
intersected with the bounds its rows imply.

**Symmetry** (:meth:`BlockMaster.classes`, :meth:`BlockMaster.lex_rows`). Two
blocks are *interchangeable in a subsystem* when a column bijection between them
maps the subsystem's rows, bounds and integralities onto themselves and fixes
every linking row. That is checked on the probed subsystem, never inherited from
the full model: deleting one row of one block breaks it. For a class of mutually
interchangeable blocks any ordering ``f(block_i) >= f(block_{i+1})`` by one linear
function of the core columns loses no solution (sort the blocks by ``f``), which
is what lets an ordinary MILP solve prove these systems; :func:`lex_decide` is
that solve, used as an independent cross-check of the master and for subsystems
the master cannot enumerate.

Measured on ``no-ip-65059`` (2547 rows, 2232 columns): structure and enumeration
0.08 s; the full model proven infeasible at the master's root in under a second
(80 master columns after aggregation), where HiGHS and SCIP without symmetry
handling do not finish in 300 s; HiGHS with the 8 ordering rows needs 2,742 nodes.

**How often the rule applies.** Rarely. It arms on 3 of the 50 MIPcc27 smallset
instances (``no-ip-65059``, ``hole9``, ``hole10``), on 2 of 43 MIPLIB 2017
infeasible instances (both ``no-ip``) and on none of 219 general MIPLIB instances.
The usual refusals are "no linking row" and "a single block". A smaller answer was
measured on the ``no-ip`` family only; on ``hole9`` the answer is the same size
with fewer full solves. On the 50-pair panel of 2026-11-26 the stack changed one
answer (a certified IIS where there was none) and made none worse, so the switches
in :func:`discopt._iis_milp._block_flags` are default ON, except the ordering rows
(``DISCOPT_IIS_BLOCK_LEX``), which cost more than they returned and stay an opt-in.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from discopt import _iis_backend as _bk

_INF = 1e20  # discopt's unbounded-bound sentinel (``math.inf`` is also "none")


class Unsupported(Exception):
    """The system is outside the rule this oracle decides exactly."""


def _finite(v) -> Any:
    return np.abs(v) < _INF


def _is_integral(v) -> bool:
    v = np.asarray(v, dtype=np.float64)
    f = v[np.abs(v) < _INF]
    return bool(np.all(f == np.rint(f)) and np.all(np.abs(f) < 2**40))


@dataclass
class _Block:
    cols: list[int]  # core columns (canonical order when ``canon``)
    rows: np.ndarray  # block rows
    coef: np.ndarray  # (len(rows), len(cols)) int64 core coefficients
    sat: np.ndarray  # satellite column of each row, or -1
    satcoef: np.ndarray  # its coefficient
    allcols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sats: list[int] = field(default_factory=list)  # every satellite of the block
    canon: bool = False
    rowsig: list[tuple] = field(default_factory=list)  # canonical tuple of each row
    satkey: dict[int, tuple] = field(default_factory=dict)


class BlockMaster:
    """The oracle for one model (see the module docstring)."""

    def __init__(
        self,
        A: Any,
        row_lo: np.ndarray,
        row_hi: np.ndarray,
        col_lb: np.ndarray,
        col_ub: np.ndarray,
        is_int: np.ndarray,
        engine: str,
        *,
        implied_int: bool = False,
        max_enum: int = 200_000,
        max_cells: int = 30_000_000,
        max_master_cols: int = 20_000,
        max_verbatim: int = 2,
    ) -> None:
        A = sp.csr_matrix(A, dtype=np.float64)
        A.eliminate_zeros()
        A.sort_indices()
        self.A = A
        m, n = A.shape
        self.m, self.n = m, n
        self.engine = engine
        self.implied_int = implied_int
        self.max_enum, self.max_cells, self.max_master_cols = max_enum, max_cells, max_master_cols
        self.max_verbatim = max_verbatim
        self.n_verbatim = 0  # master solves that carried a block as its own rows
        self.col_lb = np.asarray(col_lb, dtype=np.float64)
        self.col_ub = np.asarray(col_ub, dtype=np.float64)
        is_int = np.asarray(is_int, dtype=bool)
        self.n_calls = 0
        self.n_unsupported = 0
        self.seconds = 0.0
        self.last_master: tuple[int, int] = (0, 0)  # (columns, groups) of the last master
        self._cache: dict[tuple, Any] = {}
        self._pcache: dict[tuple, Any] = {}

        isbin = is_int & (self.col_lb == 0) & (self.col_ub == 1)
        ip, ix, dat = A.indptr, A.indices, A.data
        rowcols = [ix[ip[r] : ip[r + 1]] for r in range(m)]
        rowvals = [dat[ip[r] : ip[r + 1]] for r in range(m)]
        self.rowcols, self.rowvals = rowcols, rowvals
        link = [
            r
            for r in range(m)
            if rowcols[r].size >= 2
            and bool(np.all(isbin[rowcols[r]]))
            and bool(np.all(rowvals[r] > 0))
            and row_hi[r] >= _INF
            and 0 < row_lo[r] < _INF
        ]
        if not link:
            raise Unsupported("no linking row")
        is_link = np.zeros(m, dtype=bool)
        is_link[link] = True
        brow = [r for r in range(m) if not is_link[r] and rowcols[r].size]
        if not brow:
            raise Unsupported("no block row")
        in_link = np.zeros(n, dtype=bool)
        for r in link:
            in_link[rowcols[r]] = True
        cnt = np.zeros(n, dtype=np.int64)
        for r in brow:
            cnt[rowcols[r]] += 1
        # One satellite per block row: a linking column that lies in this block row only.
        satrow: dict[int, int] = {}
        for r in brow:
            for j in rowcols[r]:
                if cnt[j] == 1 and in_link[j]:
                    satrow[int(j)] = r
                    break
        self.satrow = satrow
        in_brow = cnt > 0
        core = [int(j) for j in np.flatnonzero(in_brow) if int(j) not in satrow]
        if not core:
            raise Unsupported("no core column")
        core_a = np.asarray(core, dtype=np.int64)
        if not np.all(is_int[core_a]):
            raise Unsupported("a continuous core column")
        if not (np.all(_finite(self.col_lb[core_a])) and np.all(_finite(self.col_ub[core_a]))):
            raise Unsupported("a core column with an infinite box")
        used = np.flatnonzero(in_brow | in_link)
        if not (
            _is_integral(dat)
            and _is_integral(row_lo)
            and _is_integral(row_hi)
            and _is_integral(self.col_lb[used])
            and _is_integral(self.col_ub[used])
        ):
            raise Unsupported("non-integer data")

        par = {j: j for j in core}

        def find(x: int) -> int:
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        rowcore: dict[int, list[int]] = {}
        for r in brow:
            cs = [int(j) for j in rowcols[r] if int(j) not in satrow]
            if not cs:
                raise Unsupported("a block row without a core column")
            rowcore[r] = cs
            for a in cs[1:]:
                par[find(a)] = find(cs[0])
        comp: dict[int, list[int]] = defaultdict(list)
        for j in core:
            comp[find(j)].append(j)
        if len(comp) < 2:
            raise Unsupported("a single block")
        blocks_cols = sorted(comp.values())
        bid = {j: b for b, cs in enumerate(blocks_cols) for j in cs}
        brows: dict[int, list[int]] = defaultdict(list)
        for r in brow:
            brows[bid[rowcore[r][0]]].append(r)
        # sat -> [(linking row, coefficient)]
        self.satlink: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.free_link: dict[int, list[tuple[int, int]]] = defaultdict(list)
        # core column -> [(linking row, coefficient)]: a core column may itself lie
        # in linking rows (a pigeonhole model has no satellites at all); it then
        # contributes its own value to them.
        self.corelink: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in link:
            for j, v in zip(rowcols[r], rowvals[r]):
                jj = int(j)
                tgt = (
                    self.satlink if jj in satrow else self.corelink if jj in par else self.free_link
                )
                tgt[jj].append((r, int(v)))
        self.link = np.asarray(link, dtype=np.int64)
        self.is_link = is_link
        self.row_block = np.full(m, -1, dtype=np.int64)
        self.col_block = np.full(n, -1, dtype=np.int64)
        self.blocks: list[_Block] = []
        for b, cs in enumerate(blocks_cols):
            size = 1
            for j in cs:
                size *= int(self.col_ub[j] - self.col_lb[j]) + 1
            if size > max_enum:
                raise Unsupported(f"block {b}: {size} assignments")
            self.blocks.append(self._make_block(b, cs, brows[b]))

    # ------------------------------------------------------------------ structure

    def _make_block(self, b: int, cs: list[int], rows: list[int]) -> _Block:
        satrow, satlink = self.satrow, self.satlink
        sats = [j for j, r in satrow.items() if r in set(rows)]
        satkey = {j: tuple(sorted(satlink[j])) for j in sats}
        rowsat = {satrow[j]: j for j in sats}
        # Canonical column order: a core column is identified by its coefficients on
        # the rows of the block's satellites, each row named by what its satellite
        # links to. Blocks that are copies of each other get the same order.
        canon = len(set(satkey.values())) == len(sats)
        colsig: dict[int, tuple] = {}
        if canon:
            acc: dict[int, list] = {j: [] for j in cs}
            for r in rows:
                if r not in rowsat:
                    continue
                key = satkey[rowsat[r]]
                for j, v in zip(self.rowcols[r], self.rowvals[r]):
                    if int(j) in acc:
                        acc[int(j)].append((key, int(v)))
            for j in cs:
                colsig[j] = (
                    tuple(sorted(acc[j])),
                    tuple(sorted(self.corelink.get(j, ()))),
                    float(self.col_lb[j]),
                    float(self.col_ub[j]),
                )
            canon = len(set(colsig.values())) == len(cs)
        if canon:
            cs = sorted(cs, key=lambda j: colsig[j])
        pos = {j: k for k, j in enumerate(cs)}
        coef = np.zeros((len(rows), len(cs)), dtype=np.int64)
        sat = np.full(len(rows), -1, dtype=np.int64)
        satcoef = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            for j, v in zip(self.rowcols[r], self.rowvals[r]):
                if int(j) in pos:
                    coef[i, pos[int(j)]] = int(v)
                else:
                    sat[i] = int(j)
                    satcoef[i] = int(v)
        blk = _Block(cs, np.asarray(rows, dtype=np.int64), coef, sat, satcoef)
        blk.sats = sorted(sats)
        blk.allcols = np.asarray(cs + blk.sats, dtype=np.int64)
        blk.canon = canon
        blk.satkey = satkey
        if canon:
            blk.rowsig = [
                (
                    tuple(int(v) for v in coef[i]),
                    satkey[int(sat[i])] if sat[i] >= 0 else None,
                    int(satcoef[i]),
                )
                for i in range(len(rows))
            ]
        self.row_block[blk.rows] = b
        self.col_block[blk.allcols] = b
        return blk

    def describe(self) -> dict[str, Any]:
        return {
            "linking_rows": int(self.link.size),
            "blocks": len(self.blocks),
            "core_columns": sum(len(b.cols) for b in self.blocks),
            "satellites": len(self.satrow),
            "canonical_blocks": sum(1 for b in self.blocks if b.canon),
        }

    # ---------------------------------------------------------------- enumeration

    def _domain(self, blk: _Block, k: int, rlo, rhi, lb, ub, integ) -> tuple[int, int]:
        """The integer domain of core column ``k`` of ``blk`` in this subsystem."""
        j = blk.cols[k]
        lo, hi = float(lb[j]), float(ub[j])
        if integ[j]:
            if not (abs(lo) < _INF and abs(hi) < _INF):
                raise Unsupported("a held core column with an infinite box")
            return int(lo), int(hi)
        if not self.implied_int:
            raise Unsupported("a relaxed core column")
        if self.corelink.get(j):
            # Its linking rows hold columns of other blocks; the interval argument
            # below is about the block's own rows only.
            raise Unsupported("a relaxed core column in a linking row")
        # Implied integer: +-1 on this column and held integers besides, in every
        # active row that holds it. The bounds those rows imply close its domain.
        lo = lo if abs(lo) < _INF else -math.inf
        hi = hi if abs(hi) < _INF else math.inf
        for i in np.flatnonzero(blk.coef[:, k]):
            r = int(blk.rows[i])
            fl, fu = abs(rlo[r]) < _INF, abs(rhi[r]) < _INF
            if not (fl or fu):
                continue
            a = int(blk.coef[i, k])
            if abs(a) != 1:
                raise Unsupported("a relaxed core column with a non-unit coefficient")
            omin = omax = 0.0  # range of the row's other terms
            others = [
                (blk.cols[t], int(blk.coef[i, t])) for t in np.flatnonzero(blk.coef[i]) if t != k
            ]
            if blk.sat[i] >= 0:
                others.append((int(blk.sat[i]), int(blk.satcoef[i])))
            for jo, v in others:
                if not integ[jo]:
                    raise Unsupported("two relaxed columns in one row")
                l2 = float(lb[jo]) if abs(lb[jo]) < _INF else -math.inf
                u2 = float(ub[jo]) if abs(ub[jo]) < _INF else math.inf
                omin += v * l2 if v > 0 else v * u2
                omax += v * u2 if v > 0 else v * l2
            # rlo <= a x + others <= rhi
            xl = (rlo[r] - omax) if fl else -math.inf
            xu = (rhi[r] - omin) if fu else math.inf
            if a < 0:
                xl, xu = -xu, -xl
            lo, hi = max(lo, xl), min(hi, xu)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise Unsupported("a relaxed core column its rows do not bound")
        return int(math.ceil(lo)), int(math.floor(hi))

    def _patterns(self, b: int, rlo, rhi, lb, ub, integ):
        """``(E, ok, hi, lo)`` of block ``b``: the assignments, which are valid, and
        per satellite with an active row its largest / smallest value on each
        assignment as ``(numerator array or None for unbounded, denominator)``."""
        blk = self.blocks[b]
        key = (
            b,
            rlo[blk.rows].tobytes(),
            rhi[blk.rows].tobytes(),
            lb[blk.allcols].tobytes(),
            ub[blk.allcols].tobytes(),
            integ[blk.allcols].tobytes(),
        )
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        doms = []
        size = 1
        for k in range(len(blk.cols)):
            lo_k, hi_k = self._domain(blk, k, rlo, rhi, lb, ub, integ)
            if hi_k < lo_k:
                res: tuple[Any, ...] = (
                    np.zeros((0, len(blk.cols)), dtype=np.int64),
                    np.zeros(0, dtype=bool),
                    {},
                    {},
                    key,
                )
                self._cache[key] = res
                return res
            doms.append(np.arange(lo_k, hi_k + 1, dtype=np.int64))
            size *= doms[-1].size
        if size > self.max_enum or size * max(1, blk.rows.size) > self.max_cells:
            raise Unsupported(f"block {b}: {size} assignments")
        E = np.array(list(itertools.product(*doms)), dtype=np.int64).reshape(size, len(doms))
        ok = np.ones(size, dtype=bool)
        hi: dict[int, tuple[Optional[np.ndarray], int]] = {}
        lo: dict[int, tuple[Optional[np.ndarray], int]] = {}
        for i, r in enumerate(blk.rows):
            fl, fu = abs(rlo[r]) < _INF, abs(rhi[r]) < _INF
            if not (fl or fu):
                continue
            act = E @ blk.coef[i]
            j = int(blk.sat[i])
            if j < 0:
                if fl:
                    ok &= act >= int(rlo[r])
                if fu:
                    ok &= act <= int(rhi[r])
                continue
            a = int(blk.satcoef[i])
            den = abs(a)
            # rlo <= act + a x <= rhi, as numerators over ``den``
            if a > 0:
                lo_n = int(rlo[r]) - act if fl else None
                hi_n = int(rhi[r]) - act if fu else None
            else:
                lo_n = act - int(rhi[r]) if fu else None
                hi_n = act - int(rlo[r]) if fl else None
            jl, ju = abs(lb[j]) < _INF, abs(ub[j]) < _INF
            if integ[j]:
                if lo_n is not None:
                    lo_n = -((-lo_n) // den)  # ceiling
                if hi_n is not None:
                    hi_n = hi_n // den  # floor
                den = 1
            if jl:
                v = np.full(size, int(lb[j]) * den, dtype=np.int64)
                lo_n = v if lo_n is None else np.maximum(lo_n, v)
            if ju:
                v = np.full(size, int(ub[j]) * den, dtype=np.int64)
                hi_n = v if hi_n is None else np.minimum(hi_n, v)
            if lo_n is not None and hi_n is not None:
                ok &= lo_n <= hi_n
            hi[j] = (hi_n, den)
            lo[j] = (lo_n, den)
        res = (E, ok, hi, lo, key)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = res
        return res

    # --------------------------------------------------------------------- decide

    def decide(
        self, rlo, rhi, lb, ub, integ, stop: float
    ) -> tuple[Optional[bool], Optional[np.ndarray]]:
        """``(True, None)`` infeasible, ``(False, x)`` feasible at the verified
        point ``x``, ``(None, None)`` undecided. Raises :class:`Unsupported`.

        The arguments are one probe's arrays: row and column bounds with the
        inactive ones at infinity, and the held-integer mask. ``stop`` is the
        caller's deadline (a ``time.perf_counter`` value); there is no default."""
        t0 = time.perf_counter()
        self.n_calls += 1
        try:
            out: tuple[Optional[bool], Optional[np.ndarray]] = self._decide(
                rlo, rhi, lb, ub, np.asarray(integ, dtype=bool), stop
            )
            return out
        except Unsupported:
            self.n_unsupported += 1
            raise
        finally:
            self.seconds += time.perf_counter() - t0

    def _decide(self, rlo, rhi, lb, ub, integ, stop):
        rlo = np.asarray(rlo, dtype=np.float64)
        rhi = np.asarray(rhi, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        act_link = [int(r) for r in self.link if abs(rlo[r]) < _INF]
        # A linking row is in the rule only as ``>=``; a probe never adds an upper
        # bound, so this is a guard on the caller.
        if any(abs(rhi[r]) < _INF for r in act_link):
            raise Unsupported("a linking row with an upper bound")
        rhs = {r: int(rlo[r]) for r in act_link}
        for j, lst in self.free_link.items():
            if abs(ub[j]) >= _INF:
                for r, _ in lst:
                    rhs.pop(r, None)  # an unbounded column meets the row alone
            else:
                for r, c in lst:
                    if r in rhs:
                        rhs[r] -= c * int(math.floor(ub[j]) if integ[j] else ub[j])
        per_block = []
        dens: dict[int, int] = defaultdict(lambda: 1)
        verb: list[int] = []
        for b, blk in enumerate(self.blocks):
            try:
                E, ok, hi, lo, pkey = self._patterns(b, rlo, rhi, lb, ub, integ)
            except Unsupported:
                # A block the enumeration refuses enters the master as itself (its
                # own rows and columns), which is exact; a few such blocks at most,
                # or the master is the original model again.
                if len(verb) >= min(self.max_verbatim, len(self.blocks) - 1):
                    raise
                verb.append(b)
                continue
            if not ok.any():
                return True, None
            sat_hi: dict[int, tuple[Optional[np.ndarray], int]] = {}
            for j in blk.sats:
                if not any(r in rhs for r, _ in self.satlink[j]):
                    continue
                if j in hi:
                    num, den = hi[j]
                    num = None if num is None else num[ok]
                elif abs(ub[j]) < _INF:  # its block row is not in the subsystem
                    num, den = np.full(int(ok.sum()), int(ub[j]), dtype=np.int64), 1
                else:
                    num, den = None, 1
                if num is not None and np.any(num < 0):
                    raise Unsupported("a satellite whose largest value is negative")
                sat_hi[j] = (num, den)
                for r, _ in self.satlink[j]:
                    if r in rhs:
                        dens[r] = math.lcm(dens[r], den)
            per_block.append((b, E, ok, hi, lo, sat_hi, pkey))
        rows = [r for r in act_link if r in rhs and rhs[r] > 0]
        rpos = {r: i for i, r in enumerate(rows)}
        cap = np.array([rhs[r] * dens[r] for r in rows], dtype=np.int64)
        groups: dict[bytes, list] = defaultdict(list)
        blk_cols = {b: self.blocks[b].cols for b in range(len(self.blocks))}
        sig = (tuple(rows), cap.tobytes(), tuple(dens[r] for r in rows))
        for b, E, ok, hi, lo, sat_hi, pkey in per_block:
            got = self._pcache.get((pkey, sig))
            if got is not None:
                groups[got[0]].append((b, got[1], got[2]))
                continue
            idx = np.flatnonzero(ok)
            P = np.zeros((idx.size, len(rows)), dtype=np.int64)
            for j, (num, den) in sat_hi.items():
                for r, c in self.satlink[j]:
                    i = rpos.get(r)
                    if i is None:
                        continue
                    if num is None:
                        P[:, i] += cap[i]
                    else:
                        P[:, i] += c * num * (dens[r] // den)
            for k, j in enumerate(blk_cols[b]):
                for r, c in self.corelink.get(j, ()):
                    i = rpos.get(r)
                    if i is None:
                        continue
                    v = E[idx, k]
                    if np.any(v < 0):
                        raise Unsupported("a core column of a linking row with a negative value")
                    P[:, i] += c * v * dens[r]
            np.minimum(P, cap[None, :], out=P)  # every term is >= 0: more than the rhs is the rhs
            P, first = np.unique(P, axis=0, return_index=True)
            if P.shape[0] <= 4000:
                keep = [
                    i
                    for i in range(P.shape[0])
                    if not np.any(np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1))
                ]
                P, first = P[keep], first[keep]
            gkey = P.tobytes() + repr(P.shape).encode()
            if len(self._pcache) > 2048:
                self._pcache.clear()
            self._pcache[(pkey, sig)] = (gkey, P, idx[first])
            groups[gkey].append((b, P, idx[first]))
        if not rows and not verb:
            lam = {key: Counter({0: len(g)}) for key, g in groups.items()}
            return self._point(per_block, groups, lam, {}, rlo, rhi, lb, ub, integ)
        ncol = sum(g[0][1].shape[0] for g in groups.values())
        if ncol > self.max_master_cols:
            raise Unsupported(f"master with {ncol} columns")
        self.last_master = (ncol, len(groups))
        ri: list[int] = []
        ci: list[int] = []
        va: list[float] = []
        m_lo: list[float] = []
        m_hi: list[float] = []
        c_ub: list[float] = []
        c0 = 0
        ng = len(groups)
        offs = {}
        for g, (key, members) in enumerate(groups.items()):
            P = members[0][1]
            k = P.shape[0]
            offs[key] = c0
            ri += [g] * k
            ci += list(range(c0, c0 + k))
            va += [1.0] * k
            m_lo.append(float(len(members)))
            m_hi.append(float(len(members)))
            c_ub += [float(len(members))] * k
            nz = np.nonzero(P)
            ri += (ng + nz[1]).tolist()
            ci += (c0 + nz[0]).tolist()
            va += P[nz].astype(np.float64).tolist()
            c0 += k
        m_lo += cap.astype(np.float64).tolist()
        m_hi += [_INF] * len(rows)
        c_lb = [0.0] * ncol
        c_int = [True] * ncol
        vcol: dict[int, int] = {}  # original column of a verbatim block -> master column
        nrow = ng + len(rows)
        for b in verb:
            blk = self.blocks[b]
            for j in blk.allcols:
                vcol[int(j)] = c0
                c_lb.append(float(lb[j]))
                c_ub.append(float(ub[j]))
                c_int.append(bool(integ[j]))
                c0 += 1
            for r in blk.rows:
                if not (abs(rlo[r]) < _INF or abs(rhi[r]) < _INF):
                    continue
                for j, v in zip(self.rowcols[r], self.rowvals[r]):
                    ri.append(nrow)
                    ci.append(vcol[int(j)])
                    va.append(float(v))
                m_lo.append(float(rlo[r]))
                m_hi.append(float(rhi[r]))
                nrow += 1
            for j in blk.allcols:
                jj = int(j)
                for r, c in self.satlink.get(jj, ()) or self.corelink.get(jj, ()):
                    i = rpos.get(r)
                    if i is not None:
                        ri.append(ng + i)
                        ci.append(vcol[jj])
                        va.append(float(c * dens[r]))
        if verb:
            self.n_verbatim += 1
        ncol = c0
        Am = sp.csr_matrix((va, (ri, ci)), shape=(nrow, ncol))
        mlo, mhi = np.asarray(m_lo), np.asarray(m_hi)
        clb, cub = np.asarray(c_lb), np.asarray(c_ub)
        allint = np.asarray(c_int, dtype=bool)
        be = _bk.make_backend(self.engine, Am, mlo, mhi, clb, cub, allint)
        inf = be.inf
        be.configure(
            np.where(mlo <= -_INF, -inf, mlo),
            np.where(mhi >= _INF, inf, mhi),
            np.where(clb <= -_INF, -inf, clb),
            np.where(cub >= _INF, inf, cub),
            allint,
        )
        budget = stop  # the caller's deadline; no budget of its own (#912)
        if budget - time.perf_counter() < 0.05:
            return None, None
        st = be.run("choose", budget)
        if st == _bk.INFEASIBLE:
            if budget - time.perf_counter() < 0.05:
                return None, None
            if be.run("off", budget) == _bk.INFEASIBLE:
                return True, None
            return None, None
        if st != _bk.OPTIMAL:
            return None, None
        raw = np.asarray(be.point(), dtype=np.float64)
        sol = np.rint(raw).astype(np.int64)
        vals = {j: (float(sol[c]) if integ[j] else float(raw[c])) for j, c in vcol.items()}
        lam = {}
        for key, members in groups.items():
            k = members[0][1].shape[0]
            cnt = Counter({i: int(sol[offs[key] + i]) for i in range(k) if sol[offs[key] + i] > 0})
            if sum(cnt.values()) != len(members):
                return None, None
            lam[key] = cnt
        return self._point(per_block, groups, lam, vals, rlo, rhi, lb, ub, integ)

    def _point(self, per_block, groups, lam, vals, rlo, rhi, lb, ub, integ):
        """The original-space point behind a master solution, checked by
        arithmetic against the probe; ``(None, None)`` when the check fails."""
        x = np.zeros(self.n)
        fl, fu = np.abs(lb) < _INF, np.abs(ub) < _INF
        x = np.where(fl, lb, np.where(fu, ub, 0.0)).astype(np.float64)
        for j in self.free_link:
            if fu[j]:
                x[j] = ub[j]
            else:  # unbounded above: large enough for every row it is in
                x[j] = max(
                    [x[j]] + [float(rlo[r]) for r, _ in self.free_link[j] if abs(rlo[r]) < _INF]
                )
        info = {b: (E, hi, lo, sat_hi) for b, E, ok, hi, lo, sat_hi, _k in per_block}
        for key, members in groups.items():
            picks = [i for i, c in sorted(lam[key].items()) for _ in range(c)]
            for (b, _P, rep), pi in zip(members, picks):
                E, hi, lo, sat_hi = info[b]
                e = int(rep[pi])
                blk = self.blocks[b]
                for k, j in enumerate(blk.cols):
                    x[j] = float(E[e, k])
                for j in blk.sats:
                    num, den = hi.get(j, (None, 1))
                    if j in hi and num is not None:
                        x[j] = float(num[e]) / den
                    elif j not in hi and fu[j]:
                        x[j] = ub[j]
                    else:
                        lo_n, lden = lo.get(j, (None, 1))
                        base = (
                            float(lo_n[e]) / lden if lo_n is not None else (lb[j] if fl[j] else 0.0)
                        )
                        need = [
                            math.ceil(float(rlo[r]) / c)
                            for r, c in self.satlink[j]
                            if abs(rlo[r]) < _INF
                        ]
                        x[j] = max([base] + need)
        for j, v in vals.items():  # verbatim blocks: the master's own values
            x[j] = v
        if not self.verify(x, rlo, rhi, lb, ub, integ):
            return None, None
        return False, x

    def verify(self, x, rlo, rhi, lb, ub, integ, tol: float = 1e-9) -> bool:
        """``x`` meets every active row and bound of the probe and is integral on
        the held columns (exact arithmetic on integer data, hence the tight tol)."""
        act = self.A @ x
        fin = np.abs(rlo) < _INF
        if np.any(act[fin] < rlo[fin] - tol):
            return False
        fin = np.abs(rhi) < _INF
        if np.any(act[fin] > rhi[fin] + tol):
            return False
        fin = np.abs(lb) < _INF
        if np.any(x[fin] < lb[fin] - tol):
            return False
        fin = np.abs(ub) < _INF
        if np.any(x[fin] > ub[fin] + tol):
            return False
        held = np.asarray(integ, dtype=bool)
        return bool(np.all(x[held] == np.rint(x[held])))

    # ------------------------------------------------------------------- symmetry

    def classes(self, rlo, rhi, lb, ub, integ) -> list[list[int]]:
        """Classes (size >= 2) of blocks that are interchangeable **in this
        subsystem**: the canonical column map between two members carries the
        active rows with their bounds, the columns' boxes and held flags, and the
        satellites (named by the linking rows they are in, with coefficients)
        onto each other. Linking rows are fixed by such a swap, so it is an
        automorphism of the subsystem."""
        by_sig: dict[tuple, list[int]] = defaultdict(list)
        for b, blk in enumerate(self.blocks):
            if not blk.canon:
                continue
            rows = Counter(
                (blk.rowsig[i], float(rlo[r]), float(rhi[r]))
                for i, r in enumerate(blk.rows)
                if abs(rlo[r]) < _INF or abs(rhi[r]) < _INF
            )
            cols = tuple(
                (
                    float(lb[j]),
                    float(ub[j]),
                    bool(integ[j]),
                    tuple(sorted(self.corelink.get(j, ()))),
                )
                for j in blk.cols
            )
            sats = frozenset(
                (blk.satkey[j], float(lb[j]), float(ub[j]), bool(integ[j])) for j in blk.sats
            )
            by_sig[(frozenset(rows.items()), cols, sats)].append(b)
        return sorted(sorted(g) for g in by_sig.values() if len(g) >= 2)

    def lex_rows(self, classes: list[list[int]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """One ordering row per consecutive pair of each class:
        ``sum_k w_k (x[b, k] - x[b', k]) >= 0`` over the canonical core columns,
        ``w`` the mixed-radix weights of the columns' declared boxes (cut off
        where they would exceed 1e9: any linear ``f`` is a valid order)."""
        idx: list[np.ndarray] = []
        val: list[np.ndarray] = []
        for cls in classes:
            cols0 = self.blocks[cls[0]].cols
            radix = [int(self.col_ub[j] - self.col_lb[j]) + 1 for j in cols0]
            w = np.ones(len(cols0))
            for k in range(len(cols0) - 2, -1, -1):
                w[k] = w[k + 1] * radix[k + 1]
            if w[0] > 1e9:
                w = w / w[0] * 1e9
                w = np.where(w < 1.0, 0.0, np.rint(w))
            for b1, b2 in zip(cls, cls[1:]):
                c1, c2 = self.blocks[b1].cols, self.blocks[b2].cols
                nz = np.flatnonzero(w)
                idx.append(np.array([c1[k] for k in nz] + [c2[k] for k in nz], dtype=np.int64))
                val.append(np.concatenate([w[nz], -w[nz]]))
        return idx, val

    # --------------------------------------------------------------------- orbits

    def link_groups(self) -> list[tuple[int, list[int], list[int]]]:
        """Per linking row: ``(row, block rows, satellites)`` of the satellites
        that lie in this linking row only -- what a deletion of the row takes with
        it. Deleting whole groups keeps every block symmetry."""
        out = []
        for r in self.link:
            sats = [
                int(j)
                for j in self.rowcols[int(r)]
                if int(j) in self.satrow and len(self.satlink[int(j)]) == 1
            ]
            out.append((int(r), [self.satrow[j] for j in sats], sats))
        return out

    def element_orbits(self, cls: list[int]) -> dict[str, list[list[int]]]:
        """Orbits of single elements under the swaps of class ``cls``: matching
        block rows (``"rows"``), matching satellites (``"sats"``) and matching
        core columns (``"cols"``), each a list of one id per block."""
        blks = [self.blocks[b] for b in cls]
        rows: dict[tuple, list[int]] = defaultdict(list)
        for blk in blks:
            seen: Counter = Counter()
            for i, r in enumerate(blk.rows):
                sig = blk.rowsig[i]
                rows[(sig, seen[sig])].append(int(r))
                seen[sig] += 1
        sats: dict[tuple, list[int]] = defaultdict(list)
        for blk in blks:
            for j in blk.sats:
                sats[blk.satkey[j]].append(j)
        cols = [[blk.cols[k] for blk in blks] for k in range(len(blks[0].cols))]
        full = len(cls)
        return {
            "rows": [v for v in rows.values() if len(v) == full],
            "sats": [v for v in sats.values() if len(v) == full],
            "cols": cols,
        }


def lex_decide(
    bm: BlockMaster,
    rlo,
    rhi,
    lb,
    ub,
    integ,
    stop: float,
    presolve: tuple[str, ...] = ("choose", "off"),
) -> tuple[Optional[bool], Optional[np.ndarray], int]:
    """Decide one probe with an ordinary MILP solve plus the ordering rows of the
    blocks interchangeable in it: ``(True, None, k)`` when every ``presolve``
    configuration returns ``Infeasible``; ``(False, x, k)`` on a feasible status
    (``x`` is the solver's point: the caller checks it); else ``(None, None, k)``.
    ``k`` is the number of ordering rows; with ``k == 0`` nothing is solved."""
    classes = bm.classes(rlo, rhi, lb, ub, integ)
    idx, val = bm.lex_rows(classes)
    k = len(idx)
    if k == 0:
        return None, None, 0
    n = bm.n
    extra = sp.csr_matrix(
        (
            np.concatenate(val),
            np.concatenate(idx),
            np.concatenate([[0], np.cumsum([i.size for i in idx])]),
        ),
        shape=(k, n),
    )
    Aa = sp.vstack([bm.A, extra], format="csr")
    integ = np.asarray(integ, dtype=bool)
    fin = lambda v: np.where(np.abs(v) < _INF, v, np.sign(v) * _INF)  # noqa: E731
    r_lo = np.concatenate([fin(np.asarray(rlo, float)), np.zeros(k)])
    r_hi = np.concatenate([fin(np.asarray(rhi, float)), np.full(k, _INF)])
    c_lb, c_ub = fin(np.asarray(lb, float)), fin(np.asarray(ub, float))
    be = _bk.make_backend(bm.engine, Aa, r_lo, r_hi, c_lb, c_ub, integ)
    inf = be.inf
    be.configure(
        np.where(r_lo <= -_INF, -inf, r_lo),
        np.where(r_hi >= _INF, inf, r_hi),
        np.where(c_lb <= -_INF, -inf, c_lb),
        np.where(c_ub >= _INF, inf, c_ub),
        integ,
    )
    for p in presolve:
        if stop - time.perf_counter() < 0.05:
            return None, None, k
        st = be.run(p, stop)
        if st == _bk.OPTIMAL:
            return False, np.asarray(be.point(), dtype=np.float64), k
        if st != _bk.INFEASIBLE:
            return None, None, k
    return True, None, k
