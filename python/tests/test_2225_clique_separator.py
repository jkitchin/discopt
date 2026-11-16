"""#2225: the Rust MILP driver proves pigeonhole infeasibility through clique cuts.

The MIPcc27 smallset's ``hole9``/``hole10`` are pigeonhole models (``k + 1`` pigeons,
``k`` holes, one ``>= 1`` row per pigeon and one *pairwise* ``x_ik + x_jk <= 1`` row per
hole and pair, written ``-x_ik - x_jk >= -1``). Their LP relaxation is feasible; with
the hole cliques ``sum_i x_ik <= 1`` it is not. HiGHS gets those cliques from its
clique table (presolve merging and ``separateCliques``) and proves ``hole9`` at the
root. Before #2225 the in-house driver had the table ported but not wired in: the IIS
engine's full-model probe on ``hole9`` branched 10,185 nodes (7.3 s), ``hole10`` ran out
of a 60 s budget, and the MIPcc entry returned the whole model.

The root separation (``lp/highs_cutgen/bridge.rs``) now extracts the clique table from
the model rows, merges it (``runCliqueMerging``) and puts every extended set-packing row
into the cut pool, then runs ``separateCliques`` each round
(``DISCOPT_HIGHS_SEPA_CLIQUE``, default ON; ``=0`` opts out).

The test builds the model in the smallset's own form (no data file) and solves it
through the IIS engine's Rust backend, both arms in the same process: the flag is read
per solve.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_backend import INFEASIBLE, OPTIMAL, make_backend

# Measured on hole(7) (8 pigeons, 7 holes, 203 rows), 2026-11-16, this branch:
# clique ON 0.0x s and <= 120 nodes; OFF 1,6xx nodes. The bars leave a wide margin both
# ways; the OFF arm is the anti-vacuity check that the instance needs the cliques.
K = 7
ON_MAX_NODES = 400
OFF_MIN_NODES = 1000


def _pigeonhole(k: int):
    """``k + 1`` pigeons into ``k`` holes, rows as MIPcc27's ``hole9`` writes them."""
    p = k + 1

    def var(i: int, j: int) -> int:
        return i * k + j

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for i in range(p):  # each pigeon in some hole
        for j in range(k):
            rows.append(r)
            cols.append(var(i, j))
            vals.append(1.0)
        lo.append(1.0)
        hi.append(1e20)
        r += 1
    for j in range(k):  # no two pigeons share a hole: -x_ij - x_lj >= -1
        for i1 in range(p):
            for i2 in range(i1 + 1, p):
                rows += [r, r]
                cols += [var(i1, j), var(i2, j)]
                vals += [-1.0, -1.0]
                lo.append(-1.0)
                hi.append(1e20)
                r += 1
    n = p * k
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    return A, np.array(lo), np.array(hi), np.zeros(n), np.ones(n), np.ones(n, dtype=bool)


def _solve(monkeypatch, flag: str):
    monkeypatch.setenv("DISCOPT_HIGHS_SEPA_CLIQUE", flag)
    A, lo, hi, lb, ub, isint = _pigeonhole(K)
    b = make_backend("rust", A, lo, hi, lb, ub, isint)
    t = time.perf_counter()
    st = b.run("off", t + 120.0)
    return st, b.last_nodes()


def test_pigeonhole_needs_few_nodes_with_the_clique_separator(monkeypatch):
    st_on, nodes_on = _solve(monkeypatch, "1")
    st_off, nodes_off = _solve(monkeypatch, "0")
    # Soundness first: the model has no point, so neither arm may find one.
    assert st_on != OPTIMAL and st_off != OPTIMAL
    assert st_on == INFEASIBLE, st_on
    assert st_off == INFEASIBLE, st_off
    # Anti-vacuity (CLAUDE.md §6): without the cliques the instance is hard.
    assert nodes_off >= OFF_MIN_NODES, f"OFF arm needed only {nodes_off} nodes"
    assert nodes_on <= ON_MAX_NODES, f"ON arm needed {nodes_on} nodes (OFF {nodes_off})"


def test_clique_flag_refuses_garbage(monkeypatch):
    monkeypatch.setenv("DISCOPT_HIGHS_SEPA_CLIQUE", "maybe")
    A, lo, hi, lb, ub, isint = _pigeonhole(2)
    b = make_backend("rust", A, lo, hi, lb, ub, isint)
    with pytest.raises(BaseException, match="DISCOPT_HIGHS_SEPA_CLIQUE"):
        b.run("off", time.perf_counter() + 10.0)
