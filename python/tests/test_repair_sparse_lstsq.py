"""The incumbent repair's sparse least-squares path (``_min_norm_lstsq_sparse``).

Above ``REPAIR_DENSE_SVD_CELLS`` the repair's Newton step is solved by LSQR on the
sparse Jacobian instead of a dense SVD (``gelsd``): the dense solve took ~1 s on
large equality systems single-threaded and ~19 s when two solves shared four cores
with OpenBLAS's threads (neos-4393408-tinui, mc7; 2026-11-14). LSQR from zero
converges to the same minimum-norm least-squares solution, rank-deficient systems
included; these tests pin that, and that the repair reaches the same verified point
through either path. The small-system path is unchanged (dense, bit-for-bit).
"""

from __future__ import annotations

import discopt.validation.feasibility as F
import numpy as np
import scipy.sparse as sp
from discopt._tape_nlp_evaluator import make_evaluator
from test_1537e_incumbent_repair import _BAD, _minimax


def test_sparse_min_norm_matches_the_dense_svd_on_rank_deficient_systems():
    rng = np.random.default_rng(1537)
    compared = 0
    for m, n, rank in [(30, 50, 20), (50, 30, 25), (40, 40, 40), (60, 120, 35)]:
        left = sp.random(m, rank, density=0.3, random_state=rng, format="csr")
        right = sp.random(rank, n, density=0.3, random_state=rng, format="csr")
        a = (left @ right).tocsr()
        b = a @ rng.standard_normal(n)  # consistent right-hand side
        dense = F._min_norm_lstsq(a.toarray(), b)
        sparse = F._min_norm_lstsq_sparse(a, b)
        scale = 1.0 + np.max(np.abs(dense))
        assert np.max(np.abs(sparse - dense)) <= 1e-8 * scale, (m, n, rank)
        compared += 1
    assert compared == 4


def test_repair_reaches_the_same_point_through_the_sparse_path(monkeypatch):
    m = _minimax(1.0)
    dense = F.repair_point(m, _BAD, evaluator=make_evaluator(m))
    assert dense.x is not None, dense.reason
    calls = {"n": 0}
    orig = F._min_norm_lstsq_sparse

    def counted(a, b):
        calls["n"] += 1
        return orig(a, b)

    monkeypatch.setattr(F, "REPAIR_DENSE_SVD_CELLS", 0)
    monkeypatch.setattr(F, "_min_norm_lstsq_sparse", counted)
    m2 = _minimax(1.0)
    sparse = F.repair_point(m2, _BAD, evaluator=make_evaluator(m2))
    assert calls["n"] > 0, "the sparse path never ran"
    assert sparse.x is not None, sparse.reason
    assert sparse.excess_after == 0.0
    assert np.max(np.abs(sparse.x - dense.x)) <= 1e-9
    check = F.verify_point(_minimax(1.0), sparse.x, with_objective=True)
    assert check.ok, check.reason
