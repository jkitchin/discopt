"""The MIQP node loop keeps a sparse-extracted model's matrices sparse.

Found re-measuring #1928 on ``unitcommit_50_20_2_mod_8``: after the slack
pseudo-inverse was gone, the solve still returned at 79 s on a 60 s limit
because ``_solve_miqp_bb`` densified the 15,690 x 20,072 ``A_eq`` (2.5 GB, ~14 s)
in its setup, again in every batched QP wave, again in ``_miqp_origin_shift`` and
again in the exit-time dual recovery. The blocks are now built once per solve
from the sparse matrix and stay sparse (every consumer takes a sparse operand);
their values must be exactly the dense route's.
"""

from pathlib import Path

import discopt.solver as S
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.problem_classifier import dense_A, dense_Q, extract_qp_data
from discopt.modeling import from_nl

_ST_MIQP5 = Path(__file__).parent / "data" / "minlplib_nl" / "st_miqp5.nl"


def _qp():
    m = from_nl(str(_ST_MIQP5))
    qp = extract_qp_data(m)
    n_orig = sum(v.size for v in m._variables)
    sparse_qp = qp._replace(Q=sp.csr_matrix(dense_Q(qp.Q)), A_eq=sp.csr_matrix(dense_A(qp.A_eq)))
    return qp, sparse_qp, n_orig


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    a = a.toarray() if sp.issparse(a) else np.asarray(a)
    b = b.toarray() if sp.issparse(b) else np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


@pytest.mark.unit
def test_blocks_from_sparse_equal_the_dense_route():
    qp, sqp, n_orig = _qp()
    A = dense_A(qp.A_eq)
    old = S._decompose_eq_slack_form(
        A,
        np.asarray(qp.b_eq),
        n_orig,
        A.shape[1] - n_orig,
        np.asarray(qp.x_u, dtype=np.float64),
        row_sense=S._declared_row_senses(qp, A),
    )
    blocks = S._miqp_blocks(sqp, n_orig)
    # A sparse-extracted model keeps its blocks sparse; their values are the
    # dense route's exactly.
    assert sp.issparse(blocks.slack_block) and sp.issparse(blocks.A_struct)
    assert sp.issparse(blocks.P_s)
    for o, n in zip(old, (blocks.A_ub, blocks.b_ub, blocks.A_eq, blocks.b_eq)):
        assert _same(o, n)
    assert _same(dense_Q(qp.Q)[:n_orig, :n_orig], blocks.P_s)
    assert _same(A[:, :n_orig], blocks.A_struct)


@pytest.mark.unit
def test_origin_shift_from_sparse_equals_dense():
    qp, sqp, n_orig = _qp()
    lb = np.full(qp.x_l.shape[0], 3.0)
    ub = lb + 5.0
    dqp = qp._replace(Q=dense_Q(qp.Q), A_eq=dense_A(qp.A_eq))
    a = S._miqp_origin_shift(sqp, lb, ub, n_orig)
    b = S._miqp_origin_shift(dqp, lb, ub, n_orig)
    assert a is not None and b is not None
    assert np.array_equal(a[0], b[0])
    for k in ("c", "b_eq", "obj_const"):
        assert np.array_equal(np.asarray(getattr(a[1], k)), np.asarray(getattr(b[1], k)))


@pytest.mark.unit
def test_the_batched_wave_does_not_densify_the_full_matrix(monkeypatch):
    qp, sqp, n_orig = _qp()
    calls = []
    real = S._dense_A

    def spy(A):
        calls.append(getattr(A, "shape", None))
        return real(A)

    monkeypatch.setattr(S, "_dense_A", spy)
    blocks = S._miqp_blocks(sqp, n_orig)
    # Node boxes are over the structural columns, as the tree hands them over.
    lb = [np.asarray(qp.x_l, dtype=np.float64)[:n_orig]]
    ub = [np.minimum(np.asarray(qp.x_u, dtype=np.float64)[:n_orig], 1e3)]
    clean, *_ = S._pounce_qp_relaxation_nodes(sqp, lb, ub, n_orig, 0.0, 1e9, blocks=blocks)
    assert calls == [], calls  # the serial fallback (which densifies) did not run
    assert clean.tolist() == [True]


@pytest.mark.unit
def test_qp_waves_are_chunked_to_the_pool_and_stop_at_the_deadline(monkeypatch):
    import time

    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    calls = []

    def fake_batch(problems, *, check_psd, time_limit):
        calls.append((len(problems), time_limit))
        time.sleep(0.2)
        return [f"r{p}" for p in problems]

    problems = list(range(5))
    out = S._solve_qp_batch_by_deadline(fake_batch, problems, time.perf_counter() + 0.3)
    # Two-wide chunks; each gets the whole remainder; none starts after the deadline.
    assert [n for n, _ in calls] == [2, 2]
    assert all(0.0 < t <= 0.3 for _, t in calls)
    assert out == ["r0", "r1", "r2", "r3", None]
    # Inside the limit every chunk runs and the result is the unchunked one.
    calls.clear()
    out = S._solve_qp_batch_by_deadline(fake_batch, problems, time.perf_counter() + 60.0)
    assert out == [f"r{p}" for p in problems] and [n for n, _ in calls] == [2, 2, 1]
