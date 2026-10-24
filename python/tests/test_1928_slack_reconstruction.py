"""#1928: the batched POUNCE QP wave reconstructs slacks without a dense SVD.

``_pounce_qp_relaxation_nodes`` computed ``np.linalg.pinv`` of the slack block of
``A_eq`` every wave. On ``unitcommit_50_20_2_mod_8`` that block is 15,690 x
14,058 and the pseudo-inverse alone ran for more than 13 minutes. The block has
one slack per inequality row (disjoint column supports), so ``S^+ rhs`` is
``rhs[r_j] / s_j``; any other block keeps the pseudo-inverse up to a size cap.
"""

import numpy as np
import pytest
from discopt import solver


@pytest.fixture
def pinv_calls(monkeypatch):
    calls = []
    real = np.linalg.pinv

    def spy(a, *args, **kw):
        calls.append(np.shape(a))
        return real(a, *args, **kw)

    monkeypatch.setattr(np.linalg, "pinv", spy)
    return calls


@pytest.mark.unit
def test_one_slack_per_row_is_solved_in_closed_form(pinv_calls):
    rng = np.random.default_rng(0)
    m, k = 30, 18
    S = np.zeros((m, k))
    rows = rng.permutation(m)[:k]
    S[rows, np.arange(k)] = rng.choice([1.0, -1.0, 2.5], size=k)
    S[:, 3] = 0.0  # an empty slack column: S^+ gives 0 there
    rhs = rng.normal(size=m)
    got = solver._slack_reconstruction(S)(rhs)
    assert pinv_calls == []
    np.testing.assert_allclose(got, np.linalg.pinv(S) @ rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_a_large_disjoint_block_needs_no_pseudo_inverse(pinv_calls):
    m = k = solver._SLACK_PINV_MAX_DIM + 500  # above the pinv cap
    S = np.zeros((m, k))
    S[np.arange(k), np.arange(k)] = -1.0
    rhs = np.arange(m, dtype=float)
    np.testing.assert_array_equal(solver._slack_reconstruction(S)(rhs), -rhs)
    assert pinv_calls == []


@pytest.mark.unit
def test_a_general_block_keeps_the_pseudo_inverse_up_to_the_cap(pinv_calls):
    rng = np.random.default_rng(1)
    S = rng.normal(size=(12, 5))  # dense columns: not the one-slack-per-row form
    rhs = rng.normal(size=12)
    np.testing.assert_allclose(
        solver._slack_reconstruction(S)(rhs), np.linalg.pinv(S) @ rhs, rtol=1e-12, atol=1e-12
    )
    assert pinv_calls[0] == (12, 5)


@pytest.mark.unit
def test_a_large_general_block_declines(pinv_calls):
    n = solver._SLACK_PINV_MAX_DIM + 1
    S = np.ones((n, n))
    assert solver._slack_reconstruction(S) is None
    assert pinv_calls == []
