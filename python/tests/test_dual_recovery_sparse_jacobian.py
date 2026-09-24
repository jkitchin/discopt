"""Dual recovery must not densify a Jacobian that arrived sparse.

THE DEFECT.  ``_duals_against_declared_box`` refits KKT multipliers against the
box the user declared (#1037), and the refit builds a dense least-squares system.
Measured on MINLPLib ``arki0014`` against ``time_limit=20``:

    evaluator.evaluate_jacobian(x)     (17525, 19305)  2.707 GB dense,  75,329 nnz
    jac_c = jac[:, cont_idx]           another 2.707 GB copy
    sub_jac = jac_c[row_select, :]                1.43  GB
    A = np.concatenate(cols, axis=1)   (19305, 16676)  2.575 GB dense,  55,333 nnz

~9.4 GB of dense arrays carrying 75k numbers — the matrix is **0.017 % dense**,
about 0.7 MB as CSR — and then a bounded dense least squares on it.  Two
``faulthandler`` stack dumps 60 s apart showed the identical
``numpy.linalg.lstsq`` frame, and the solve ran >600 s against a 20 s limit.

None of that work is even necessary upstream: ``evaluate_jacobian``'s own
docstring says that above ``_DENSE_JACOBIAN_COMPILE_LIMIT`` it computes the
Jacobian through the sparse colouring path and then **densifies it** to honour a
dense return contract.  The sparse matrix exists and is thrown away.

WHY THE REPRESENTATION FOLLOWS THE INPUT.  ``lsq_linear`` solves a sparse system
with LSMR and a dense one with an exact factorisation, and the two do not agree
bit for bit: measured on a 200x120 system at 1.7 % density, the costs agree to
1.2e-13 relative but the arguments differ by 1.1e-06.  Multipliers are reported
output, so that difference is visible.  Therefore ``recover_multipliers`` keeps
**exactly** today's dense path for a dense ``jac`` — every existing caller is bit
identical — and takes the sparse path only for a ``jac`` that arrived sparse,
which only the large-model route asks for.  The change is confined to the cases
that today take minutes or exhaust memory.

The two traps this module walks past are the ones CLAUDE.md names explicitly:
``np.asarray()`` on a scipy sparse matrix returns a 0-d **object** array instead
of raising, and ``.size`` on a sparse matrix is ``nnz``, not rows x columns —
which is why ``jac.size`` may never be used as "does this Jacobian have rows".
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._dual_recovery import recover_multipliers
from discopt.validation.feasibility import jacobian_row_scales


def _problem(n=40, m=30, seed=0):
    """A sparse KKT system with a genuinely nonempty active set.

    Rows are built to be exactly active at ``x`` so ``row_select`` is not empty —
    an empty active set returns before the least squares and would make every
    assertion below vacuous (CLAUDE.md §6).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 1.5, size=n)
    jac = np.zeros((m, n))
    for i in range(m):
        cols = rng.choice(n, 3, replace=False)
        jac[i, cols] = rng.normal(size=3)
    rhs = jac @ x  # every row exactly active at x
    body = jac @ x
    sense = np.array(["<="] * m)
    grad = rng.normal(size=n)
    lb = np.full(n, -10.0)
    ub = np.full(n, 10.0)
    is_cont = np.ones(n, dtype=bool)
    return dict(
        grad=grad,
        jac=jac,
        body=body,
        sense_arr=sense,
        rhs_arr=rhs,
        x_flat=x,
        lb=lb,
        ub=ub,
        is_continuous=is_cont,
    )


# --------------------------------------------------------------------------
# jacobian_row_scales: the shared helper both paths go through
# --------------------------------------------------------------------------


def test_row_scales_accept_a_sparse_jacobian():
    """The #1151 shared row-scale helper must read a sparse J, not choke on it.

    ``np.asarray(csr)`` yields a 0-d object array, so the pre-fix helper raised
    ``expected a 2-D Jacobian, got shape ()`` — loudly, at least, but it made the
    sparse route impossible to take.
    """
    p = _problem()
    dense = jacobian_row_scales(p["jac"], p["x_flat"])
    sparse = jacobian_row_scales(sp.csr_matrix(p["jac"]), p["x_flat"])
    assert dense.shape == sparse.shape == (p["jac"].shape[0],)
    assert np.allclose(dense, sparse, rtol=0, atol=0), "row scales must be EXACT, not close"


def test_row_scales_sparse_keeps_the_non_finite_rule():
    """A non-finite term still yields 0.0 — the strictest answer (#1157)."""
    p = _problem()
    jac = p["jac"].copy()
    jac[2, np.nonzero(jac[2])[0][0]] = np.inf
    dense = jacobian_row_scales(jac, p["x_flat"])
    sparse = jacobian_row_scales(sp.csr_matrix(jac), p["x_flat"])
    assert dense[2] == 0.0 and sparse[2] == 0.0
    assert np.array_equal(dense, sparse)


def test_row_scales_a_non_finite_x_poisons_every_row_sparse_too():
    """The one place sparsity could LOOSEN the answer, pinned against dense.

    A dense row has an entry in every column, so a non-finite ``x_j`` makes every
    row unestimatable: ``inf`` where ``J_ij != 0``, ``|0| * inf = NaN`` where it is
    zero. Sparsity drops exactly the second half, so without a guard the rows with
    a structural zero in that column stay finite and get a LARGER scale — a more
    permissive activity test, the #1151 failure direction.
    """
    p = _problem()
    x = p["x_flat"].copy()
    # A column that is structurally absent from most rows: the interesting case.
    col = int(np.argmin((p["jac"] != 0).sum(axis=0)))
    x[col] = np.inf
    dense = jacobian_row_scales(p["jac"], x)
    sparse = jacobian_row_scales(sp.csr_matrix(p["jac"]), x)
    assert (p["jac"][:, col] == 0).any(), "test needs a structurally absent entry to be real"
    assert np.array_equal(dense, sparse)
    assert np.all(dense == 0.0), "dense marks every row unestimatable"


def test_row_scales_sparse_rejects_a_column_count_mismatch():
    """The shape guard must survive the sparse path rather than be skipped."""
    p = _problem()
    with pytest.raises(ValueError, match="columns"):
        jacobian_row_scales(sp.csr_matrix(p["jac"]), p["x_flat"][:-1])


# --------------------------------------------------------------------------
# recover_multipliers
# --------------------------------------------------------------------------


def test_sparse_and_dense_jacobians_agree():
    """The sparse route must solve the same problem, within LSMR's tolerance."""
    p = _problem()
    dense = recover_multipliers(**p)
    sparse = recover_multipliers(**{**p, "jac": sp.csr_matrix(p["jac"])})
    assert dense.ok and sparse.ok
    assert not dense.empty_active_set, "empty active set would make this vacuous"
    assert dense.n_active_cons == sparse.n_active_cons > 0
    assert np.allclose(dense.mu_full, sparse.mu_full, atol=1e-6)
    assert np.allclose(dense.lam_lb_full, sparse.lam_lb_full, atol=1e-6)
    assert np.allclose(dense.lam_ub_full, sparse.lam_ub_full, atol=1e-6)
    assert sparse.residual_max == pytest.approx(dense.residual_max, abs=1e-6)


def test_a_sparse_jacobian_never_materialises_a_dense_system(monkeypatch):
    """The point of the change: no dense (n_cont x n_active) array is built.

    Asserted on the object handed to ``lsq_linear`` rather than on wall time, so
    the test states the property instead of measuring the machine.
    """
    import discopt._dual_recovery as DR

    seen = {}
    orig = DR.lsq_linear

    def spy(A, b, **kw):
        seen["sparse"] = sp.issparse(A)
        seen["shape"] = A.shape
        return orig(A, b, **kw)

    monkeypatch.setattr(DR, "lsq_linear", spy)

    p = _problem()
    rec = recover_multipliers(**{**p, "jac": sp.csr_matrix(p["jac"])})
    assert rec.ok
    assert seen, "lsq_linear was never called — the assertion below would be vacuous"
    assert seen["sparse"], "a sparse Jacobian must produce a sparse least-squares system"


def test_a_dense_jacobian_still_takes_the_exact_dense_path(monkeypatch):
    """Bit-identity for every existing caller is the safety argument; pin it."""
    import discopt._dual_recovery as DR

    seen = {}
    orig = DR.lsq_linear

    def spy(A, b, **kw):
        seen["sparse"] = sp.issparse(A)
        return orig(A, b, **kw)

    monkeypatch.setattr(DR, "lsq_linear", spy)

    rec = recover_multipliers(**_problem())
    assert rec.ok
    assert seen, "lsq_linear was never called"
    assert seen["sparse"] is False, "a dense Jacobian must keep the dense exact solver"


def test_empty_active_set_with_a_sparse_jacobian_sizes_mu_by_ROWS_not_nnz():
    """``jac.size`` is ``nnz`` for a sparse matrix — the CLAUDE.md trap.

    On the empty-active-set return, ``mu_full`` is sized ``jac.shape[0]``. Written
    as ``np.zeros(jac.shape[0]) if jac.size else np.zeros(0)`` that is still
    correct for a dense Jacobian but silently returns a LENGTH-ZERO ``mu_full``
    for an all-zero sparse one, because ``nnz == 0``. The caller then scatters
    row multipliers into an empty vector.
    """
    n, m = 6, 4
    x = np.ones(n)
    jac = np.zeros((m, n))  # no structure at all => nnz == 0
    p = dict(
        grad=np.zeros(n),
        jac=sp.csr_matrix(jac),
        body=np.full(m, -5.0),  # every row strictly inactive
        sense_arr=np.array(["<="] * m),
        rhs_arr=np.zeros(m),
        x_flat=x,
        lb=np.full(n, -10.0),
        ub=np.full(n, 10.0),
        is_continuous=np.ones(n, dtype=bool),
    )
    rec = recover_multipliers(**p)
    assert rec.empty_active_set, "this test is about the empty-active-set return"
    assert rec.mu_full is not None
    assert rec.mu_full.shape == (m,), (
        f"mu_full must be one entry per ROW ({m}), got {rec.mu_full.shape} — "
        "sized from nnz instead of shape[0]"
    )


def test_sparse_path_preserves_the_ge_row_sign_convention():
    """``>=`` rows are flipped into ``<=`` form and un-flipped on the way out.

    Row-wise scaling of a CSR matrix is where a sparse rewrite most easily goes
    wrong, so the convention is pinned against the dense answer rather than
    assumed to have survived.
    """
    p = _problem(seed=3)
    sense = np.array(["<="] * p["jac"].shape[0])
    sense[::2] = ">="
    p["sense_arr"] = sense
    dense = recover_multipliers(**p)
    sparse = recover_multipliers(**{**p, "jac": sp.csr_matrix(p["jac"])})
    assert dense.ok and sparse.ok and dense.n_active_cons > 0
    ge = np.nonzero(sense == ">=")[0]
    assert np.allclose(dense.mu_full[ge], sparse.mu_full[ge], atol=1e-6)
    assert np.allclose(np.sign(dense.mu_full), np.sign(sparse.mu_full))


# --------------------------------------------------------------------------
# jacobian_for_recovery: which representation the refit asks for
# --------------------------------------------------------------------------


class _FakeEvaluator:
    """Records which Jacobian entry point the refit called."""

    def __init__(self, m, n):
        self.m, self.n = m, n
        self.calls = []

    def evaluate_jacobian(self, x):
        self.calls.append("dense")
        return np.zeros((self.m, self.n))

    def evaluate_sparse_jacobian(self, x):
        self.calls.append("sparse")
        return sp.csc_matrix((self.m, self.n))


def test_small_model_still_asks_for_the_dense_jacobian():
    """Below the limit the dense jacfwd path is cheap and must not change."""
    from discopt._dual_recovery import jacobian_for_recovery

    ev = _FakeEvaluator(10, 20)
    jacobian_for_recovery(ev, np.zeros(20), m=10)
    assert ev.calls == ["dense"]


def test_large_model_asks_for_the_sparse_jacobian():
    """Above the limit ``evaluate_jacobian`` would densify; ask for it sparse.

    The threshold is the evaluator's own ``_DENSE_JACOBIAN_COMPILE_LIMIT``, so
    the sparse route is requested exactly where the dense contract was being met
    by building the sparse matrix and throwing it away.
    """
    from discopt._dual_recovery import jacobian_for_recovery
    from discopt._relax.nlp_evaluator import _DENSE_JACOBIAN_COMPILE_LIMIT

    n = 4000
    m = _DENSE_JACOBIAN_COMPILE_LIMIT // n + 1
    assert m * n > _DENSE_JACOBIAN_COMPILE_LIMIT, "test must sit ABOVE the limit"
    ev = _FakeEvaluator(m, n)
    J = jacobian_for_recovery(ev, np.zeros(n), m=m)
    assert ev.calls == ["sparse"]
    assert sp.issparse(J)


def test_an_evaluator_without_the_sparse_entry_point_still_works():
    """No ``evaluate_sparse_jacobian`` means the old behaviour, not an error."""
    from discopt._dual_recovery import jacobian_for_recovery

    class _DenseOnly:
        def __init__(self):
            self.calls = []

        def evaluate_jacobian(self, x):
            # Deliberately NOT a real (100000, 4000) array: this test is about
            # which entry point is chosen, and materialising 3.2 GB to prove it
            # would be the very thing under repair.
            self.calls.append("dense")
            return "dense-jacobian"

    ev = _DenseOnly()
    jacobian_for_recovery(ev, np.zeros(4000), m=100_000)
    assert ev.calls == ["dense"]


# --------------------------------------------------------------------------
# TapeNLPEvaluator: the evaluator the default path actually uses
# --------------------------------------------------------------------------


def test_tape_evaluator_sparse_jacobian_equals_the_dense_one_exactly():
    """The default solve path uses ``TapeNLPEvaluator``, not ``NLPEvaluator``.

    Which is the whole reason the first cut of this fix was a no-op on
    ``arki0014``: the routing asked ``hasattr(evaluator,
    'evaluate_sparse_jacobian')`` and the tape evaluator did not have it, so the
    refit silently kept the dense 2.707 GB array. The tape is natively COO, so
    the sparse form must be EXACT — ``coo_matrix`` sums duplicate entries on
    conversion exactly as ``np.add.at`` does in the dense scatter.
    """
    import discopt.modeling as dm
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(x * y + z <= 3)
    m.subject_to(x**2 - y <= 1)
    m.subject_to(dm.exp(z) + x <= 10)
    m.minimize(x + y + z)

    ev = TapeNLPEvaluator(m)
    pt = np.array([0.3, -0.7, 1.1])
    dense = ev.evaluate_jacobian(pt)
    sparse = ev.evaluate_sparse_jacobian(pt)
    assert sp.issparse(sparse), "must return a sparse matrix, not a dense one"
    assert sparse.shape == dense.shape
    assert np.array_equal(sparse.toarray(), dense), "sparse form must be EXACT"
    assert np.count_nonzero(dense) > 0, "an all-zero Jacobian would make this vacuous"


def test_tape_evaluator_is_routed_to_the_sparse_jacobian_when_large():
    """End of the wire: a large tape-backed model must not be densified."""
    import discopt.modeling as dm
    from discopt._dual_recovery import jacobian_for_recovery
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    m = dm.Model()
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x * y <= 3)
    m.minimize(x + y)
    ev = TapeNLPEvaluator(m)

    # Routing is a shape decision, so it is exercised by lying about the shape
    # rather than by building a model big enough to reproduce the 2.7 GB array.
    from discopt._relax.nlp_evaluator import _DENSE_JACOBIAN_COMPILE_LIMIT

    J = jacobian_for_recovery(ev, np.array([0.3, -0.7]), m=_DENSE_JACOBIAN_COMPILE_LIMIT)
    assert sp.issparse(J), "a tape evaluator over the limit must yield a sparse Jacobian"
