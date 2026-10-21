"""#1845: ``sum(B * (A @ X))`` with a 2-D ``X`` was certified wrong.

The first LP-extraction rung reads the Rust arena's ``quadratic_form``, which
combined array values by flat length: a matmul contracted both operands' flat
vectors into ONE scalar whatever their rank (``@`` only for ``(k,) @ (k,)``), and
an element-wise op zipped two equal-length operands even when their shapes
broadcast to something larger (``(2,1) * (1,2)``). The model below certified
``y = 3.8``; the optimum is 7.6. The walk now follows the operands' shapes.

Oracle: numpy. Each model fixes ``X`` at a random point (``lb == ub``) so the
optimum of ``max y s.t. f(X) + y <= 9`` is exactly ``9 - f(Xv)``.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import problem_classifier as pc
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke


def _solve_fixed(f_dm, Xv, Yv=None):
    m = dm.Model()
    X = m.continuous("X", shape=Xv.shape, lb=Xv, ub=Xv)
    args = [X]
    if Yv is not None:
        args.append(m.continuous("Y", shape=Yv.shape, lb=Yv, ub=Yv))
    y = m.continuous("y", lb=-1e3, ub=1e3)
    m.subject_to(f_dm(*args) + y <= 9)
    m.maximize(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=60)


def test_issue_repro():
    A = np.array([[2, -1], [0.5, 1]])
    B = np.array([[1, 0], [3, -2]])
    Xv = np.array([[1, 0.5], [0.2, 1]])
    r = _solve_fixed(lambda X: dm.sum(B * (A @ X)), Xv)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(7.6, abs=1e-6)


FORMS = {
    "sum(B*(A@X))": (lambda A, B, X: dm.sum(B * (A @ X)), lambda A, B, X: np.sum(B * (A @ X))),
    "sum(A@X)": (lambda A, B, X: dm.sum(A @ X), lambda A, B, X: np.sum(A @ X)),
    "sum(X@A.T)": (lambda A, B, X: dm.sum(X @ A.T), lambda A, B, X: np.sum(X @ A.T)),
    "sum(B*(X@A.T))": (
        lambda A, B, X: dm.sum(B.T * (X @ A.T)),
        lambda A, B, X: np.sum(B.T * (X @ A.T)),
    ),
    "(A@X)[last,0]": (
        lambda A, B, X: (A @ X)[A.shape[0] - 1, 0],
        lambda A, B, X: (A @ X)[A.shape[0] - 1, 0],
    ),
    "sum(col*row)": (
        # (m,1) * (1,n) broadcasts to (m,n): an outer product, not a zip.
        lambda A, B, X: dm.sum(Constant(A[:, :1]) * X[:1, :]),
        lambda A, B, X: np.sum(A[:, :1] * X[:1, :]),
    ),
    "sum(c@X)": (lambda A, B, X: dm.sum(A[0] @ X), lambda A, B, X: np.sum(A[0] @ X)),
}


@pytest.mark.parametrize("form", sorted(FORMS))
def test_random_shapes_against_numpy(form):
    f_dm, f_np = FORMS[form]
    rng = np.random.default_rng(1845)
    checked = 0
    for _ in range(4):
        m_, k, n = (int(v) for v in rng.integers(1, 4, size=3))
        A = np.round(rng.normal(size=(m_, k)), 3)
        Xv = np.round(rng.uniform(-2, 2, size=(k, n)), 3)
        if form in ("sum(X@A.T)", "sum(B*(X@A.T))"):
            Xv = np.round(rng.uniform(-2, 2, size=(n, k)), 3)
        B = np.round(rng.normal(size=(m_, n)), 3)
        r = _solve_fixed(lambda X: f_dm(A, B, X), Xv)
        assert r.status == "optimal", (form, A.shape, Xv.shape, r.status)
        assert r.objective == pytest.approx(9 - f_np(A, B, Xv), abs=1e-6), (form, A, Xv)
        checked += 1
    assert checked == 4


def test_repr_rung_agrees_with_tape_rung():
    # The coefficients of the first (arena) rung must equal the per-element tape
    # rung's on a free box, for every form above. Indexing a compound node is a
    # documented decline of the arena walk (the next rung takes the row).
    rng = np.random.default_rng(18450)
    compared = 0
    for form, (f_dm, _f_np) in sorted(FORMS.items()):
        A = rng.normal(size=(3, 2))
        B = rng.normal(size=(3, 2))
        m = dm.Model()
        X = m.continuous("X", shape=(2, 2), lb=-1, ub=1)
        y = m.continuous("y", lb=-1, ub=1)
        m.subject_to(f_dm(A, B, X) + y <= 9)
        m.maximize(y)
        if form == "(A@X)[last,0]":
            with pytest.raises(pc._NotQuadraticError):
                pc._extract_lp_data_from_repr(m)
            continue
        repr_lp = pc._extract_lp_data_from_repr(m)
        tape_lp = pc._extract_lp_data_tape(m)
        np.testing.assert_allclose(
            pc.dense_A(repr_lp.A_eq), pc.dense_A(tape_lp.A_eq), atol=1e-12, err_msg=form
        )
        compared += 1
    assert compared == len(FORMS) - 1
