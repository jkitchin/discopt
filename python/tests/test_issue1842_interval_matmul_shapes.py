"""#1842: interval matmul enclosures had the wrong shape outside ``(m,k) @ (k,)``.

``_eval_matmul`` enclosed only a matrix times a vector; every other product
fell through to an unbounded interval of shape ``A.shape[:-1]`` -- 0-d for
``(k,) @ (k, n)`` -- so ``(x + c @ M - 10)[0]`` raised ``IndexError: array is
0-dimensional`` in the root relaxation build. With the shape alone fixed the
constant ``sqrt(sum(M @ M))`` still had no finite enclosure and the dual bound
stayed at the variable box (status ``feasible``, bound 20, 8000+ nodes in 15 s).
Every product now reports the product's shape (default ON: unbounded in either
shape is the same enclosure), and so do the sibling fallbacks (``**`` with a
non-scalar exponent, multi-argument calls, unknown binary ops, and a division by
an interval containing zero). The finite interval dot product for the general
1-D / 2-D forms is bound-changing -- ``c @ x``, ``x @ x``, ``x @ Q @ x`` went
from unbounded to finite and moved node counts on a solve panel -- so it was
behind ``DISCOPT_MATMUL_INTERVAL_ENCLOSURE`` until the CLAUDE.md §5 panel
graduated it default ON (``=0`` is the opt-out). The repro now certifies on the
default path.
"""

from __future__ import annotations

import discopt._relax.convexity.interval_eval as ie
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke

FLAG = "DISCOPT_MATMUL_INTERVAL_ENCLOSURE"
M = np.array([[3.0, -1.0], [-2.0, 1.0]])
# (ones @ M)[0] = 1, sum(M @ M) = 2.
OPT = 9.0 - np.sqrt(2.0)


@pytest.fixture
def enclosure_on(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)  # the default since graduation


@pytest.fixture
def enclosure_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")


def _repro():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    row = (x + Constant(np.ones(2)) @ Constant(M) - 10)[0] + dm.sqrt(
        dm.sum(Constant(M) @ Constant(M))
    )
    m.subject_to(row <= 0)
    m.maximize(x)
    return m, x


def test_issue_repro_solves_to_optimum(enclosure_on):
    m, x = _repro()
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(OPT, abs=1e-6)


def test_issue_repro_flag_off_does_not_crash_and_is_sound(enclosure_off):
    # With the opt-out the IndexError stays gone, but the constant ``sum(M @ M)`` has
    # no finite enclosure, so the dual bound stays at the box (measured: status
    # ``feasible``, bound 20.0, ~7000 nodes in 5 s). Whatever it reports must hold.
    m, x = _repro()
    r = m.solve(time_limit=2)
    assert r.status in ("optimal", "feasible", "time_limit")
    assert float(r.value(x)) <= OPT + 1e-6
    assert r.bound >= OPT - 1e-6
    if r.status == "optimal":
        assert float(r.value(x)) == pytest.approx(OPT, abs=1e-6)


SHAPES = [((3,), (3, 2)), ((2, 3), (3, 4)), ((3,), (3,)), ((2, 3), (3,))]


@pytest.mark.parametrize("sa,sb", SHAPES, ids=[f"{a}@{b}" for a, b in SHAPES])
def test_matmul_enclosure_shape_and_soundness(sa, sb, enclosure_on):
    rng = np.random.default_rng(1842)
    m = dm.Model()
    a = m.continuous("a", shape=sa, lb=-2.0, ub=3.0)
    b = m.continuous("b", shape=sb, lb=-1.5, ub=0.5)
    enc = evaluate_interval(a @ b, m)
    want = np.matmul(np.zeros(sa), np.zeros(sb)).shape
    assert enc.lo.shape == want and enc.hi.shape == want
    assert np.all(np.isfinite(enc.lo)) and np.all(np.isfinite(enc.hi))
    checks = 0
    for _ in range(2000):
        av = rng.uniform(-2.0, 3.0, sa)
        bv = rng.uniform(-1.5, 0.5, sb)
        val = av @ bv
        assert np.all(enc.lo <= val) and np.all(val <= enc.hi)
        checks += 1
    # Corners attain the interval dot-product extremes; check them too.
    for av in (np.full(sa, -2.0), np.full(sa, 3.0)):
        for bv in (np.full(sb, -1.5), np.full(sb, 0.5)):
            val = av @ bv
            assert np.all(enc.lo <= val) and np.all(val <= enc.hi)
            checks += 1
    assert checks == 2004


def test_constant_matmul_enclosure_is_tight(enclosure_on):
    enc = evaluate_interval(dm.sum(Constant(M) @ Constant(M)), None)
    assert float(enc.lo) == pytest.approx(2.0) and float(enc.hi) == pytest.approx(2.0)
    assert float(enc.lo) <= 2.0 <= float(enc.hi)


@pytest.mark.parametrize("sa,sb", SHAPES[:3], ids=[f"{a}@{b}" for a, b in SHAPES[:3]])
def test_flag_off_is_unbounded_in_the_product_shape(sa, sb, enclosure_off):
    # Bound-neutral against the pre-#1842 evaluator: still (-inf, inf), but in the
    # product's shape, so ``(a @ b)[0]`` no longer raises.
    m = dm.Model()
    a = m.continuous("a", shape=sa, lb=-2.0, ub=3.0)
    b = m.continuous("b", shape=sb, lb=-1.5, ub=0.5)
    enc = evaluate_interval(a @ b, m)
    assert enc.lo.shape == np.matmul(np.zeros(sa), np.zeros(sb)).shape
    assert np.all(np.isneginf(enc.lo)) and np.all(np.isposinf(enc.hi))


def test_flag_off_matrix_vector_is_still_finite(enclosure_off):
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-1.0, ub=2.0)
    enc = evaluate_interval(Constant(np.ones((2, 3))) @ x, m)
    assert enc.lo.shape == (2,) and np.all(np.isfinite(enc.lo)) and np.all(np.isfinite(enc.hi))


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize(
    "build,shape",
    [
        (lambda u: np.array([1.0, 2.0]) @ u, ()),
        (lambda u: u @ np.array([[0.0, 1.0], [1.0, 0.0]]), (2,)),
        (lambda u: np.array([[0.0, 1.0], [1.0, 0.0]]) @ u, (2,)),
        (lambda u: u @ np.zeros((2, 3)), (3,)),
    ],
    ids=["c@(1/x)", "(1/x)@M", "M@(1/x)", "(1/x)@zeros"],
)
def test_division_through_zero_then_matmul(build, shape, flag, monkeypatch):
    # ``Interval.__truediv__`` built its unbounded result in the numerator's 0-d
    # shape, so ``c @ (1/x)`` handed np.matmul a scalar and raised ValueError.
    monkeypatch.setenv(FLAG, flag)
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    u = 1.0 / x
    assert evaluate_interval(u, m).lo.shape == (2,)
    enc = evaluate_interval(build(u), m)
    assert enc.lo.shape == shape
    assert np.all(np.isneginf(enc.lo)) and np.all(np.isposinf(enc.hi))


def test_matmul_shape_tolerates_a_scalar_operand():
    # A 0-d operand whose shape cannot be recovered keeps the pre-#1842 rule
    # (``A.shape[:-1]``) instead of raising inside np.matmul.
    assert ie._matmul_shape((), (3,)) == ()
    assert ie._matmul_shape((2, 3), ()) == (2,)
    assert ie._matmul_shape((3,), (3, 4)) == (4,)


def test_batched_matmul_unbounded_with_product_shape():
    m = dm.Model()
    a = m.continuous("a", shape=(2, 2, 3), lb=0, ub=1)
    enc = evaluate_interval(a @ Constant(np.ones((3, 4))), m)
    assert enc.lo.shape == (2, 2, 4)


def test_sibling_fallbacks_report_broadcast_shape():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=2)
    enc = evaluate_interval(x ** Constant(np.array([1.0, 2.0, 3.0])), m)
    assert enc.lo.shape == (3,)
