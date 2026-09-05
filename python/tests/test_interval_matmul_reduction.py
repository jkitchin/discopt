"""Soundness of ``_eval_matmul``'s interval enclosure (#1161).

``_relax/convexity/interval_eval.py::_eval_matmul`` reduces a ``k``-term dot
product with ``prods_lo.sum()`` / ``prods_hi.sum()``. Before this suite it
widened each endpoint by a **single** ``np.nextafter`` — the right convention
for a *binary* operation, but not a bound on the floating-point accumulation of
a ``k``-term sum whose summands are themselves rounded products. The returned
enclosure could therefore be strictly narrower than the true image.

``evaluate_interval`` is on the solve path (nonlinear bound tightening, the
uniform/OA relaxations, the g-convex injection, the interval-AD Hessian
propagator), where a too-narrow enclosure is an unsound FBBT tightening — the
class that cuts the optimum out of the box.

The reference is ``fractions.Fraction``, deliberately **not** ``np.longdouble``
(which is float64 on arm64, degrading the comparison to ``fl == fl`` and making
the probe measure nothing — CLAUDE.md §6).

``MatMulExpression`` comes from the Python ``@`` operator and the ``.nl`` reader
never emits one, so no ``.nl`` panel covers this class; these tests are the
coverage.
"""

from __future__ import annotations

import math
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.dag_compiler import compile_expression
from discopt.modeling.core import Model

# CLAUDE.md §6: a probe that silently compares nothing reads as a pass. Every
# exact comparison bumps this counter and ``teardown_module`` refuses a run in
# which it stayed at zero. It is a teardown hook rather than a test because this
# suite runs under ``pytest-randomly``, and as a test a shuffle could place it
# first — where the guard against measuring nothing would itself measure
# nothing.
_COMPARISONS = 0


def teardown_module(module):  # noqa: D103 - pytest hook
    print(f"\n[#1161] exact enclosure comparisons executed: {_COMPARISONS}")
    assert _COMPARISONS > 0, "the #1161 suite executed zero exact comparisons — it measured nothing"


# ─────────────────────────── exact reference ───────────────────────────


def _exact_image(A_lo, A_hi, b_lo, b_hi) -> tuple[Fraction, Fraction]:
    """Exact ``[min, max]`` of the interval dot product, summed in ``Fraction``.

    Each term ``a_j * b_j`` is independent of the others, so the image of the
    dot product over the box is exactly the sum of the term-wise extremes. No
    floating point enters: every product and every partial sum is rational.
    """
    lo = Fraction(0)
    hi = Fraction(0)
    for al, ah, bl, bh in zip(A_lo, A_hi, b_lo, b_hi):
        cands = [Fraction(float(a)) * Fraction(float(b)) for a in (al, ah) for b in (bl, bh)]
        lo += min(cands)
        hi += max(cands)
    return lo, hi


def _matmul_enclosure(A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Interval:
    """``evaluate_interval`` of ``A @ x`` for ``x`` in ``[lb, ub]``, via the API."""
    m = Model("mm")
    x = m.continuous("x", shape=(lb.size,), lb=lb, ub=ub)
    return evaluate_interval(A @ x, m)


# ─────────────────────────── the containment test ───────────────────────────


@pytest.mark.unit
def test_the_enclosure_contains_the_exact_dot_product_under_cancellation():
    """The enclosure must contain the exact image of ``A @ x`` over the box.

    Fails before the fix: with a single ``nextafter`` per endpoint, 166 of these
    400 products came back with an enclosure that did **not** contain the truth,
    worst shortfall 8.5e-07.
    """
    global _COMPARISONS
    rng = np.random.default_rng(1161)
    misses = 0
    worst = 0.0
    for _ in range(400):
        k = int(rng.integers(4, 400))
        # Heavy cancellation: terms ~1e8 in magnitude summing to ~1e9, so the
        # accumulated rounding is many ULPs of the result.
        A = rng.normal(0.0, 1e4, size=(1, k))
        center = rng.normal(0.0, 1e4, size=k)
        half = np.abs(rng.normal(0.0, 1e2, size=k))
        lb, ub = center - half, center + half

        enc = _matmul_enclosure(A, lb, ub)
        lo = float(np.asarray(enc.lo).reshape(()))
        hi = float(np.asarray(enc.hi).reshape(()))
        exact_lo, exact_hi = _exact_image(A[0], A[0], lb, ub)

        _COMPARISONS += 1
        short = max(float(Fraction(lo) - exact_lo), float(exact_hi - Fraction(hi)), 0.0)
        if short > 0.0:
            misses += 1
            worst = max(worst, short)

    assert misses == 0, (
        f"matmul enclosure missed the exact image on {misses}/400 products, "
        f"worst shortfall {worst:.3e}"
    )


@pytest.mark.unit
def test_variable_times_variable_matmul_also_encloses_the_exact_image():
    """The same guarantee when *both* operands are variables (``Y @ x``).

    The constant-matrix case above exercises a degenerate left interval; this one
    makes every term a genuine interval product, so the enclosure's four-product
    min/max is on the path too.
    """
    global _COMPARISONS
    rng = np.random.default_rng(11610)
    for _ in range(40):
        k = int(rng.integers(4, 200))
        a_c = rng.normal(0.0, 1e3, size=k)
        a_h = np.abs(rng.normal(0.0, 1e2, size=k))
        b_c = rng.normal(0.0, 1e3, size=k)
        b_h = np.abs(rng.normal(0.0, 1e2, size=k))

        m = Model("mm2")
        Y = m.continuous("Y", shape=(1, k), lb=(a_c - a_h)[None, :], ub=(a_c + a_h)[None, :])
        x = m.continuous("x", shape=(k,), lb=b_c - b_h, ub=b_c + b_h)
        enc = evaluate_interval(Y @ x, m)

        lo = float(np.asarray(enc.lo).reshape(()))
        hi = float(np.asarray(enc.hi).reshape(()))
        exact_lo, exact_hi = _exact_image(a_c - a_h, a_c + a_h, b_c - b_h, b_c + b_h)
        _COMPARISONS += 1
        assert Fraction(lo) <= exact_lo, f"lo {lo!r} above the exact minimum"
        assert Fraction(hi) >= exact_hi, f"hi {hi!r} below the exact maximum"


# ─────────────────────── the widening must stay small ───────────────────────


@pytest.mark.unit
def test_widening_is_a_relative_epsilon_not_a_visible_loss():
    """Control: the fix must widen, but only by ``O(eps · Σ|terms|)``.

    A blanket ``[-inf, +inf]`` would satisfy every containment assertion above
    vacuously. This pins the other side: the enclosure stays within a relative
    ``1e-12`` of the exact image, so the bound it feeds is unchanged for any
    practical purpose.
    """
    global _COMPARISONS
    rng = np.random.default_rng(42)
    worst_rel = 0.0
    for _ in range(60):
        k = int(rng.integers(4, 400))
        A = rng.normal(0.0, 1e4, size=(1, k))
        center = rng.normal(0.0, 1e4, size=k)
        half = np.abs(rng.normal(0.0, 1e2, size=k))
        enc = _matmul_enclosure(A, center - half, center + half)
        lo = float(np.asarray(enc.lo).reshape(()))
        hi = float(np.asarray(enc.hi).reshape(()))
        exact_lo, exact_hi = _exact_image(A[0], A[0], center - half, center + half)
        exact_width = float(exact_hi - exact_lo)
        assert exact_width > 0.0
        _COMPARISONS += 1
        worst_rel = max(worst_rel, ((hi - lo) - exact_width) / exact_width)
    assert worst_rel < 1e-12, f"enclosure inflated by {worst_rel:.3e} of its width"


@pytest.mark.unit
def test_small_exact_matmul_stays_exact():
    """A tiny integer-valued product must still read as its exact range.

    ``A = [1, -2, 3]``, ``x ∈ [-1, 2]^3`` has image ``[-8, 10]`` exactly.
    """
    global _COMPARISONS
    A = np.array([[1.0, -2.0, 3.0]])
    enc = _matmul_enclosure(A, np.full(3, -1.0), np.full(3, 2.0))
    lo = float(np.asarray(enc.lo).reshape(()))
    hi = float(np.asarray(enc.hi).reshape(()))
    _COMPARISONS += 1
    assert -8.0 - 1e-12 <= lo <= -8.0
    assert 10.0 <= hi <= 10.0 + 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("k", [8, 128, 1024, 4096])
def test_point_box_stays_degenerate_within_the_point_evaluation_slack(k):
    """Item 3 of #1161: the widening must not break point evaluation.

    A residual reporter evaluates an expression at a *point* by handing the
    interval evaluator a degenerate box and refusing anything that does not come
    back degenerate — ``width <= 1e-6 * max(1, |mid|)`` (``mpec_report``'s
    ``_POINT_EVAL_REL_WIDTH``; the criterion is replicated here so the property
    is pinned on this tree regardless of merge order). The accumulation widening
    is relative to ``Σ|terms|``, so it must stay far under that slack at large
    ``k``, and the midpoint must still be the true dot product.
    """
    global _COMPARISONS
    rng = np.random.default_rng(1000 + k)
    A = rng.normal(0.0, 1.0, size=(1, k))
    point = rng.normal(0.0, 1.0, size=k)
    enc = _matmul_enclosure(A, point, point)
    lo = float(np.asarray(enc.lo).reshape(()))
    hi = float(np.asarray(enc.hi).reshape(()))
    mid = 0.5 * (lo + hi)
    slack = 1e-6 * max(1.0, abs(mid))
    _COMPARISONS += 1
    assert hi - lo <= slack, (
        f"k={k}: enclosure width {hi - lo:.3e} exceeds the point-evaluation slack {slack:.3e}"
    )
    exact = sum(
        (Fraction(float(a)) * Fraction(float(p)) for a, p in zip(A[0], point)),
        Fraction(0),
    )
    assert Fraction(lo) <= exact <= Fraction(hi)


@pytest.mark.unit
def test_an_unbounded_operand_widens_to_infinity_not_nan():
    """A non-finite term must widen the endpoint, never produce ``inf - inf``.

    The accumulation term is ``factor * Σ|terms|``; with an infinite bound in the
    box that sum is ``+inf``, and subtracting it from an infinite total would
    give ``nan`` — an enclosure that compares false against everything and so
    silently stops enclosing anything.
    """
    global _COMPARISONS
    A = np.array([[1.0, 1.0, 1.0]])
    enc = _matmul_enclosure(A, np.array([-np.inf, 0.0, 0.0]), np.array([1.0, 1.0, np.inf]))
    lo = float(np.asarray(enc.lo).reshape(()))
    hi = float(np.asarray(enc.hi).reshape(()))
    _COMPARISONS += 1
    assert not math.isnan(lo) and not math.isnan(hi)
    assert lo == -np.inf and hi == np.inf


# ─────────────────── enclosure vs. real pointwise values ───────────────────


@pytest.mark.unit
def test_enclosure_contains_sampled_values_of_a_nonlinear_matmul_model():
    """Base contract: every pointwise value in the box lies inside the enclosure.

    ``sin(A @ x)`` puts the matmul enclosure underneath a nonlinear atom, which
    is how ``discopt.nn`` and ``discopt.dae`` models reach this code. The check is
    per output element; ``dm.sum`` is deliberately not used, because
    ``SumExpression`` is a separate unreduced-enclosure defect (#1158) that this
    change neither causes nor fixes.
    """
    global _COMPARISONS
    rng = np.random.default_rng(7)
    k = 24
    A = rng.normal(0.0, 2.0, size=(3, k))
    m = Model("nl")
    x = m.continuous("x", shape=(k,), lb=-1.0, ub=1.5)
    expr = dm.sin(A @ x)
    enc = evaluate_interval(expr, m)
    lo = np.asarray(enc.lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(enc.hi, dtype=np.float64).reshape(-1)
    assert lo.size == A.shape[0]
    f = compile_expression(expr, m)
    for _ in range(64):
        pt = rng.uniform(-1.0, 1.5, size=k)
        vals = np.asarray(f(pt), dtype=np.float64).reshape(-1)
        for j, val in enumerate(vals):
            _COMPARISONS += 1
            assert lo[j] <= val <= hi[j], (
                f"row {j}: value {val} outside enclosure [{lo[j]}, {hi[j]}]"
            )


# ───────────────── differential: bound vs. the true optimum ─────────────────


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dual_bound_never_exceeds_the_true_optimum_on_matmul_models(seed):
    """Item 4 of #1161: the solve-path differential check the ``.nl`` panel cannot do.

    ``MatMulExpression`` is unreachable from a ``.nl`` file, so no panel or gate
    covers models built with ``A @ x``. These are bound-constrained linear
    least-squares problems — convex, so ``scipy.optimize.lsq_linear`` gives the
    true optimum independently — and the dual bound must never exceed it.
    """
    from scipy.optimize import lsq_linear

    rng = np.random.default_rng(seed)
    n, rows = 5, 4
    A = rng.normal(size=(rows, n))
    b = rng.normal(size=rows)
    lb, ub = np.full(n, -1.0), np.full(n, 1.0)

    m = Model(f"mmsolve{seed}")
    x = m.continuous("x", shape=(n,), lb=lb, ub=ub)
    resid = A @ x - b
    m.minimize(dm.sum(resid * resid))
    res = m.solve(time_limit=60)

    ref = lsq_linear(A, b, bounds=(lb, ub), tol=1e-14)
    true_opt = float(np.sum((A @ ref.x - b) ** 2))

    # Not `if res.bound is not None` — a run that produced no bound would then
    # pass while measuring nothing (CLAUDE.md §6).
    assert res.status == "optimal", f"seed {seed}: status {res.status!r}"
    assert res.bound is not None and res.objective is not None

    tol = 1e-7 * max(1.0, abs(true_opt))
    globals()["_COMPARISONS"] += 1
    assert float(res.bound) <= true_opt + tol, (
        f"seed {seed}: dual bound {res.bound!r} exceeds the true optimum "
        f"{true_opt!r} — the bound cut the optimum out of the box"
    )
    assert float(res.objective) >= true_opt - tol, (
        f"seed {seed}: incumbent {res.objective!r} is below the true optimum {true_opt!r}"
    )
