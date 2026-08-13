"""#1017: the Farkas certificate may not be built from result magnitudes.

``farkas_ray_certifies_cols`` fathoms a node by proving ``bᵀy > max_x (Aᵀy)ᵀx`` — a
strict inequality between two floating-point *accumulations*. Pre-#1017 the margin
that kept it strict was ``1e-9·(1 + |bᵀy| + Σ|boxmax_j|)``: scaled off the sizes of
the **results**, which bound the rounding of nothing. On the reported relaxation LP
``bᵀy = 3e-8`` came out of terms whose absolute sum was ``600`` — 5e-11 relative,
pure rounding — and the engine returned ``LpStatus::Infeasible`` for an LP that
SciPy/HiGHS, three perturbed pivot paths of the same engine, and an elastic
feasibility LP all solve feasible. A node-LP ``Infeasible`` is a fathoming proof, so
that is a region of the search space deleted on a rounding artifact (CLAUDE.md §1).

This test pins the fix at the shipped-extension level on an LP small enough to
reason about exactly, and feasible by construction rather than by an external
oracle:

* ``A_j = e_j − e_{j+1}`` (a path incidence matrix), so ``Aᵀ1 = 0`` exactly and
  ``range(A) = {v : Σv_i = 0}``;
* ``b = [1e16, 1.5×8, −1e16, −12]`` sums to **exactly zero** as a real number, so
  ``b ∈ range(A)``: ``Ax = b`` has a real solution (``x_j = Σ_{i≤j} b_i``), well
  inside the box.

Yet the naive left-to-right accumulation the check performs returns ``+4``, because
each ``1.5`` added onto ``1e16`` — where the ulp is ``2`` — rounds up by ``0.5``.
Four is 1e9× the old margin, so the certificate was issued. The rigorous margin
scales off ``Σ|b_i·y_i| = 2e16`` (``γ_12 · 2e16 ≈ 53``), and rejects it.

The engine may report ``optimal`` or the honest, non-fathoming ``numerical`` here
(the exact solution is not representable at this scale); the one verdict it may
never return is ``infeasible``. Measured on the pre-#1017 extension: ``infeasible``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._rust import solve_lp_py

pytestmark = pytest.mark.smoke


def _cancellation_lp(big: float = 1e16, k: int = 8, step: float = 1.5):
    """A feasible standard-form LP whose ``bᵀ1`` is exactly 0 but sums to > 0."""
    smalls = [step] * k
    b = np.array([big, *smalls, -big, -float(sum(smalls))])
    m = b.size
    n = m - 1
    a = np.zeros((m, n))
    for j in range(n):
        a[j, j] = 1.0
        a[j + 1, j] = -1.0
    w = 4.0 * (big + sum(smalls))
    return a, b, np.full(n, -w), np.full(n, w), np.zeros(n)


def test_exactly_zero_row_side_is_not_a_certificate() -> None:
    a, b, lo, hi, c = _cancellation_lp()

    # The premise, asserted rather than asserted-about: the certificate quantity is
    # exactly zero (math.fsum is exact), while the naive accumulation reports +4.
    assert math_fsum(b) == 0.0
    naive = 0.0
    for v in b:
        naive += v
    assert naive > 1e-9 * (1.0 + abs(naive)), (
        f"premise: the naive bᵀy ({naive}) must clear the pre-#1017 margin"
    )

    status, x, _obj, _iters = solve_lp_py(c, a, b, lo, hi)
    assert status != "infeasible", (
        "a feasible LP (b ⊥ 1 exactly, range(A) = 1⊥) was fathomed on a rounding "
        f"artifact; got {status} with x={x}"
    )


@pytest.mark.parametrize("big", [1e12, 1e14, 1e16, 1e18])
@pytest.mark.parametrize("k", [3, 8, 17])
def test_cancellation_family_is_never_fathomed(big: float, k: int) -> None:
    """The class, not the instance: sweep the magnitude gap and the term count."""
    a, b, lo, hi, c = _cancellation_lp(big=big, k=k, step=2.5)
    assert math_fsum(b) == 0.0, "generator: b must sum to exactly 0"
    status, _x, _obj, _iters = solve_lp_py(c, a, b, lo, hi)
    assert status != "infeasible", f"false fathom at big={big:g}, k={k}"


def math_fsum(values) -> float:
    """Exact (correctly-rounded) sum — the ground truth the naive one is compared to."""
    import math

    return math.fsum(float(v) for v in values)
