"""#863: QP extraction cost must not scale with variables the objective ignores.

The extractor this file was written against recovered off-diagonal Q entries with
one model evaluation per variable pair::

    Q[i,j] = f(e_i + e_j) - f(e_i) - f(e_j) + d

Sweeping all pairs is O(n^2) evaluations, each allocating an O(n) probe vector. A
variable that does not appear in the objective has ``f(e_j) == f(-e_j) == d``, so
every product involving it is identically zero — probing that pair is pure waste.

That extractor is **gone**: ``extract_qp_data`` now reads the coefficients off the
expression arena and never evaluates the model. The tests below were written
against the public dispatcher, not the probe, so they still say exactly what they
said — and the cost assertion is now satisfied with room to spare, because the
walk is O(nodes) rather than O(n) in the diagonal too. Keeping them is the
regression guard that the ladder's *first* rung stays cheap: a silent fall-through
to a per-variable rung would put the old scaling back.

This bites hard on wide models with a narrow objective. ``watercontamination0202``
has **106,711 variables whose objective touches only 101 of them** (measured: 101
nonzeros in the objective gradient), so the sweep issues **5.69e9** probes to
discover that ~10^4 entries could be nonzero. Measured scaling of the old sweep was
~8x per doubling, extrapolating to tens of hours at that n — which is why the solve
blew past its ``time_limit`` by >8x *without ever reaching a solver*.

The fix restricts the pair loop to the objective's support, which is free: the
``f(e_j)`` / ``f(-e_j)`` probes that identify it are already computed in O(n) for
the diagonal.

Measured A/B on identical models (objective touches 5 of n variables):

======  =========  =========  =======
n       before     after      speedup
======  =========  =========  =======
200     0.017 s    0.003 s     5.7x
400     0.068 s    0.005 s    13.6x
800     0.341 s    0.016 s    21x
1600    1.980 s    0.051 s    **39x**
======  =========  =========  =======

That A/B measured the pair-sweep fix alone, which removed the quadratic term from
the cost but not the linear one: at n = 106,711 the remaining O(n) diagonal
probing (~213k objective evaluations) and the dense ``(n, n)`` ``Q`` allocation
(91 GB) were still fatal. Both are since resolved — the sparse ``QPData``
representation for the second, and deleting the probe outright for the first.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.problem_classifier import extract_qp_data  # noqa: E402


def _wide_model_narrow_objective(n: int, support: int = 5):
    """``n`` variables, but the objective involves only the first ``support``."""
    m = dm.Model(f"wide{n}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    m.minimize(sum((xs[i] - 1.0) ** 2 for i in range(support)))
    m.subject_to(sum(xs) >= 1)
    return m


@pytest.mark.parametrize("n", [200, 800])
def test_extraction_is_correct_on_a_narrow_objective(n):
    """Correctness first: restricting the sweep must not change the recovered form.
    ``sum (x_i - 1)^2`` over the first 5 variables gives Q = 2I on those, c = -2
    there and 0 elsewhere."""
    data = extract_qp_data(_wide_model_narrow_objective(n))
    q = np.asarray(data.Q)
    c = np.asarray(data.c).ravel()
    assert int((q != 0).sum()) == 5, f"expected 5 nonzero Q entries, got {int((q != 0).sum())}"
    for i in range(5):
        assert q[i, i] == pytest.approx(2.0, rel=1e-9)
        assert c[i] == pytest.approx(-2.0, rel=1e-9)
    assert np.allclose(c[5:n], 0.0), "variables outside the objective picked up coefficients"


def test_cost_does_not_scale_quadratically_with_unrelated_variables():
    """Adding variables the objective never mentions must not blow up extraction.

    Pre-fix this was O(n^2) probes: 0.068 s at n=400 rising to 1.98 s at n=1600
    (~4x per doubling). With the support restriction it was 0.005 s -> 0.051 s;
    with the probe deleted the walk does not visit an absent variable at all.

    The assertion is a ratio rather than an absolute time so it does not encode this
    machine's speed; a quadratic sweep cannot come near it even on a slow runner.
    """
    t0 = time.perf_counter()
    extract_qp_data(_wide_model_narrow_objective(400))
    t_small = time.perf_counter() - t0

    t0 = time.perf_counter()
    extract_qp_data(_wide_model_narrow_objective(1600))
    t_large = time.perf_counter() - t0

    # 4x the variables. Quadratic would be ~16x; measured post-fix is ~10x on the
    # O(n) diagonal scan, and was ~29x pre-fix. 14x separates them with margin.
    ratio = t_large / max(t_small, 1e-6)
    assert ratio < 14.0, (
        f"extraction cost grew {ratio:.1f}x for 4x the variables — the pair sweep "
        "looks quadratic again (pre-fix this was ~29x)"
    )
