"""A quadratic sign test must not allocate a whole-model Hessian (#1456).

``_quadratic_data`` hands back a dense ``(n_total, n_total)`` Q for *every*
call, and ``is_homogeneous_psd_quadratic`` / ``quadratic_curvature`` then throw
almost all of it away — they only need eigenvalue SIGNS on the support.  #814
and #1458 cut the eigendecomposition down to that support, which left the
allocation as the entire remaining cost.

Measured on ``glider400`` (5215 variables), one ``has_factorable_work`` scan:

    _quadratic_data          26.8 s over 401 calls
      of which 0.5*(Q+Q.T)   26.2 s   -- three dense 218 MB temporaries per call
    _quadratic_sign_form      0.002 s over the same 401 calls   (16,709x)
    has_factorable_work      27.4 s  ->  0.12 s                    (228x)

for quadratics that touch a handful of variables each.

Two negative results are pinned in the module docstring rather than here
because they produced no code (CLAUDE.md §4): restricting the support *before*
symmetrising while keeping the dense extraction measured 3.5x against a
pre-stated 5x kill criterion, and caching is worthless because all 401
expressions are distinct objects.  The allocation had to go.

The change is deliberately BIT-IDENTICAL, so #1456's gating change stays
bound-neutral (§5) and is verifiable by "nothing but the clock moved" rather
than by a differential panel.  That is what this file pins: the new path must
return the same submatrix, the same ``c`` and the same ``const`` as
``_support_restricted(_quadratic_data(...))`` — not merely the same
classification, which would let a drift hide until some tolerance straddled it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.convexity.lattice import Curvature  # noqa: E402
from discopt._relax.convexity.patterns import (  # noqa: E402
    _quadratic_data,
    _quadratic_sign_form,
    _support_restricted,
    is_homogeneous_psd_quadratic,
    quadratic_curvature,
)

pytestmark = pytest.mark.unit


def _old_form(expr, model):
    """Exactly what the two sign-test callers computed before #1456."""
    data = _quadratic_data(expr, model)
    if data is None:
        return None
    Q, c, const = data
    return _support_restricted(Q), np.asarray(c, dtype=np.float64), float(const)


def _wide_model(n: int = 400):
    """A model with many variables where each quadratic touches only a few —
    ``glider400``'s shape, which is what makes the whole-model Q absurd."""
    m = dm.Model("wide")
    x = m.continuous("x", shape=(n,), lb=-2.0, ub=2.0)
    return m, x


def _forms():
    """(label, expr, model) covering every branch of the new path."""
    m, x = _wide_model()
    return [
        # PSD: a sum of squares.
        ("psd", x[0] * x[0] + x[1] * x[1], m),
        # NSD: its negation.
        ("nsd", -(x[2] * x[2]) - x[3] * x[3], m),
        # Indefinite: a bilinear term, eigenvalues +-0.5.
        ("indefinite", x[4] * x[5], m),
        # Off-diagonal only in ONE triangle: pins the symmetrisation.
        ("one_triangle", 3.0 * x[6] * x[7] + x[6] * x[6], m),
        # With a linear part and a constant: pins c / const pass-through.
        ("with_linear", x[8] * x[8] + 5.0 * x[9] + 2.0, m),
        # Exactly zero quadratic part: the empty-support branch.
        ("affine_only", 4.0 * x[10] + 1.0, m),
        # Repeated variable across terms, so cells accumulate.
        ("accumulating", x[11] * x[11] + 2.0 * x[11] * x[11] + x[11] * x[12], m),
    ]


def test_sign_form_is_bit_identical_to_the_dense_path():
    """THE INVARIANT. Not "same classification" — same bytes."""
    compared = 0
    for label, expr, model in _forms():
        old, new = _old_form(expr, model), _quadratic_sign_form(expr, model)
        assert (old is None) == (new is None), f"{label}: extractability disagrees"
        if old is None:
            continue
        Qo, co, ko = old
        Qn, cn, kn = new
        assert Qo.shape == Qn.shape, f"{label}: shape {Qo.shape} vs {Qn.shape}"
        assert np.array_equal(Qo, Qn), f"{label}: submatrix differs\n{Qo}\n{Qn}"
        assert np.array_equal(co, cn), f"{label}: linear part differs"
        assert ko == kn, f"{label}: constant differs"
        compared += 1
    assert compared == len(_forms()) - 0, f"only {compared} forms compared"
    assert compared >= 7, "the form table shrank; this test is weaker than it reads"


def test_the_submatrix_is_support_sized_not_model_sized():
    """THE POINT. A 2-variable quadratic in a 400-variable model must produce a
    tiny matrix — otherwise the cost is still O(n_total**2) and nothing was
    fixed, however correct the answer is."""
    m, x = _wide_model(400)
    Q, _c, _const = _quadratic_sign_form(x[0] * x[0] + x[0] * x[1], m)
    assert Q.shape == (2, 2), f"support-sized submatrix expected, got {Q.shape}"


def test_classifications_are_unchanged():
    """The consumers, not just the helper: a bit-identical Q that some caller
    reads differently would still be a regression."""
    m, x = _wide_model()
    assert is_homogeneous_psd_quadratic(x[0] * x[0] + x[1] * x[1], m) is True
    assert is_homogeneous_psd_quadratic(x[2] * x[3], m) is False
    assert quadratic_curvature(x[0] * x[0] + x[1] * x[1], m) == Curvature.CONVEX
    assert quadratic_curvature(-(x[0] * x[0]) - x[1] * x[1], m) == Curvature.CONCAVE
    assert quadratic_curvature(x[4] * x[5], m) == Curvature.UNKNOWN
    assert quadratic_curvature(3.0 * x[6] + 1.0, m) == Curvature.AFFINE


def test_an_all_zero_quadratic_yields_an_empty_spectrum():
    """``_support_restricted`` returns 0x0 for an all-zero Q and lets callers
    decide what that means; the replacement must not start guessing instead."""
    m, x = _wide_model()
    Q, _c, _const = _quadratic_sign_form(4.0 * x[0] + 1.0, m)
    assert Q.shape == (0, 0)
    # The two callers' documented readings of an empty spectrum, unchanged.
    assert is_homogeneous_psd_quadratic(0.0 * x[0] * x[0], m) is True
    assert quadratic_curvature(4.0 * x[0] + 1.0, m) == Curvature.AFFINE


def test_a_non_quadratic_expression_is_refused():
    m, x = _wide_model()
    assert _quadratic_sign_form(x[0] * x[1] * x[2], m) is None
    assert _quadratic_data(x[0] * x[1] * x[2], m) is None, "the two paths must agree on refusal"


@pytest.mark.slow
def test_a_wide_model_scan_does_not_cost_the_whole_hessian():
    """END TO END: the defect as it showed up — pre-solve scanning a wide model
    with small quadratics. Ungated this is O(n_total**2) per sqrt node."""
    import time

    n = 1200
    m = dm.Model("wide_sqrt")
    x = m.continuous("x", shape=(n,), lb=0.5, ub=4.0)
    # Many sqrt nodes, each over a 2-variable quadratic: one _quadratic_data
    # call per node, each of which used to allocate an (n, n) dense matrix.
    for k in range(0, 200, 2):
        m.subject_to(dm.sqrt(x[k] * x[k] + x[k + 1] * x[k + 1]) <= 10.0)
    m.minimize(dm.sum([x[i] for i in range(n)]))

    from discopt._relax.factorable_reform import has_factorable_work

    t0 = time.perf_counter()
    has_factorable_work(m)
    dt = time.perf_counter() - t0
    # Generous: the point is that it is not quadratic in n_total. With the dense
    # path this shape took tens of seconds; the support path is milliseconds.
    assert dt < 10.0, f"scan took {dt:.1f}s on a {n}-variable model with 2-var quadratics"
