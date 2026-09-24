"""#1456: ``is_homogeneous_psd_quadratic`` must not decompose the whole model.

#814 restricted ``quadratic_curvature``'s ``eigvalsh`` to the nonzero SUPPORT of
``Q`` — ``_quadratic_data`` returns ``Q`` as a full ``n_total x n_total`` dense
matrix, so a quadratic touching three variables inside a 2186-variable model
still cost an ``O(n_total**3)`` eigendecomposition.  Its neighbour
``is_homogeneous_psd_quadratic`` was missed.

Measured on MINLPLib's ``glider400`` (5215 variables): 20 calls, each a
5215x5215 ``eigvalsh``, spending 119.47 s — 97.6% of a solve given a
``time_limit`` of 20 s.  This is a pre-solve/relaxation-build path, so the time
limit cannot reach it.  After the fix the largest matrix decomposed is 2x2 and
the total ``eigvalsh`` time is 0.01 s.

The restriction is *exact* here, not an approximation: the omitted rows and
columns are all-zero and contribute exactly-zero eigenvalues, which cannot flip
a ``min >= -1e-10`` sign test.  These tests pin both halves — the verdicts are
unchanged (``test_..._verdicts_are_identical_to_the_full_matrix``) and the cost
is the support, not the model (``test_..._decomposes_only_the_support``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.convexity import patterns as pat  # noqa: E402

pytestmark = pytest.mark.unit

N_EXTRA = 120


def _wide_model():
    """x, y, z plus ``N_EXTRA`` untouched variables, so any quadratic below has
    support 1-3 inside a (N_EXTRA + 3)-dimensional ``Q``."""
    m = dm.Model("psd1456")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=-2.0, ub=2.0)
    for i in range(N_EXTRA):
        m.continuous(f"w{i}", lb=-1.0, ub=1.0)
    return m, x, y, z


def _cases(x, y, z):
    """(expression, expected PSD verdict).  Homogeneous only — a linear or
    constant term makes the function return False before any eigenproblem."""
    return [
        (x * x + 2.0 * (y * y), True),
        (x * x + y * y + z * z, True),
        (x * x - y * y, False),  # indefinite
        (-(x * x) - (y * y), False),  # negative definite
        (x * x + 2.0 * (x * y) + y * y, True),  # (x+y)^2, PSD but singular
        (x * x + 4.0 * (x * y) + y * y, False),  # off-diagonal dominates
    ]


class _EigSpy:
    """Record every matrix ``patterns`` hands to ``eigvalsh``.

    Asserted non-empty by both users (CLAUDE.md §6): a spy that records nothing
    would make the dimension assertion below vacuously true.
    """

    def __enter__(self) -> "_EigSpy":
        self.shapes: list[tuple[int, ...]] = []
        self._real = pat.np.linalg.eigvalsh

        def spy(a, *args, **kwargs):
            self.shapes.append(np.shape(a))
            return self._real(a, *args, **kwargs)

        self._patch = pytest.MonkeyPatch()
        self._patch.setattr(pat.np.linalg, "eigvalsh", spy)
        return self

    def __exit__(self, *exc) -> None:
        self._patch.undo()


def test_verdicts_are_identical_to_the_full_matrix():
    """CORRECTNESS. The support restriction changes no classification.

    The reference is computed here from the *full* ``Q`` that ``_quadratic_data``
    returns — the exact computation the function used to do — so this compares
    the fix against the thing it replaced rather than against a hand-written
    expectation.
    """
    m, x, y, z = _wide_model()
    checked = 0
    for expr, expected in _cases(x, y, z):
        data = pat._quadratic_data(expr, m)
        assert data is not None, "fixture is not a quadratic any more"
        full_q, c, const = data
        assert np.allclose(c, 0.0) and abs(const) <= 1e-10, "fixture is not homogeneous"
        assert full_q.shape[0] == N_EXTRA + 3, f"model shrank to {full_q.shape[0]} vars"
        reference = bool(float(np.min(np.linalg.eigvalsh(full_q))) >= -1e-10)
        assert reference is expected, f"the test's own expectation is wrong for {expr}"
        assert pat.is_homogeneous_psd_quadratic(expr, m) is reference
        checked += 1
    assert checked == len(_cases(x, y, z))


def test_decomposes_only_the_support():
    """THE REGRESSION. ``glider400`` decomposed 5215x5215 once per ``sqrt`` node."""
    m, x, y, z = _wide_model()
    with _EigSpy() as spy:
        for expr, _ in _cases(x, y, z):
            pat.is_homogeneous_psd_quadratic(expr, m)
    assert spy.shapes, "no eigendecomposition happened at all: the spy is not wired up"
    worst = max(s[0] for s in spy.shapes)
    assert worst <= 3, (
        f"decomposed a {worst}x{worst} matrix for a quadratic of support <= 3 "
        f"inside a {N_EXTRA + 3}-variable model"
    )


def test_the_sum_of_squares_branch_is_also_restricted():
    """The ``sum((A@x)*(A@x))`` branch formed the full Gram matrix ``A^T A``,
    with a row and column per variable in the model — the same blow-up on a
    different branch.  Dropping ``A``'s all-zero columns is the same exact
    restriction, so the verdict is still computed, not assumed."""
    m = dm.Model("sos1456")
    v = m.continuous("v", shape=(3,), lb=-1.0, ub=1.0)
    for i in range(N_EXTRA):
        m.continuous(f"w{i}", lb=-1.0, ub=1.0)
    with _EigSpy() as spy:
        assert pat.is_homogeneous_psd_quadratic(dm.sum(v * v), m) is True
    assert spy.shapes, "the sum-of-squares branch was not taken"
    worst = max(s[0] for s in spy.shapes)
    assert worst <= 3, f"formed a {worst}x{worst} Gram matrix for 3 variables"


def test_a_full_support_quadratic_is_unchanged():
    """CHARACTERIZATION: when the support IS the model, nothing is restricted and
    the old and new code are the same code."""
    m = dm.Model("full1456")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    with _EigSpy() as spy:
        assert pat.is_homogeneous_psd_quadratic(x * x + y * y, m) is True
        assert pat.is_homogeneous_psd_quadratic(x * x - y * y, m) is False
    assert spy.shapes and max(s[0] for s in spy.shapes) == 2
