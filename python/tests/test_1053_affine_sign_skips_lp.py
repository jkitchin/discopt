"""#1053: sign queries must not pay for an LP the variable box answers.

``_refine_sign`` asks one question -- is this affine argument strictly
positive / negative / zero? -- and used to answer it via
``LinearContext.affine_range``, which solves a POUNCE LP pair whenever the
model has any linear row. But ``affine_range`` intersects the LP result with
the free box-only enclosure, so the refined interval is a *subset* of the box
interval. When the box alone already gives a strict sign, the LP cannot change
the answer.

Measured on MINLPLib ``hda`` (722 vars): 12 ``affine_range`` calls at ~1.05 s
each -- 12.7 s, 21% of a 60 s solve -- and all 12 were box-conclusive, with the
LP-refined sign equal to the box sign every time.

The tests below pin both halves: the LP is skipped when the box settles the
sign, and still runs when it does not. The second is the anti-vacuity control:
without it, deleting the LP path entirely would pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._relax.convexity.lattice import Sign
from discopt._relax.convexity.linear_context import LinearContext, build_linear_context


def _context_with_a_linear_row(lb: float, ub: float) -> LinearContext:
    """A 2-var model with one linear row, so the LP path is reachable.

    ``affine_range`` short-circuits to the box when there are no linear
    constraints at all, so a row is required for these tests to mean
    anything.
    """
    m = Model("ctx")
    x = m.continuous("x", lb=lb, ub=ub)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y <= ub + 1.0)
    ctx = build_linear_context(m)
    assert ctx is not None
    assert ctx.A_ub.size or ctx.A_eq.size, "no linear row: the LP path is unreachable"
    return ctx


@pytest.mark.smoke
def test_box_conclusive_sign_does_not_solve_an_lp(monkeypatch):
    """x in [2, 5] makes ``x`` strictly positive without any LP."""
    ctx = _context_with_a_linear_row(2.0, 5.0)
    coeffs = np.array([1.0, 0.0])

    calls = 0

    def counting_affine_range(self, c, k):
        nonlocal calls
        calls += 1
        raise AssertionError("affine_sign solved an LP for a box-conclusive sign (#1053)")

    monkeypatch.setattr(LinearContext, "affine_range", counting_affine_range)

    assert ctx.affine_sign(coeffs, 0.0) is Sign.POS
    assert calls == 0

    # Same expression shifted negative, and the ZERO case: all strict, all free.
    assert ctx.affine_sign(-coeffs, 0.0) is Sign.NEG
    assert ctx.affine_sign(np.zeros(2), 0.0) is Sign.ZERO
    assert calls == 0


@pytest.mark.smoke
def test_box_inconclusive_sign_still_uses_the_lp(monkeypatch):
    """ANTI-VACUITY CONTROL: the LP path must survive the short-circuit.

    ``x`` spans zero, so the box yields ``UNKNOWN`` and the linear rows are
    the only thing that could sharpen it. If the short-circuit swallowed
    this case the refinement would be silently dead.
    """
    ctx = _context_with_a_linear_row(-3.0, 5.0)
    coeffs = np.array([1.0, 0.0])

    calls = 0
    real = LinearContext.affine_range

    def counting_affine_range(self, c, k):
        nonlocal calls
        calls += 1
        return real(self, c, k)

    monkeypatch.setattr(LinearContext, "affine_range", counting_affine_range)

    ctx.affine_sign(coeffs, 0.0)
    assert calls == 1, "the LP refinement was skipped on a box-inconclusive sign"


@pytest.mark.smoke
def test_classify_model_does_not_solve_an_lp_for_a_box_positive_denominator(monkeypatch):
    """End-to-end: the solve path itself stops paying for these LPs.

    This is the test that fails before the fix -- ``_refine_sign`` called
    ``affine_range`` unconditionally, so classifying ``1/(x - y)`` solved two
    LPs to learn what the declared bounds already say.

    The denominator has to be a *combination*: the syntactic walker reads a
    lone ``x`` with ``lb=2`` as POS directly and never asks for refinement,
    while ``x - y`` is syntactically UNKNOWN yet box-conclusively POS -- which
    is the shape ``hda`` presents.
    """
    from discopt._relax.convexity import rules

    m = Model("div")
    x = m.continuous("x", lb=3.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y <= 6.0)
    m.minimize(1.0 / (x - y) + y)

    refinements = 0
    real_refine = rules._refine_sign

    def counting_refine(expr, model, cache, current_sign):
        nonlocal refinements
        refinements += 1
        return real_refine(expr, model, cache, current_sign)

    def forbidden_affine_range(self, c, k):
        raise AssertionError("classify_model solved an LP for a box-positive denominator (#1053)")

    monkeypatch.setattr(rules, "_refine_sign", counting_refine)
    monkeypatch.setattr(LinearContext, "affine_range", forbidden_affine_range)

    rules.classify_model(m)

    # CLAUDE.md 6: without this the test passes when the refinement path is
    # never reached at all, which would make the LP assertion meaningless.
    assert refinements > 0, "_refine_sign never ran -- the probe measured nothing"


@pytest.mark.smoke
def test_affine_sign_agrees_with_affine_range_everywhere():
    """The short-circuit is exact, not an approximation.

    Sweeps boxes that are strictly positive, strictly negative, zero-width
    and straddling; ``affine_sign`` must equal the sign of the full
    ``affine_range`` in every one.
    """
    cases = [(2.0, 5.0), (-5.0, -2.0), (0.0, 0.0), (-3.0, 5.0), (0.0, 4.0), (-4.0, 0.0)]
    coeff_sets = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([1.0, 1.0])]
    consts = [0.0, 1.5, -1.5]

    from discopt._relax.convexity.lattice import sign_from_bounds

    compared = 0
    for lb, ub in cases:
        ctx = _context_with_a_linear_row(lb, ub)
        for coeffs in coeff_sets:
            for const in consts:
                expected = sign_from_bounds(*ctx.affine_range(coeffs, const))
                assert ctx.affine_sign(coeffs, const) is expected, (
                    f"box=[{lb},{ub}] coeffs={coeffs} const={const}"
                )
                compared += 1

    # CLAUDE.md 6: a comparison count, so a loop that silently ran zero
    # times cannot read as a pass.
    assert compared == len(cases) * len(coeff_sets) * len(consts) == 54
