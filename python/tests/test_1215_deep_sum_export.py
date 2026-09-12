"""`sum()` over many variables must export to LP, MPS and GAMS (#1215).

The array scalarizer expands a reduction like ``dm.sum(x)`` over ``x`` of shape
``(n,)`` into a LEFT-DEEP ``+`` fold of depth ``n``. Three exporters descended
that one Python frame per term, so ``m.minimize(dm.sum(x))`` -- about the
commonest objective there is -- raised ``RecursionError`` from ``to_lp``,
``to_mps`` and ``to_gams`` once the model had roughly 1000 variables. ``.nl`` was
unaffected on both its writers, which is why it went unnoticed: the format the
solver path uses was fine, and the three that only matter on the way *out* were
not.

GAMS is the only route to a full-license BARON (the bundled AMPL binary is
demo-limited to 10 variables), so this also silently blocked BARON comparison
for any model with a sum objective.

The fold itself is deliberately NOT changed -- both ``.nl`` writers mirror its
exact shape and the emitted bytes depend on it (``performance-plan.md`` §46).
Only the traversal became iterative, and the emitted text is byte-identical: the
sizes below bracket the old limit so a regression shows up as an exception, and
`test_emitted_text_keeps_the_fold_nesting` pins the shape the bytes depend on.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.export import to_gams, to_lp, to_mps, to_nl
from discopt.modeling import Model

WRITERS = {"lp": to_lp, "mps": to_mps, "gams": to_gams, "nl": to_nl}
# 800 is under the old failure point, 2000 and 20000 well past it.
SIZES = (800, 2000, 20000)


def _sum_objective(n):
    m = Model("obj")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0, name="c")
    m.minimize(dm.sum(x))
    return m


def _sum_in_a_row(n):
    m = Model("row")
    y = m.continuous("y", shape=(n,), lb=0.0, ub=10.0)
    m.subject_to(dm.sum(y) <= 5.0, name="agg")
    m.minimize(y[0])
    return m


def _mixed_signs(n):
    """A `+`/`-` spine, so the flattening must carry each node's own operator."""
    m = Model("mixed")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    y = m.continuous("y", shape=(n,), lb=-2.0, ub=3.0)
    m.subject_to(x + 2.0 * y <= 5.0, name="c")
    m.minimize(dm.sum(x) - 2.0 * dm.sum(y) + 7.0)
    return m


SHAPES = {
    "sum objective": _sum_objective,
    "sum in a row": _sum_in_a_row,
    "mixed signs": _mixed_signs,
}


# The n=20000 MPS cases cost 8-18 s uninstrumented but 35-74 s under
# `--cov` (measured, M4 Pro), and CI's coverage lane runs `--timeout=120` on a
# slower runner -- so the default timeout was deciding this test rather than the
# assertion, exactly what the `python-correctness-slow` lane's `--timeout=1800`
# comment in `ci.yml` warns against. The assertion is unchanged and the sizes are
# untouched (`test_the_old_failure_point_is_actually_crossed` pins them); only the
# hang backstop is sized for an instrumented run.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("writer", sorted(WRITERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("n", SIZES)
def test_a_deep_sum_exports(writer, shape, n):
    text = WRITERS[writer](SHAPES[shape](n))
    assert text, f"{writer}/{shape}/n={n} produced no output"


def test_the_old_failure_point_is_actually_crossed():
    """§6: the sizes above must exceed Python's recursion limit, or the test is a no-op.

    If someone lowers SIZES below the limit the parametrised tests keep passing
    while testing nothing, so assert the premise rather than trusting it.
    """
    import sys

    assert max(SIZES) > sys.getrecursionlimit(), (
        f"largest size {max(SIZES)} no longer exceeds the recursion limit "
        f"{sys.getrecursionlimit()}; these tests would pass without the fix"
    )


def test_emitted_text_keeps_the_fold_nesting():
    """The traversal changed; the emitted text must not.

    Both `.nl` writers mirror the fold's shape, so a flattening that emitted
    `a + b + c` instead of `((a + b) + c)` would silently change every exported
    file. Checked on GAMS, which writes the nesting literally.
    """
    text = to_gams(_sum_objective(6))
    assert "((((" in text, f"the fold nesting is gone:\n{text[:400]}"


def test_quadratic_objective_with_a_deep_sum():
    """The quadratic extractor has its own descent; it needed the same fix."""
    n = 2000
    m = Model("quad")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0)
    m.subject_to(x <= 5.0, name="c")
    m.minimize(dm.sum(x) + x[0] * x[1])
    assert to_lp(m)
    assert to_mps(m)
