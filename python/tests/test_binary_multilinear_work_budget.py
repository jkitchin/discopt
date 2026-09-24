"""A size budget does not bound work: the multilinear expansion needs a work one.

``binary_multilinear_reform`` bounded the *size* of what it produced
(``_MAX_MONOMIALS``) and the *work* of one product (``_MAX_PRODUCT_OPS``), but
nothing bounded the cumulative work of the ``flat = _poly_add(flat, ...)``
accumulate loop.  ``_poly_add`` starts with ``out = dict(p)``, so accumulating a
polynomial that sits just *under* the monomial budget is quadratic in the number
of addends — and every individual call passes every existing check.

Measured on MINLPLib's ``hadamard_9`` with the work budget lifted out of the way:
200,000 adds copying 2.0e10 entries over 154 s, ending in
``_Unsupported("monomial budget exceeded")``.  It never had a reformulation to
deliver — the whole 154 s was spent arriving at "unchanged".  It is a pre-solve
pass, so a ``time_limit`` cannot reach it (issue #1456).

Charging adds to the same running total as products (``_MAX_EXPAND_OPS``) reaches
that same answer in 4.6 s.  The budget value is unchanged at 10,000,000 — only
what counts against it changed — and that is deliberate: surveyed over all 1610
MINLPLib instances, the pass fires on 52 and the most cumulative work any
instance that actually *gets* a reformulation asks for is 959,794
(``autocorr_bern35-09``), a 10.4x margin.  So no corpus instance loses a
reformulation; see ``test_a_reformulated_model_is_nowhere_near_the_budget``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax import binary_multilinear_reform as bml  # noqa: E402

pytestmark = pytest.mark.unit


def _accumulate_model(n_addends: int, n_factors: int = 3, width: int = 8):
    """A constraint body that is a long sum of products of sums over binaries.

    Each addend expands to ``width ** n_factors`` monomials over its own
    variable block, so the addends contribute *distinct* monomials and the
    accumulator ``flat`` grows linearly in ``n_addends`` — making the accumulate
    loop quadratic.  This is ``hadamard_9``'s shape, not a contrived one.
    """
    m = dm.Model("accumulate")
    per = n_factors * width
    b = m.binary("b", shape=(n_addends * per,))
    body = None
    for j in range(n_addends):
        prod = None
        for f in range(n_factors):
            s = None
            for w in range(width):
                v = b[j * per + f * width + w]
                s = v if s is None else s + v
            prod = s if prod is None else prod * s
        body = prod if body is None else body + prod
    m.subject_to(body <= float(n_addends))
    m.minimize(dm.sum([b[i] for i in range(n_addends * per)]))
    return m


class _Counter:
    """Instrument ``_poly_add``/``_poly_mul`` in place.

    Counts are asserted non-zero by every test that uses them (CLAUDE.md §6): an
    instrument that silently measures nothing reads as a pass.
    """

    def __init__(self) -> None:
        self.add_work = 0
        self.mul_work = 0
        self.calls = 0
        self.max_call = 0
        self.max_result = 0

    def __enter__(self) -> "_Counter":
        self._add, self._mul = bml._poly_add, bml._poly_mul

        def add(p, q, ctx, sign=1.0):
            cost = len(p) + len(q)
            self.add_work += cost
            self.calls += 1
            self.max_call = max(self.max_call, cost)
            out = self._add(p, q, ctx, sign)
            self.max_result = max(self.max_result, len(out))
            return out

        def mul(p, q, ctx):
            self.mul_work += len(p) * len(q)
            out = self._mul(p, q, ctx)
            self.max_result = max(self.max_result, len(out))
            return out

        bml._poly_add, bml._poly_mul = add, mul
        return self

    def __exit__(self, *exc) -> None:
        bml._poly_add, bml._poly_mul = self._add, self._mul

    @property
    def work(self) -> int:
        return self.add_work + self.mul_work


def test_the_size_budgets_do_not_bound_the_work():
    """THE PREMISE. If this ever goes vacuous every test below is meaningless.

    A model whose every intermediate stays well inside ``_MAX_MONOMIALS`` still
    does work that is orders of magnitude larger than anything it ever holds —
    so bounding what is *produced* bounds nothing about what is *spent*.
    """
    m = _accumulate_model(100)
    with _Counter() as c:
        bml.reformulate_binary_multilinear(m)
    assert c.calls > 0, "the instrument never fired"
    assert c.max_result < bml._MAX_MONOMIALS, (
        f"fixture drifted: max result {c.max_result:,} already trips the "
        f"{bml._MAX_MONOMIALS:,} monomial budget, so it proves nothing about work"
    )
    assert c.work > 50 * c.max_result, (
        f"work {c.work:,} is not dominated by the accumulate: the fixture no "
        f"longer exhibits the quadratic copy (max result {c.max_result:,})"
    )


def test_oversized_expansion_aborts_on_the_work_budget():
    """THE REGRESSION. ``hadamard_9`` spent 154 s to reach "unchanged"."""
    m = _accumulate_model(200)
    t0 = time.perf_counter()
    with pytest.raises(bml._Unsupported, match="expansion work budget"):
        bml._reformulate(m)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"abort took {dt:.1f}s"


def test_aborting_returns_the_model_unchanged():
    """SOUNDNESS. A budget abort must fall back, never emit a partial rewrite.

    ``_reformulate`` raises; the public entry point is what the solver calls and
    it must hand back the *same object*, so every existing path still applies.
    """
    m = _accumulate_model(200)
    assert bml.has_binary_multilinear_work(m), "premise gone: the pass would not fire"
    assert bml.reformulate_binary_multilinear(m) is m


def test_the_budget_is_a_running_total_not_a_per_call_limit():
    """Bounding each call bounds nothing: 4,472 calls of at most 4,472 entries
    each still cost 1e7.  (The same hole the #1455 term budget had in review.)"""
    m = _accumulate_model(200)
    with _Counter() as c:
        bml.reformulate_binary_multilinear(m)
    assert c.calls > 100, f"only {c.calls} calls: fixture is not an accumulate loop"
    assert c.max_call * 100 < bml._MAX_EXPAND_OPS, (
        f"largest single call is {c.max_call:,}, not small relative to the "
        f"{bml._MAX_EXPAND_OPS:,} budget — this fixture cannot distinguish a "
        f"per-call bound from a running total"
    )
    assert c.work > bml._MAX_EXPAND_OPS, (
        f"total work {c.work:,} did not exceed the budget, so nothing was bounded"
    )


def test_a_reformulated_model_is_nowhere_near_the_budget():
    """BOUND-NEUTRAL (CLAUDE.md §5). The corpus must be untouched.

    A model the pass genuinely reformulates must still be reformulated.  The
    corpus-wide version of this is the 1610-instance A/B in the PR; this pins the
    margin locally so a future budget cut cannot quietly start truncating
    instances that were being served.
    """
    n, n_lags, degree = 20, 6, 3
    m = dm.Model("reformulable")
    b = m.binary("b", shape=(n,))
    terms = []
    for lag in range(1, n_lags + 1):
        s = None
        for i in range(n - degree * lag):
            t = b[i]
            for d in range(1, degree):
                t = t * b[i + d * lag]
            s = t if s is None else s + t
        if s is not None:
            terms.append(s * s)
    assert terms, "fixture built no terms"
    obj = terms[0]
    for t in terms[1:]:
        obj = obj + t
    m.minimize(obj)

    with _Counter() as c:
        out = bml.reformulate_binary_multilinear(m)
    assert out is not m, "premise gone: the pass no longer reformulates this model"
    assert c.calls > 0, "the instrument never fired"
    assert c.work * 1000 < bml._MAX_EXPAND_OPS, (
        f"a reformulated model costs {c.work:,}, uncomfortably close to the "
        f"{bml._MAX_EXPAND_OPS:,} budget"
    )


@pytest.mark.slow
def test_an_accumulate_blowup_model_honours_its_time_limit():
    """END TO END: the defect as the user met it — a pre-solve pass a
    ``time_limit`` cannot reach."""
    m = _accumulate_model(200)
    t0 = time.perf_counter()
    m.solve(time_limit=5.0)
    dt = time.perf_counter() - t0
    assert dt < 120.0, f"solve ran {dt:.1f}s against a 5s time_limit"
