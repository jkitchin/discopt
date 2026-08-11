"""#844: the LP-per-node spatial engine must honour its wall-clock ``time_limit``.

Measured before this fix, on `ball_mk2_30` (**30 integer variables, 1 constraint**):

| config | limit | wall | overrun |
|---|---|---|---|
| default path | 30 s | 30.4 s / 30.0 s | 1.01x / 1.00x |
| `lp_spatial=True` | 30 s | 486.7 s / 500.0 s | **16.22x / 16.67x** |

Reproducible across repeats, so a deterministic engine defect rather than noise, and
specific to this engine — the default path honoured the same limit exactly, twice.
It is also **not** the residual accepted in #845 (closed NOT_PLANNED), whose rationale
is IPM + sparse-direct factorization on a *very large* single NLP; that cannot apply
to a 30-variable, 1-row model.

Three unbounded loops were responsible, none of which consulted the deadline:

* the root OBBT call, which ran ``|vars| x 2 x rounds`` LPs at up to
  ``time_limit_per_lp`` each entirely outside ``time_limit`` (up to ~150 s on
  ball_mk2_30's 30 integers) — ``obbt_tighten_root`` already accepted a ``deadline``,
  the caller simply never passed one;
* ``dive``, which solves an LP per iteration for ``2n+2`` iterations, at the root and
  again at every node;
* ``feasibility_pump``, likewise one LP per iteration for ``max_iter`` iterations.

After the fix the engine honours its budget to 1.00x while still finding the same
incumbents (tln6 83.1 and tln4 13.3 at a 14 s budget).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.lp_spatial_bb import _is_in_scope, solve_lp_spatial_bb  # noqa: E402


def _integer_model(n: int = 12) -> dm.Model:
    """A pure-integer MINIMIZE model with bilinear coupling — in scope for the engine
    and hard enough that it will not terminate before the deadline."""
    m = dm.Model("deadline_probe")
    xs = [m.integer(f"x{i}", lb=0, ub=20) for i in range(n)]
    m.minimize(sum((i + 1) * xs[i] for i in range(n)))
    for i in range(n - 1):
        m.subject_to(xs[i] * xs[i + 1] >= 12)
    return m


def _incremental_model(n: int = 12) -> dm.Model:
    """Like :func:`_integer_model`, but with the bilinear terms in the OBJECTIVE so the
    incremental McCormick structure actually builds."""
    m = dm.Model("incremental_probe")
    xs = [m.integer(f"x{i}", lb=0, ub=20) for i in range(n)]
    m.minimize(sum(xs[i] * xs[i + 1] for i in range(n - 1)))
    for i in range(n - 1):
        m.subject_to(xs[i] + xs[i + 1] >= 7)
    return m


def _no_incremental_model(n: int = 8) -> dm.Model:
    """In scope for the engine, but its incremental structure does NOT build — the
    control for anything asserting cold-path behaviour.

    Uses a TRILINEAR coupling. ``_integer_model``'s bilinear *constraints* used to
    serve this role, because a constraint on the product lifted to a 5th row over
    the term's own columns and the structure declined counting it as an envelope
    row. That was a defect (#861), not a property: envelope rows are now identified
    numerically and ``_integer_model`` builds a structure, which silently turned
    every test relying on it into a fast-path test. Trilinear lifts are outside the
    closed-form patch's covered families by design (it regenerates bilinear /
    integer-power / affine-square envelopes only), so this decline is a stable
    property of the mathematics rather than of a bug.

    Callers must still assert :func:`_declines_incremental` rather than trust this
    docstring — that is what makes the rot loud instead of silent.
    """
    m = dm.Model("no_incremental_probe")
    xs = [m.integer(f"x{i}", lb=0, ub=20) for i in range(n)]
    m.minimize(sum((i + 1) * xs[i] for i in range(n)))
    for i in range(n - 2):
        m.subject_to(xs[i] * xs[i + 1] * xs[i + 2] >= 12)
    return m


def _declines_incremental(model) -> tuple[bool, str]:
    """``(declined, reason)`` for ``model``'s incremental structure."""
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.term_classifier import classify_nonlinear_terms

    inc = IncrementalMcCormickLP(model, classify_nonlinear_terms(model), deadline=None)
    return (not inc.ok), (inc.decline_reason or "")


def test_model_is_in_scope():
    """Guard: if this model stopped being in scope the deadline test below would
    vacuously pass by bailing out immediately."""
    assert _is_in_scope(_integer_model()) is True
    assert _is_in_scope(_incremental_model()) is True


@pytest.mark.parametrize("limit", [3.0, 6.0])
def test_engine_honours_its_time_limit(limit):
    """The engine must return within a small multiple of its budget.

    Pre-fix this overran by 16x on ball_mk2_30; the bound here is deliberately loose
    (3x) so the test asserts "the deadline is actually polled" rather than pinning
    machine-specific timing, yet still fails hard on a 16x regression.
    """
    t0 = time.perf_counter()
    solve_lp_spatial_bb(_integer_model(), time_limit=limit, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0
    assert wall < 3.0 * limit, (
        f"engine overran its {limit}s budget: {wall:.1f}s ({wall / limit:.1f}x)"
    )


def test_root_obbt_is_budgeted():
    """The root OBBT pass must be deadline-bounded. It previously ran outside
    ``time_limit`` entirely, which was the single largest contributor."""
    t0 = time.perf_counter()
    solve_lp_spatial_bb(_integer_model(16), time_limit=3.0, gap_tolerance=1e-4, use_obbt=True)
    wall = time.perf_counter() - t0
    assert wall < 9.0, f"root OBBT escaped the budget: {wall:.1f}s against a 3s limit"


def test_expired_ambient_deadline_does_not_disable_the_incremental_structure():
    """``model._solve_deadline`` is written once per ``solve_model`` and never cleared.

    A *later* in-process consumer with a budget of its own — the #844 no-incumbent
    fallback, which by construction runs after a primary solve that spent its whole
    budget — therefore read an already-expired deadline, and the #654 guard in
    ``IncrementalMcCormickLP`` declined to build the structure at all. The engine then
    silently degraded to the trusted-but-~30x-slower per-node cold build. Measured on
    tln5 at a 21 s budget: ``ok=False`` gave 5 nodes in 43.8 s (2.08x, slowest node
    42.4 s) where ``ok=True`` gives 13158 nodes in 21.0 s (1.00x, slowest node 0.04 s).
    """
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.term_classifier import classify_nonlinear_terms

    m = _incremental_model(6)
    terms = classify_nonlinear_terms(m)
    # Control: with no ambient deadline the structure builds. If this ever fails the
    # rest of the test is vacuous.
    assert IncrementalMcCormickLP(m, terms).ok is True

    m._solve_deadline = time.perf_counter() - 1000.0
    # The #654 guard still binds on the ambient stash (that behaviour is deliberate)…
    assert IncrementalMcCormickLP(m, terms).ok is False
    # …but an explicit caller budget overrides it, which is what the engine passes.
    assert IncrementalMcCormickLP(m, terms, deadline=time.perf_counter() + 60.0).ok is True


def test_engine_does_not_degrade_after_a_previous_solve(monkeypatch):
    """End-to-end form of the above: a stale ``_solve_deadline`` must not cost the
    engine its fast path.

    The structure's ``ok`` is the observable, not node throughput — no synthetic model
    small enough for a unit test is *also* hard enough for the cold path's slowness to
    show up in a node count, so a throughput assertion here would pass vacuously.
    """
    import discopt._relax.incremental_mccormick as im

    built: list[bool] = []
    real = im.IncrementalMcCormickLP

    class _Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.ok)

    monkeypatch.setattr(im, "IncrementalMcCormickLP", _Spy)

    m = _incremental_model()
    m._solve_deadline = time.perf_counter() - 1000.0  # a completed prior solve
    solve_lp_spatial_bb(m, time_limit=3.0, gap_tolerance=1e-4)

    assert built, "spy never fired — the engine built no structure, so this asserts nothing"
    assert built[0] is True, (
        "a stale _solve_deadline from a previous solve degraded the engine to the "
        "per-node cold build"
    )


def test_require_incremental_declines_when_the_fast_path_is_unavailable():
    """``require_incremental=True`` must decline rather than run the cold path.

    ``_no_incremental_model`` is in scope but its structure does not build, so it is
    exactly the case the #844 fallback must refuse: on ball_mk2_30 (same situation)
    the cold path spent 61 s on the root LP alone against a 21 s reserve and returned
    nothing.
    """
    m = _no_incremental_model(8)
    # PRECONDITION, asserted rather than assumed: this guard is only meaningful for a
    # model whose fast path is genuinely unavailable. #861 made the previous fixture
    # (bilinear constraints) build a structure, which turned this into a test that the
    # engine declines a model it can actually serve — it failed loudly, but only
    # because the assertion below happened to be `is None`. Assert the premise so a
    # future coverage widening points at the fixture instead.
    declined, reason = _declines_incremental(m)
    assert declined, (
        "fixture no longer declines the incremental structure — pick a family the "
        "closed-form patch still does not cover, else this guard is vacuous"
    )
    assert solve_lp_spatial_bb(m, time_limit=4.0, require_incremental=True) is None, (
        f"engine ran the cold path despite require_incremental (decline reason: {reason})"
    )
    # …while the default (cold path allowed) still runs, so the decline is caused by
    # the flag and not by the model being out of scope.
    assert solve_lp_spatial_bb(_no_incremental_model(8), time_limit=4.0) is not None
    # And a model whose structure DOES build is unaffected by the flag.
    assert (
        solve_lp_spatial_bb(_incremental_model(8), time_limit=4.0, require_incremental=True)
        is not None
    )


def test_result_is_sound_when_one_is_returned():
    """Honouring the deadline must not cost soundness: any incumbent returned still
    carries a dual bound that does not cross it."""
    res = solve_lp_spatial_bb(_integer_model(), time_limit=6.0, gap_tolerance=1e-4)
    if res is not None and res.objective is not None and res.bound is not None:
        assert res.bound <= res.objective + 1e-6 * (1 + abs(res.objective)), (
            f"UNSOUND: bound {res.bound} > incumbent {res.objective}"
        )
