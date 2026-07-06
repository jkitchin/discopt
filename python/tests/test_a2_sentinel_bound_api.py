"""A2 soundness lock: the 1e30 failure sentinel must never escape a bound/gap API.

Internally the solver stores ``INFEASIBILITY_SENTINEL`` (1e30) as the "lower bound"
of a node whose relaxation failed / was declared infeasible, and on the
*no-relaxation* class (a model whose relaxation omits rows, so no dual bound is ever
produced — e.g. hda / heatexch_gen3) the tree's ``global_lower_bound`` stays at that
sentinel through to result assembly. It is *finite* (``np.isfinite(1e30)`` is True),
so the pre-A2 non-finite guard did not catch it: a raw result-assembly path could
surface ``SolveResult.bound = 1e30`` — and a ``gap`` computed from it — as if it were
a real dual bound.

A1 (#498) fixed only the callback ``best_bound`` dataclass surface. A2 establishes
the *invariant* across every public bound/gap surface at the single chokepoint every
``SolveResult`` construction passes through (``SolveResult.__post_init__``): a
sentinel-magnitude ``bound``/``root_bound`` (either sense) is mapped to ``None`` and
its gap cleared. Consumers (``SolveResult.bound``/``.gap``, the benchmark JSON via
``runner``, the LLM serializer/diagnosis, the callback) then never do arithmetic on
the sentinel. This is a representation/reporting normalisation only; the internal
sentinel is unchanged.

Fail-before: on pre-A2 code ``SolveResult(bound=1e30, ...)`` retained ``bound=1e30``
(``np.isfinite`` passed it), and the callback ``gap`` was surfaced even when
``best_bound`` was ``None``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.constants import INFEASIBILITY_SENTINEL, SENTINEL_THRESHOLD
from discopt.modeling.core import SolveResult


@pytest.mark.smoke
def test_solveresult_bound_sentinel_scrubbed_to_none():
    """A ``SolveResult`` built with the raw failure sentinel as its bound must
    expose ``bound=None`` and ``gap=None`` — never 1e30."""
    r = SolveResult(
        status="feasible",
        objective=10.0,
        bound=INFEASIBILITY_SENTINEL,
        gap=0.5,
        gap_certified=False,
    )
    assert r.bound is None, f"sentinel leaked through SolveResult.bound: {r.bound}"
    assert r.gap is None, f"gap computed from sentinel leaked: {r.gap}"


@pytest.mark.smoke
def test_solveresult_negated_sentinel_scrubbed():
    """A MAXIMIZE model negates the internal minimization bound, so the sentinel can
    arrive as ``-1e30``; that too must be scrubbed to ``None``."""
    r = SolveResult(
        status="feasible",
        objective=10.0,
        bound=-INFEASIBILITY_SENTINEL,
        gap=0.5,
        gap_certified=False,
    )
    assert r.bound is None
    assert r.gap is None


@pytest.mark.smoke
def test_solveresult_root_bound_sentinel_scrubbed():
    """The root-node instrumentation bound funnels through the same guard: a
    sentinel ``root_bound`` -> ``None`` with ``root_gap`` cleared, and the real
    primary ``bound`` is untouched."""
    r = SolveResult(
        status="feasible",
        objective=10.0,
        bound=5.0,
        gap=0.5,
        root_bound=INFEASIBILITY_SENTINEL,
        root_gap=0.99,
        gap_certified=False,
    )
    assert r.root_bound is None
    assert r.root_gap is None
    assert r.bound == 5.0, "a real primary bound must not be scrubbed"


@pytest.mark.smoke
def test_solveresult_real_bound_preserved():
    """A genuine finite bound below the sentinel threshold must pass through
    untouched — the guard must not over-scrub legitimate large objectives."""
    r = SolveResult(status="optimal", objective=1e25, bound=1e25, gap=0.0, gap_certified=True)
    assert r.bound == 1e25
    assert r.gap == 0.0
    # A value just under the threshold is still a real bound.
    r2 = SolveResult(
        status="feasible",
        objective=SENTINEL_THRESHOLD * 0.9,
        bound=SENTINEL_THRESHOLD * 0.9,
        gap=0.1,
        gap_certified=False,
    )
    assert r2.bound == SENTINEL_THRESHOLD * 0.9


@pytest.mark.smoke
def test_solveresult_non_finite_bound_still_scrubbed():
    """Regression guard: the existing non-finite (+/-inf) scrub is preserved
    alongside the new sentinel scrub."""
    r = SolveResult(status="feasible", objective=1.0, bound=-np.inf, gap=0.5, gap_certified=True)
    assert r.bound is None
    assert r.gap is None
    assert r.gap_certified is False


@pytest.mark.smoke
def test_callback_gap_none_when_best_bound_none():
    """A2 callback-surface consistency: when the certified ``best_bound`` is ``None``
    (tainted tree / failure sentinel), a real solve's callback must not hand out a
    ``gap`` derived from that non-bound. Exercised end-to-end on a small model whose
    node callback records every (best_bound, gap) pair."""
    from discopt import Model
    from discopt.callbacks import CallbackContext

    m = Model("cb_gap_consistency")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=0, ub=2)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y + z)

    pairs: list[tuple[float | None, float | None]] = []

    def _cb(ctx: CallbackContext, _model) -> None:
        pairs.append((ctx.best_bound, ctx.gap))

    m.solve(time_limit=10.0, node_callback=_cb)

    # The invariant: no callback ever reports a gap while best_bound is None.
    for bb, gap in pairs:
        assert not (bb is None and gap is not None), (
            f"callback surfaced gap={gap} with best_bound=None (gap from a non-bound)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
