"""Issue #1114: alphaBB must not run alongside the reduced-space engine.

``_use_alphabb`` is enabled whenever the *lifted* LP relaxer is absent
(``solver.py``: ``_alphabb_eligible and _mc_lp_relaxer is None``). That is exactly
the ``dm.custom``/``CustomCall`` class, on which the reduced-space McCormick engine
supplies every node bound — so alphaBB ran at every node beside the engine that
replaced it.

WHY THE SUPPRESSION IS BOUND-NEUTRAL, not a bound trade. ``_compute_alphabb_bound``
derives its bound from ``rigorous_alpha``, which encloses the Hessian with the
interval-AD walker in ``_relax/convexity/interval_ad.py``. That walker has no rule
for ``CustomCall`` (an opaque user callable has no symbolic second derivative), the
node falls through to ``_unbounded``, and the flag propagates through every
enclosing operator. So ``rigorous_alpha`` returns ``+inf`` on EVERY box — narrowing
the box never recovers it — and ``_compute_alphabb_bound`` abstains with ``-inf``,
whose only effect at both node-loop call sites is ``max(lb, -inf)``.

``test_rigorous_alpha_abstains_on_customcall_but_not_native`` pins that implication
against the native twin of the same function. It is the test that must fail if
someone later teaches the interval-AD walker about ``CustomCall``: at that point
alphaBB would start producing real bounds on this class and the gate in
``solve_model`` would be silently discarding them.

Measured before/after over an 8-model CustomCall family (``scratchpad/issue1114``):
51 alphaBB node boxes -> 0, every one of which had abstained, with
status/node_count/bound/objective bit-identical on all 8 (32/32 exact-identity
assertions) — the §5 bound-neutral regime.

Every test counts the assertions it executed and the file fails if a count is zero
(CLAUDE.md §6). No exception is swallowed (§7).
"""

from __future__ import annotations

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._alphabb_rigorous import rigorous_alpha
from discopt._relax.mcbox import MCBox

pytestmark = [pytest.mark.relaxation]

_CHECKS = {"n": 0}


def _check(condition: bool, message: str) -> None:
    """An assertion that RECORDS that it ran, so a silent no-op cannot pass."""
    _CHECKS["n"] += 1
    assert condition, message


def _mexp(x):
    """``exp`` dispatched to MCBox for the relaxation, ``jnp`` for the value path.

    The value path is traced by the fused JIT in ``_relax/dag_compiler.py``, so the
    non-MCBox branch must be ``jnp``: a ``np.exp`` there raises
    ``TracerArrayConversionError`` and the model never reaches the code under test.
    """
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def _custom_model():
    """A nonconvex, reduced-space-admissible CustomCall model.

    All the nonlinearity is inside the opaque node: a bilinear term left outside it
    would give the lifted LP relaxer something to bound, ``_mc_lp_relaxer`` would not
    be None, and alphaBB would never have been enabled in the first place.
    """
    m = dm.Model("custom")
    x = m.continuous("x", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
    f = dm.custom(lambda a, b: a * _mexp(-b) + b * _mexp(-a), name="f")
    m.minimize(f(x[0], x[1]))
    m.subject_to(x[0] + x[1] >= 1.0)
    return m


def _native_model():
    """The SAME function written in native expression ops — the control."""
    m = dm.Model("native")
    y = m.continuous("y", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
    m.minimize(y[0] * dm.exp(-y[1]) + y[1] * dm.exp(-y[0]))
    m.subject_to(y[0] + y[1] >= 1.0)
    return m


def _alpha_over_box(model):
    from discopt.solver import _alphabb_node_box

    n = sum(v.size for v in model._variables)
    box = _alphabb_node_box(model, np.full(n, 0.1), np.full(n, 2.0))
    return np.asarray(rigorous_alpha(model._objective.expression, model, box), dtype=np.float64)


def test_rigorous_alpha_abstains_on_customcall_but_not_native():
    """The implication the suppression rests on, asserted directly."""
    alpha_custom = _alpha_over_box(_custom_model())
    alpha_native = _alpha_over_box(_native_model())
    _check(
        not np.all(np.isfinite(alpha_custom)),
        f"rigorous_alpha certified convexity through a CustomCall: {alpha_custom}. "
        "If the interval-AD walker learned CustomCall, the #1114 gate in solve_model "
        "is now discarding real alphaBB bounds and must be revisited.",
    )
    _check(
        np.all(np.isfinite(alpha_native)),
        f"the native control must still yield a finite alpha, got {alpha_native}; "
        "otherwise this file proves nothing about the CustomCall wrapper.",
    )


def test_alpha_stays_unbounded_on_a_narrow_subbox():
    """Subdividing cannot recover it — the reason a per-node retry is pointless."""
    from discopt.solver import _alphabb_node_box

    model = _custom_model()
    n = sum(v.size for v in model._variables)
    box = _alphabb_node_box(model, np.full(n, 0.90), np.full(n, 0.91))
    alpha = np.asarray(rigorous_alpha(model._objective.expression, model, box), dtype=np.float64)
    _check(
        not np.all(np.isfinite(alpha)),
        f"a 0.01-wide box certified convexity through a CustomCall: {alpha}",
    )


def test_alphabb_is_not_invoked_on_a_customcall_model(monkeypatch):
    """The regression: before #1114 this counted 11 calls, all returning -inf."""
    import discopt.solver as solver_mod

    calls = {"n": 0, "finite": 0}
    original = solver_mod._compute_alphabb_bound

    def _counting(evaluator, model, expr, node_lb, node_ub):
        value = original(evaluator, model, expr, node_lb, node_ub)
        calls["n"] += 1
        if np.isfinite(value):
            calls["finite"] += 1
        return value

    monkeypatch.setattr(solver_mod, "_compute_alphabb_bound", _counting)
    result = _custom_model().solve(max_nodes=200, time_limit=60.0)

    _check(
        result.node_count is not None and result.node_count > 0,
        "the solve explored no node, so the node loop under test never ran",
    )
    _check(
        calls["n"] == 0,
        f"alphaBB ran {calls['n']} times on a CustomCall model "
        f"({calls['finite']} produced a finite bound)",
    )
    # The bound source that remains must still be a real spatial bound.
    _check(
        result.bound is not None and np.isfinite(result.bound),
        f"no dual bound after suppressing alphaBB: {result.bound}",
    )
    _check(
        result.objective is not None and result.bound <= result.objective + 1e-6,
        f"certificate invariant violated: bound={result.bound} > obj={result.objective}",
    )


def test_predicate_separates_custom_from_native():
    from discopt.solver import _expression_contains_custom_call

    _check(
        _expression_contains_custom_call(_custom_model()._objective.expression),
        "the CustomCall objective was not detected",
    )
    _check(
        not _expression_contains_custom_call(_native_model()._objective.expression),
        "the native objective was misreported as containing a CustomCall",
    )
    _check(not _expression_contains_custom_call(None), "None must be handled")


@pytest.fixture(autouse=True)
def _assert_probe_fired():
    """§6: a test whose model stops reaching the path must fail, not pass silently."""
    before = _CHECKS["n"]
    yield
    assert _CHECKS["n"] > before, "this test executed no recorded assertion"
