"""A ``SolverTuning`` installed with ``set_current()`` must survive ``solve()`` (#1117).

``solve()`` is wrapped by ``solver._scoped_tuning``, which published the ``tuning=``
kwarg for the call. With the kwarg omitted — every plain ``m.solve()`` — it used to
publish a *fresh env-resolved* ``SolverTuning()``, overwriting whatever the caller had
installed with ``solver_tuning.set_current(...)``. No warning, no error: the solve just
ran on env defaults.

That is not a cosmetic API wart. It is the instrument defect behind two retracted #1113
measurement panels: a probe that delivered ``singular_tangent`` via ``set_current()``
measured zero effect on a full solve, and "both arms bit-identical" was written up as a
neutrality result when it was the signature of a probe that never fired (CLAUDE.md §6).

The contract pinned here is a precedence order — explicit ``tuning=`` kwarg > ambient
context > environment defaults — plus the two ways a scope can be lost silently:
``set_current`` semantics (``None`` means "env defaults", which callers of ``enter_scope``
must not inherit) and the deep-recursion worker thread, which starts with an EMPTY
contextvars context and so would have kept the deep-model path on env defaults while the
shallow path honored the caller.
"""

from __future__ import annotations

import contextvars
import os
import sys

import discopt.modeling as dm
import discopt.solver_tuning as solver_tuning
import pytest
from discopt._relax import uniform_relax
from discopt._relax.convexity.rules import _run_with_deep_recursion
from discopt.solver import _scoped_tuning

pytestmark = [pytest.mark.relaxation]

FLAG = "DISCOPT_SINGULAR_TANGENT"
LAZY = "DISCOPT_SINGULAR_TANGENT_LAZY"


@pytest.fixture
def no_env_flag():
    """Guarantee the flag is OFF in the environment, so any ON observation is
    attributable to the delivery under test and not to an inherited env var."""
    saved = {k: os.environ.pop(k, None) for k in (FLAG, LAZY)}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _eager_tuning():
    return solver_tuning.current().replace(singular_tangent=True, singular_tangent_lazy=False)


# --------------------------------------------------------------------------- #
# 1. enter_scope's precedence, at the context layer
# --------------------------------------------------------------------------- #
def test_enter_scope_inherits_the_ambient_context(no_env_flag):
    tuning = _eager_tuning()
    outer = solver_tuning.set_current(tuning)
    try:
        inner = solver_tuning.enter_scope(None)
        try:
            assert solver_tuning.current().singular_tangent is True, (
                "enter_scope(None) means 'no override was requested' and must inherit; "
                "publishing a fresh env-default here is exactly bug #1117"
            )
        finally:
            solver_tuning.reset_current(inner)
        assert solver_tuning.current().singular_tangent is True
    finally:
        solver_tuning.reset_current(outer)
    assert solver_tuning.current().singular_tangent is False


def test_explicit_tuning_still_wins_over_the_ambient_context(no_env_flag):
    outer = solver_tuning.set_current(_eager_tuning())
    try:
        inner = solver_tuning.enter_scope(solver_tuning.SolverTuning())
        try:
            assert solver_tuning.current().singular_tangent is False
        finally:
            solver_tuning.reset_current(inner)
    finally:
        solver_tuning.reset_current(outer)


def test_enter_scope_outside_any_context_is_env_defaults(no_env_flag):
    token = solver_tuning.enter_scope(None)
    try:
        assert solver_tuning.current().singular_tangent is False
    finally:
        solver_tuning.reset_current(token)


def test_set_current_none_still_means_env_defaults(no_env_flag):
    """``set_current`` keeps its documented meaning; ``enter_scope`` is the new one."""
    outer = solver_tuning.set_current(_eager_tuning())
    try:
        inner = solver_tuning.set_current(None)
        try:
            assert solver_tuning.current().singular_tangent is False
        finally:
            solver_tuning.reset_current(inner)
    finally:
        solver_tuning.reset_current(outer)


# --------------------------------------------------------------------------- #
# 2. The decorator that guards every solve entry point
# --------------------------------------------------------------------------- #
def test_scoped_tuning_decorator_honors_the_ambient_context(no_env_flag):
    @_scoped_tuning
    def observe():
        return solver_tuning.current().singular_tangent

    token = solver_tuning.set_current(_eager_tuning())
    try:
        assert observe() is True
        assert observe(tuning=solver_tuning.SolverTuning()) is False
    finally:
        solver_tuning.reset_current(token)
    assert observe() is False


# --------------------------------------------------------------------------- #
# 3. End to end: the issue's own reproducer
# --------------------------------------------------------------------------- #
def _sqrt_model():
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=-1e3, ub=1e3)
    m.subject_to(y == dm.sqrt(x))
    m.minimize(2 * x - y)
    return m


def _count_tangent_calls(monkeypatch, run):
    calls = {"n": 0}
    orig = uniform_relax._interior_tangent_point

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(uniform_relax, "_interior_tangent_point", counting)
    run()
    return calls["n"]


def test_set_current_reaches_a_plain_solve(monkeypatch, no_env_flag):
    """The reproducer from #1117: identical delivery via kwarg and via context."""
    tuning = _eager_tuning()

    via_kwarg = _count_tangent_calls(
        monkeypatch, lambda: _sqrt_model().solve(tuning=tuning, max_nodes=20)
    )
    assert via_kwarg > 0, "control arm never fired — the probe measures nothing (§6)"

    def _via_context():
        token = solver_tuning.set_current(tuning)
        try:
            assert solver_tuning.current().singular_tangent is True
            _sqrt_model().solve(max_nodes=20)
        finally:
            solver_tuning.reset_current(token)

    via_context = _count_tangent_calls(monkeypatch, _via_context)
    assert via_context > 0, (
        "set_current() around a plain solve() was discarded at the solve boundary (#1117)"
    )

    off = _count_tangent_calls(monkeypatch, lambda: _sqrt_model().solve(max_nodes=20))
    assert off == 0, "flag-OFF control fired the recovery — the env fixture is not clean"


# --------------------------------------------------------------------------- #
# 4. The deep-model worker thread must carry the context across
# --------------------------------------------------------------------------- #
_PROBE: contextvars.ContextVar[str] = contextvars.ContextVar("discopt_test_1117", default="unset")


def test_deep_recursion_worker_sees_the_callers_context():
    """``_run_with_deep_recursion`` runs ``fn`` on a fresh thread for deep models.

    A new thread's contextvars context is EMPTY, so without an explicit copy the
    ``SolverTuning`` published for the solve would be invisible on exactly the large
    models — the fix for #1117 would be inert where it matters most.
    """
    need = sys.getrecursionlimit() + 1000  # above the current limit -> the thread path
    token = _PROBE.set("published")
    try:
        assert _run_with_deep_recursion(_PROBE.get, depth_need=need) == "published"

        # ...and writes on the worker do not leak back to the caller's context.
        def _writes():
            _PROBE.set("mutated")
            return _PROBE.get()

        assert _run_with_deep_recursion(_writes, depth_need=need) == "mutated"
        assert _PROBE.get() == "published"
    finally:
        _PROBE.reset(token)
