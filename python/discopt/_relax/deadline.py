"""Wall-clock deadline plumbing for JAX-compiled solver loops.

Python orchestration enforces ``time_limit`` between B&B nodes and between
sub-solver calls, but once a ``jax.lax.while_loop`` (e.g. IPM iteration) is
running the XLA runtime spins until ``cond_fn`` returns false. There is no
path for Python to interrupt it mid-execution.

This module exposes a process-global deadline that a ``cond_fn`` can poll via
a host callback (``jax.experimental.io_callback``). One callback per outer
iteration is cheap compared to the iteration body (matrix factor / KKT solve),
so the predicate short-circuits the loop within ``time_limit + ε``.

Usage::

    from discopt._relax.deadline import deadline_scope, deadline_exceeded_jax

    with deadline_scope(time_limit_seconds):
        ...  # any JAX call whose cond_fn uses deadline_exceeded_jax()

The deadline is cleared on scope exit; if no deadline is set,
``deadline_exceeded_jax()`` returns ``False`` unconditionally.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp

# NOTE: ``jax`` is intentionally NOT imported at module scope. Only the
# JAX-side predicate ``deadline_exceeded_jax`` needs it, and it imports jax
# lazily inside the function body. This keeps the pure-Python deadline plumbing
# (``deadline_scope``/``set_deadline``/...) — imported on every ``Model.solve``
# — free of the heavy JAX/XLA initialization, so LP/MILP/MIQP solves that never
# touch JAX do not pay its cold-start cost.

_deadline_monotonic: float | None = None


def set_deadline(seconds_from_now: float | None) -> None:
    """Set a process-global wall-clock deadline ``seconds_from_now`` ahead.

    ``None`` clears any existing deadline.
    """
    global _deadline_monotonic
    if seconds_from_now is None:
        _deadline_monotonic = None
        return
    _deadline_monotonic = time.monotonic() + max(0.0, float(seconds_from_now))


def clear_deadline() -> None:
    """Clear the process-global deadline."""
    global _deadline_monotonic
    _deadline_monotonic = None


def get_deadline() -> float | None:
    """Return the monotonic deadline timestamp, or None if unset."""
    return _deadline_monotonic


def deadline_exceeded() -> bool:
    """True iff a deadline is set and the wall clock has reached it."""
    d = _deadline_monotonic
    return d is not None and time.monotonic() >= d


class deadline_scope:
    """Context manager that installs (and on exit restores) the global deadline."""

    def __init__(self, seconds_from_now: float | None):
        self._seconds = seconds_from_now
        self._prev: float | None = None

    def __enter__(self) -> "deadline_scope":
        self._prev = _deadline_monotonic
        set_deadline(self._seconds)
        return self

    def __exit__(self, *_exc: object) -> None:
        global _deadline_monotonic
        _deadline_monotonic = self._prev


def _host_check() -> np.ndarray:
    return np.asarray(deadline_exceeded(), dtype=np.bool_)


def deadline_exceeded_jax() -> jnp.ndarray:
    """JAX-side predicate: 0-d bool array, True iff the deadline has passed.

    Safe to call inside ``jax.lax.while_loop`` ``cond_fn``. Uses
    ``ordered=True`` so the host callback fires once per iteration in
    order, which is what we want for a self-terminating loop. Note this
    means the predicate cannot be used under ``jax.vmap`` — call sites
    that vmap must compile the cond_fn without the deadline check.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import io_callback

    result: jnp.ndarray = io_callback(
        _host_check,
        jax.ShapeDtypeStruct((), jnp.bool_),
        ordered=True,
    )
    return result


# ── abortable setup (#2079) ──────────────────────────────────────────────────
#
# Some setup work cannot return a partial result: a native GDP algorithm has
# nothing to run until its lowering (``GDPSpace.from_model``: a model copy, the
# bound pretransform, the big-M lowering) and its evaluator and master exist,
# and none of that produces a bound or a point. On GDPlib ``stranded_gas`` it
# took ~25 s, so ``time_limit=10`` returned after ~33 s. Such work runs inside
# :class:`abortable_setup`; the loops it is made of call :func:`setup_checkpoint`,
# which raises :class:`SetupDeadlineExpired` once the deadline has passed. The
# caller turns that into a ``time_limit`` result with no incumbent and no bound,
# which is the truth: the work was abandoned, not truncated into a weaker answer.
#
# Outside such a scope a checkpoint is a no-op, so code shared with other routes
# (``Model.__deepcopy__``, ``reformulate_gdp``) behaves exactly as before there.


class SetupDeadlineExpired(BaseException):
    """The deadline of an :class:`abortable_setup` scope passed during the setup.

    A ``BaseException``, like ``asyncio.CancelledError``: it is a cancellation,
    not a failure, and must not be caught by an ``except Exception`` fallback
    inside the setup (which would read it as "this step failed, degrade") on its
    way out to the scope's owner.
    """

    def __init__(self, where: str) -> None:
        super().__init__(f"time limit reached during {where}")
        self.where = where


_setup_expired: contextvars.ContextVar[Callable[[], bool] | None] = contextvars.ContextVar(
    "discopt_setup_expired", default=None
)


class abortable_setup:
    """Make :func:`setup_checkpoint` raise once ``expired()`` is true, inside this scope.

    ``expired`` is the owner's own deadline test (for example
    ``Deadline.expired`` of a native GDP algorithm), so the checkpoints read the
    same clock the owner reports against.
    """

    def __init__(self, expired: Callable[[], bool]) -> None:
        self._expired = expired
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "abortable_setup":
        self._token = _setup_expired.set(self._expired)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _setup_expired.reset(self._token)
            self._token = None


def setup_checkpoint(where: str) -> None:
    """Raise :class:`SetupDeadlineExpired` if an enclosing :class:`abortable_setup`
    has run out of time; otherwise (including outside any scope) do nothing."""
    expired = _setup_expired.get()
    if expired is not None and expired():
        raise SetupDeadlineExpired(where)


def setup_abortable() -> bool:
    """Whether an :class:`abortable_setup` scope is active (a cheap gate for callers
    that would otherwise do extra bookkeeping per checkpoint)."""
    return _setup_expired.get() is not None
