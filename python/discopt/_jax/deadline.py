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

    from discopt._jax.deadline import deadline_scope, deadline_exceeded_jax

    with deadline_scope(time_limit_seconds):
        ...  # any JAX call whose cond_fn uses deadline_exceeded_jax()

The deadline is cleared on scope exit; if no deadline is set,
``deadline_exceeded_jax()`` returns ``False`` unconditionally.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

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
    result: jnp.ndarray = io_callback(
        _host_check,
        jax.ShapeDtypeStruct((), jnp.bool_),
        ordered=True,
    )
    return result
