"""Interactive branch-and-bound debugger for discopt — a "pdb for B&B".

Pause a solve at well-defined node-lifecycle checkpoints, inspect the tree /
nodes / incumbent / relaxations, set breakpoints by iteration / condition /
event, and *safely* steer (inject an incumbent, hint a branch). It has **zero
effect on the solve when not attached**: every fire-site is a single ``None``
check, and the state context is built lazily only once a debugger is attached.

Typical use::

    result = m.solve(debug=True)          # drop into the REPL at the first pause
    result = m.solve(debug="on-error")    # enter only if the solve fails/limits

Programmatic / notebook use::

    from discopt import debug
    with debug.attach():
        result = m.solve()

Correctness note: debugger steering can never invalidate the solver's
certificate — see :mod:`discopt.debug.steer`.

The active session is **thread-local**: attaching in one thread never affects
solves running in other threads, so concurrent solves can each carry their own
debugger (or none). Attach in the same thread that calls ``solve()``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from .checkpoints import Checkpoint
from .context import DebugContext
from .session import DebugQuit, DebugSession

if TYPE_CHECKING:
    from .session import Frontend

__all__ = [
    "Checkpoint",
    "DebugContext",
    "DebugQuit",
    "DebugSession",
    "attach",
    "detach",
    "current",
    "is_attached",
    "fire",
    "fire_rust",
    "rust_hook",
    "make_session",
]

# Thread-local active session. ``None`` => detached => fire() is a no-op.
# Thread-local (not module-global) so concurrent solves in different threads
# never share a frontend or stop each other.
_TLS = threading.local()


def _active() -> Optional[DebugSession]:
    return getattr(_TLS, "session", None)


def attach(session: Optional[DebugSession] = None, **kwargs: Any) -> "_AttachGuard":
    """Attach a debugger session for the duration of a ``with`` block (or until
    :func:`detach`). With no argument, builds a default REPL session. The
    attachment is thread-local: only solves on this thread see the debugger.
    """
    if session is None:
        session = make_session(**kwargs)
    _TLS.session = session
    return _AttachGuard(session)


def detach() -> None:
    """Remove this thread's active debugger; subsequent ``fire`` calls are no-ops."""
    _TLS.session = None


def current() -> Optional[DebugSession]:
    return _active()


def is_attached() -> bool:
    return _active() is not None


def make_session(
    kind: Any = True,
    *,
    frontend: "Optional[Frontend]" = None,
    script: Any = None,
) -> DebugSession:
    """Build a :class:`DebugSession` from a ``debug=`` argument.

    ``kind`` may be ``True`` / ``"repl"`` (human REPL), ``"json"`` (agent
    protocol on stdin/stdout), ``"on-error"``, or a ready-made
    :class:`DebugSession` (returned as-is). Anything else raises ``ValueError``
    — a typo like ``debug="jsn"`` must not silently fall back to the REPL.
    """
    if isinstance(kind, DebugSession):
        return kind
    if kind is not True and kind not in ("repl", "json", "on-error"):
        raise ValueError(
            f"debug: unknown mode {kind!r} (expected True, 'repl', 'json', "
            "'on-error', or a DebugSession)"
        )
    enter_on_error = kind == "on-error"
    if frontend is None:
        if kind == "json":
            from .jsonproto import JsonFrontend

            frontend = JsonFrontend()
        else:
            from .repl import ReplFrontend

            frontend = ReplFrontend(script=script)
    return DebugSession(frontend, enter_on_error=enter_on_error)


def fire(
    checkpoint: Checkpoint,
    *,
    tree: Any = None,
    model: Any = None,
    iteration: int = 0,
    elapsed: float = 0.0,
    batch_lb: Any = None,
    batch_ub: Any = None,
    batch_ids: Any = None,
    result_lbs: Any = None,
    result_sols: Any = None,
    result_feas: Any = None,
    event: Optional[str] = None,
    error: Any = None,
    validator: Any = None,
) -> bool:
    """Fire a checkpoint. **Hot path** — returns immediately when detached.

    Arguments are passed as already-existing loop locals, so a detached solve
    incurs only the attribute read and one ``is None`` test.

    Returns
    -------
    bool
        ``True`` if the user requested the solve stop (``quit``); the loop
        should ``break`` promptly. Always ``False`` when detached.
    """
    session = _active()
    if session is None:
        return False
    ctx = DebugContext.build(
        checkpoint,
        tree=tree,
        model=model,
        iteration=iteration,
        elapsed=elapsed,
        batch_lb=batch_lb,
        batch_ub=batch_ub,
        batch_ids=batch_ids,
        result_lbs=result_lbs,
        result_sols=result_sols,
        result_feas=result_feas,
        event=event,
        validator=validator,
    )
    if error is not None:
        ctx.extra["error"] = error
    return session.on_checkpoint(ctx)


def fire_rust(state: dict) -> bool:
    """Checkpoint entry for the pure-Rust MILP hook (see ``lp_bindings.rs``).

    Called across the PyO3 boundary with an aggregate state dict. Returns
    ``True`` to stop the search (``quit``). A no-op / detached debugger returns
    ``False`` — matching the Rust guard that only installs a hook when attached.
    """
    session = _active()
    if session is None:
        return False
    return session.on_checkpoint(DebugContext.from_rust(state))


def rust_hook() -> "Optional[Callable[[dict], bool]]":
    """Return the callable to pass as ``solve_milp_py(debug_hook=…)``, or None.

    A debugger must be attached *now* for a hook to be installed, so the Rust
    search stays bound-neutral whenever no debugger is present.
    """
    if _active() is None:
        return None
    return fire_rust


class _AttachGuard:
    """Context manager returned by :func:`attach`; detaches on exit."""

    def __init__(self, session: DebugSession) -> None:
        self.session = session

    def __enter__(self) -> DebugSession:
        return self.session

    def __exit__(self, *exc: Any) -> None:
        detach()
