"""The debugger session: pause-decision state machine over the command engine.

A ``DebugSession`` decides *whether* to pause at each checkpoint (based on the
current step mode plus the engine's breakpoints) and, when it pauses, hands the
context to the active frontend (REPL or JSON) to interact. It is deliberately
frontend-agnostic; ``frontend.interact(ctx, session)`` drives the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from .checkpoints import DEFAULT_STOP, Checkpoint
from .engine import Control, DebugCommandEngine

if TYPE_CHECKING:
    from .context import DebugContext


class DebugQuit(Exception):
    """Reserved for a future hard-abort path. ``quit`` currently requests a
    graceful stop via :attr:`DebugSession.stop_requested` so the solver still
    builds a valid "stopped early" result from the partial tree state.
    """


class Frontend(Protocol):
    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        """Drive the prompt until the user resumes; return the resume control."""
        ...


class DebugSession:
    """Owns step mode + breakpoints and mediates checkpoint pauses."""

    def __init__(self, frontend: "Frontend", *, enter_on_error: bool = False) -> None:
        self.engine = DebugCommandEngine()
        self.frontend = frontend
        self.stop_checkpoints: set[Checkpoint] = set(DEFAULT_STOP)
        self.enter_on_error = enter_on_error
        # Step mode: None means "continue" (only breakpoints/stop-set pause).
        self._mode: Optional[Control] = Control.STEP  # first checkpoint pauses
        self._detached = False
        self._stop_requested = False
        self._last_incumbent: Optional[float] = None

    # ── attachment lifecycle ──

    def detach(self) -> None:
        self._detached = True

    @property
    def detached(self) -> bool:
        return self._detached

    @property
    def stop_requested(self) -> bool:
        """True once the user issued ``quit``; the loop should break promptly."""
        return self._stop_requested

    # ── the hot path: called at every fired checkpoint ──

    def on_checkpoint(self, ctx: "DebugContext") -> bool:
        """Handle a checkpoint; return True if the solve should stop now."""
        if self._detached:
            # A detached session is inert. Returning _stop_requested here would
            # let a stale session (quit during an earlier solve, then reused)
            # kill every future solve at its first checkpoint — the quit that
            # triggered the stop already propagated True through the non-
            # detached path below.
            return False
        # Derive the new_incumbent event for event breakpoints.
        if ctx.checkpoint is Checkpoint.INCUMBENT_FOUND:
            ctx.event = "new_incumbent"

        if self._should_pause(ctx):
            control = self.frontend.interact(ctx, self)
            self._apply_control(control)
        return self._stop_requested

    def _should_pause(self, ctx: "DebugContext") -> bool:
        # on-error sessions only ever pause at a failed termination. The solve
        # loops pass ``error=<non-"optimal" status>`` at their TERMINATED fire
        # (None on a certified optimum), so this triggers on limit hits,
        # debugger interrupts, "unknown", and certified infeasibility.
        if self.enter_on_error:
            if ctx.checkpoint is Checkpoint.TERMINATED and ctx.extra.get("error"):
                ctx.extra["reason"] = f"on-error: {ctx.extra['error']}"
                return True
            return False

        # Explicit breakpoint hit always pauses (and records the reason).
        reason = self.engine.hit_reason(ctx)
        if reason is not None:
            ctx.extra["reason"] = reason
            return True

        # Step modes.
        if self._mode is Control.STEPI:
            ctx.extra["reason"] = "stepi"
            return True
        if self._mode is Control.STEP:
            if ctx.checkpoint is Checkpoint.ITER_START or ctx.checkpoint is Checkpoint.TERMINATED:
                ctx.extra["reason"] = "step"
                return True
            return False

        # Continue mode: pause only at requested checkpoints.
        if ctx.checkpoint in self.stop_checkpoints and self._mode is None:
            # Honor stop-at only when the user explicitly added a checkpoint
            # beyond the defaults; defaults are for step mode.
            if ctx.checkpoint not in DEFAULT_STOP or ctx.checkpoint is Checkpoint.TERMINATED:
                ctx.extra["reason"] = f"stop-at {ctx.checkpoint.value}"
                return True
        return False

    def _apply_control(self, control: Control) -> None:
        if control is Control.QUIT:
            # Graceful stop: detach so no further pauses, and signal the loop.
            self._stop_requested = True
            self._detached = True
            return
        if control is Control.STEP:
            self._mode = Control.STEP
        elif control is Control.STEPI:
            self._mode = Control.STEPI
        elif control is Control.CONTINUE:
            self._mode = None
        # NONE never returns here (interact loops until a resume control).
