"""Human REPL frontend for the interactive debugger.

Prints a pause banner and reads commands until the user issues a resume verb
(``step``/``continue``/…). Uses ``readline`` for history and tab-completion
when a TTY is available, and falls back to a plain line reader on a pipe so the
same command stream can drive scripted tests (as pounce does).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from .engine import Control

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession


class ReplFrontend:
    """Interactive stdin/stderr REPL sharing the session's command engine."""

    def __init__(
        self,
        *,
        input_fn: Optional[Callable[[str], str]] = None,
        out=None,
        script: Optional[Iterable[str]] = None,
    ) -> None:
        # ``script`` lets tests / ``--debug-script`` feed commands without a TTY.
        self._script = iter(script) if script is not None else None
        self._input_fn = input_fn or input
        self._out = out or sys.stderr
        self._banner_shown = False

    def _emit(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def _read(self, prompt: str) -> str:
        if self._script is not None:
            try:
                line = next(self._script)
                self._emit(f"{prompt}{line}")
                return line
            except StopIteration:
                # Script exhausted: detach and run to completion.
                return "detach"
        return self._input_fn(prompt)

    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        if not self._banner_shown:
            self._emit("discopt debugger — 'help' for commands, 'c' to continue")
            self._banner_shown = True

        reason = ctx.extra.get("reason", "")
        self._emit(
            f"\n── paused at {ctx.checkpoint.value}"
            + (f"  ({reason})" if reason else "")
            + f"  iter={ctx.iteration} nodes={ctx.node_count} "
            f"incumbent={_fmt(ctx.incumbent_obj)} bound={_fmt(ctx.best_bound)}"
        )
        # Auto-print watches.
        for target in session.engine.watches:
            res = session.engine.execute(f"print {target}", ctx, session)
            for ln in res.output:
                self._emit(f"  {ln}")

        while True:
            try:
                raw = self._read("(discopt-dbg) ")
            except (EOFError, KeyboardInterrupt):
                self._emit("")
                return Control.CONTINUE
            res = session.engine.execute(raw, ctx, session)
            for ln in res.output:
                self._emit(ln)
            if res.control is not Control.NONE:
                return res.control


def _fmt(x) -> str:
    if x is None:
        return "none"
    return f"{x:.6g}" if isinstance(x, float) else str(x)
