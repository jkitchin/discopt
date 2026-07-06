"""Newline-delimited JSON protocol frontend for the debugger.

This is the agent-facing twin of the human REPL: it shares the exact same
:class:`~discopt.debug.engine.DebugCommandEngine`, so an LLM agent, script, or
GUI drives the solver with identical semantics. Modeled on pounce's
``pounce-dbg/1`` protocol.

Wire format (all newline-delimited JSON on a single stream):

* first output line is a ``hello`` handshake advertising every checkpoint,
  event, command, metric, and capability — clients feature-detect off it and
  never need out-of-band docs;
* at each pause the frontend emits a ``pause`` event (or a ``terminated`` event
  at the final checkpoint), then reads command objects until one resumes;
* each command yields a ``result`` event echoing the client's ``id`` as
  ``request_id``; ``ok`` is ``false`` when the command failed (unknown verb,
  raised, or unavailable at this checkpoint), so agents can branch on it.

Input commands are either a bare string (``"continue"``) or an object
(``{"cmd": "print", "args": ["node", "0"], "id": 7}``); ``{"cmd": "break if
gap<0.2", "id": 8}`` (whole command in ``cmd``) is also accepted.
"""

from __future__ import annotations

import json
import math
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

from .checkpoints import EVENTS, Checkpoint
from .engine import Control

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession

PROTOCOL = "discopt-dbg/1"

# Canonical command verbs advertised in the handshake (aliases omitted).
_COMMANDS = [
    "step",
    "stepi",
    "continue",
    "run",
    "stop-at",
    "detach",
    "quit",
    "break",
    "tbreak",
    "info",
    "print",
    "watch",
    "inject",
    "hint",
    "help",
]

_METRICS = ["nodes", "open", "incumbent", "bound", "gap", "iter", "elapsed"]


def _capabilities() -> dict[str, Any]:
    """What this debugger build can do — clients branch on these, not version."""
    return {
        "inspect": True,
        "safe_steer": True,  # inject (validated vs original problem) + branch hint
        "mutate_iterate": False,  # certificate-safe: no node-box/bound edits
        "conditional_breakpoints": "compound_and",  # '&&' only; no '||'/grouping
        "event_breakpoints": True,
        "request_ids": True,
        "rewind": False,
        "resolve": False,
        "progress_events": False,
        "terminal_checkpoint": True,
    }


class JsonFrontend:
    """Drives the command engine over a newline-delimited JSON stream."""

    def __init__(
        self,
        *,
        read_fn: Optional[Callable[[], Optional[str]]] = None,
        write_fn: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._read = read_fn or _stdin_reader()
        self._write = write_fn or _stdout_writer()
        self._hello_sent = False

    # ── event emitters ──

    def _emit(self, obj: dict) -> None:
        # Non-finite floats (inf/nan) are not valid JSON; strict agent parsers
        # reject them. Coerce to null so the stream is always RFC-8259 clean.
        self._write(_json_safe(obj))

    def _hello(self) -> dict:
        return {
            "event": "hello",
            "protocol": PROTOCOL,
            "checkpoints": [c.value for c in Checkpoint],
            "events": sorted(EVENTS),
            "commands": _COMMANDS,
            "metrics": _METRICS,
            "capabilities": _capabilities(),
        }

    def _pause_event(self, ctx: "DebugContext") -> dict:
        m = ctx.metrics()
        event = "terminated" if ctx.checkpoint is Checkpoint.TERMINATED else "pause"
        payload = {
            "event": event,
            "checkpoint": ctx.checkpoint.value,
            "iter": ctx.iteration,
            "nodes": ctx.node_count,
            "open_nodes": ctx.open_nodes,
            "incumbent": ctx.incumbent_obj,
            "bound": ctx.best_bound if ctx.best_bound not in (float("-inf"),) else None,
            "gap": ctx.gap,
            "elapsed": m["elapsed"],
            "n_batch": ctx.n_batch,
            "reason": ctx.extra.get("reason"),
        }
        return payload

    def _result_event(self, res, command: str, req_id: Any) -> dict:
        return {
            "event": "result",
            "request_id": req_id,
            "command": command,
            "ok": res.ok,
            "output": res.output,
            "data": res.data,
        }

    # ── the Frontend protocol ──

    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        if not self._hello_sent:
            self._emit(self._hello())
            self._hello_sent = True

        self._emit(self._pause_event(ctx))

        while True:
            line = self._read()
            if line is None:  # EOF: client hung up -> run to completion
                return Control.CONTINUE
            line = line.strip()
            if not line:
                continue
            command, req_id = _parse_command(line)
            if command is None:
                self._emit(
                    {
                        "event": "result",
                        "request_id": req_id,
                        "ok": False,
                        "output": [f"malformed command: {line!r}"],
                        "data": None,
                    }
                )
                continue
            res = session.engine.execute(command, ctx, session)
            self._emit(self._result_event(res, command, req_id))
            if res.control is not Control.NONE:
                return res.control


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parse_command(line: str) -> tuple[Optional[str], Any]:
    """Return ``(command_string, request_id)`` from one input line.

    Accepts a bare string, a bare JSON string literal, or an object with
    ``cmd`` (+ optional ``args`` list and ``id``).
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        # Treat a raw, unquoted line as the command itself.
        return line, None

    if isinstance(obj, str):
        return obj, None
    if isinstance(obj, dict):
        cmd = obj.get("cmd")
        if not isinstance(cmd, str):
            return None, obj.get("id")
        args = obj.get("args") or []
        if args:
            cmd = cmd + " " + " ".join(str(a) for a in args)
        return cmd, obj.get("id")
    return None, None


def _stdin_reader() -> Callable[[], Optional[str]]:
    def read() -> Optional[str]:
        line = sys.stdin.readline()
        return None if line == "" else line

    return read


def _stdout_writer() -> Callable[[dict], None]:
    def write(obj: dict) -> None:
        # default=str is a safety net for any stray numpy scalar; _json_safe has
        # already handled non-finite floats upstream.
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()

    return write
