"""Frontend-agnostic command engine for the interactive debugger.

Pounce's design is "one command engine, two frontends". The engine owns all
breakpoint state and command semantics; the REPL and the JSON protocol are thin
adapters that feed it command strings and render its results. This keeps human
and agent debugging behaviourally identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .checkpoints import EVENTS, Checkpoint, resolve_checkpoint

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession


class Control(Enum):
    """What a command tells the debugger to do after it runs."""

    NONE = auto()  # stay at the prompt (inspection command)
    STEP = auto()  # resume until next ITER_START
    STEPI = auto()  # resume until next checkpoint of any kind
    CONTINUE = auto()  # resume until next breakpoint / termination
    QUIT = auto()  # abort the solve


@dataclass
class CommandResult:
    """Outcome of one command: rendered text plus a control decision."""

    output: list[str] = field(default_factory=list)
    control: Control = Control.NONE
    data: Optional[dict] = None  # structured payload for the JSON frontend
    ok: bool = True  # False when the command failed (unknown / raised / unavailable)

    def line(self, text: str) -> "CommandResult":
        self.output.append(text)
        return self


_COND_RE = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|==|<|>)\s*([-+0-9.eE]+)\s*$")


@dataclass
class _Condition:
    raw: str
    clauses: list[tuple[str, str, float]]  # AND-joined (metric, op, value)

    def eval(self, metrics: dict[str, float]) -> bool:
        for metric, op, val in self.clauses:
            m = metrics.get(metric)
            if m is None:
                return False
            if not _apply(m, op, val):
                return False
        return True


def _apply(lhs: float, op: str, rhs: float) -> bool:
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    # float-tolerant equality (pounce semantics)
    return abs(lhs - rhs) <= 1e-9 * max(1.0, abs(rhs))


def _parse_condition(expr: str) -> _Condition:
    """Parse a ``&&``-joined conjunction of ``metric OP value`` atoms.

    ``||`` and grouping are not supported in v1 (register multiple conditions,
    as pounce recommends). Parentheses are stripped.
    """
    cleaned = expr.replace("(", " ").replace(")", " ")
    if "||" in cleaned:
        raise ValueError("'||' not supported; register separate 'break if' conditions")
    clauses: list[tuple[str, str, float]] = []
    for atom in cleaned.split("&&"):
        m = _COND_RE.match(atom)
        if not m:
            raise ValueError(f"cannot parse condition atom: {atom.strip()!r}")
        clauses.append((m.group(1), m.group(2), float(m.group(3))))
    return _Condition(raw=expr.strip(), clauses=clauses)


class DebugCommandEngine:
    """Holds breakpoint state and interprets debugger commands."""

    def __init__(self) -> None:
        self.iter_breaks: set[int] = set()
        self.temp_breaks: set[int] = set()
        self.conditions: list[_Condition] = []
        self.events: set[str] = set()
        self.watches: list[str] = []

    # ── breakpoint predicate (consulted by the session at each checkpoint) ──

    def hit_reason(self, ctx: "DebugContext") -> Optional[str]:
        """Return a human reason if a breakpoint fires at ``ctx``, else None.

        Iteration / conditional breakpoints are evaluated at ITER_START only;
        event breakpoints fire wherever the event surfaces.
        """
        if ctx.event and ctx.event in self.events:
            return f"event: {ctx.event}"
        if ctx.checkpoint is Checkpoint.ITER_START:
            it = ctx.iteration
            if it in self.iter_breaks or it in self.temp_breaks:
                self.temp_breaks.discard(it)
                return f"iteration {it}"
            metrics = ctx.metrics()
            for cond in self.conditions:
                if cond.eval(metrics):
                    return f"condition: {cond.raw}"
        return None

    # ── command dispatch ──

    def execute(self, raw: str, ctx: "DebugContext", session: "DebugSession") -> CommandResult:
        raw = raw.strip()
        if not raw:
            return CommandResult()
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        handler = _DISPATCH.get(cmd)
        if handler is None:
            return CommandResult([f"unknown command: {cmd!r} (try 'help')"], ok=False)
        try:
            return handler(self, args, ctx, session)
        except Exception as exc:  # keep the prompt alive on user error
            return CommandResult([f"error: {exc}"], ok=False)


# ── individual command handlers ────────────────────────────────────────────
# Each takes (engine, args, ctx, session) and returns a CommandResult.


def _fmt(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _cmd_help(engine, args, ctx, session):
    return CommandResult(
        [
            "flow:   step(s/n) stepi(si) continue(c) run N stop-at <cp> detach quit(q)",
            "breaks: break N | tbreak N | break if <m op v [&& ...]> | break on <event>",
            "        break (list) | break del N | break clear [cond|events]",
            "watch:  watch <target> | watch | watch clear",
            "look:   info(i) | print(p) incumbent|bound|gap|stats|nodes|node <i>|relax <i>",
            "steer:  inject <i>   (validate relax sol of batch node i; adopt if feasible)",
            "        hint <node_id> <var>",
            f"metrics: {', '.join(sorted(ctx.metrics()))}",
            f"checkpoints: {', '.join(c.value for c in Checkpoint)}",
        ]
    )


def _cmd_step(engine, args, ctx, session):
    return CommandResult(control=Control.STEP)


def _cmd_stepi(engine, args, ctx, session):
    return CommandResult(control=Control.STEPI)


def _cmd_continue(engine, args, ctx, session):
    return CommandResult(control=Control.CONTINUE)


def _cmd_quit(engine, args, ctx, session):
    return CommandResult(["quitting solve"], control=Control.QUIT)


def _cmd_detach(engine, args, ctx, session):
    session.detach()
    return CommandResult(["detached; solve runs to completion"], control=Control.CONTINUE)


def _cmd_run(engine, args, ctx, session):
    if not args:
        return CommandResult(["usage: run N"], ok=False)
    # One-shot (temp) break: `run N` means "pause once at iteration N", not
    # "leave a breakpoint at N behind".
    engine.temp_breaks.add(int(args[0]))
    return CommandResult([f"running to iteration {int(args[0])}"], control=Control.CONTINUE)


def _cmd_stop_at(engine, args, ctx, session):
    if not args:
        names = ", ".join(c.value for c in sorted(session.stop_checkpoints, key=lambda c: c.value))
        return CommandResult([f"stopping at: {names or '(none)'}"])
    if args[0] == "clear":
        session.stop_checkpoints = set()
        return CommandResult(["stop-at cleared"])
    cp = resolve_checkpoint(args[0])
    session.stop_checkpoints.add(cp)
    return CommandResult([f"will stop at {cp.value}"])


def _cmd_break(engine, args, ctx, session):
    if not args:  # list
        out = [f"iter breaks: {sorted(engine.iter_breaks) or '(none)'}"]
        out += [f"temp breaks: {sorted(engine.temp_breaks) or '(none)'}"]
        out += [f"conditions:  {[c.raw for c in engine.conditions] or '(none)'}"]
        out += [f"events:      {sorted(engine.events) or '(none)'}"]
        return CommandResult(out)
    head = args[0]
    if head == "if":
        cond = _parse_condition(" ".join(args[1:]))
        engine.conditions.append(cond)
        return CommandResult([f"break if {cond.raw}"])
    if head == "on":
        if not args[1:] or args[1] not in EVENTS:
            return CommandResult([f"events: {', '.join(sorted(EVENTS))}"])
        engine.events.add(args[1])
        return CommandResult([f"break on {args[1]}"])
    if head == "del":
        engine.iter_breaks.discard(int(args[1]))
        engine.temp_breaks.discard(int(args[1]))
        return CommandResult([f"removed break {int(args[1])}"])
    if head == "clear":
        what = args[1] if len(args) > 1 else "all"
        if what in ("cond", "conditions"):
            engine.conditions.clear()
        elif what == "events":
            engine.events.clear()
        else:
            engine.iter_breaks.clear()
            engine.temp_breaks.clear()
            engine.conditions.clear()
            engine.events.clear()
        return CommandResult([f"cleared {what} breakpoints"])
    engine.iter_breaks.add(int(head))
    return CommandResult([f"break at iteration {int(head)}"])


def _cmd_tbreak(engine, args, ctx, session):
    engine.temp_breaks.add(int(args[0]))
    return CommandResult([f"one-shot break at iteration {int(args[0])}"])


def _cmd_info(engine, args, ctx, session):
    m = ctx.metrics()
    return CommandResult(
        [
            f"[{ctx.checkpoint.value}] iter={ctx.iteration} "
            f"nodes={ctx.node_count} open={ctx.open_nodes} "
            f"incumbent={_fmt(ctx.incumbent_obj)} bound={_fmt(ctx.best_bound)} "
            f"gap={_fmt(ctx.gap)} t={m['elapsed']:.2f}s"
        ],
        data={"metrics": m, "checkpoint": ctx.checkpoint.value},
    )


def _cmd_print(engine, args, ctx, session):
    if not args:
        return CommandResult(["usage: print <target>"])
    target = args[0].lower()
    if target in ("incumbent", "inc"):
        return CommandResult(
            [f"incumbent = {_fmt(ctx.incumbent_obj)}"],
            data={"incumbent": ctx.incumbent_obj},
        )
    if target in ("bound", "lb"):
        return CommandResult([f"bound = {_fmt(ctx.best_bound)}"], data={"bound": ctx.best_bound})
    if target == "gap":
        return CommandResult([f"gap = {_fmt(ctx.gap)}"], data={"gap": ctx.gap})
    if target == "stats":
        return _cmd_info(engine, args, ctx, session)
    if target == "nodes":
        if ctx.batch_ids is None:
            return CommandResult(["no batch live at this checkpoint"])
        lines = [f"{ctx.n_batch} open node(s) in batch:"]
        for i in range(min(ctx.n_batch, 20)):
            lines.append(f"  [{i}] id={int(ctx.batch_ids[i])}")
        return CommandResult(lines, data={"node_ids": [int(x) for x in ctx.batch_ids]})
    if target == "node":
        i = int(args[1])
        if ctx.batch_lb is None:
            return CommandResult(["no batch live at this checkpoint"])
        lb = np.asarray(ctx.batch_lb[i])
        ub = np.asarray(ctx.batch_ub[i])
        return CommandResult(
            [
                f"node[{i}] id={int(ctx.batch_ids[i])} box:",
                f"  lb = {np.array2string(lb, precision=4, threshold=20)}",
                f"  ub = {np.array2string(ub, precision=4, threshold=20)}",
            ],
            data={"lb": lb.tolist(), "ub": ub.tolist()},
        )
    if target == "relax":
        i = int(args[1])
        if ctx.result_sols is None:
            return CommandResult(["no relaxation results live at this checkpoint"])
        sol = np.asarray(ctx.result_sols[i])
        return CommandResult(
            [
                f"relax[{i}] bound={_fmt(float(ctx.result_lbs[i]))} "
                f"feasible={bool(ctx.result_feas[i]) if ctx.result_feas is not None else '?'}",
                f"  x = {np.array2string(sol, precision=4, threshold=20)}",
            ],
            data={"bound": float(ctx.result_lbs[i]), "x": sol.tolist()},
        )
    return CommandResult([f"unknown print target: {target!r}"])


def _cmd_watch(engine, args, ctx, session):
    if not args:
        return CommandResult([f"watches: {engine.watches or '(none)'}"])
    if args[0] == "clear":
        engine.watches.clear()
        return CommandResult(["watches cleared"])
    engine.watches.append(" ".join(args))
    return CommandResult([f"watching: {' '.join(args)}"])


def _cmd_inject(engine, args, ctx, session):
    if ctx.steer is None or ctx.result_sols is None:
        return CommandResult(["inject unavailable at this checkpoint"], ok=False)
    if not ctx.steer.can_inject:
        return CommandResult(
            [
                "inject unavailable: this solve path wires no candidate validator, "
                "so a point's true feasibility/objective cannot be verified "
                "(refusing — certificate safety)"
            ],
            ok=False,
        )
    i = int(args[0])
    sol = np.asarray(ctx.result_sols[i], dtype=np.float64)
    # The candidate is validated against the ORIGINAL problem (integrality,
    # constraints, true objective) — never trusted with its relaxation bound.
    adopted, obj, reason = ctx.steer.inject(sol)
    if obj is None:
        return CommandResult([f"inject node[{i}]: {reason}"])
    return CommandResult([f"inject node[{i}] obj={_fmt(obj)}: {reason}"])


def _cmd_hint(engine, args, ctx, session):
    if ctx.steer is None:
        return CommandResult(["hint unavailable at this checkpoint"], ok=False)
    node_id, var = int(args[0]), int(args[1])
    ctx.steer.hint([node_id], [var])
    return CommandResult([f"branch hint: node {node_id} -> var {var}"])


_DISPATCH: dict[str, Callable[..., CommandResult]] = {
    "help": _cmd_help,
    "h": _cmd_help,
    "?": _cmd_help,
    "step": _cmd_step,
    "s": _cmd_step,
    "n": _cmd_step,
    "stepi": _cmd_stepi,
    "si": _cmd_stepi,
    "continue": _cmd_continue,
    "c": _cmd_continue,
    "quit": _cmd_quit,
    "q": _cmd_quit,
    "exit": _cmd_quit,
    "detach": _cmd_detach,
    "run": _cmd_run,
    "r": _cmd_run,
    "stop-at": _cmd_stop_at,
    "break": _cmd_break,
    "b": _cmd_break,
    "tbreak": _cmd_tbreak,
    "tb": _cmd_tbreak,
    "info": _cmd_info,
    "i": _cmd_info,
    "print": _cmd_print,
    "p": _cmd_print,
    "watch": _cmd_watch,
    "inject": _cmd_inject,
    "hint": _cmd_hint,
}
