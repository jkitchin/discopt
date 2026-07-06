"""Command-line entry point for the interactive branch-and-bound debugger.

Lets an external agent, GUI, or human spawn a debugged solve of an AMPL ``.nl``
model as a standalone process — the counterpart to pounce's ``--debug`` /
``--debug-json`` CLI. The three modes mirror ``Model.solve(debug=...)``::

    python -m discopt.debug model.nl              # human REPL
    python -m discopt.debug model.nl --json       # JSON agent protocol (stdin/stdout)
    python -m discopt.debug model.nl --on-error   # enter only if the solve fails
    python -m discopt.debug model.nl --script cmds.pdbg   # scripted REPL

In ``--json`` mode the debugger's protocol owns **stdout** (newline-delimited
JSON); the solver banner and the final result summary go to **stderr** so the
stream stays machine-parseable.
"""

from __future__ import annotations

import argparse
import sys
from typing import IO, TYPE_CHECKING, Optional, Sequence, cast

if TYPE_CHECKING:
    from discopt.modeling.core import SolveResult


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m discopt.debug",
        description="Interactively debug a branch-and-bound solve of an AMPL .nl model.",
    )
    p.add_argument("model", help="path to an AMPL .nl model file")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--json",
        action="store_true",
        help="drive via the self-describing JSON agent protocol on stdin/stdout",
    )
    mode.add_argument(
        "--on-error",
        action="store_true",
        help="run freely; enter the debugger only if the solve fails or hits a limit",
    )
    p.add_argument(
        "--script",
        metavar="FILE",
        help="run debugger commands from FILE at each pause (human REPL mode only)",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        metavar="SECONDS",
        help="solve wall-clock time limit (default: 3600)",
    )
    p.add_argument(
        "--nlp-solver",
        default=None,
        metavar="NAME",
        help="override the node solver, e.g. 'simplex' for the pure-Rust MILP path",
    )
    return p


def _resolve_kind(args: argparse.Namespace):
    """Map parsed flags to the ``debug=`` argument understood by ``Model.solve``."""
    if args.json:
        return "json"
    if args.on_error:
        return "on-error"
    return True  # human REPL


def _read_script(path: Optional[str]) -> Optional[list[str]]:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        # One command per line; skip blanks and '#' / '//' comments (pounce style).
        out: list[str] = []
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            out.append(s)
        return out


def main(argv: Optional[Sequence[str]] = None, *, err: Optional[IO[str]] = None) -> int:
    """Run a debugged solve. Returns a process exit code (0 = solved)."""
    err = err or sys.stderr
    args = build_parser().parse_args(argv)

    if args.script and (args.json or args.on_error):
        print("error: --script applies to the human REPL only", file=err)
        return 2

    from discopt import debug as _debug
    from discopt.modeling import from_nl

    try:
        model = from_nl(args.model)
    except FileNotFoundError:
        print(f"error: no such model file: {args.model}", file=err)
        return 2
    except Exception as exc:  # malformed .nl, parser error, etc.
        print(f"error: could not load {args.model}: {exc}", file=err)
        return 2

    kind = _resolve_kind(args)
    if kind == "json":
        session = _debug.make_session("json")
    else:
        # Route the REPL onto the same ``err`` stream as the summary, so the
        # whole human-facing session is one stream (and testable via a buffer).
        from discopt.debug.repl import ReplFrontend

        frontend = ReplFrontend(script=_read_script(args.script), out=err)
        session = _debug.make_session(kind, frontend=frontend)

    solve_kwargs = {"time_limit": args.time_limit, "debug": session}
    if args.nlp_solver:
        solve_kwargs["nlp_solver"] = args.nlp_solver

    # A non-streaming solve always returns a SolveResult (not an update iterator).
    result = cast("SolveResult", model.solve(**solve_kwargs))

    # Result summary always to stderr so --json keeps stdout pure.
    obj = "n/a" if result.objective is None else f"{result.objective:.6g}"
    print(
        f"[discopt.debug] status={result.status} objective={obj} nodes={result.node_count}",
        file=err,
    )
    return 0 if result.status in ("optimal", "feasible") else 1


if __name__ == "__main__":  # pragma: no cover - process entry
    sys.exit(main())
