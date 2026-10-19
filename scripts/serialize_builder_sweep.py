"""Classify every model-producing builder's serialization outcome (WP-6, D12).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-6. Builds one small model
per case in ``python/tests/serialize_builder_cases.py`` and classifies it:

* ``round-trips`` -- ``dumps(loads(dumps(m))) == dumps(m)``, equal
  ``model_fingerprint``, and ``pickle`` fingerprint-equal too;
* ``refused: <message>`` -- ``dumps`` raised ``SerializationError`` (unexpected
  unless the message matches the case's own ``Refusal.match``);
* ``crashed: <exc>`` -- anything else (a missing classification or encoding).

With ``--solve`` it also solves each round-tripping model and its reload and
reports wall time and whether ``(status, node_count, objective)`` agree (a solve
stopped by its time limit is reported, not compared: see ``_COMPARED``). Prints
one line per case and ``executed: N, crashed: C, unexpected: U``; exits non-zero
on a crash, an outcome that differs from the case's table, or nothing executed.

    python -u scripts/serialize_builder_sweep.py [--solve]
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

# torch and cyipopt each link their own OpenMP runtime, and importing torch after a
# solve has loaded cyipopt aborts the process ("OMP: Error #15"). The reader cases
# import torch, so load it first, when it is installed.
if importlib.util.find_spec("torch") is not None:
    import torch  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "tests"))

import serialize_builder_cases as cases  # noqa: E402


def classify(build) -> tuple[str, str, object]:
    """``(kind, detail, model)``: kind is round-trips / refused / crashed."""
    from discopt.serialize import SerializationError, dumps, loads
    from discopt.transformations import model_fingerprint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build()
        try:
            text = dumps(model, provenance=False)
        except SerializationError as exc:
            return "refused", str(exc), model
        reloaded = loads(text)
    problems = []
    if dumps(reloaded, provenance=False) != text:
        problems.append("dumps not stable")
    fp = model_fingerprint(model)
    if model_fingerprint(reloaded) != fp:
        problems.append("fingerprint differs after reload")
    if model_fingerprint(pickle.loads(pickle.dumps(model))) != fp:
        problems.append("fingerprint differs after pickle")
    if problems:
        return "crashed", "; ".join(problems), model
    return "round-trips", "", model


def solve_pair(model) -> tuple[float, bool, str]:
    from discopt.serialize import dumps, loads

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = loads(dumps(model))
        t0 = time.perf_counter()
        a = _solve_outcome(model)
        wall = time.perf_counter() - t0
        b = _solve_outcome(reloaded)
    if a[0] not in _COMPARED or b[0] not in _COMPARED:
        return wall, None, f"{a}"  # stopped by the time limit: wall-clock dependent
    return wall, a == b, f"{a}"


#: Outcomes that do not depend on the machine's speed. A solve stopped by its
#: time limit reports however many nodes fit in the time, so its node count
#: differs between two runs of the *same* model (measured: fit_trajectories gave
#: 337, 345 and 361 nodes at the same objective in three runs).
_COMPARED = frozenset({"optimal", "infeasible", "unbounded", "raised"})


def _solve_outcome(model) -> tuple:
    """``(status, node_count, objective)``, or the refusal a solve raised: a
    builder may produce a model the solver declines (both sides must agree)."""
    try:
        r = model.solve(time_limit=30)
    except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
        return ("raised", type(exc).__name__, str(exc)[:80])
    return (r.status, r.node_count, r.objective)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solve", action="store_true", help="also check bound neutrality")
    args = ap.parse_args(argv)

    table = [(k, "round-trips", b) for k, b in cases.ROUND_TRIPS.items()]
    for k, (modules, b) in cases.READERS.items():
        missing = [mod for mod in modules if importlib.util.find_spec(mod) is None]
        if missing:
            print(f"{k:58s} skipped: {', '.join(missing)} not installed", flush=True)
        else:
            table.append((k, "round-trips", b))
    table += [(k, "refused", r.build) for k, r in cases.REFUSED.items()]
    executed = crashed = unexpected = 0
    for key, expected, build in table:
        try:
            kind, detail, model = classify(build)
        except Exception as exc:  # noqa: BLE001 - classified, counted, printed in full
            kind, detail, model = "crashed", f"{type(exc).__name__}: {exc}", None
            traceback.print_exc(file=sys.stderr)
        executed += 1
        crashed += kind == "crashed"
        # A refusal is expected only with the case's own message: one raised for a
        # different, unlisted reason is the failure this sweep exists to catch.
        wrong_reason = kind == "refused" and not re.search(cases.REFUSED[key].match, detail)
        unexpected += kind != expected or wrong_reason
        mark = "" if kind == expected else f"   <-- expected {expected}"
        if wrong_reason:
            mark = f"   <-- expected a refusal matching {cases.REFUSED[key].match!r}"
        line = f"{key:58s} {kind}{': ' + detail[:110] if detail else ''}{mark}"
        if args.solve and kind == "round-trips":
            wall, same, outcome = solve_pair(model)
            verdict = {True: "same", False: "DIFFERENT", None: "time-limited, not compared"}
            line += f"   solve {wall:5.2f}s {verdict[same]} {outcome}"
            unexpected += same is False
        print(line, flush=True)
    print(f"executed: {executed}, crashed: {crashed}, unexpected: {unexpected}", flush=True)
    return 2 if executed == 0 else (1 if crashed or unexpected else 0)


if __name__ == "__main__":
    sys.exit(main())
