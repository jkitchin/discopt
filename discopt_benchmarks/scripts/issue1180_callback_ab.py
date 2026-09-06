#!/usr/bin/env python
"""#1180 build gate -- the callback-path change, A/B, interleaved, in one process.

The change is **bound-neutral by construction** (it computes identical numbers;
bit-identity is pinned separately over the corpus), so under CLAUDE.md §5 the bar
is: ``node_count`` and the certified ``objective`` **exactly unchanged**, and a
wall improvement that is measured rather than asserted.

Arms, toggled at runtime so they interleave in ONE process on ONE box rather than
running as two sequential jobs on a machine whose clock speed can drift:

* ``new`` -- the shipped code: ``TapeNLPEvaluator._x`` hands pounce a contiguous
  ``float64`` array, and ``_timing.charge`` is a ``__slots__`` context-manager
  class.
* ``old`` -- the pre-change code restored: ``_x`` rebuilds a Python list per
  callback, and ``charge`` is the ``@contextlib.contextmanager`` generator.

Each arm asserts which variant is live before it runs (CLAUDE.md §8: verify the
code you actually loaded), so an arm that silently ran the other one fails
instead of reporting a 1.00x.

Neutrality is checked on a **deterministic** budget (``deterministic=True``,
which renders the wall-clock role-2 budgets inert) so a faster arm cannot simply
do more nodes and look like a behavior change. The wall comparison uses the
ordinary wall budget, which is what a user actually experiences.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time

import numpy as np

ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


def _old_x(self, x):
    """The pre-#1180 marshaling: a Python list rebuilt on every callback."""
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def _make_old_charge(timing):
    @contextlib.contextmanager
    def charge(bucket: str):
        """The pre-#1180 generator-based context manager, verbatim."""
        if bucket not in timing.BUCKETS:
            raise ValueError(f"unknown timing bucket {bucket!r}")
        stack = timing._stack()

        class _F:
            __slots__ = ("bucket", "started", "child")

        frame = _F()
        frame.bucket = bucket
        frame.started = time.perf_counter()
        frame.child = 0.0
        stack.append(frame)
        try:
            yield
        finally:
            stack.pop()
            elapsed = time.perf_counter() - frame.started
            timing._totals()[bucket] += elapsed - frame.child
            if stack:
                stack[-1].child += elapsed

    return charge


class Arms:
    """Install/remove the OLD code path; ``new`` is simply the un-patched tree."""

    def __init__(self):
        from discopt import _timing
        from discopt._tape_nlp_evaluator import TapeNLPEvaluator

        self.timing = _timing
        self.tape = TapeNLPEvaluator
        self.new_x = TapeNLPEvaluator._x
        self.new_charge = _timing.charge
        self.old_charge = _make_old_charge(_timing)
        # nlp_ipopt captured ``_timing`` as a module, not ``charge`` as a value
        # (the import is inside ``_charge_evaluator``), so patching the module
        # attribute reaches the hot wrapper. Verified by the arm assertion below.

    def install(self, arm: str) -> None:
        if arm == "old":
            self.tape._x = _old_x
            self.timing.charge = self.old_charge
        else:
            self.tape._x = self.new_x
            self.timing.charge = self.new_charge

    def verify(self, arm: str) -> None:
        """Assert the arm that is live is the arm we asked for."""
        from discopt import _timing

        cm = _timing.charge("rust")
        is_generator_cm = type(cm).__name__ not in ("_Charge",)
        with cm:
            pass
        probe = np.array([1.0, 2.0, 3.0])

        class _Fake:
            pass

        marshalled = self.tape._x(_Fake(), probe)
        if arm == "old":
            check(is_generator_cm, "old arm: charge is not the generator context manager")
            check(isinstance(marshalled, list), "old arm: _x did not return a Python list")
        else:
            check(not is_generator_cm, "new arm: charge is not the _Charge class")
            check(
                isinstance(marshalled, np.ndarray) and marshalled.dtype == np.float64,
                "new arm: _x did not return a float64 array",
            )


def solve_once(nl: str, time_limit: float, deterministic: bool, max_nodes: int | None) -> dict:
    from discopt.modeling.core import from_nl

    model = from_nl(nl)
    kw: dict = {"time_limit": time_limit, "gap_tolerance": 1e-4}
    if deterministic:
        kw["deterministic"] = True
    if max_nodes is not None:
        kw["max_nodes"] = max_nodes
    t0 = time.perf_counter()
    r = model.solve(**kw)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "nodes": int(r.node_count),
        "status": str(r.status),
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import discopt

    check(
        os.path.abspath(discopt.__file__).startswith(os.path.abspath("python/discopt")),
        f"discopt imported from {discopt.__file__}, not the worktree under test",
    )
    arms = Arms()

    records = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        nl = os.path.join(args.nl_dir, f"{name}.nl")
        check(os.path.exists(nl), f"missing instance {nl}")
        runs: dict[str, list[dict]] = {"old": [], "new": []}
        # One discarded warm-up. The first solve in a process pays lazy imports
        # and first-touch allocation that belong to neither arm: on ``alan`` it
        # read as 0.35 s against a 0.04 s steady state, i.e. a 0.21x "regression"
        # invented entirely by which arm happened to go first.
        arms.install("new")
        solve_once(nl, args.time_limit, args.deterministic, args.max_nodes)
        for rep in range(args.reps):
            # Alternate which arm goes first, so a warm-up or a drift in the box
            # cannot land on the same arm every time.
            order = ("new", "old") if rep % 2 == 0 else ("old", "new")
            for arm in order:
                arms.install(arm)
                arms.verify(arm)
                rec = solve_once(nl, args.time_limit, args.deterministic, args.max_nodes)
                rec["arm"] = arm
                rec["rep"] = rep
                runs[arm].append(rec)
                print(
                    f"  [{name}] rep{rep} {arm}: {rec['wall_s']:.2f}s "
                    f"{rec['nodes']} nodes obj={rec['objective']} bound={rec['bound']}",
                    flush=True,
                )
        arms.install("new")

        def med(arm, key, _runs=runs):
            return statistics.median(r[key] for r in _runs[arm])

        old_w, new_w = med("old", "wall_s"), med("new", "wall_s")
        nodes_match = {r["nodes"] for r in runs["old"]} == {r["nodes"] for r in runs["new"]}
        obj_match = {r["objective"] for r in runs["old"]} == {r["objective"] for r in runs["new"]}
        bound_match = {r["bound"] for r in runs["old"]} == {r["bound"] for r in runs["new"]}
        rec = {
            "instance": name,
            "median_wall_old_s": old_w,
            "median_wall_new_s": new_w,
            "speedup": old_w / new_w if new_w > 0 else None,
            "sd_old_s": statistics.stdev([r["wall_s"] for r in runs["old"]])
            if args.reps > 1
            else 0.0,
            "sd_new_s": statistics.stdev([r["wall_s"] for r in runs["new"]])
            if args.reps > 1
            else 0.0,
            "nodes_identical": nodes_match,
            "objective_identical": obj_match,
            "bound_identical": bound_match,
            "runs": runs,
        }
        records.append(rec)
        print(
            f"[{name}] old {old_w:.2f}s vs new {new_w:.2f}s -> {rec['speedup']:.3f}x  "
            f"nodes_identical={nodes_match} objective_identical={obj_match} "
            f"bound_identical={bound_match}",
            flush=True,
        )

    speedups = [r["speedup"] for r in records if r["speedup"]]
    neutral = all(
        r["nodes_identical"] and r["objective_identical"] and r["bound_identical"] for r in records
    )
    out = {
        "probe": "issue1180_callback_ab",
        "deterministic": args.deterministic,
        "max_nodes": args.max_nodes,
        "time_limit_s": args.time_limit,
        "reps": args.reps,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "all_neutral": neutral,
        "records": records,
        "executed_assertions": ASSERTS["n"],
    }
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(out, indent=1))
    print(f"\nmedian speedup (old/new): {out['median_speedup']}")
    print(f"neutral on every instance: {neutral}")
    print(f"executed assertions: {ASSERTS['n']}")
    if ASSERTS["n"] == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
