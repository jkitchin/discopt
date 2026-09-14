"""Entry experiment for the trace-once template idea (#1215 / plan §40 direction 1).

Two gates, both measured, neither assumed:

  GATE 1 (reachability): what absolute wall does per-element construction cost
      at the row counts that actually occur?  If a real model's whole
      construction is ~1 ms, no construction lever is worth building.

  GATE 2 (eligibility): of the families that ARE built through the per-element
      rule path, what fraction could a trace-once template legally handle?
      A template traces the rule ONCE against a symbolic index and instantiates
      N times below the Python line.  That is only sound when every member
      produces a structurally identical DAG.  Any index-dependent branching,
      any `Skip`, any varying operator mix makes the family ineligible.

This module is BOTH a pytest plugin (collects families from a real workload)
and a standalone script (--mode size for gate 1, --mode report for gate 2).

Per CLAUDE.md §6 every mode prints an executed-assertion / comparison count and
exits non-zero when it is zero.  Per §7 nothing here swallows an exception.
Per §8 the loaded discopt is asserted before anything is measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import Counter

RECORDS: list[dict] = []
_OUT = os.environ.get("TEMPLATE_PROBE_OUT", "")

#: Marker asserted by the §8 load gate of anything importing this module.
PROBE_VERSION = "issue1215-template-entry-v1"


# --------------------------------------------------------------------------
# §8: prove which code is loaded before measuring anything.
# --------------------------------------------------------------------------
def assert_loaded_discopt():
    import discopt
    import discopt.modeling.core as core

    path = os.path.realpath(core.__file__)
    # The marker: this tree's `Model.constraint` carries the lazy-name change
    # from plan §63.  Its absence means a different (older) discopt is loaded.
    src_marker = "_name_family"
    with open(path) as fh:
        src = fh.read()
    if src_marker not in src:
        raise SystemExit(
            f"LOADED THE WRONG TREE: {path} lacks marker {src_marker!r}. Refusing to measure."
        )
    return discopt.__file__, path


# --------------------------------------------------------------------------
# Structural fingerprint of a constraint body.
#
# A trace-once template binds per-row LEAVES (which variable position, which
# constant value); it cannot vary the SHAPE of the DAG or the operators in it.
# So abstract leaves, keep structure.  Two rows with the same fingerprint are
# template-compatible; two with different fingerprints are not.
#
# This is deliberately the UPPER BOUND on eligibility: it abstracts constant
# values and index positions entirely, so it can only over-count eligible
# families, never under-count them.  If even this bound is low, the idea dies.
# --------------------------------------------------------------------------
def fingerprint(node) -> str:
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        CustomCall,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    out: list[str] = []
    # Iterative walk: bodies can be thousands deep (plan §36: `sum()` is the trap)
    # and a recursive walker would hit the recursion limit rather than measure.
    stack: list = [node]
    while stack:
        n = stack.pop()
        if type(n) is str:  # a literal token pushed back for ordering
            out.append(n)
            continue
        if isinstance(n, Constant):
            # Value abstracted: a template can carry a per-row constant array.
            out.append("C")
        elif isinstance(n, Parameter):
            out.append(f"P:{n.name}")
        elif isinstance(n, Variable):
            # Identity NOT abstracted: rows of one family must reference the same
            # declared variable for a template to bind a position per row.
            out.append(f"V:{n.name}")
        elif isinstance(n, IndexExpression):
            # Index abstracted: that is exactly the axis a template varies.
            out.append("I(")
            stack.append(")")
            stack.append(n.base)
        elif isinstance(n, BinaryOp):
            out.append(f"B:{n.op}(")
            stack.append(")")
            stack.append(n.right)
            stack.append(",")
            stack.append(n.left)
        elif isinstance(n, MatMulExpression):
            out.append("B:@(")
            stack.append(")")
            stack.append(n.right)
            stack.append(",")
            stack.append(n.left)
        elif isinstance(n, UnaryOp):
            out.append(f"U:{n.op}(")
            stack.append(")")
            stack.append(n.operand)
        elif isinstance(n, FunctionCall):
            out.append(f"F:{n.func_name}(")
            stack.append(")")
            for a in reversed(n.args):
                stack.append(a)
        elif isinstance(n, CustomCall):
            out.append(f"X:{n.name}(")
            stack.append(")")
            for a in reversed(n.args):
                stack.append(a)
        elif isinstance(n, SumExpression):
            out.append(f"S:axis={n.axis}(")
            stack.append(")")
            stack.append(n.operand)
        elif isinstance(n, SumOverExpression):
            # The TERM COUNT is part of the shape: a template instantiates one
            # traced body N times, so two rows summing different numbers of terms
            # are not the same template.  This is the `sum(... for j in J)`
            # pattern, where J may depend on the outer index.
            out.append(f"SO:{len(n.terms)}(")
            stack.append(")")
            for t in reversed(n.terms):
                stack.append(t)
        else:
            # §7: do not swallow.  An unknown node type means the fingerprint is
            # not covering the DAG and every "eligible" verdict is suspect.
            raise TypeError(
                f"fingerprint: unhandled node type {type(n).__name__}. "
                "The probe does not cover the DAG; refusing to report."
            )
    return "".join(out)


# --------------------------------------------------------------------------
# The instrument: wrap Model.constraint, record every family built.
# --------------------------------------------------------------------------
def install(origin: str = "workload"):
    from discopt.modeling.core import Model

    if getattr(Model.constraint, "_template_probe", False):
        return

    original = Model.constraint

    def probed(self, index_set, rule, name=None, fast=True):
        t0 = time.perf_counter()
        result = original(self, index_set, rule, name=name, fast=fast)
        elapsed = time.perf_counter() - t0

        # Fingerprint the rows the family ACTUALLY produced, read off the
        # returned IndexedConstraint.  An earlier version re-ran the rule per
        # member; that perturbs any rule with a side effect and is not what the
        # model contains.  `_members` holds the real Constraint objects (fast
        # path included -- they are kept for introspection).
        prints: Counter = Counter()
        senses: set = set()
        n_members = sum(1 for _ in index_set)
        rows = result._members
        for c in rows.values():
            prints[fingerprint(c.body)] += 1
            senses.add(c.sense)
        n_rows = len(rows)

        RECORDS.append(
            {
                "origin": origin,
                "name": name or "<unnamed>",
                "n_members": n_members,
                "n_rows": n_rows,
                "n_skip": n_members - n_rows,
                "n_shapes": len(prints),
                "n_senses": len(senses),
                "wall_s": elapsed,
                "fast": bool(getattr(result, "fast", False)),
                "eligible": bool(
                    n_rows == n_members and len(prints) == 1 and len(senses) == 1 and n_rows > 0
                ),
            }
        )
        return result

    probed._template_probe = True  # type: ignore[attr-defined]
    Model.constraint = probed  # type: ignore[assignment]


def dump():
    if not _OUT:
        return
    with open(_OUT, "w") as fh:
        for r in RECORDS:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------
# pytest plugin hooks
# --------------------------------------------------------------------------
def pytest_configure(config):
    assert_loaded_discopt()
    install(origin="pytest")


def pytest_unconfigure(config):
    dump()
    sys.stderr.write(f"\n[template-probe] recorded {len(RECORDS)} families\n")


# --------------------------------------------------------------------------
# GATE 1: absolute construction wall at real corpus row counts
# --------------------------------------------------------------------------
def mode_size(args):
    import glob

    import discopt.modeling as dm
    from discopt.modeling.core import Model

    print(f"# discopt: {assert_loaded_discopt()[1]}")
    print(f"# load at start: {os.getloadavg()[0]:.2f}")

    # --- the in-repo .nl corpus row-count distribution ---
    rows = []
    files = sorted(glob.glob(args.corpus))
    for path in files:
        # The .nl header is ASCII even for binary-format ("b") files, so read
        # bytes and decode latin-1: several corpus instances are binary and a
        # utf-8 read raises on them.
        n_con = None
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("latin-1")
        lines = head.splitlines()
        if len(lines) < 2:
            raise SystemExit(f"{path}: header shorter than 2 lines")
        n_con = int(lines[1].split()[1])  # line 2: nvar ncon nobj ...
        if n_con is None:
            raise SystemExit(f"could not read row count from {path}")
        rows.append(n_con)
    if not rows:
        raise SystemExit(f"GATE 1 read ZERO instances from {args.corpus!r}")
    rows.sort()

    def pct(p):
        return rows[min(len(rows) - 1, int(round(p / 100 * (len(rows) - 1))))]

    print(f"\n## corpus row counts ({len(rows)} instances from {args.corpus})")
    print(
        f"min={rows[0]} p25={pct(25)} p50={pct(50)} p75={pct(75)} "
        f"p90={pct(90)} p99={pct(99)} max={rows[-1]} total={sum(rows)}"
    )

    # --- absolute wall of per-element construction at those sizes ---
    # Same idiom and same body mix as the standing benchmark's `build_discopt`
    # (discopt_benchmarks/scripts/bench_model_construction.py) so the numbers are
    # comparable to plan §63's, just at corpus sizes instead of 200 000 rows.
    sizes = sorted({max(1, pct(p)) for p in (25, 50, 75, 90, 99)} | {rows[-1]})
    print("\n## per-element construction (m.constraint + rule), absolute wall")
    print("# body mix: bilinear / exp / square / log -- nonlinear, no fast path")
    print(f"{'rows':>8} {'wall_ms':>10} {'sd_ms':>8} {'us/row':>9}")
    n_timed = 0
    for n in sizes:
        samples = []
        for _ in range(args.reps):
            gc.collect()
            m = Model()
            idx = m.set("I", list(range(n)))
            x = m.continuous("x", over=idx, lb=0.5, ub=4.0)
            y = m.continuous("y", over=idx, lb=0.5, ub=4.0)
            t0 = time.perf_counter()
            for f, kind in enumerate(("bilinear", "exp", "square", "log")):
                c = 4.0 + f
                if kind == "bilinear":

                    def rule(i, x=x, y=y, c=c):
                        return x[i] * y[i] <= c
                elif kind == "exp":

                    def rule(i, x=x, y=y, c=c):
                        return dm.exp(x[i]) + y[i] <= c
                elif kind == "square":

                    def rule(i, x=x, y=y, c=c):
                        return x[i] ** 2 + y[i] <= c
                else:

                    def rule(i, x=x, y=y, c=c):
                        return dm.log(x[i]) + y[i] <= c

                m.constraint(idx, rule, name=f"c{f}")
            samples.append(time.perf_counter() - t0)
            n_timed += 1
        best = min(samples)
        sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
        # 4 families of n rows each
        print(f"{n * 4:>8} {best * 1e3:>10.3f} {sd * 1e3:>8.3f} {best / (n * 4) * 1e6:>9.2f}")

    if n_timed == 0:
        raise SystemExit("GATE 1 executed ZERO timings")
    print(f"\n[gate1] executed timings: {n_timed}")
    return 0


# --------------------------------------------------------------------------
# GATE 2 report
# --------------------------------------------------------------------------
def mode_report(args):
    recs = []
    for path in args.inputs:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"GATE 2 loaded ZERO family records from {args.inputs}")

    total_wall = sum(r["wall_s"] for r in recs)
    total_rows = sum(r["n_rows"] for r in recs)
    elig = [r for r in recs if r["eligible"]]
    elig_wall = sum(r["wall_s"] for r in elig)
    elig_rows = sum(r["n_rows"] for r in elig)

    sizes = sorted(r["n_members"] for r in recs)

    def pct(p):
        return sizes[min(len(sizes) - 1, int(round(p / 100 * (len(sizes) - 1))))]

    print(f"## GATE 2 -- {len(recs)} rule-based families observed")
    print(f"family size: min={sizes[0]} p50={pct(50)} p90={pct(90)} max={sizes[-1]}")
    print(f"total rows built through the rule path: {total_rows}")
    print(f"total wall in Model.constraint:         {total_wall * 1e3:.2f} ms")
    print()
    print(f"eligible families: {len(elig)}/{len(recs)} ({100 * len(elig) / len(recs):.1f}%)")
    print(
        f"eligible by rows:  {elig_rows}/{total_rows} ({100 * elig_rows / max(1, total_rows):.1f}%)"
    )
    print(
        f"eligible by wall:  {elig_wall * 1e3:.2f}/{total_wall * 1e3:.2f} ms "
        f"({100 * elig_wall / max(1e-12, total_wall):.1f}%)"
    )
    print()
    reasons = Counter()
    for r in recs:
        if r["eligible"]:
            continue
        if r["n_members"] == 0:
            reasons["empty family"] += 1
        if r["n_skip"]:
            reasons["uses Skip"] += 1
        if r["n_shapes"] > 1:
            reasons["non-uniform DAG shape"] += 1
        if r["n_senses"] > 1:
            reasons["mixed sense"] += 1
    print("## why ineligible (a family can hit more than one)")
    for k, v in reasons.most_common():
        print(f"  {k:28} {v}")

    print("\n## the 10 largest families by wall")
    print(f"{'wall_ms':>9} {'rows':>7} {'shapes':>7} {'skip':>5}  name")
    for r in sorted(recs, key=lambda r: -r["wall_s"])[:10]:
        flag = "OK " if r["eligible"] else "NO "
        print(
            f"{r['wall_s'] * 1e3:>9.3f} {r['n_rows']:>7} {r['n_shapes']:>7} "
            f"{r['n_skip']:>5}  {flag}{r['name']}"
        )

    print(f"\n[gate2] families compared: {len(recs)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("size", "report"), required=True)
    ap.add_argument("--corpus", default="python/tests/data/minlplib_nl/*.nl")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inputs", nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.mode == "size":
        return mode_size(args)
    return mode_report(args)


if __name__ == "__main__":
    sys.exit(main())
