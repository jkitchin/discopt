#!/usr/bin/env python
"""Issue #1215 entry experiment: what does an arena-shaped construction path cost?

Answers the two questions the issue's "open question worth settling early"
poses -- **does a flat arena perform, and does it break compatibility** --
*before* any representation change is attempted (CLAUDE.md §4: run the
falsifying experiment first).

This is a **prototype, not a proposal.** It builds the same 40-form x 5,000
model as ``bench_model_construction.py`` through a flat arena (parallel lists
plus integer handles) so the arena's ceiling can be compared against the
measured object-per-node path before anyone invests in the change.

Four arms, each adding back one thing the real modeling layer must do, so the
cost of each is attributable rather than bundled:

  current        today's path: one unslotted Python object per node
  arena-raw      flat arena, integer handles, no names, no bookkeeping
  arena-named    + per-row names and the name-collision check
  arena-api      + a slotted handle OBJECT per node, so operator overloading
                 and ``x[i]*y[i] <= c`` work as public API

``arena-api`` is the arm that decides the question: discopt's public API *is*
operator overloading on objects, so a path that only works through explicit
function calls is not a candidate. ``arena-raw`` is the microbenchmark floor
the issue quotes (0.25 us) and is reported only to show how much of the gap is
representation versus API shape.

``--mode verify`` is the soundness gate: a fast arena that encodes a different
model is worthless, so it evaluates every prototype row against the current
path's compiled row on random points and requires agreement to 1e-12.

Measurement discipline: every arm counts the rows it built and the arena nodes
it emitted, and the arena arms are cross-checked against each other for an
identical node count; the script exits non-zero if any count is wrong or zero
(§6). It prints the module file it loaded (§8). Timing arms are interleaved
A/B/A/B in one process behind a load gate, with a standard deviation (§9).
Per-rep progress is unbuffered (§10).

Usage (from repo root, extension built, venv active)::

    python -u discopt_benchmarks/scripts/issue1215_arena_prototype.py
    python -u discopt_benchmarks/scripts/issue1215_arena_prototype.py --mode verify

Measured 2026-09-09, 4 cores, 1-min loadavg 0.33, `main` @ c052e85,
40 x 5,000 = 200,000 rows, 5 interleaved reps:

=============  ==========  ========  ========  =======  =========
arm            median s    us/row    RSS MB    B/row    speedup
=============  ==========  ========  ========  =======  =========
current             2.656     13.28     182.2      911       1.0x
arena-api           0.363      1.81      74.9      374       7.3x
arena-named         0.174      0.87      74.3      372      15.3x
arena-raw           0.105      0.52      51.3      257      25.3x
=============  ==========  ========  ========  =======  =========

For scale: Pyomo 6.10.1 is 6.67 us/row and oximo 0.6.0 is 0.66 us/row on the
same model. So an arena-backed construction path through discopt's *public*
operator API lands at ~1.8 us/row -- 3.7x faster than Pyomo, still 2.7x off
oximo. The ``arena-api`` -> ``arena-named`` step (1.81 -> 0.87) is the cost of
allocating one Python handle object per node, which operator overloading makes
unavoidable; that gap, not the representation, is what separates discopt from
oximo once the arena is in place.

Caveat on the 7.3x: the prototype does NOT do static shape inference, the
out-of-bounds guard, variable registration into the Rust builder, or the full
operator set, and it does not hash-cons (it emits 565,002 nodes where the Rust
arena interns the same model to 245,043). Restoring the guards will spend some
of the 7.3x back, so treat it as a ceiling, not a forecast.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import subprocess
import sys
import time

KINDS = ("bilinear", "exp", "square", "log")

# op codes
VAR, CONST, ADD, SUB, MUL, POW, EXP, LOG = range(8)


def _rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


# ─────────────────────────────────────────────────────────────
# arm 1 + 2: flat arena, integer handles
# ─────────────────────────────────────────────────────────────


class Arena:
    """Flat expression arena: parallel lists, integer handles.

    ``op[h]`` is the opcode, ``a[h]``/``b[h]`` the operand handles (-1 when
    unused), ``k[h]`` the literal payload (variable block index for VAR, the
    value for CONST). Rows are ``(body_handle, sense, rhs)``.
    """

    __slots__ = ("op", "a", "b", "k", "rows", "row_names", "_names", "_const")

    def __init__(self):
        self.op: list[int] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.k: list[float] = []
        self.rows: list[tuple[int, int, float]] = []
        self.row_names: list[str] = []
        self._names: set[str] = set()
        self._const: dict[float, int] = {}

    def node(self, op: int, a: int = -1, b: int = -1, k: float = 0.0) -> int:
        h = len(self.op)
        self.op.append(op)
        self.a.append(a)
        self.b.append(b)
        self.k.append(k)
        return h

    def const(self, v: float) -> int:
        """Interned constant -- the arena analogue of hash-consing a literal."""
        h = self._const.get(v)
        if h is None:
            h = self._const[v] = self.node(CONST, k=v)
        return h

    def row(self, body: int, sense: int, rhs: float) -> int:
        self.rows.append((body, sense, rhs))
        return len(self.rows) - 1

    def named_row(self, body: int, sense: int, rhs: float, name: str) -> int:
        if name in self._names:
            raise ValueError(f"duplicate constraint name {name!r}")
        self._names.add(name)
        self.row_names.append(name)
        return self.row(body, sense, rhs)


def build_arena(n_forms: int, n_inst: int, named: bool):
    ar = Arena()
    # Leaves pre-created once per position -- Pyomo's trick, and the thing
    # discopt currently does not do.
    xv = ar.node(VAR, k=0.0)
    yv = ar.node(VAR, k=1.0)
    xs = [ar.node(ADD, xv, ar.const(float(i))) for i in range(n_inst)]  # x[i] leaf
    ys = [ar.node(ADD, yv, ar.const(float(i))) for i in range(n_inst)]  # y[i] leaf

    built = 0
    for f in range(n_forms):
        kind = KINDS[f % 4]
        c = ar.const(4.0 + f)
        nm = f"c{f}"
        for i in range(n_inst):
            xi, yi = xs[i], ys[i]
            if kind == "bilinear":
                t = ar.node(MUL, xi, yi)
            elif kind == "exp":
                t = ar.node(ADD, ar.node(EXP, xi), yi)
            elif kind == "square":
                t = ar.node(ADD, ar.node(POW, xi, ar.const(2.0)), yi)
            else:
                t = ar.node(ADD, ar.node(LOG, xi), yi)
            body = ar.node(SUB, t, c)
            if named:
                ar.named_row(body, 0, 0.0, f"{nm}[{i}]")
            else:
                ar.row(body, 0, 0.0)
            built += 1
    return ar, built, len(ar.op)


# ─────────────────────────────────────────────────────────────
# arm 3: same arena, but a slotted handle OBJECT per node so the
# public operator-overloading API still works
# ─────────────────────────────────────────────────────────────


class H:
    """Handle: a slotted proxy for one arena node. 2 slots, no __dict__.

    This is the smallest object that can carry discopt's public API
    (``x[i] * y[i] <= c``) while keeping the node payload in flat arrays.
    """

    __slots__ = ("ar", "h")

    def __init__(self, ar: Arena, h: int):
        self.ar = ar
        self.h = h

    def _w(self, other):
        return other if type(other) is H else H(self.ar, self.ar.const(other))

    def __add__(self, other):
        o = self._w(other)
        return H(self.ar, self.ar.node(ADD, self.h, o.h))

    def __sub__(self, other):
        o = self._w(other)
        return H(self.ar, self.ar.node(SUB, self.h, o.h))

    def __mul__(self, other):
        o = self._w(other)
        return H(self.ar, self.ar.node(MUL, self.h, o.h))

    def __pow__(self, other):
        o = self._w(other)
        return H(self.ar, self.ar.node(POW, self.h, o.h))

    def __le__(self, other):
        # Same normalization discopt uses: body - rhs, rhs == 0.
        o = self._w(other)
        return Row(self.ar, self.ar.node(SUB, self.h, o.h), 0, 0.0)


class Row:
    """A constraint row -- the arena analogue of ``Constraint``."""

    __slots__ = ("ar", "body", "sense", "rhs", "name")

    def __init__(self, ar, body, sense, rhs):
        self.ar = ar
        self.body = body
        self.sense = sense
        self.rhs = rhs
        self.name = None


def h_exp(x: H) -> H:
    return H(x.ar, x.ar.node(EXP, x.h))


def h_log(x: H) -> H:
    return H(x.ar, x.ar.node(LOG, x.h))


def build_arena_api(n_forms: int, n_inst: int):
    """Arena arm through the *public* operator-overloading shape, plus names."""
    ar = Arena()
    xv = ar.node(VAR, k=0.0)
    yv = ar.node(VAR, k=1.0)
    # IndexedVar with cached leaf handles (candidate #1 in the issue).
    xs = [H(ar, ar.node(ADD, xv, ar.const(float(i)))) for i in range(n_inst)]
    ys = [H(ar, ar.node(ADD, yv, ar.const(float(i)))) for i in range(n_inst)]

    built = 0
    for f in range(n_forms):
        kind = KINDS[f % 4]
        c = 4.0 + f
        nm = f"c{f}"
        if kind == "bilinear":

            def rule(i, xs=xs, ys=ys, c=c):
                return xs[i] * ys[i] <= c

        elif kind == "exp":

            def rule(i, xs=xs, ys=ys, c=c):
                return h_exp(xs[i]) + ys[i] <= c

        elif kind == "square":

            def rule(i, xs=xs, ys=ys, c=c):
                return xs[i] ** 2 + ys[i] <= c

        else:

            def rule(i, xs=xs, ys=ys, c=c):
                return h_log(xs[i]) + ys[i] <= c

        for i in range(n_inst):
            r = rule(i)
            ar.named_row(r.body, r.sense, r.rhs, f"{nm}[{i}]")
            built += 1
    return ar, built, len(ar.op)


# ─────────────────────────────────────────────────────────────
# current discopt path, for the same-process interleaved control
# ─────────────────────────────────────────────────────────────


def build_current(n_forms: int, n_inst: int):
    import discopt.modeling as dm
    from discopt.modeling import Model

    m = Model()
    idx = m.set("I", list(range(n_inst)))
    x = m.continuous("x", over=idx, lb=0.5, ub=4.0)
    y = m.continuous("y", over=idx, lb=0.5, ub=4.0)
    built = 0
    for f in range(n_forms):
        kind = KINDS[f % 4]
        c = 4.0 + f
        if kind == "bilinear":
            rule = lambda i, x=x, y=y, c=c: x[i] * y[i] <= c  # noqa: E731
        elif kind == "exp":
            rule = lambda i, x=x, y=y, c=c: dm.exp(x[i]) + y[i] <= c  # noqa: E731
        elif kind == "square":
            rule = lambda i, x=x, y=y, c=c: x[i] ** 2 + y[i] <= c  # noqa: E731
        else:
            rule = lambda i, x=x, y=y, c=c: dm.log(x[i]) + y[i] <= c  # noqa: E731
        built += len(m.constraint(idx, rule, name=f"c{f}"))
    return m, built, None


ARMS = {
    "current": build_current,
    "arena-raw": lambda f, n: build_arena(f, n, named=False),
    "arena-named": lambda f, n: build_arena(f, n, named=True),
    "arena-api": build_arena_api,
}


def _memory_child(arm: str, n_forms: int, n_inst: int) -> None:
    import json

    fn = ARMS[arm]
    if arm == "current":
        import discopt.modeling  # noqa: F401
    gc.collect()
    base = _rss_mb()
    obj, built, n_nodes = fn(n_forms, n_inst)
    gc.collect()
    retained = _rss_mb() - base
    assert obj is not None
    print(json.dumps({"arm": arm, "built": built, "nodes": n_nodes, "retained_mb": retained}))


# ─────────────────────────────────────────────────────────────
# mode: verify -- does the prototype encode the SAME model?
# ─────────────────────────────────────────────────────────────


def _eval_arena(ar, h, xv, yv):
    """Evaluate arena node *h* given the two variable value vectors."""
    op = ar.op[h]
    if op == CONST:
        return ar.k[h]
    if op == VAR:
        raise AssertionError("bare VAR reached: leaves are VAR+CONST offsets")
    if op == ADD:
        # x[i] leaves are encoded as ADD(VAR, CONST(i)) -- recognize and resolve.
        la, rb = ar.a[h], ar.b[h]
        if ar.op[la] == VAR:
            vec = xv if ar.k[la] == 0.0 else yv
            return vec[int(ar.k[rb])]
        return _eval_arena(ar, la, xv, yv) + _eval_arena(ar, rb, xv, yv)
    left = _eval_arena(ar, ar.a[h], xv, yv)
    if op == EXP:
        return math.exp(left)
    if op == LOG:
        return math.log(left)
    right = _eval_arena(ar, ar.b[h], xv, yv)
    if op == SUB:
        return left - right
    if op == MUL:
        return left * right
    if op == POW:
        return left**right
    raise AssertionError(f"unhandled opcode {op}")


def mode_verify(n_forms: int, n_inst: int, n_pts: int) -> int:
    """Every prototype row must agree with the current path's compiled row."""
    import discopt.modeling.core as core
    import numpy as np
    from discopt._relax.dag_compiler import compile_constraint

    print(f"# discopt loaded from: {core.__file__}")
    m, built_cur, _ = build_current(n_forms, n_inst)
    ar, built_ar, _ = build_arena_api(n_forms, n_inst)
    n_total = n_forms * n_inst
    if not (built_cur == built_ar == n_total):
        print(f"FAIL: built {built_cur} / {built_ar} rows, expected {n_total}")
        return 1

    cons = list(m._constraints)
    print(f"# {len(cons)} Constraint objects, {len(ar.rows)} arena rows")
    if len(cons) != len(ar.rows):
        print(f"FAIL: row-count mismatch {len(cons)} vs {len(ar.rows)}")
        return 1

    rng = np.random.default_rng(0)
    compiled = [compile_constraint(c, m) for c in cons]

    n_cmp = 0
    max_err = 0.0
    for _ in range(n_pts):
        xv = rng.uniform(0.6, 3.9, size=n_inst)
        yv = rng.uniform(0.6, 3.9, size=n_inst)
        xfull = np.concatenate([xv, yv])  # x block then y block
        for r, (fn, (body, sense, rhs)) in enumerate(zip(compiled, ar.rows, strict=True)):
            cur = float(fn(xfull))
            new = _eval_arena(ar, body, xv, yv)
            err = abs(cur - new)
            max_err = max(max_err, err)
            n_cmp += 1
            if err > 1e-12:
                print(f"FAIL: row {r} disagrees: current {cur!r} vs arena {new!r} (err {err:.3e})")
                return 1
            if sense != 0 or rhs != 0.0:
                print(f"FAIL: row {r} sense/rhs {sense}/{rhs}, expected 0/0.0")
                return 1

    print(f"# executed comparisons: {n_cmp}, max abs err {max_err:.3e}")
    if n_cmp == 0:
        print("FAIL: probe compared nothing")
        return 1
    print("PASS: arena prototype encodes the same model as the current path")
    return 0


def main() -> int:
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", type=int, default=40)
    ap.add_argument("--instances", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="current,arena-api,arena-named,arena-raw")
    ap.add_argument("--mode", choices=("bench", "verify"), default="bench")
    ap.add_argument("--points", type=int, default=5, help="verify mode: random points")
    ap.add_argument("--memory-child", default=None)
    args = ap.parse_args()

    if args.memory_child:
        _memory_child(args.memory_child, args.forms, args.instances)
        return 0

    if args.mode == "verify":
        return mode_verify(args.forms, args.instances, args.points)

    import discopt.modeling.core as core

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_total = args.forms * args.instances
    print(f"# discopt loaded from: {core.__file__}")
    print(f"# load gate: 1-min loadavg = {os.getloadavg()[0]:.2f}, cpus = {os.cpu_count()}")
    print(
        f"# model: {args.forms} forms x {args.instances} = {n_total} rows, "
        f"{args.reps} interleaved reps",
        flush=True,
    )

    times: dict[str, list[float]] = {a: [] for a in arms}
    counts: dict[str, set[int]] = {a: set() for a in arms}
    nodes: dict[str, set] = {a: set() for a in arms}
    for rep in range(args.reps):
        for a in arms:
            gc.collect()
            t0 = time.perf_counter()
            obj, built, n_nodes = ARMS[a](args.forms, args.instances)
            dt = time.perf_counter() - t0
            times[a].append(dt)
            counts[a].add(built)
            nodes[a].add(n_nodes)
            del obj
            gc.collect()
            print(
                f"  rep {rep + 1}/{args.reps} {a:12s} {dt:7.3f} s  ({built} rows"
                + (f", {n_nodes} nodes)" if n_nodes else ")"),
                flush=True,
            )

    mem: dict[str, float] = {}
    for a in arms:
        out = subprocess.run(
            [
                sys.executable,
                "-u",
                os.path.abspath(__file__),
                "--memory-child",
                a,
                "--forms",
                str(args.forms),
                "--instances",
                str(args.instances),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        mem[a] = rec["retained_mb"]
        counts[a].add(rec["built"])
        print(f"  memory {a:12s} {rec['retained_mb']:7.1f} MB retained", flush=True)

    executed = 0
    for a in arms:
        if len(counts[a]) != 1 or counts[a].copy().pop() != n_total:
            print(f"FAIL: arm {a} built {sorted(counts[a])} rows, expected {n_total}")
            return 1
        executed += n_total * (args.reps + 1)
    # cross-check: every arena arm must emit the same node count
    arena_nodes = {a: sorted(nodes[a])[0] for a in arms if a.startswith("arena")}
    if len(set(arena_nodes.values())) > 1:
        print(f"FAIL: arena arms disagree on node count: {arena_nodes}")
        return 1
    print(f"# executed: {executed} rows built; arena node count {arena_nodes}")
    if executed == 0:
        print("FAIL: probe built nothing")
        return 1

    print()
    base_med = statistics.median(times["current"]) if "current" in times else None
    print(
        f"{'arm':14s} {'median s':>10s} {'sd':>7s} {'us/row':>8s} {'RSS MB':>8s} "
        f"{'B/row':>7s} {'speedup':>8s}"
    )
    for a in arms:
        ts = times[a]
        med = statistics.median(ts)
        sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
        sp = f"{base_med / med:.1f}x" if base_med else "-"
        print(
            f"{a:14s} {med:10.3f} {sd:7.3f} {med / n_total * 1e6:8.2f} "
            f"{mem[a]:8.1f} {mem[a] * 1e6 / n_total:7.0f} {sp:>8s}"
        )
    print("\n# reference: oximo 0.66 us/row, ~205 B/row (Rust, flat arena)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
