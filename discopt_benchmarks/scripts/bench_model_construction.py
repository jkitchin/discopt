#!/usr/bin/env python
"""Model-construction benchmark: discopt vs Pyomo (issue #1215).

The standing instrument for **construction cost** in the Python modeling
layer -- building a :class:`~discopt.modeling.Model`, before any solve. This
is the layer that had no external baseline, which is why the gap went
unmeasured; Pyomo is the recorded baseline here so a regression or an
improvement is visible against something other than discopt's own history.

Three modes, all over the same model family::

    --mode headline     build wall + retained RSS, discopt vs Pyomo   (default)
    --mode attribution  where the wall goes, stage by stage, and the
                        per-op gap on the two hottest operations
    --mode alloc        retained allocation by source line (tracemalloc)

The model is ``--forms`` distinct equation forms x ``--instances`` set
members, cycling through four nonlinear kinds (bilinear / ``exp`` / square /
``log``). The default 40 x 5,000 = 200,000 constraint instances is the size
issue #1215 and ``docs/dev/performance-plan.md`` §30 report against.

Method, per the measurement discipline in CLAUDE.md:

* §6  every mode counts what it actually built and exits non-zero if that
  count is zero or disagrees with the requested model size.
* §8  each arm prints the module file it loaded before measuring.
* §9  a load gate (1-minute loadavg) is reported and a warning is printed
  when the machine is busy; the timing arms are interleaved A/B/A/B across
  reps in one process, never run sequentially; a standard deviation is
  reported alongside every mean.
* §10 per-rep progress is printed unbuffered as it happens.

Retained RSS is measured in a **fresh subprocess per arm** -- retained RSS is
not recoverable in a process that has already built the other arm's model --
as (RSS after build and ``gc.collect()``) minus (RSS after imports, before
build).

Usage (from repo root, extension built, venv active)::

    python -u discopt_benchmarks/scripts/bench_model_construction.py
    python -u discopt_benchmarks/scripts/bench_model_construction.py --mode attribution
    python -u discopt_benchmarks/scripts/bench_model_construction.py --mode alloc

    # quick smoke
    python -u discopt_benchmarks/scripts/bench_model_construction.py \
        --forms 4 --instances 200 --reps 2

**The recorded numbers below are container-relative: compare arms WITHIN one
run, never across sessions.** The absolute discopt figure drifted 3.225 s ->
3.776 s over a few hours in one container, on *identical* code -- verified by
checking `python/discopt/` back out to the baseline commit (marker
`_additive_chain` count 0 vs 3) and re-running: 3.776 s for the old Python
against 3.823 s for the current, a 1.2% difference inside the spread, while the
gap to the session's own opening measurement was 17%. Retained RSS, by contrast,
is deterministic (183.4 -> 182.2 MB here, Pyomo identical to the byte at 94.3),
so **memory is the reliable cross-session control and wall time is not**. Reading
the drift as a regression is the trap: it nearly cost this instrument's author a
published 18% slowdown that did not exist. When a wall-time change matters,
bisect it in one container with a marker assertion on the loaded code, and treat
an unchanged memory figure as evidence that nothing structural moved.

**Two discopt arms, and they are not interchangeable.** ``discopt`` builds the
model per-element (one ``Constraint`` object per row) -- the idiom Pyomo and
oximo are limited to, since neither has an array-valued constraint body.
``discopt-vec`` builds the same model with one array-valued body per form, which
is discopt's own idiom. Reporting only the first published the slower of the two
as "discopt's construction cost"; both are default-on now, and a claim about
"discopt" has to say which it means. See :func:`build_discopt_vectorised`.

Opening baseline, 40 x 5,000 = 200,000 instances, 5 reps, 4 cores, 1-minute
loadavg 0.70, `main` @ c052e85, Pyomo 6.10.1 -- the state issue #1215 was opened
against:

=========== ========== ========== ========= ========
arm         median s   us/inst    RSS MB    B/inst
=========== ========== ========== ========= ========
discopt     3.225      16.12      183.4     917
pyomo       1.334       6.67       94.3     471
=========== ========== ========== ========= ========

-- discopt 2.42x slower and 1.95x heavier at construction.

Same model, same machine, 2026-09-10, loadavg 0.24, with the vectorised arm
added:

=========== ========== ========== ========= ========
arm         median s   us/inst    RSS MB    B/inst
=========== ========== ========== ========= ========
discopt     0.776       3.88       93.6     468
discopt-vec 0.0006      0.00304     0.14      0.696
pyomo       1.299       6.49       94.3     471
pyomo-gc    0.669       3.34       94.3     471
=========== ========== ========== ========= ========

-- per-element discopt is now 1.16x pyomo-gc (the like-for-like arm) and at
memory parity; the vectorised arm is **1275x** cheaper still and retains
essentially nothing.

**What the vectorised arm's near-zero does and does not mean.** An array-valued
body defers its per-row work to the writer or the solve, so 0.003 us/inst is a
real measurement of *construction* and a partial measurement of anything else.
The end-to-end figure (model to written ``.nl``) is 3.05-4.42 us/row against
Pyomo's 35.5-48.1 and oximo's 2.97-4.11 -- see
``discopt_benchmarks/scripts/issue1215_cross_tool_panel.py`` and
``docs/dev/performance-plan.md`` §48. Quote that one for end-to-end claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import linecache
import os
import statistics
import subprocess
import sys
import time
from collections import Counter

#: The four nonlinear equation kinds the forms cycle through.
KINDS = ("bilinear", "exp", "square", "log")


# ─────────────────────────────────────────────────────────────
# discopt arm
# ─────────────────────────────────────────────────────────────


def build_discopt(n_forms: int, n_inst: int):
    """Build the model in discopt; return ``(model, instances built)``."""
    import discopt.modeling as dm
    from discopt.modeling import Model

    m = Model()
    idx = m.set("I", list(range(n_inst)))
    x = m.continuous("x", over=idx, lb=0.5, ub=4.0)
    y = m.continuous("y", over=idx, lb=0.5, ub=4.0)

    built = 0
    for f in range(n_forms):
        kind = KINDS[f % len(KINDS)]
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

        built += len(m.constraint(idx, rule, name=f"c{f}"))
    return m, built


def build_discopt_vectorised(n_forms: int, n_inst: int):
    """The SAME model in discopt's own idiom: one array-valued body per form.

    ``build_discopt`` above is the *per-element* idiom -- one ``Constraint``
    object per row -- which is what Pyomo and oximo are limited to, because
    neither has an array-valued constraint body. discopt does, and the cross-tool
    panel (`docs/dev/performance-plan.md` §48) measured the difference between
    discopt's two idioms at **6.9-11.1x**: larger than the whole discopt-to-Pyomo
    gap, and enough to put discopt at oximo parity (0.91-1.27x).

    So the per-element arm alone is not "discopt's construction cost"; it is the
    cost of the idiom discopt shares with the tools it is being compared to.
    Reporting only that arm published the slower of two numbers as our result.
    Both arms are here now, and a claim about "discopt" must say which.
    """
    import discopt.modeling as dm
    from discopt.modeling import Model

    m = Model()
    x = m.continuous("x", shape=(n_inst,), lb=0.5, ub=4.0)
    y = m.continuous("y", shape=(n_inst,), lb=0.5, ub=4.0)

    built = 0
    for f in range(n_forms):
        kind = KINDS[f % len(KINDS)]
        c = 4.0 + f
        if kind == "bilinear":
            body = x * y
        elif kind == "exp":
            body = dm.exp(x) + y
        elif kind == "square":
            body = x**2 + y
        else:
            body = dm.log(x) + y
        m.subject_to(body <= c, name=f"c{f}")
        # §6: count the rows the body actually carries, not the rows intended.
        # A body that silently collapsed to a scalar would otherwise be counted
        # as `n_inst` and the arm would report a 200,000-row build of one row.
        shape = getattr(m._constraints[-1].body, "shape", ())
        rows = 1
        for dim in shape:
            rows *= dim
        built += rows
    return m, built


def discopt_marker() -> str:
    """§8: the file the discopt arm actually loaded."""
    import discopt.modeling.core as core

    return core.__file__


# ─────────────────────────────────────────────────────────────
# Pyomo arm (the recorded external baseline)
# ─────────────────────────────────────────────────────────────


def build_pyomo(n_forms: int, n_inst: int):
    """Build the same model in Pyomo; return ``(model, instances built)``."""
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n_inst - 1)
    m.x = pyo.Var(m.I, bounds=(0.5, 4.0))
    m.y = pyo.Var(m.I, bounds=(0.5, 4.0))

    built = 0
    for f in range(n_forms):
        kind = KINDS[f % len(KINDS)]
        c = 4.0 + f
        if kind == "bilinear":

            def rule(mm, i, c=c):
                return mm.x[i] * mm.y[i] <= c

        elif kind == "exp":

            def rule(mm, i, c=c):
                return pyo.exp(mm.x[i]) + mm.y[i] <= c

        elif kind == "square":

            def rule(mm, i, c=c):
                return mm.x[i] ** 2 + mm.y[i] <= c

        else:

            def rule(mm, i, c=c):
                return pyo.log(mm.x[i]) + mm.y[i] <= c

        con = pyo.Constraint(m.I, rule=rule)
        setattr(m, f"c{f}", con)
        built += len(con)
    return m, built


def pyomo_marker() -> str:
    """§8: the tree the Pyomo arm actually loaded."""
    import pyomo.environ as pyo

    return os.path.dirname(pyo.__file__)


def build_pyomo_gc_tuned(n_forms: int, n_inst: int):
    """Pyomo, under the SAME GC treatment discopt now applies to itself.

    **This arm exists so the headline cannot lie.** discopt raises CPython's
    gen-0 GC threshold inside ``Model.constraint`` (see
    ``modeling.bulk_construction_gc``), and GC turned out to be ~40% of a
    200 000-row build's wall. Pyomo exposes the same idea as ``PauseGC`` but
    leaves it to the caller, so comparing tuned-discopt against untuned-Pyomo
    measures *whose library does it for you*, not which modelling layer is
    cheaper -- and the two answers differ by a factor of 1.6.

    Both are worth reporting and they mean different things:

    * ``discopt`` vs ``pyomo``    -- what a user gets writing the natural code;
    * ``discopt`` vs ``pyomo-gc`` -- like-for-like cost of the two layers.

    Never quote the first as though it were the second.
    """
    import gc

    old = gc.get_threshold()
    gc.set_threshold(old[0] * 100, old[1], old[2])
    try:
        return build_pyomo(n_forms, n_inst)
    finally:
        gc.set_threshold(*old)


ARMS = {
    "discopt": (build_discopt, discopt_marker),
    "discopt-vec": (build_discopt_vectorised, discopt_marker),
    "pyomo": (build_pyomo, pyomo_marker),
    "pyomo-gc": (build_pyomo_gc_tuned, pyomo_marker),
}

#: Arms run when ``--arms`` is not given. Both discopt idioms are in the default
#: set: reporting only the per-element one published the slower of the two as
#: "discopt" (see :func:`build_discopt_vectorised`).
DEFAULT_ARMS = ("discopt", "discopt-vec", "pyomo", "pyomo-gc")


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────


def _rss_mb() -> float:
    """Resident set size in MB, from ``/proc/self/statm`` (Linux)."""
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def _best(fn, reps: int = 3) -> float:
    """Minimum wall over *reps* calls, each after a ``gc.collect()``."""
    out = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def _load_gate() -> float:
    """§9: report the 1-minute loadavg and warn when timings are untrustworthy."""
    load1 = os.getloadavg()[0]
    print(f"# load gate: 1-min loadavg = {load1:.2f}, cpus = {os.cpu_count()}", flush=True)
    if load1 > 2.0:
        print("# WARNING: machine is loaded -- timing numbers are NOT trustworthy", flush=True)
    return load1


# ─────────────────────────────────────────────────────────────
# mode: headline
# ─────────────────────────────────────────────────────────────


def _memory_child(arm: str, n_forms: int, n_inst: int) -> None:
    """Build one arm in a fresh process and report its *retained* RSS."""
    build, marker = ARMS[arm]
    marker_path = marker()  # force the import before taking the baseline
    gc.collect()
    base = _rss_mb()
    t0 = time.perf_counter()
    model, built = build(n_forms, n_inst)
    wall = time.perf_counter() - t0
    gc.collect()
    retained = _rss_mb() - base
    assert model is not None  # keep the model alive across the measurement
    print(
        json.dumps(
            {
                "arm": arm,
                "marker": marker_path,
                "built": built,
                "retained_mb": retained,
                "wall_s": wall,
            }
        )
    )


def mode_headline(args, arms: list[str]) -> int:
    n_total = args.forms * args.instances
    load1 = _load_gate()
    print(
        f"# model: {args.forms} forms x {args.instances} instances "
        f"= {n_total} constraint instances; {args.reps} interleaved reps",
        flush=True,
    )
    for a in arms:
        print(f"# {a} loaded from: {ARMS[a][1]()}", flush=True)

    # Timing: interleaved A/B/A/B in one process (§9).
    times: dict[str, list[float]] = {a: [] for a in arms}
    counts: dict[str, set[int]] = {a: set() for a in arms}
    for rep in range(args.reps):
        for a in arms:
            build, _ = ARMS[a]
            gc.collect()
            t0 = time.perf_counter()
            model, built = build(args.forms, args.instances)
            dt = time.perf_counter() - t0
            times[a].append(dt)
            counts[a].add(built)
            del model
            gc.collect()
            print(
                f"  rep {rep + 1}/{args.reps} {a:8s} {dt:7.3f} s  ({built} built)",
                flush=True,
            )

    # Memory: fresh subprocess per arm.
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
        print(
            f"  memory  {a:8s} {rec['retained_mb']:7.1f} MB retained ({rec['built']} built)",
            flush=True,
        )

    # §6: prove the probe fired, and built exactly the model it claims.
    executed = 0
    for a in arms:
        if len(counts[a]) != 1:
            print(f"FAIL: arm {a} built inconsistent instance counts {sorted(counts[a])}")
            return 1
        n = counts[a].pop()
        if n != n_total:
            print(f"FAIL: arm {a} built {n} instances, expected {n_total}")
            return 1
        executed += n * (args.reps + 1)
    print(f"# executed: {executed} constraint instances built across all arms/reps")
    if executed == 0:
        print("FAIL: probe built nothing")
        return 1

    print()
    print(
        f"{'arm':12s} {'median s':>10s} {'mean s':>10s} {'sd':>8s} "
        f"{'us/inst':>9s} {'RSS MB':>9s} {'B/inst':>9s}"
    )
    results = {}
    for a in arms:
        ts = times[a]
        med = statistics.median(ts)
        mean = statistics.fmean(ts)
        sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
        results[a] = {
            "median_s": med,
            "mean_s": mean,
            "sd_s": sd,
            "us_per_inst": med / n_total * 1e6,
            "retained_mb": mem[a],
            "b_per_inst": mem[a] * 1e6 / n_total,
            "times_s": ts,
        }
        r = results[a]
        print(
            f"{a:12s} {med:10.4f} {mean:10.4f} {sd:8.4f} "
            f"{r['us_per_inst']:9.3g} {r['retained_mb']:9.2f} {r['b_per_inst']:9.3g}"
        )

    if "discopt" in results and "pyomo" in results:
        d, p = results["discopt"], results["pyomo"]
        print()
        print(
            f"discopt / pyomo (as written):  time {d['median_s'] / p['median_s']:.3g}x   "
            f"memory {d['retained_mb'] / p['retained_mb']:.3g}x"
        )
        if "pyomo-gc" in results:
            g = results["pyomo-gc"]
            print(
                f"discopt / pyomo (same GC):    time {d['median_s'] / g['median_s']:.3g}x   "
                f"memory {d['retained_mb'] / g['retained_mb']:.3g}x"
            )
            print(
                "  ^ the LIKE-FOR-LIKE number. The line above credits discopt for "
                "tuning GC\n    inside Model.constraint, which Pyomo leaves to the "
                "caller (PauseGC); quoting\n    it as a statement about the modelling "
                "layers would overstate discopt by ~1.6x."
            )

    if "discopt-vec" in results and "pyomo" in results:
        v, p = results["discopt-vec"], results["pyomo"]
        print()
        print(
            f"discopt-vec / pyomo:          time {v['median_s'] / p['median_s']:.3g}x   "
            f"memory {v['retained_mb'] / p['retained_mb']:.3g}x"
        )
        if "discopt" in results:
            d = results["discopt"]
            print(
                f"discopt-vec / discopt:        time {v['median_s'] / d['median_s']:.3g}x   "
                f"memory {v['retained_mb'] / d['retained_mb']:.3g}x"
            )
        print(
            "  ^ THE IDIOM GAP. Both discopt rows build the same model; they differ\n"
            "    only in whether the constraint body is array-valued. Any claim about\n"
            "    'discopt's construction cost' has to say which of the two it means.\n"
            "    Note what this mode does NOT measure: an array-valued body defers its\n"
            "    per-row work to the writer or the solve, so discopt-vec's near-zero\n"
            "    construction time is real but partial. End to end (model -> .nl text)\n"
            "    the honest figure is 3.05-4.42 us/row against Pyomo's 35.5-48.1 --\n"
            "    see issue1215_cross_tool_panel.py and performance-plan.md section 48."
        )

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {"config": vars(args), "loadavg_1min": load1, "results": results},
                fh,
                indent=2,
            )
        print(f"# wrote {args.json_out}")
    return 0


# ─────────────────────────────────────────────────────────────
# mode: attribution
# ─────────────────────────────────────────────────────────────


def _count_nodes(con) -> Counter:
    """Distinct expression nodes reachable from a discopt constraint body."""
    from discopt.modeling.core import Expression

    seen: set[int] = set()
    stack = [con.body]
    c: Counter = Counter({"Constraint": 1})
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        c[type(n).__name__] += 1
        for attr in ("left", "right", "operand", "arg", "base", "args"):
            v = getattr(n, attr, None)
            if isinstance(v, Expression):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(a for a in v if isinstance(a, Expression))
    return c


def mode_attribution(args) -> int:
    import pyomo.environ as pyo
    from discopt.modeling import Model
    from discopt.modeling.core import Constant

    _load_gate()
    print(f"# discopt loaded from: {discopt_marker()}", flush=True)
    print(f"# pyomo loaded from:   {pyomo_marker()}", flush=True)

    checks = 0
    n_inst = args.instances

    m = Model()
    idx = m.set("I", list(range(n_inst)))
    x = m.continuous("x", over=idx, lb=0.5, ub=4.0)
    y = m.continuous("y", over=idx, lb=0.5, ub=4.0)

    pm = pyo.ConcreteModel()
    pm.I = pyo.RangeSet(0, n_inst - 1)
    pm.x = pyo.Var(pm.I, bounds=(0.5, 4.0))
    pm.y = pyo.Var(pm.I, bounds=(0.5, 4.0))

    # ── leaf identity ──
    print("\n== leaf identity ==", flush=True)
    print(f"  discopt  x[0] is x[0] : {x[0] is x[0]}   -> {type(x[0]).__name__}")
    print(f"  pyomo    x[0] is x[0] : {pm.x[0] is pm.x[0]}   -> {type(pm.x[0]).__name__}")
    checks += 2

    reps = max(1, args.reps)
    n_ops = 10 * n_inst
    td = _best(lambda: [x[i % n_inst] for i in range(n_ops)], reps)
    tp = _best(lambda: [pm.x[i % n_inst] for i in range(n_ops)], reps)
    print(
        f"  indexing alone: discopt {td / n_ops * 1e6:.2f} us   "
        f"pyomo {tp / n_ops * 1e6:.2f} us   ratio {td / tp:.1f}x"
    )
    checks += 1

    # ── nodes per constraint ──
    print("\n== nodes per constraint (x[i]*y[i] <= 4.0) ==", flush=True)
    con_d = x[0] * y[0] <= 4.0
    cd = _count_nodes(con_d)
    print(f"  discopt: {dict(cd)}  total = {sum(cd.values())}")
    print(f"           body = {con_d.body!r}   rhs = {float(con_d.rhs)}")
    expr_p = pm.x[0] * pm.y[0] <= 4.0
    print(f"  pyomo:   {type(expr_p).__name__} args = {[type(a).__name__ for a in expr_p.args]}")
    checks += 2

    # ── scalar literals ──
    print("\n== scalar literals ==", flush=True)
    k = Constant(4.0)
    print(
        f"  Constant(4.0).value is a {type(k.value).__name__} "
        f"at {sys.getsizeof(k.value)} B; a python float is {sys.getsizeof(4.0)} B"
    )
    node = x[0] * y[0]
    print(
        f"  discopt BinaryOp has __dict__: {hasattr(node, '__dict__')} "
        f"{sorted(getattr(node, '__dict__', {}))}"
    )
    print(f"  pyomo   ProductExpression has __dict__: {hasattr(expr_p.args[0], '__dict__')}")
    checks += 3

    # ── stage attribution ──
    print(f"\n== stage attribution, {n_inst} instances of x[i]*y[i] <= 4.0 ==", flush=True)

    def setup():
        mm = Model()
        ii = mm.set("I", list(range(n_inst)))
        mm.continuous("x", over=ii, lb=0.5, ub=4.0)
        mm.continuous("y", over=ii, lb=0.5, ub=4.0)

    n_fam = [0]

    def full():
        m2 = Model()
        i2 = m2.set("I", list(range(n_inst)))
        x2 = m2.continuous("x", over=i2, lb=0.5, ub=4.0)
        y2 = m2.continuous("y", over=i2, lb=0.5, ub=4.0)
        n_fam[0] = len(m2.constraint(i2, lambda i: x2[i] * y2[i] <= 4.0, name="c"))

    t_setup = _best(setup, reps)
    t_expr = _best(lambda: [x[i] * y[i] for i in range(n_inst)], reps)
    t_con = _best(lambda: [x[i] * y[i] <= 4.0 for i in range(n_inst)], reps)
    t_full = _best(full, reps)
    for label, t in (
        ("model setup (sets + vars)", t_setup),
        ("expression tree only (x*y)", t_expr),
        ("+ Constraint wrapper (<= 4.0)", t_con),
        ("full m.constraint(...)", t_full),
    ):
        print(f"  {label:32s} {t:7.3f} s   {t / n_inst * 1e6:6.2f} us/inst", flush=True)
        checks += 1
    print(f"  expression-node share of the full build: {t_con / t_full * 100:.0f}%")
    if n_fam[0] != n_inst:
        print(f"FAIL: family built {n_fam[0]} rows, expected {n_inst}")
        return 1
    checks += 1

    # ── one constraint, head to head ──
    print("\n== one constraint, head to head ==", flush=True)
    td = _best(lambda: [x[i] * y[i] <= 4.0 for i in range(n_inst)], reps)
    tp = _best(lambda: [pm.x[i] * pm.y[i] <= 4.0 for i in range(n_inst)], reps)
    print(
        f"  discopt {td / n_inst * 1e6:.2f} us   pyomo {tp / n_inst * 1e6:.2f} us   "
        f"ratio {td / tp:.1f}x"
    )
    checks += 1

    print(f"\n# executed checks: {checks}")
    if checks == 0:
        print("FAIL: probe asserted nothing")
        return 1
    return 0


# ─────────────────────────────────────────────────────────────
# mode: alloc
# ─────────────────────────────────────────────────────────────


def mode_alloc(args) -> int:
    """Retained allocation by source line (tracemalloc, after ``gc.collect()``)."""
    import tracemalloc

    print(f"# discopt loaded from: {discopt_marker()}", flush=True)
    n_total = args.forms * args.instances

    gc.collect()
    tracemalloc.start(1)
    model, built = build_discopt(args.forms, args.instances)
    gc.collect()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    assert model is not None  # keep it alive: this measures *retained*, not peak

    stats = snap.statistics("lineno")
    total = sum(s.size for s in stats)
    print(
        f"# built {built} constraint instances "
        f"({args.forms} forms x {args.instances}); tracemalloc total "
        f"{total / 1e6:.1f} MB = {total / n_total:.0f} B/instance"
    )
    print(f"{'B/inst':>8s}  {'MB':>7s}  site")
    shown = 0
    for s in stats[: args.top]:
        fr = s.traceback[0]
        src = linecache.getline(fr.filename, fr.lineno).strip()
        short = fr.filename.split("/discopt/")[-1]
        print(f"{s.size / n_total:8.0f}  {s.size / 1e6:7.1f}  {short}:{fr.lineno}  {src[:70]}")
        shown += 1

    print(f"# executed: {built} instances built, {shown} sites reported")
    if built != n_total or shown == 0:
        print(f"FAIL: built {built} (expected {n_total}), reported {shown} sites")
        return 1
    return 0


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=("headline", "attribution", "alloc"),
        default="headline",
        help="headline: wall + retained RSS vs Pyomo. attribution: where the "
        "wall goes. alloc: retained allocation by source line.",
    )
    ap.add_argument("--forms", type=int, default=40, help="distinct equation forms")
    ap.add_argument("--instances", type=int, default=5000, help="members per form")
    ap.add_argument("--reps", type=int, default=5, help="interleaved timing reps")
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="headline mode: arms to run. Both discopt idioms are default-on "
        "(see build_discopt_vectorised); see build_pyomo_gc_tuned for pyomo-gc",
    )
    ap.add_argument("--top", type=int, default=12, help="alloc mode: sites to report")
    ap.add_argument("--json-out", default=None, help="headline mode: write results JSON")
    ap.add_argument("--memory-child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.memory_child:
        _memory_child(args.memory_child, args.forms, args.instances)
        return 0

    if args.mode == "attribution":
        return mode_attribution(args)
    if args.mode == "alloc":
        return mode_alloc(args)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            ap.error(f"unknown arm {a!r}; choose from {sorted(ARMS)}")
    return mode_headline(args, arms)


if __name__ == "__main__":
    sys.exit(main())
