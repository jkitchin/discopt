"""Cross-tool modelling panel: discopt vs Pyomo vs oximo (#1215).

Answers "where is the action" across a fuller set than a single synthetic shape.

Same four model families and three sizes as the oximo arm, same mathematics,
each timed from an empty model to `.nl` text. discopt is measured in BOTH
idioms: per-element (comparable to oximo's and Pyomo's, which have no
array-valued constraint body) and vectorised (its own).

Row and variable counts come from each written file's own header, so an arm
cannot look fast by building a smaller model than it claims; the combiner
asserts every arm agrees on both before reporting a ratio.

oximo is a **Rust** crate (``pip install oximo`` finds an unrelated package), so
its arm is built separately -- see ``scripts/oximo_arm/README.md`` -- and merged
here via ``--oximo <tsv>``::

    python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
        --oximo oximo.tsv

With no flags the Python arms are measured, their per-point TSV streamed to
stdout as each point lands, and the combined table printed at the end.
``--from-tsv <tsv>`` (repeatable) re-reports a stored TSV instead of measuring,
so a panel can be recombined without paying for it twice. Both flags feed the
same combiner; the run prints its executed-comparison count to stderr and exits
non-zero if it reported nothing.

``--memory`` measures **retained memory** instead of wall time -- the other half
of the #1215 goal, and the half that had never been measured against oximo. Each
point runs in a fresh process (an allocator pools what a previous model freed),
reads ``VmRSS`` from ``/proc/self/status`` before and after construction with the
model held alive, and warms up on a tiny model first so the first build's imports
land in the baseline. The oximo arm does exactly the same thing (``--memory``
there too), which is why RSS and not ``tracemalloc``: RSS counts the Rust arena
behind discopt's PyO3 handles and oximo's ``Vec<ExprNode>`` alike. It also counts
allocator slack, so a difference under ~10% is noise.

Measured 2026-09-10, load 0.05, 4 families x 3 sizes x 4 arms. Total µs/row,
model to ``.nl`` text, at 100 000 rows:

    family        oximo   discopt vec   discopt elem   pyomo
    linear         3.97          3.63          40.28   38.99
    sep_nl         2.97          3.05          25.29   35.51
    coupled_nl     3.47          4.42          36.99   48.14
    minlp          4.11          4.24          29.13   46.92

**The action is the idiom, not the tool.** Vectorised discopt is at oximo
parity at scale (0.91-1.27x, and faster on ``linear``) and 10.7-11.6x faster
than Pyomo; per-element discopt is within noise of Pyomo, and both are
7.1-10.7x off oximo. The swing between discopt's own two idioms is 6.9-11.1x --
larger than the whole discopt-to-Pyomo gap. See
``docs/dev/performance-plan.md`` section 48.

Retained B/row at 100 000 rows, same panel (``--memory``):

    family        oximo   discopt vec   discopt elem   pyomo
    linear         1072            17           1301    1311
    sep_nl          615            17            942     984
    coupled_nl      684            17           1135    1225
    minlp           853            17           1108    1150

Memory tells the same story more sharply: vectorised discopt is **0.02-0.03x
oximo** -- 40-60x leaner, because it retains little more than the numpy data the
user handed it -- while per-element discopt (1.21-1.66x) and Pyomo (1.22-1.79x)
are both somewhat heavier than oximo. Section 50.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

import discopt.modeling as dm
import numpy as np
import pyomo.environ as pyo
from discopt.export import to_nl
from discopt.modeling import Model

#: Default sizes, kept at the values the recorded oximo TSVs in
#: ``scripts/oximo_arm/`` were measured at so a stored arm still combines.
#: They are ROUND NUMBERS, not corpus row counts, and ``performance-plan.md``
#: §62 retracted the headline multipliers measured at them: MINLPLib's row
#: distribution is p25=9, p50=111, p75=945, p90=4009, p99=68072, so 100 000 is
#: past the corpus p99 and no in-repo instance reaches even 1 000. §62's binding
#: rule is that a panel must sample that distribution -- pass ``--sizes`` (and
#: the oximo arm's matching ``--sizes``) to do so:
#:
#:     --sizes 9,111,945,4009,68072
#:
#: The default is left alone rather than changed in place because both arms
#: hardcode it and the combiner joins on ``rows``: moving it here alone would
#: make every recorded oximo TSV uncombinable.
SIZES = (1_000, 10_000, 100_000)
REPS = 3
INF = 1e20

# ── discopt, vectorised ─────────────────────────────────────────────────────


def d_vec_linear(n):
    m = Model("linear")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=INF)
    z = m.continuous("z", shape=(n,), lb=0.0, ub=INF)
    b = 10.0 + np.arange(n) % 7
    m.subject_to(x + 2.0 * y + 3.0 * z <= b, name="c")
    m.minimize(dm.sum(x))
    return m


def d_vec_sep_nl(n):
    m = Model("sep_nl")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.1, ub=INF)
    m.subject_to(dm.exp(x) + y <= 9.0 + np.arange(n) % 5, name="c")
    m.minimize(dm.sum(x))
    return m


def d_vec_coupled_nl(n):
    m = Model("coupled_nl")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.1, ub=INF)
    m.subject_to(x * y + dm.log(x + 1.0) <= 9.0 + np.arange(n) % 5, name="c")
    m.minimize(dm.sum(x))
    return m


def d_vec_minlp(n):
    m = Model("minlp")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=INF)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=INF)
    z = m.binary("z", shape=(n,))
    m.subject_to(x * y + z <= 4.0 + np.arange(n) % 3, name="c")
    m.minimize(dm.sum(x))
    return m


# ── discopt, per-element ────────────────────────────────────────────────────


def _d_elem(name, n, decl, rule):
    m = Model(name)
    idx = m.set("I", list(range(n)))
    v = decl(m, n)
    m.constraint(idx, lambda i: rule(v, i), name="c")
    m.minimize(dm.sum([v[0][i] for i in range(n)]))
    return m


def d_elem_linear(n):
    return _d_elem(
        "linear",
        n,
        lambda m, n: (
            m.continuous("x", shape=(n,), lb=0.0, ub=INF),
            m.continuous("y", shape=(n,), lb=0.0, ub=INF),
            m.continuous("z", shape=(n,), lb=0.0, ub=INF),
        ),
        lambda v, i: v[0][i] + 2.0 * v[1][i] + 3.0 * v[2][i] <= 10.0 + i % 7,
    )


def d_elem_sep_nl(n):
    return _d_elem(
        "sep_nl",
        n,
        lambda m, n: (
            m.continuous("x", shape=(n,), lb=0.1, ub=INF),
            m.continuous("y", shape=(n,), lb=0.1, ub=INF),
        ),
        lambda v, i: dm.exp(v[0][i]) + v[1][i] <= 9.0 + i % 5,
    )


def d_elem_coupled_nl(n):
    return _d_elem(
        "coupled_nl",
        n,
        lambda m, n: (
            m.continuous("x", shape=(n,), lb=0.1, ub=INF),
            m.continuous("y", shape=(n,), lb=0.1, ub=INF),
        ),
        lambda v, i: v[0][i] * v[1][i] + dm.log(v[0][i] + 1.0) <= 9.0 + i % 5,
    )


def d_elem_minlp(n):
    return _d_elem(
        "minlp",
        n,
        lambda m, n: (
            m.continuous("x", shape=(n,), lb=0.0, ub=INF),
            m.continuous("y", shape=(n,), lb=0.0, ub=INF),
            m.binary("z", shape=(n,)),
        ),
        lambda v, i: v[0][i] * v[1][i] + v[2][i] <= 4.0 + i % 3,
    )


# ── Pyomo ───────────────────────────────────────────────────────────────────


def p_linear(n):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(n))
    m.x = pyo.Var(m.I, bounds=(0.0, None))
    m.y = pyo.Var(m.I, bounds=(0.0, None))
    m.z = pyo.Var(m.I, bounds=(0.0, None))
    m.c = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.x[i] + 2.0 * mm.y[i] + 3.0 * mm.z[i] <= 10.0 + i % 7
    )
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I))
    return m


def p_sep_nl(n):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(n))
    m.x = pyo.Var(m.I, bounds=(0.1, None))
    m.y = pyo.Var(m.I, bounds=(0.1, None))
    m.c = pyo.Constraint(m.I, rule=lambda mm, i: pyo.exp(mm.x[i]) + mm.y[i] <= 9.0 + i % 5)
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I))
    return m


def p_coupled_nl(n):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(n))
    m.x = pyo.Var(m.I, bounds=(0.1, None))
    m.y = pyo.Var(m.I, bounds=(0.1, None))
    m.c = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.x[i] * mm.y[i] + pyo.log(mm.x[i] + 1.0) <= 9.0 + i % 5
    )
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I))
    return m


def p_minlp(n):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(n))
    m.x = pyo.Var(m.I, bounds=(0.0, None))
    m.y = pyo.Var(m.I, bounds=(0.0, None))
    m.z = pyo.Var(m.I, domain=pyo.Binary)
    m.c = pyo.Constraint(m.I, rule=lambda mm, i: mm.x[i] * mm.y[i] + mm.z[i] <= 4.0 + i % 3)
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I))
    return m


ARMS = {
    ("discopt", "vectorised"): {
        "linear": d_vec_linear,
        "sep_nl": d_vec_sep_nl,
        "coupled_nl": d_vec_coupled_nl,
        "minlp": d_vec_minlp,
    },
    ("discopt", "per-element"): {
        "linear": d_elem_linear,
        "sep_nl": d_elem_sep_nl,
        "coupled_nl": d_elem_coupled_nl,
        "minlp": d_elem_minlp,
    },
    ("pyomo", "per-element"): {
        "linear": p_linear,
        "sep_nl": p_sep_nl,
        "coupled_nl": p_coupled_nl,
        "minlp": p_minlp,
    },
}


def write_text(tool, model, path):
    if tool == "pyomo":
        model.write(path, format="nl")
        with open(path) as fh:
            return fh.read()
    text = to_nl(model)
    with open(path, "w") as fh:
        fh.write(text)
    return text


COLS = ("tool", "family", "idiom", "rows", "vars", "construct_s", "write_s")
MEM_COLS = ("tool", "family", "idiom", "rows", "vars", "retained_b")
_FIELD_TYPE = {
    "rows": int,
    "vars": int,
    "construct_s": float,
    "write_s": float,
    "retained_b": int,
}

# What each mode divides by its row count, and how the table is labelled.
_MODES = {
    "time": {
        "cols": COLS,
        "value": lambda r: (r["construct_s"] + r["write_s"]) * 1e6,
        "unit": "total us/row (model -> .nl text)",
        "fmt": "{:.2f}",
    },
    "memory": {
        "cols": MEM_COLS,
        "value": lambda r: float(r["retained_b"]),
        "unit": "retained B/row (built model, held alive)",
        "fmt": "{:.0f}",
    },
}


def _parse_tsv(path, cols=COLS):
    """Read an arm's TSV back in. Raises on a malformed row -- never skips one."""
    out = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != cols:
            raise ValueError(f"{path}: header is {header!r}, expected {list(cols)!r}")
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) != len(cols):
                raise ValueError(f"{path}:{lineno}: {len(f)} fields, expected {len(cols)}")
            out.append({c: _FIELD_TYPE.get(c, str)(v) for c, v in zip(cols, f, strict=True)})
    if not out:
        raise ValueError(f"{path}: no data rows")
    return out


def _emit_tsv(row, cols):
    print("\t".join(str(row[c]) for c in cols), flush=True)


def _arm(row):
    return f"{row['tool']} {row['idiom']}" if row["tool"] == "discopt" else row["tool"]


def _index(rows):
    """Order-preserving (arms, families, sizes) plus the (arm, family, size) map."""
    arms, families, sizes = [], [], []
    for r in rows:
        for seq, val in ((arms, _arm(r)), (families, r["family"]), (sizes, r["rows"])):
            if val not in seq:
                seq.append(val)
    sizes.sort()
    by = {}
    for r in rows:
        key = (_arm(r), r["family"], r["rows"])
        if key in by:
            raise ValueError(f"duplicate measurement for {key}")
        by[key] = r
    return arms, families, sizes, by


def _identity_gate(arms, families, sizes, by):
    """Every arm that measured a cell must agree on the model it built.

    Row and variable counts come from the header of the ``.nl`` each arm wrote
    itself. Without this check an arm looks fast, or lean, by building less.
    Returns the number of comparisons made -- 0 means the gate did nothing.
    """
    checked = 0
    for fam in families:
        for n in sizes:
            present = [by[(a, fam, n)] for a in arms if (a, fam, n) in by]
            if len(present) < 2:
                continue
            ref = present[0]
            for other in present[1:]:
                if (other["rows"], other["vars"]) != (ref["rows"], ref["vars"]):
                    raise AssertionError(
                        f"{fam}/{n}: {_arm(other)} built "
                        f"{other['rows']}x{other['vars']} but {_arm(ref)} built "
                        f"{ref['rows']}x{ref['vars']} -- arms are not the same model"
                    )
                checked += 1
    print(f"# model-identity comparisons: {checked}", file=sys.stderr)
    return checked


def _report(rows, mode="time", out=None):
    """Combined table. Asserts the arms built the same model before dividing.

    Returns the number of (arm, family, size) cells reported; a caller that gets
    0 back has measured nothing and must fail.
    """
    out = sys.stdout if out is None else out
    spec = _MODES[mode]
    value, fmt = spec["value"], spec["fmt"]
    arms, families, sizes, by = _index(rows)
    _identity_gate(arms, families, sizes, by)

    cells = 0
    unresolved = 0
    width = max(len(a) for a in arms) + 2
    for n in sizes:
        print(f"\n## {spec['unit']}, {n} rows\n", file=out)
        print("family".ljust(12) + "".join(a.rjust(width) for a in arms), file=out)
        for fam in families:
            line = fam.ljust(12)
            for a in arms:
                r = by.get((a, fam, n))
                if r is None:
                    line += "-".rjust(width)
                    continue
                per_row = value(r) / r["rows"]
                line += fmt.format(per_row).rjust(width)
                cells += 1
                if mode == "memory" and per_row < _RSS_RESOLUTION_B / r["rows"]:
                    unresolved += 1
            print(line, file=out)
        if mode == "memory" and unresolved:
            print(
                f"  (a cell below {_RSS_RESOLUTION_B / n:.2f} B/row is under the "
                f"{_RSS_RESOLUTION_B} B RSS resolution at this size -- read it as "
                "'too small to measure', not as zero)",
                file=out,
            )
            unresolved = 0

    if "oximo" in arms:
        biggest = sizes[-1]
        print(f"\n## ratio to oximo at {biggest} rows (>1 = worse)\n", file=out)
        print("family".ljust(12) + "".join(a.rjust(width) for a in arms), file=out)
        for fam in families:
            b = by.get(("oximo", fam, biggest))
            if b is None or value(b) == 0:
                continue
            line = fam.ljust(12)
            for a in arms:
                r = by.get((a, fam, biggest))
                line += (
                    "-".rjust(width) if r is None else f"{value(r) / value(b):.2f}x".rjust(width)
                )
            print(line, file=out)
    return cells


def _nl_shape(tool, model):
    """``(rows, vars)`` read out of the ``.nl`` this model actually writes."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as fh:
        path = fh.name
    try:
        text = write_text(tool, model, path)
    finally:
        os.unlink(path)
    header = text.split("\n")[1].split()
    return int(header[1]), int(header[0])


def _measure(emit, sizes=SIZES):
    """Run the Python arms, calling ``emit(row)`` per point as it completes."""
    import tempfile

    rows_out = []
    for (tool, idiom), fams in ARMS.items():
        for fam, build in fams.items():
            for n in sizes:
                cons, writes = [], []
                rows = vars_ = 0
                for _ in range(REPS):
                    gc.collect()
                    t0 = time.perf_counter()
                    m = build(n)
                    cons.append(time.perf_counter() - t0)
                    with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as fh:
                        p = fh.name
                    try:
                        t1 = time.perf_counter()
                        text = write_text(tool, m, p)
                        writes.append(time.perf_counter() - t1)
                    finally:
                        os.unlink(p)
                    hdr = text.split("\n")[1].split()
                    vars_, rows = int(hdr[0]), int(hdr[1])
                    del m
                assert rows == n, f"{tool}/{idiom}/{fam}/{n}: wrote {rows} rows"
                row = {
                    "tool": tool,
                    "family": fam,
                    "idiom": idiom,
                    "rows": rows,
                    "vars": vars_,
                    "construct_s": statistics.median(cons),
                    "write_s": statistics.median(writes),
                }
                rows_out.append(row)
                emit(row)
    return rows_out


# ── memory ──────────────────────────────────────────────────────────────────
#
# Measured exactly as the oximo arm measures it (`oximo_arm/main.rs --memory`):
# a FRESH PROCESS per point, resident set read from /proc/self/statm before and
# after construction, the model held alive across the reading. RSS rather than a
# language-level counter is what makes the two comparable -- it counts the Rust
# arena behind discopt's PyO3 handles and oximo's `Vec<ExprNode>` alike, which no
# `tracemalloc` figure would. It also counts allocator slack, so treat a
# difference under ~10% as noise.


#: One `VmRSS` reading is quantised to this many bytes, so a model smaller than
#: this is indistinguishable from no model at all. It is a floor on what the
#: memory mode can resolve, not a measured zero.
_RSS_RESOLUTION_B = 1024


def _rss_bytes():
    """Resident set size, from ``/proc/self/status``.

    ``VmRSS`` rather than ``/proc/self/statm`` because its unit is stated in the
    file (kB), so the oximo arm reads the same field the same way with no shared
    assumption about page size.
    """
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                value, unit = line.split()[1:3]
                if unit != "kB":
                    raise ValueError(f"VmRSS reported in {unit!r}, expected 'kB'")
                return int(value) * 1024
    raise ValueError("no VmRSS line in /proc/self/status")


def _memory_child(tool, idiom, family, n):
    """Build one arm in this (fresh) process and print its retained RSS as JSON."""
    build = ARMS[(tool, idiom)][family]

    # Warm-up: a tiny model of the same shape, discarded. Whatever the first
    # build imports or caches is then already resident and is NOT charged to the
    # measured model.
    warm = build(8)
    del warm
    gc.collect()

    base = _rss_bytes()
    model = build(n)
    gc.collect()
    retained = _rss_bytes() - base
    # `model` must still be alive at the reading above, and the shape must come
    # from the file it really writes -- so read it after, not before.
    rows, vars_ = _nl_shape(tool, model)
    assert rows == n, f"{tool}/{idiom}/{family}/{n}: wrote {rows} rows"
    print(
        json.dumps(
            {
                "tool": tool,
                "family": family,
                "idiom": idiom,
                "rows": rows,
                "vars": vars_,
                "retained_b": retained,
            }
        )
    )


def _measure_memory(emit, sizes=SIZES):
    """Spawn one child per point; a shared process would pool freed memory."""
    rows_out = []
    for (tool, idiom), fams in ARMS.items():
        for fam in fams:
            for n in sizes:
                proc = subprocess.run(
                    [sys.executable, "-u", __file__, "--memory-child", tool, idiom, fam, str(n)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                rows_out.append(row)
                emit(row)
    return rows_out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--memory",
        action="store_true",
        help="measure retained memory per built model instead of wall time",
    )
    ap.add_argument(
        "--oximo",
        metavar="TSV",
        help="TSV from the oximo arm (see scripts/oximo_arm/README.md); merged "
        "into the combined table. Must match the mode: a --memory run needs the "
        "oximo arm's --memory TSV",
    )
    ap.add_argument(
        "--from-tsv",
        metavar="TSV",
        action="append",
        default=[],
        help="report a previously measured TSV instead of re-running the Python arms; repeatable",
    )
    ap.add_argument(
        "--memory-child",
        nargs=4,
        metavar=("TOOL", "IDIOM", "FAMILY", "ROWS"),
        help=argparse.SUPPRESS,  # internal: one measurement, one fresh process
    )
    ap.add_argument(
        "--sizes",
        metavar="N,N,...",
        help="row counts to measure, comma-separated (default: "
        f"{','.join(str(n) for n in SIZES)}). Per performance-plan.md §62 a panel "
        "must sample the corpus row-count distribution -- 9,111,945,4009,68072 are "
        "MINLPLib's p25/p50/p75/p90/p99. The oximo arm must be run at the same "
        "sizes for the arms to combine.",
    )
    args = ap.parse_args(argv)

    if args.memory_child:
        tool, idiom, family, n = args.memory_child
        _memory_child(tool, idiom, family, int(n))
        return 0

    mode = "memory" if args.memory else "time"
    cols = _MODES[mode]["cols"]
    if args.sizes:
        sizes = tuple(int(t) for t in args.sizes.split(","))
        if not sizes or any(n < 1 for n in sizes):
            raise ValueError(f"--sizes must be positive row counts, got {args.sizes!r}")
    else:
        sizes = SIZES
    print(f"# load: {os.getloadavg()[0]:.2f}  sizes: {list(sizes)}", file=sys.stderr)

    rows = []
    if args.from_tsv:
        for path in args.from_tsv:
            rows += _parse_tsv(path, cols)
    else:
        print("\t".join(cols))
        measure = _measure_memory if args.memory else _measure
        rows = measure(lambda r: _emit_tsv(r, cols), sizes)

    if args.oximo:
        rows += _parse_tsv(args.oximo, cols)

    print(f"# executed: {len(rows)} (arm, family, size) points", file=sys.stderr)
    cells = _report(rows, mode)
    print(f"# reported: {cells} cells", file=sys.stderr)
    return 0 if rows and cells else 1


if __name__ == "__main__":
    sys.exit(main())
