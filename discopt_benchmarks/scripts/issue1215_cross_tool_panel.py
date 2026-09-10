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
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

import discopt.modeling as dm
import numpy as np
import pyomo.environ as pyo
from discopt.export import to_nl
from discopt.modeling import Model

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


def _parse_tsv(path):
    """Read an arm's TSV back in. Raises on a malformed row -- never skips one."""
    out = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != COLS:
            raise ValueError(f"{path}: header is {header!r}, expected {list(COLS)!r}")
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) != len(COLS):
                raise ValueError(f"{path}:{lineno}: {len(f)} fields, expected {len(COLS)}")
            out.append(
                {
                    "tool": f[0],
                    "family": f[1],
                    "idiom": f[2],
                    "rows": int(f[3]),
                    "vars": int(f[4]),
                    "construct_s": float(f[5]),
                    "write_s": float(f[6]),
                }
            )
    if not out:
        raise ValueError(f"{path}: no data rows")
    return out


def _arm(row):
    return f"{row['tool']} {row['idiom']}" if row["tool"] == "discopt" else row["tool"]


def _report(rows, out=None):
    """Combined table. Asserts the arms built the same model before dividing.

    Returns the number of (arm, family, size) cells reported; a caller that gets
    0 back has measured nothing and must fail.
    """
    out = sys.stdout if out is None else out
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

    # Model-identity gate: every arm that measured a (family, size) cell must
    # agree with every other on the row and variable counts it read back out of
    # its own .nl header. Without this an arm could look fast by building less.
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

    cells = 0
    width = max(len(a) for a in arms) + 2
    for n in sizes:
        print(f"\n## total us/row (model -> .nl text), {n} rows\n", file=out)
        print("family".ljust(12) + "".join(a.rjust(width) for a in arms), file=out)
        for fam in families:
            line = fam.ljust(12)
            for a in arms:
                r = by.get((a, fam, n))
                if r is None:
                    line += "-".rjust(width)
                    continue
                line += f"{(r['construct_s'] + r['write_s']) * 1e6 / r['rows']:.2f}".rjust(width)
                cells += 1
            print(line, file=out)

    base = "oximo" if "oximo" in arms else None
    if base is not None:
        print(f"\n## ratio to {base} at {sizes[-1]} rows (>1 = slower)\n", file=out)
        print("family".ljust(12) + "".join(a.rjust(width) for a in arms), file=out)
        for fam in families:
            b = by.get((base, fam, sizes[-1]))
            if b is None:
                continue
            bt = b["construct_s"] + b["write_s"]
            line = fam.ljust(12)
            for a in arms:
                r = by.get((a, fam, sizes[-1]))
                line += (
                    "-".rjust(width)
                    if r is None
                    else f"{(r['construct_s'] + r['write_s']) / bt:.2f}x".rjust(width)
                )
            print(line, file=out)
    return cells


def _measure(emit):
    """Run the Python arms, calling ``emit(row)`` per point as it completes."""
    import tempfile

    rows_out = []
    for (tool, idiom), fams in ARMS.items():
        for fam, build in fams.items():
            for n in SIZES:
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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--oximo",
        metavar="TSV",
        help="TSV from the oximo arm (see scripts/oximo_arm/README.md); merged "
        "into the combined table",
    )
    ap.add_argument(
        "--from-tsv",
        metavar="TSV",
        action="append",
        default=[],
        help="report a previously measured TSV instead of re-running the Python arms; repeatable",
    )
    args = ap.parse_args(argv)

    print(f"# load: {os.getloadavg()[0]:.2f}", file=sys.stderr)

    rows = []
    if args.from_tsv:
        for path in args.from_tsv:
            rows += _parse_tsv(path)
    else:
        print("\t".join(COLS))

        def emit(row):
            print(
                f"{row['tool']}\t{row['family']}\t{row['idiom']}\t{row['rows']}\t"
                f"{row['vars']}\t{row['construct_s']:.6f}\t{row['write_s']:.6f}",
                flush=True,
            )

        rows = _measure(emit)

    if args.oximo:
        rows += _parse_tsv(args.oximo)

    print(f"# executed: {len(rows)} (arm, family, size) points", file=sys.stderr)
    cells = _report(rows)
    print(f"# reported: {cells} cells", file=sys.stderr)
    return 0 if rows and cells else 1


if __name__ == "__main__":
    sys.exit(main())
