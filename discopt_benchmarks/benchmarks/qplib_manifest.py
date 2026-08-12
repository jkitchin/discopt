"""Build the QPLIB corpus manifest (issue #830).

QPLIB ships an ``instancedata.csv`` describing every instance and a
``qplib.solu`` oracle of best-known objective values, but neither says whether
*this* reader reproduces those values. That question has to be answered per
instance, because a handful of instances are not usable as correctness oracles
at all -- four of them carry an identically zero objective in the ``.qplib``
file while publishing a nonzero optimum.

This script answers it by parsing every instance, evaluating the published
reference point through the parsed model, and recording the outcome. The result
is ``qplib_manifest.csv`` in the corpus directory: one row per instance with the
structural facts, the reference objective, whether that objective was
*reproduced* from the file, and how badly the reference point violates the
parsed model.

Downstream selection reads the manifest instead of hardcoding instance names, so
no instance-specific special cases leak into the solver or the suites (CLAUDE.md
development philosophy #2).

Usage::

    python -m benchmarks.qplib_manifest                  # default corpus dir
    python -m benchmarks.qplib_manifest --corpus DIR --out FILE
    python -m benchmarks.qplib_manifest --max-bytes 20000000   # skip giants

Exits non-zero if it processed zero instances or if any instance failed to
parse -- a manifest that silently covered nothing would read as a pass.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from discopt.interfaces import qplib as qp

DEFAULT_CORPUS = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")

#: A reference point is treated as reproducing its published objective when the
#: relative difference is within this. QPLIB values are published to ~10
#: significant digits, so this is loose enough for accumulation error over
#: millions of terms and far tighter than any real convention error (the
#: quadratic-scaling mistake this guards against shows up as a factor of ~2).
_OBJ_RTOL = 1e-6

#: Best-known points are *approximately* feasible -- they come from floating
#: point solvers and are published as-is. A point is recorded as feasible when
#: its worst violation is within this, scaled by the row magnitude.
_FEAS_TOL = 1e-4

FIELDS = [
    "name",
    "probtype",
    "sense",
    "nvars",
    "ncons",
    "n_integral",
    "n_continuous",
    "nonconvex",
    "obj_quad_nnz",
    "obj_lin_nnz",
    "declared_obj_nz",
    "con_quad_nnz",
    "con_lin_nnz",
    "reference",
    "sol_file",
    "sol_objvar",
    "ref_reproduced",
    "ref_max_violation",
    "ref_feasible",
    "usable_oracle",
    "bytes",
    "parse_seconds",
    "note",
]


def build_rows(corpus: str, max_bytes: int | None, verbose: bool = True):
    """Parse every instance and yield one manifest row each.

    Parse failures are *not* swallowed -- they are recorded with a note and
    counted, and the caller turns a nonzero count into a nonzero exit.
    """
    data_path = os.path.join(corpus, "instancedata.csv")
    with open(data_path, encoding="utf-8") as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    solu = qp.read_solu(os.path.join(corpus, "qplib.solu"))

    rows = []
    failures = []
    for n, (name, r) in enumerate(sorted(meta.items()), start=1):
        path = os.path.join(corpus, "qplib", f"{name}.qplib")
        if not os.path.exists(path):
            failures.append(f"{name}: missing {path}")
            continue
        size = os.path.getsize(path)
        if max_bytes is not None and size > max_bytes:
            if verbose:
                print(f"[{n:3d}/{len(meta)}] {name} SKIP ({size} bytes)", flush=True)
            continue

        t0 = time.perf_counter()
        try:
            inst = qp.read_qplib(path)
        except ValueError as exc:  # narrow: the reader's own structural error
            failures.append(f"{name}: parse failed: {exc}")
            if verbose:
                print(f"[{n:3d}/{len(meta)}] {name} PARSE-FAIL {exc}", flush=True)
            continue
        parse_s = time.perf_counter() - t0

        # Upstream consistency check, stated as a general rule rather than an
        # exclusion list of names: instancedata.csv counts the variables that
        # appear in the objective, so a positive count with nothing stored in
        # the file means the objective was lost upstream. Four instances trip
        # this (QPLIB_10035/10036/10037/10039) -- their published optimum is
        # nonzero but their file encodes the zero function, and their reference
        # point is also grossly infeasible. They cannot serve as oracles.
        obj_lin_nnz = int(np.count_nonzero(inst.obj_lin))
        declared_obj_nz = int(r["nobjnz"])
        objective_lost = declared_obj_nz > 0 and obj_lin_nnz + len(inst.obj_quad) == 0

        solpath = os.path.join(corpus, "sol", f"{name}.sol")
        reference = solu.get(name)
        sol_file = os.path.exists(solpath)
        objvar = None
        reproduced = ""
        feasible = ""
        violation = ""
        note = (
            f"upstream defect: instancedata.csv declares {declared_obj_nz} objective "
            "variable(s) but the file stores no objective terms"
            if objective_lost
            else ""
        )

        if sol_file:
            # Deliberately unguarded: a malformed reference point is a reader
            # defect, and hiding it would make the manifest meaningless.
            x, objvar = qp.read_solution(solpath, inst)
            recomputed = inst.evaluate_objective(x)
            target = objvar if objvar is not None else reference
            if target is not None:
                denom = max(1.0, abs(target))
                # bool(), not the bare comparison: numpy returns np.bool_, and
                # `np.bool_(True) is True` is False, which silently sank the
                # usable-oracle count from 421 to 216 the first time this ran.
                reproduced = bool(abs(recomputed - target) / denom <= _OBJ_RTOL)
                if not reproduced:
                    note = (note + "; " if note else "") + (
                        f"objective at reference point {recomputed:.12g} != published {target:.12g}"
                    )
            violation = float(inst.max_violation(x))
            feasible = bool(violation <= _FEAS_TOL)
            if not feasible:
                note = (note + "; " if note else "") + (
                    f"reference point violates the parsed model by {violation:.3e}"
                )

        usable = bool(reference is not None and reproduced is True and feasible is True)
        rows.append(
            {
                "name": name,
                "probtype": inst.probtype,
                "sense": inst.sense,
                "nvars": inst.n_vars,
                "ncons": inst.n_cons,
                "n_integral": inst.n_integral,
                "n_continuous": inst.n_vars - inst.n_integral,
                "nonconvex": r["convex"].strip().lower() != "true",
                "obj_quad_nnz": len(inst.obj_quad),
                "obj_lin_nnz": obj_lin_nnz,
                "declared_obj_nz": declared_obj_nz,
                "con_quad_nnz": len(inst.con_quad),
                "con_lin_nnz": len(inst.con_lin),
                "reference": "" if reference is None else repr(reference),
                "sol_file": sol_file,
                "sol_objvar": "" if objvar is None else repr(objvar),
                "ref_reproduced": reproduced,
                "ref_max_violation": "" if violation == "" else f"{violation:.6e}",
                "ref_feasible": feasible,
                "usable_oracle": usable,
                "bytes": size,
                "parse_seconds": f"{parse_s:.3f}",
                "note": note,
            }
        )
        if verbose:
            flag = "ok " if usable else "-- "
            print(
                f"[{n:3d}/{len(meta)}] {flag}{name} {inst.probtype} "
                f"nvars={inst.n_vars} ncons={inst.n_cons} {parse_s:.2f}s {note}",
                flush=True,
            )
    return rows, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=None, help="default: <corpus>/qplib_manifest.csv")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="skip instances larger than this many bytes (default: no limit)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or os.path.join(args.corpus, "qplib_manifest.csv")
    rows, failures = build_rows(args.corpus, args.max_bytes, verbose=not args.quiet)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    usable = sum(1 for r in rows if r["usable_oracle"])
    nonconvex = sum(1 for r in rows if r["nonconvex"])
    no_sol = sum(1 for r in rows if not r["sol_file"])
    bad_obj = sum(1 for r in rows if r["ref_reproduced"] is False)
    bad_feas = sum(1 for r in rows if r["ref_feasible"] is False)
    print(f"\nwrote {out}")
    print(f"instances in manifest : {len(rows)}")
    print(f"  usable as oracle    : {usable}")
    print(f"  nonconvex           : {nonconvex}")
    print(f"  no reference point  : {no_sol}")
    print(f"  objective not repro : {bad_obj}")
    print(f"  reference infeasible: {bad_feas}")
    print(f"  parse failures      : {len(failures)}")
    for r in rows:
        if r["note"]:
            print(f"  NOTE {r['name']}: {r['note']}")
    for f in failures[:20]:
        print("  FAIL", f)

    # Prove the run actually did something (CLAUDE.md measurement discipline #6).
    if not rows:
        print("ERROR: manifest is empty -- nothing was parsed", file=sys.stderr)
        return 2

    # Recompute usable_oracle by truthiness rather than identity. The two
    # formulations agree only if the flags are genuine Python bools; an
    # np.bool_ leaking back in would diverge here instead of silently
    # understating the count, which is how this went wrong once already.
    recheck = sum(1 for r in rows if r["reference"] and r["ref_reproduced"] and r["ref_feasible"])
    if recheck != usable:
        print(
            f"ERROR: usable_oracle inconsistent ({usable} by identity, {recheck} by "
            "truthiness) -- a non-bool leaked into the flags",
            file=sys.stderr,
        )
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
