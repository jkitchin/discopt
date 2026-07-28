"""Oracle certification check for the incremental-McCormick work (issue #861).

Solves the named in-repo instances and checks each result against MINLPLib's
reference optima, read from ``minlplib.solu`` — **never typed in by hand**.

That is the whole point of this script. Two runs during #861 were derailed by
hand-transcribed oracle values: ``st_e11`` was checked against ``0.067038`` (true
value ``189.3116297``) and reported as a false optimum, and ``st_e17`` was checked
against a value invented outright — it has no ``=opt=`` entry at all. Both
"violations" evaporated on re-check. A number typed into a probe is not a
measurement (CLAUDE.md: re-derive any figure before stating it; look it up rather
than guess).

Two distinct properties are checked, and they are NOT the same:

* **objective** — only meaningful when the solver claims ``optimal``. A
  ``time_limit`` result carries an incumbent, not a certificate; it being worse
  than the reference is a performance observation, never a correctness violation.
* **dual bound** — checked ALWAYS, whatever the status. For a minimize-sense
  model the bound must never exceed the reference optimum; crossing it is the
  false-bound class and is a hard failure regardless of how the run ended.

An instance with no reference entry is reported as ``NO-ORACLE`` and excluded from
the verdict rather than silently passing.

Usage::

    python -u discopt_benchmarks/scripts/incremental_oracle_check.py \\
        --instances prob02,st_e01,nvs17 --time-limit 30

Prints an executed-check count and exits non-zero if it is zero (a check that
checked nothing must never read as a pass — CLAUDE.md rule 6) or if any
correctness property failed.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORPUS = os.path.join(_REPO, "python", "tests", "data", "minlplib")
_DEFAULT_SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
# Reference optima for the corpus instances MINLPLib's snapshot does not carry
# (st_e17, meanvar), each established by running independent external global
# solvers on the exact .nl in this repo. See that file's _README for why it exists.
_LOCAL_ORACLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local_oracle.json"
)


def load_oracle(path):
    """Parse ``minlplib.solu`` -> ``{name: (marker, value)}``.

    Lines look like ``=opt=      name      value`` / ``=best=`` / ``=inf=`` /
    ``=unbounded=``. Only ``=opt=`` is a proven optimum; ``=best=`` is the best
    value known and is a valid *upper* bound on a minimize optimum, so a dual
    bound above it is still a violation, but an objective differing from it is not.
    """
    oracle = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or not parts[0].startswith("="):
                continue
            marker = parts[0].strip("=")
            name = parts[1]
            value = None
            if len(parts) >= 3:
                try:
                    value = float(parts[2])
                except ValueError:
                    value = None
            oracle[name] = (marker, value)
    return oracle


def merge_local_oracle(oracle, path=_LOCAL_ORACLE):
    """Add locally-established references for instances MINLPLib does not carry.

    minlplib.solu WINS on any overlap — a local measurement may fill a gap, never
    override the upstream reference."""
    if not os.path.exists(path):
        return oracle, 0
    import json

    with open(path) as fh:
        blob = json.load(fh)
    added = 0
    for name, entry in blob.get("instances", {}).items():
        if name in oracle:
            continue
        oracle[name] = (entry.get("marker", "opt"), float(entry["value"]))
        added += 1
    return oracle, added


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True, help="comma-separated instance names")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--solu", default=_DEFAULT_SOLU)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    if not os.path.exists(args.solu):
        print(f"ERROR: oracle file not found: {args.solu}", file=sys.stderr)
        return 2
    oracle = load_oracle(args.solu)
    oracle, n_local = merge_local_oracle(oracle)
    if n_local:
        print(f"(merged {n_local} locally-established references from {_LOCAL_ORACLE})")

    from discopt.modeling.core import ObjectiveSense, from_nl

    checked = 0
    violations = []
    no_oracle = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        path = os.path.join(_CORPUS, f"{name}.nl")
        if not os.path.exists(path):
            print(f"{name:12s} NOT IN CORPUS")
            continue
        entry = oracle.get(name)
        model = from_nl(path)
        res = model.solve(time_limit=args.time_limit)
        obj = res.objective
        bound = getattr(res, "bound", None)
        maximize = (
            model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE
        )
        if entry is None or entry[1] is None:
            no_oracle.append(name)
            print(f"{name:12s} {res.status:11s} obj={obj!s:22s} bound={bound!s:22s} NO-ORACLE")
            continue
        marker, ref = entry
        checked += 1
        tol = args.rel_tol * (1.0 + abs(ref))
        flags = []
        # Objective: only a CERTIFIED result makes a claim worth checking.
        certified = res.status == "optimal" and obj is not None and marker == "opt"
        if certified and abs(obj - ref) > tol:
            flags.append(f"OBJ-MISMATCH(certified {obj} vs opt {ref})")
        # Dual bound: checked at every status. Minimize -> bound must not exceed the
        # reference; maximize -> the engine reports in model units, so the bound must
        # not fall below it.
        if bound is not None:
            if not maximize and bound > ref + tol:
                flags.append(f"BOUND-ABOVE-OPTIMUM({bound} > {ref})")
            if maximize and bound < ref - tol:
                flags.append(f"BOUND-BELOW-OPTIMUM({bound} < {ref})")
        note = "" if res.status == "optimal" else "  (incumbent, not a certificate)"
        print(
            f"{name:12s} {res.status:11s} obj={obj!s:22s} bound={bound!s:22s} "
            f"{marker}={ref:<14g} {' '.join(flags) if flags else 'OK' + note}"
        )
        if flags:
            violations.append((name, flags))

    print(f"\ninstances checked against a reference: {checked}")
    if no_oracle:
        print(f"no reference entry (excluded from the verdict): {', '.join(no_oracle)}")
    print(f"violations: {len(violations)}")
    for name, flags in violations:
        print(f"  {name}: {'; '.join(flags)}")
    if not checked:
        print("CHECKED NOTHING", file=sys.stderr)
        return 2
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
