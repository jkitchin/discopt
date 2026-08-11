"""Incremental-McCormick admission sweep (issue #861).

Builds :class:`IncrementalMcCormickLP` over the in-repo MINLPLib corpus exactly
as the default relaxer does and reports, per instance, whether the structure was
ADMITTED (``ok=True`` -> the ~30x-faster patched node path serves it) or DECLINED
(``ok=False`` -> every node falls back to the trusted cold build), with the
stored ``decline_reason``.

This is the progress meter for #861: every task's PR quotes a before/after run.
The admitted count must be monotone non-decreasing, and an instance that was
admitted before must never become declined (see the plan doc's §0.5 — such a flip
is either a bug in the change or a real patch/cold divergence the old synthetic
validation boxes never exercised; investigate, never paper over).

Declining is always SOUND — the caller falls back to the per-node cold build — so
this script measures *coverage and speed*, never correctness. Bound-neutrality is
gated separately by ``IncrementalMcCormickLP._validate`` and by the panels in §4
of ``docs/dev/issue-861-incremental-admission-plan.md``.

Usage::

    python -u discopt_benchmarks/scripts/incremental_admission_sweep.py \\
        --out discopt_benchmarks/results/issue861_admission_<stamp>.json
    # compare against a previous run (fails on any admitted->declined flip):
    python -u ... --out new.json --baseline old.json

Exits non-zero when no instance was measured (a sweep that measures nothing must
never read as a pass — CLAUDE.md rule 6), when the corpus directory is missing,
or when ``--baseline`` shows a regression.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

# The in-repo MINLPLib corpus (81 instances) the issue's sweep is defined over.
DEFAULT_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
    "tests",
    "data",
    "minlplib",
)


def bucket(reason: str | None) -> str:
    """Coarse decline bucket, matching the issue's triage table."""
    if not reason:
        return "admitted"
    for needle, label in (
        ("bounds mismatch", "bounds mismatch"),
        ("column-count mismatch", "column-count mismatch"),
        ("rows, expected 4", "envelope row count != 4"),
        ("no valid bound", "no valid bound / no rows"),
        ("odd power on a root box spanning zero", "odd power on straddling root"),
        ("row-set mismatch", "row-set mismatch"),
        ("objective offset is box-dependent", "box-dependent objective offset"),
        ("exceeds budget", "structure too large"),
        ("deadline", "deadline spent"),
    ):
        if needle in reason:
            return label
    return reason.split(":")[0].strip() or "other"


def measure(path: str) -> dict:
    """Build the structure for one instance; return its result row."""
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import from_nl

    t0 = time.perf_counter()
    row: dict = {}
    # NOTE: no blanket try/except around the measurement itself — an instrument
    # that swallows exceptions turns "this path is broken" into "this path is
    # fine" (CLAUDE.md rule 7). Only the *model load* is guarded, and it records
    # the failure explicitly rather than silently counting as a decline.
    try:
        model = from_nl(path)
    except Exception as exc:  # corpus/parse problem — report, don't hide
        return {
            "admitted": False,
            "load_error": f"{type(exc).__name__}: {exc}",
            "reason": None,
            "bucket": "load error",
            "secs": round(time.perf_counter() - t0, 3),
        }
    lb = np.array([float(np.min(v.lb)) for v in model._variables])
    ub = np.array([float(np.max(v.ub)) for v in model._variables])
    row["n"] = int(lb.size)
    row["finite_root"] = bool(np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)))
    terms = classify_nonlinear_terms(model)
    # deadline=None: measure structural admission, never a budget artifact (#844).
    inc = IncrementalMcCormickLP(model, terms, deadline=None)
    row["admitted"] = bool(inc.ok)
    row["reason"] = inc.decline_reason
    row["bucket"] = bucket(inc.decline_reason)
    if inc.ok:
        row["ncol"] = int(inc.ncol)
        row["n_bilinear"] = len(inc.bilinear)
        row["n_monomial"] = len(inc.monomial)
        row["n_affine_square"] = len(inc.affine_square)
    row["secs"] = round(time.perf_counter() - t0, 3)
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="directory of .nl instances")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--baseline", default=None, help="compare against a previous run's JSON")
    ap.add_argument("--only", default=None, help="comma-separated instance names (debug)")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.corpus, "*.nl")))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-3] in want]
    if not paths:
        print(f"ERROR: no .nl instances under {args.corpus}", file=sys.stderr)
        return 2

    results: dict[str, dict] = {}
    print(f"sweeping {len(paths)} instances from {args.corpus}", flush=True)
    for k, path in enumerate(paths, 1):
        name = os.path.basename(path)[:-3]
        row = measure(path)
        results[name] = row
        tag = "ADMIT  " if row["admitted"] else "DECLINE"
        note = "" if row["admitted"] else f"  {row['bucket']}"
        print(f"[{k:3d}/{len(paths)}] {name:24s} {tag} ({row['secs']:6.2f}s){note}", flush=True)

    admitted = sum(1 for r in results.values() if r["admitted"])
    declined = len(results) - admitted
    print(f"\n=== admitted {admitted}/{len(results)}   declined {declined} ===")
    hist = Counter(r["bucket"] for r in results.values() if not r["admitted"])
    for label, count in hist.most_common():
        names = sorted(n for n, r in results.items() if not r["admitted"] and r["bucket"] == label)
        print(f"  {count:3d}  {label:32s} {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(
                {"admitted": admitted, "declined": declined, "instances": results}, fh, indent=1
            )
        print(f"\nwrote {args.out}")

    status = 0
    if args.baseline:
        with open(args.baseline) as fh:
            base = json.load(fh)
        base_inst = base.get("instances", base)
        regressed = sorted(
            n
            for n, r in base_inst.items()
            if r.get("admitted") and not results.get(n, {}).get("admitted", False)
        )
        gained = sorted(
            n
            for n, r in results.items()
            if r["admitted"] and not base_inst.get(n, {}).get("admitted", False)
        )
        print(f"\nvs baseline {args.baseline}: {base.get('admitted', '?')} -> {admitted} admitted")
        if gained:
            print(f"  NEWLY ADMITTED ({len(gained)}): {', '.join(gained)}")
        if regressed:
            print(f"  REGRESSED ({len(regressed)}): {', '.join(regressed)}")
            for n in regressed:
                print(f"    {n}: {results.get(n, {}).get('reason')}")
            status = 1

    # Executed-measurement count: a sweep that measured nothing must not read as
    # a pass (CLAUDE.md rule 6).
    print(f"\nexecuted structure builds: {len(results)}")
    if not results:
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
