#!/usr/bin/env python
"""End-to-end smoke test for discopt as a GAMS solver.

For every model in ``python/tests/data/gams/manifest.json`` this script runs the
``.gms`` through a real GAMS system with the solver forced to discopt, reads back
the objective value and the GAMS model/solve status, and checks them against the
known optimum.  It is the missing end-to-end check that the unit tests cannot do
without a GAMS installation.

Prerequisites
-------------
* A GAMS system on ``PATH`` (or pass ``--gams /path/to/gams``).
* discopt registered as a GAMS solver and importable, e.g. ``make gams-install``
  (which runs ``pip install -e ".[gams]"`` and ``discopt gams-register``).

Usage
-----
    python scripts/verify_gams_link.py
    python scripts/verify_gams_link.py --gams /opt/gams/gams --solver discopt
    python scripts/verify_gams_link.py --keep   # keep the scratch dir for debugging

Exit code is 0 only if every model solves to its known optimum.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "python" / "tests" / "data" / "gams"

# GAMS model-status codes considered a usable optimum/solution.
OK_MODEL_STATS = {1, 2, 8}  # Optimal, LocallyOptimal, IntegerSolution


def _epilogue(model_name: str, objective_var: str, result_path: Path) -> str:
    """GAMS put-statement epilogue that records obj + status to a text file."""
    res = str(result_path).replace("\\", "/")
    return (
        f"\nfile discoptres / '{res}' /;\n"
        f"discoptres.nd = 12;\n"
        f"put discoptres;\n"
        f"put {objective_var}.l /;\n"
        f"put {model_name}.modelStat /;\n"
        f"put {model_name}.solveStat /;\n"
        f"putclose discoptres;\n"
    )


def _run_one(entry: dict, gams: str, solver: str, scratch: Path) -> tuple[bool, str]:
    src = DATA_DIR / entry["file"]
    if not src.exists():
        return False, f"missing model file {src}"

    work = scratch / entry["file"].replace(".gms", "")
    work.mkdir(parents=True, exist_ok=True)
    result_file = work / "result.txt"

    # Force the discopt solver for this model's type and append the epilogue.
    gms_text = src.read_text() + _epilogue(entry["model_name"], entry["objective_var"], result_file)
    run_gms = work / src.name
    run_gms.write_text(gms_text)

    cmd = [
        gams,
        str(run_gms),
        f"{entry['model_type']}={solver}",
        "lo=2",
        "limrow=0",
        "limcol=0",
        "solprint=off",
        f"curdir={work}",
        f"o={work / 'out.lst'}",
    ]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if not result_file.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        return False, "GAMS produced no result file. Tail:\n      " + "\n      ".join(tail)

    nums = [float(t) for t in re.findall(r"[-+0-9.eE]+", result_file.read_text())]
    if len(nums) < 3:
        return False, f"could not parse result file: {result_file.read_text()!r}"
    obj, model_stat, solve_stat = nums[0], int(nums[1]), int(nums[2])

    if solve_stat != 1:
        return False, f"solveStat={solve_stat} (expected 1=Normal)"
    if model_stat not in OK_MODEL_STATS:
        return False, f"modelStat={model_stat} (expected one of {sorted(OK_MODEL_STATS)})"
    expected = entry["objective"]
    tol = entry.get("tol", 1e-4)
    if abs(obj - expected) > tol + 1e-4 * abs(expected):
        return False, f"objective {obj:.6g} != expected {expected:.6g} (tol {tol:g})"
    return True, f"obj={obj:.6g}, modelStat={model_stat}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gams", default="gams", help="Path to the gams executable.")
    parser.add_argument("--solver", default="discopt", help="GAMS solver name to force.")
    parser.add_argument(
        "--manifest", default=str(DATA_DIR / "manifest.json"), help="Path to the model manifest."
    )
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    args = parser.parse_args(argv)

    if shutil.which(args.gams) is None and not Path(args.gams).exists():
        print(f"error: GAMS executable not found: {args.gams!r}", file=sys.stderr)
        print("       Install GAMS and/or pass --gams /path/to/gams.", file=sys.stderr)
        return 2

    models = json.loads(Path(args.manifest).read_text())["models"]
    scratch = Path(tempfile.mkdtemp(prefix="discopt-gams-verify-"))
    print(f"Running {len(models)} models through GAMS solver={args.solver!r}\n")

    failures = 0
    for entry in models:
        ok, detail = _run_one(entry, args.gams, args.solver, scratch)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {entry['file']:<22} {entry['model_type']:<6} {detail}")
        if not ok:
            failures += 1

    if args.keep:
        print(f"\nScratch kept at {scratch}")
    else:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{len(models) - failures}/{len(models)} models passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
