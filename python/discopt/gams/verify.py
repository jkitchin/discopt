"""End-to-end verification of discopt as a GAMS solver.

For every model in the packaged corpus (:func:`data_dir` / ``manifest.json``)
this runs the ``.gms`` through a real GAMS system with the solver forced to
discopt, reads back the objective and the GAMS model/solve status via a
``put`` epilogue, and checks them against the known optimum. It is the
end-to-end check the unit tests cannot do without a GAMS installation.

Prerequisites: a GAMS system on ``PATH`` (or ``--gams``), and discopt registered
and importable (``pip install "discopt[gams]"`` + ``discopt gams-register``).

Run it via ``discopt gams-verify`` or ``python -m discopt.gams.verify``.
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

# GAMS model-status codes considered a usable optimum/solution.
OK_MODEL_STATS = {1, 2, 8}  # Optimal, LocallyOptimal, IntegerSolution


def data_dir() -> Path:
    """Directory of the packaged ``.gms`` smoke corpus + ``manifest.json``."""
    return Path(__file__).resolve().parent / "data"


def load_manifest(manifest: str | Path | None = None) -> list[dict]:
    """Load the corpus manifest (defaults to the packaged one)."""
    path = Path(manifest) if manifest else data_dir() / "manifest.json"
    return list(json.loads(path.read_text())["models"])


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


def _run_one(entry: dict, gams: str, solver: str, scratch: Path, corpus: Path) -> tuple[bool, str]:
    src = corpus / entry["file"]
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


def verify(
    gams: str = "gams",
    solver: str = "discopt",
    manifest: str | Path | None = None,
    keep: bool = False,
) -> int:
    """Run the corpus through GAMS with ``solver`` forced to discopt.

    Returns a process exit code: 0 if every model solves to its known optimum,
    1 on any mismatch, 2 if the GAMS executable cannot be found.
    """
    if shutil.which(gams) is None and not Path(gams).exists():
        sys.stderr.write(
            f"error: GAMS executable not found: {gams!r}\n"
            "       Install GAMS and/or pass --gams /path/to/gams.\n"
        )
        return 2

    models = load_manifest(manifest)
    corpus = Path(manifest).parent if manifest else data_dir()
    scratch = Path(tempfile.mkdtemp(prefix="discopt-gams-verify-"))
    print(f"Running {len(models)} models through GAMS solver={solver!r}\n")

    failures = 0
    for entry in models:
        ok, detail = _run_one(entry, gams, solver, scratch, corpus)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {entry['file']:<22} {entry['model_type']:<6} {detail}"
        )
        if not ok:
            failures += 1

    if keep:
        print(f"\nScratch kept at {scratch}")
    else:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{len(models) - failures}/{len(models)} models passed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="discopt gams-verify", description=__doc__)
    parser.add_argument("--gams", default="gams", help="Path to the gams executable.")
    parser.add_argument("--solver", default="discopt", help="GAMS solver name to force.")
    parser.add_argument("--manifest", default=None, help="Override the corpus manifest path.")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    args = parser.parse_args(argv)
    return verify(gams=args.gams, solver=args.solver, manifest=args.manifest, keep=args.keep)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
