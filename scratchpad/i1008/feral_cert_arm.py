"""Acceptance panel for the feral bump: one arm of the arm-vs-arm cert diff.

`crates/discopt-core/Cargo.toml`'s pin comment sets the regime for any feral bump
that moves LU arithmetic (CLAUDE.md §5, bound-neutral): `node_count` and the
certified `objective` must be EXACTLY unchanged on a certifying panel — any
drift, even an apparent improvement, means the bump is wrong. The same comment
records why the arms are diffed against each other rather than against the
committed `cert-baseline.jsonl`: that reference is stale on a POUNCE-equipped
machine (18 violations on an UNMODIFIED tree).

This script runs ONE arm over the 52 certifying instances at the baseline budgets
and writes a jsonl row per instance. `feral_cert_report.py` diffs two arms.

§8: asserts both `discopt.__file__` under the arm's worktree AND a marker string
that discriminates the builds — cargo bakes the dependency's source path, and so
the literal `feral-<version>`, into the extension's panic locations, so the arm
name IS the marker. A stale .so fails here rather than reporting one arm twice.
§6: prints an executed-instance count and exits non-zero if it is zero.
§7: nothing is caught.
§10: per-instance progress, unbuffered.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import discopt

WT = os.environ["WT"]
ARM = os.environ["FERAL_ARM"]  # a crates.io version, e.g. "0.15.1" or "0.16.0"
OUT = os.environ["FERAL_OUT"]

assert discopt.__file__.startswith(WT), discopt.__file__
import discopt._rust as _rust  # noqa: E402

assert _rust.__file__.startswith(WT), _rust.__file__
blob = open(_rust.__file__, "rb").read()
want = f"feral-{ARM}".encode()
other = b"feral-0.16.0" if ARM != "0.16.0" else b"feral-0.15.1"
assert want in blob, f"arm {ARM}: marker {want!r} absent from {_rust.__file__}"
assert other not in blob, f"arm {ARM}: foreign marker {other!r} present in {_rust.__file__}"
print(f"# arm={ARM} marker={want!r} present, {other!r} absent so={_rust.__file__}", flush=True)

_BENCH_ROOT = Path(WT) / "discopt_benchmarks"
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, WT)

from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from scripts.gen_cert_baseline import _instance_budgets  # noqa: E402
from utils.cert_neutrality import load_baseline  # noqa: E402

baseline = load_baseline(Path(WT) / "docs" / "dev" / "data" / "cert-baseline.jsonl")
budgets = _instance_budgets(60.0)
solver = SolverConfig(name="discopt", command="", solver_type="internal")

fh = open(OUT, "w")
n = 0
names = sorted(baseline)
for i, name in enumerate(names, 1):
    cfg = BenchmarkConfig(
        suite_name="feral-bump-cert",
        time_limit=int(budgets.get(name, 60)),
        num_runs=1,
        solvers=[solver],
    )
    res = BenchmarkRunner(cfg)._run_discopt(solver, name, 0)
    row = res.to_dict()
    row["arm"] = ARM
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    n += 1
    print(
        f"  [{i}/{len(names)}] {name:20s} {res.status.value:10s} "
        f"nodes={res.node_count} obj={res.objective} ({res.wall_time:.2f}s)",
        flush=True,
    )

print(f"\nexecuted: instances={n}", flush=True)
sys.exit(0 if n else 1)
