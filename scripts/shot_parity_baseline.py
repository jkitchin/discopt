#!/usr/bin/env python
"""Build the SHOT parity baseline for the MIP-NLP port roadmap.

The command is intentionally read-only with respect to solver behavior. It
solves a curated fixture set with discopt, exports AMPL ``.nl`` files for SHOT,
and runs SHOT when a built executable is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/discopt-jax-cache")

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import discopt.modeling as dm  # noqa: E402
from discopt.export.nl import to_nl  # noqa: E402
from discopt.result_io import serialize_result  # noqa: E402


@dataclass(frozen=True)
class Fixture:
    key: str
    title: str
    problem_class: str
    convexity: str
    intent: str
    builder: Callable[[], dm.Model]
    solve_options: dict[str, object]
    expected_certification: str


def _convex_nlp() -> dm.Model:
    m = dm.Model("shot_convex_nlp_quadratic")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x >= 0.25)
    m.minimize((x - 1.5) ** 2)
    return m


def _convex_minlp() -> dm.Model:
    m = dm.Model("shot_convex_minlp_onoff")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.binary("y")
    m.subject_to(x <= 0.75 + 2.5 * y)
    m.subject_to(x >= 0.25)
    m.minimize((x - 2.0) ** 2 + 0.05 * y)
    return m


def _miqp() -> dm.Model:
    m = dm.Model("shot_miqp_binary_quadratic")
    x = m.continuous("x", lb=0.0, ub=3.0)
    y = m.binary("y")
    m.subject_to(x >= 0.25 + y)
    m.minimize((x - 1.75) ** 2 + 0.2 * y)
    return m


def _nonconvex_fp() -> dm.Model:
    m = dm.Model("shot_nonconvex_fp_uncertified")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.binary("y")
    m.minimize(x * y - 2.0 * x)
    return m


def fixture_specs() -> list[Fixture]:
    return [
        Fixture(
            key="convex_nlp",
            title="Convex continuous NLP",
            problem_class="NLP",
            convexity="convex",
            intent="continuous convex NLP routing and KKT-certified objective",
            builder=_convex_nlp,
            solve_options={"time_limit": 30},
            expected_certification="global",
        ),
        Fixture(
            key="convex_minlp",
            title="Convex binary MINLP",
            problem_class="MINLP",
            convexity="convex",
            intent="OA baseline with SHOT profile trace and fixed-integer NLP calls",
            builder=_convex_minlp,
            solve_options={
                "solver": "mip-nlp",
                "mip_nlp_method": "oa",
                "mip_nlp_profile": "shot",
                "time_limit": 30,
                "max_nodes": 100,
            },
            expected_certification="global",
        ),
        Fixture(
            key="miqp",
            title="Convex MIQP-style model",
            problem_class="MIQP",
            convexity="convex",
            intent="quadratic objective with one binary decision and direct routing pressure",
            builder=_miqp,
            solve_options={"time_limit": 30, "max_nodes": 100},
            expected_certification="global",
        ),
        Fixture(
            key="nonconvex_fp",
            title="Nonconvex heuristic MINLP",
            problem_class="MINLP",
            convexity="nonconvex",
            intent="feasibility-pump incumbent with explicit uncertified bound caveat",
            builder=_nonconvex_fp,
            solve_options={
                "solver": "mip-nlp",
                "mip_nlp_method": "fp",
                "time_limit": 30,
                "max_nodes": 20,
            },
            expected_certification="heuristic",
        ),
    ]


def _json_float(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(number):
        return number
    return None


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def summarize_discopt_result(fixture: Fixture, result, elapsed: float) -> dict[str, object]:
    payload = serialize_result(result)
    trace = payload.get("mip_nlp_trace") or {}
    bound_validity = trace.get("bound_validity")
    if bound_validity is None:
        bound_validity = "global" if payload.get("gap_certified") else "heuristic"
    caveat = "none"
    if not payload.get("gap_certified"):
        caveat = "reported incumbent is not supported by a certified finite dual bound"
    return {
        "backend": "discopt",
        "available": True,
        "status": payload.get("status"),
        "objective": _json_float(payload.get("objective")),
        "bound": _json_float(payload.get("bound")),
        "gap": _json_float(payload.get("gap")),
        "wall_time_seconds": round(float(payload.get("wall_time") or elapsed), 6),
        "elapsed_seconds": round(elapsed, 6),
        "node_count": payload.get("node_count"),
        "mip_count": payload.get("mip_count"),
        "gap_certified": payload.get("gap_certified"),
        "bound_validity": bound_validity,
        "certification_caveat": caveat,
        "convex_fast_path": payload.get("convex_fast_path"),
        "nlp_bb": payload.get("nlp_bb"),
        "solution": payload.get("x"),
        "trace_summary": trace.get("summary"),
        "solve_options": fixture.solve_options,
    }


def run_discopt(fixture: Fixture) -> dict[str, object]:
    model = fixture.builder()
    start = time.perf_counter()
    try:
        result = model.solve(**fixture.solve_options)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "backend": "discopt",
            "available": True,
            "status": "error",
            "objective": None,
            "bound": None,
            "gap": None,
            "wall_time_seconds": round(elapsed, 6),
            "gap_certified": False,
            "bound_validity": "unknown",
            "certification_caveat": f"{type(exc).__name__}: {exc}",
            "solve_options": fixture.solve_options,
        }
    return summarize_discopt_result(fixture, result, time.perf_counter() - start)


def discover_shot_executable(explicit: str | None, shot_root: Path) -> str | None:
    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    env_exe = os.environ.get("SHOT_EXECUTABLE")
    if env_exe:
        candidates.append(env_exe)
    candidates.extend(
        [
            shot_root / "build" / "SHOT",
            shot_root / "build" / "src" / "SHOT",
            shot_root / "build" / "bin" / "SHOT",
            shot_root / "bin" / "SHOT",
        ]
    )
    on_path = shutil.which("SHOT")
    if on_path:
        candidates.append(on_path)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def unavailable_shot_result(reason: str, shot_root: Path) -> dict[str, object]:
    return {
        "backend": "SHOT",
        "available": False,
        "status": "unavailable",
        "objective": None,
        "bound": None,
        "gap": None,
        "wall_time_seconds": 0.0,
        "gap_certified": False,
        "bound_validity": "not_run",
        "certification_caveat": reason,
        "shot_root": str(shot_root),
    }


def _parse_shot_metric(patterns: Iterable[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _parse_shot_status(text: str, returncode: int) -> str:
    if returncode != 0:
        return "error"
    lowered = text.lower()
    if "optimal" in lowered:
        return "optimal"
    if "infeasible" in lowered:
        return "infeasible"
    if "time limit" in lowered or "timelimit" in lowered:
        return "time_limit"
    if "solution" in lowered:
        return "feasible"
    return "completed"


def run_shot(
    fixture: Fixture,
    executable: str | None,
    shot_root: Path,
    workdir: Path,
) -> dict[str, object]:
    if executable is None:
        return unavailable_shot_result(
            "No built SHOT executable found. Set SHOT_EXECUTABLE or build the local checkout.",
            shot_root,
        )

    model = fixture.builder()
    nl_path = workdir / f"{fixture.key}.nl"
    osrl_path = workdir / f"{fixture.key}.osrl"
    trace_path = workdir / f"{fixture.key}.trc"
    to_nl(model, nl_path)

    time_limit = int(float(fixture.solve_options.get("time_limit", 30)))
    cmd = [
        executable,
        str(nl_path),
        "--AMPL",
        f"--timelimit={time_limit}",
        "--osrl",
        str(osrl_path),
        "--trc",
        str(trace_path),
        "Console.LogLevel=0",
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        capture_output=True,
        timeout=time_limit + 30,
        check=False,
    )
    elapsed = time.perf_counter() - start
    combined = "\n".join([completed.stdout, completed.stderr])
    if osrl_path.exists():
        combined = "\n".join([combined, osrl_path.read_text(errors="replace")])

    objective = _parse_shot_metric(
        [
            r"primal bound\s*[:=]\s*([-+0-9.eE]+)",
            r"objective value\s*[:=]?\s*([-+0-9.eE]+)",
            r"<obj[^>]*>\s*([-+0-9.eE]+)\s*</obj>",
        ],
        combined,
    )
    bound = _parse_shot_metric(
        [
            r"dual bound\s*[:=]\s*([-+0-9.eE]+)",
            r"lower bound\s*[:=]\s*([-+0-9.eE]+)",
        ],
        combined,
    )
    gap = _parse_shot_metric(
        [
            r"absolute objective gap\s*[:=]\s*([-+0-9.eE]+)",
            r"relative objective gap\s*[:=]\s*([-+0-9.eE]+)",
        ],
        combined,
    )
    return {
        "backend": "SHOT",
        "available": True,
        "status": _parse_shot_status(combined, completed.returncode),
        "objective": _json_float(objective),
        "bound": _json_float(bound),
        "gap": _json_float(gap),
        "wall_time_seconds": round(elapsed, 6),
        "gap_certified": None,
        "bound_validity": "reported_by_shot",
        "certification_caveat": (
            "parsed from SHOT stdout/OSrL; inspect OSrL for authoritative details"
        ),
        "returncode": completed.returncode,
        "command": cmd,
        "nl_file": str(nl_path),
        "osrl_file": str(osrl_path) if osrl_path.exists() else None,
        "trace_file": str(trace_path) if trace_path.exists() else None,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def collect_baseline(
    fixture_keys: set[str] | None = None,
    *,
    include_shot: bool = True,
    shot_executable: str | None = None,
    shot_root: Path = Path("/home/bernalde/repos/SHOT"),
    workdir: Path | None = None,
) -> dict[str, object]:
    specs = fixture_specs()
    if fixture_keys is not None:
        unknown = sorted(fixture_keys - {fixture.key for fixture in specs})
        if unknown:
            raise ValueError(f"unknown fixture(s): {', '.join(unknown)}")
        specs = [fixture for fixture in specs if fixture.key in fixture_keys]

    shot_exe = discover_shot_executable(shot_executable, shot_root) if include_shot else None
    own_tempdir = None
    if workdir is None:
        own_tempdir = tempfile.TemporaryDirectory(prefix="discopt-shot-baseline-")
        workdir = Path(own_tempdir.name)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        fixtures = []
        for fixture in specs:
            results = [run_discopt(fixture)]
            if include_shot:
                results.append(run_shot(fixture, shot_exe, shot_root, workdir))
            fixtures.append(
                {
                    "key": fixture.key,
                    "title": fixture.title,
                    "problem_class": fixture.problem_class,
                    "convexity": fixture.convexity,
                    "intent": fixture.intent,
                    "expected_certification": fixture.expected_certification,
                    "results": results,
                }
            )
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "shot_root": str(shot_root),
            "shot_executable": shot_exe,
            "workdir": str(workdir),
            "workdir_persistent": own_tempdir is None,
            "fixtures": fixtures,
        }
    finally:
        if own_tempdir is not None:
            own_tempdir.cleanup()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/dev/data/shot-parity-baseline.json"),
        help="JSON baseline path.",
    )
    parser.add_argument(
        "--fixtures",
        default=None,
        help="Comma-separated fixture keys to run. Defaults to all fixtures.",
    )
    parser.add_argument(
        "--shot-executable",
        default=None,
        help="Path to a built SHOT executable. Also honored via SHOT_EXECUTABLE.",
    )
    parser.add_argument(
        "--shot-root",
        type=Path,
        default=Path("/home/bernalde/repos/SHOT"),
        help="Local SHOT checkout used for discovery and audit metadata.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory for exported .nl, OSrL, and trace files.",
    )
    parser.add_argument(
        "--discopt-only",
        action="store_true",
        help="Skip SHOT execution and only record discopt results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    keys = None
    if args.fixtures:
        keys = {key.strip() for key in args.fixtures.split(",") if key.strip()}
    baseline = collect_baseline(
        keys,
        include_shot=not args.discopt_only,
        shot_executable=args.shot_executable,
        shot_root=args.shot_root,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
