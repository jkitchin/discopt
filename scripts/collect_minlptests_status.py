#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
TEST_FILE = REPO_ROOT / "python" / "tests" / "test_minlptests.py"
ALPINE_HELPER = REPO_ROOT / "scripts" / "alpine_minlptests_status.jl"


def load_test_module():
    sys.path.insert(0, str(PYTHON_ROOT))
    if "pytest" not in sys.modules:
        class _ParameterSet:
            def __init__(self, value):
                self.values = (value,)

        class _MarkProxy:
            def __getattr__(self, _name):
                def decorator(*args, **kwargs):
                    if args and callable(args[0]) and len(args) == 1 and not kwargs:
                        return args[0]

                    def wrapper(obj):
                        return obj

                    return wrapper

                return decorator

        pytest_stub = types.SimpleNamespace()
        pytest_stub.mark = _MarkProxy()
        pytest_stub.param = lambda value, **_kwargs: _ParameterSet(value)
        pytest_stub.xfail = lambda *, reason="": (_ for _ in ()).throw(
            RuntimeError(f"Unexpected pytest.xfail call: {reason}")
        )
        sys.modules["pytest"] = pytest_stub

    spec = importlib.util.spec_from_file_location("discopt_test_minlptests", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_case(case):
    return case.values[0] if hasattr(case, "values") else case


def case_catalog(
    mod,
    *,
    include_convex: bool,
    per_instance_time_limit: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    if include_convex:
        for raw in mod.NLP_CVX_INSTANCES:
            inst = unwrap_case(raw)
            cases.append(
                {
                    "category": "nlp_cvx",
                    "directory": "nlp-cvx",
                    "symbol": inst.problem_id,
                    "instance": inst,
                    "time_limit": per_instance_time_limit,
                    "gap_tolerance": 1e-6,
                    "expected_status": "optimal",
                }
            )

    for raw in mod.NLP_INSTANCES:
        inst = unwrap_case(raw)
        cases.append(
            {
                "category": "nlp",
                "directory": "nlp",
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": 1e-6,
                "expected_status": "optimal",
            }
        )

    for raw in mod.NLP_MI_INSTANCES:
        inst = unwrap_case(raw)
        cases.append(
            {
                "category": "nlp_mi",
                "directory": "nlp-mi",
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": 1e-6,
                "expected_status": "optimal",
            }
        )

    for raw in mod.INFEASIBLE_INSTANCES:
        inst = unwrap_case(raw)
        directory = "nlp-mi" if inst.problem_id.startswith("nlp_mi_") else "nlp"
        cases.append(
            {
                "category": "infeasible",
                "directory": directory,
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": None,
                "expected_status": "infeasible",
            }
        )

    return cases


def validate_discopt_result(mod, case: dict[str, Any], result) -> None:
    inst = case["instance"]
    if case["expected_status"] == "infeasible":
        mod.assert_infeasible(result, inst.problem_id)
        return

    mod.assert_optimal(result, inst.expected_obj, inst.problem_id)
    if inst.is_convex and getattr(result, "convex_fast_path", False) is not True:
        raise AssertionError(f"[{inst.problem_id}] Expected discopt convex fast path")


def run_discopt_cases(
    mod,
    cases: list[dict[str, Any]],
    *,
    solver_mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for case in cases:
        inst = case["instance"]

        t0 = time.perf_counter()
        result = None
        try:
            model = inst.build_fn()
            solve_kwargs: dict[str, Any] = {"time_limit": case["time_limit"]}
            if solver_mode == "amp":
                solve_kwargs["solver"] = "amp"
                solve_kwargs["nlp_solver"] = "ipm"
                if case["gap_tolerance"] is not None:
                    solve_kwargs["gap_tolerance"] = 1e-3
            elif case["gap_tolerance"] is not None:
                solve_kwargs["gap_tolerance"] = case["gap_tolerance"]
            result = model.solve(**solve_kwargs)
            validate_discopt_result(mod, case, result)
            outcome = "pass"
            note = ""
        except AssertionError as err:
            outcome = "fail"
            note = str(err)
        except Exception as err:  # pragma: no cover - exercised in real benchmark runs
            outcome = "error"
            note = f"{type(err).__name__}: {err}"

        wall_time = time.perf_counter() - t0
        records.append(
            {
                "solver": f"discopt_{solver_mode}",
                "category": case["category"],
                "problem_id": inst.problem_id,
                "outcome": outcome,
                "status": getattr(result, "status", None),
                "objective": getattr(result, "objective", None),
                "wall_time_sec": wall_time,
                "note": note,
            }
        )

    return records


def run_alpine_cases(
    cases: list[dict[str, Any]],
    alpine_project: Path,
    minlptests_path: Path,
    julia_bin: str,
    julia_channel: str,
    per_instance_time_limit: float,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="alpine-minlptests-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        request_path = tmpdir_path / "request.tsv"
        output_path = tmpdir_path / "results.jsonl"

        request_lines = [
            "\t".join((case["instance"].problem_id, case["category"], case["symbol"]))
            for case in cases
        ]
        request_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")

        cmd = [julia_bin]
        if julia_channel:
            cmd.append(julia_channel)
        cmd.append(f"--project={alpine_project}")
        cmd.extend(
            [
                str(ALPINE_HELPER),
                str(request_path),
                str(output_path),
                str(minlptests_path),
                str(per_instance_time_limit),
            ]
        )
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        records: list[dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    payload["solver"] = "alpine"
                    records.append(payload)
        return records


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        summary[record["category"]][record["outcome"]] += 1
        summary[record["category"]]["total"] += 1
    return {category: dict(counter) for category, counter in summary.items()}


def compare_outcomes(
    discopt_records: list[dict[str, Any]],
    alpine_records: list[dict[str, Any]],
) -> dict[str, int]:
    alpine_by_problem = {record["problem_id"]: record for record in alpine_records}
    counts = Counter(
        {
            "both_pass": 0,
            "discopt_only_pass": 0,
            "alpine_only_pass": 0,
            "both_fail": 0,
        }
    )
    for discopt in discopt_records:
        alpine = alpine_by_problem.get(discopt["problem_id"])
        discopt_pass = discopt["outcome"] == "pass"
        alpine_pass = alpine is not None and alpine["outcome"] == "pass"
        if discopt_pass and alpine_pass:
            counts["both_pass"] += 1
        elif discopt_pass:
            counts["discopt_only_pass"] += 1
        elif alpine_pass:
            counts["alpine_only_pass"] += 1
        else:
            counts["both_fail"] += 1
    return dict(counts)


def build_markdown(
    discopt_records: list[dict[str, Any]],
    alpine_records: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# MINLPTests Status",
        "",
        (
            "Generated from the discopt AMP run and the matching Alpine.jl run "
            "on the same translated MINLPTests problem IDs."
        ),
        "",
    ]

    def add_summary_table(title: str, records: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Category | Pass | Fail | Error | Total |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        summary = summarize(records)
        for category in ("nlp", "nlp_mi", "infeasible", "nlp_cvx"):
            row = summary.get(category, {})
            lines.append(
                (
                    f"| {category} | {row.get('pass', 0)} | {row.get('fail', 0)} | "
                    f"{row.get('error', 0)} | {row.get('total', 0)} |"
                )
            )
        lines.append("")

    add_summary_table("discopt", discopt_records)
    if alpine_records:
        add_summary_table("Alpine.jl", alpine_records)

        comparison = compare_outcomes(discopt_records, alpine_records)
        lines.extend(
            [
                "## Head-to-Head Comparison",
                "",
                "| Outcome split | Count |",
                "| --- | ---: |",
                f"| both_pass | {comparison['both_pass']} |",
                f"| discopt_only_pass | {comparison['discopt_only_pass']} |",
                f"| alpine_only_pass | {comparison['alpine_only_pass']} |",
                f"| both_fail | {comparison['both_fail']} |",
                "",
            ]
        )

        alpine_failure_modes = Counter(
            record["note"] for record in alpine_records if record["outcome"] != "pass"
        )
        lines.extend(
            [
                "## Alpine Failure Modes",
                "",
                "| Failure mode | Count |",
                "| --- | ---: |",
            ]
        )
        for note, count in alpine_failure_modes.most_common():
            escaped_note = note.replace("|", "\\|")
            lines.append(f"| {escaped_note} | {count} |")
        lines.append("")

        alpine_by_problem = {record["problem_id"]: record for record in alpine_records}
        gap_rows = []
        for record in discopt_records:
            if record["outcome"] not in {"fail", "error"}:
                continue
            alpine = alpine_by_problem.get(record["problem_id"])
            if alpine is None:
                continue
            if alpine["outcome"] == "pass":
                gap_rows.append((record, alpine))

        lines.extend(["## Discopt Gaps Where Alpine Passes", ""])
        if gap_rows:
            lines.extend(
                [
                    "| Problem | Category | discopt | Alpine | Note |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for discopt, alpine in gap_rows:
                note = discopt["note"].replace("|", "\\|")
                lines.append(
                    (
                        f"| {discopt['problem_id']} | {discopt['category']} | "
                        f"{discopt['outcome']} | {alpine['outcome']} | {note} |"
                    )
                )
        else:
            lines.append("No discopt-only gaps were found in cases that Alpine passes.")
        lines.append("")

    lines.extend(
        [
            "## Discopt Failures",
            "",
            "| Problem | Category | Outcome | Status | Note |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    shown = False
    for record in discopt_records:
        if record["outcome"] == "pass":
            continue
        shown = True
        status = "" if record["status"] is None else str(record["status"])
        note = record["note"].replace("|", "\\|")
        lines.append(
            (
                f"| {record['problem_id']} | {record['category']} | "
                f"{record['outcome']} | {status} | {note} |"
            )
        )
    if not shown:
        lines.append("| none | - | - | - | - |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run discopt and Alpine MINLPTests status sweeps."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write the merged JSON result payload.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to write a Markdown summary table.",
    )
    parser.add_argument(
        "--skip-alpine",
        action="store_true",
        help="Run only the translated discopt suite and skip the Alpine comparison.",
    )
    parser.add_argument(
        "--include-convex",
        action="store_true",
        help=(
            "Include the convex nlp-cvx cases. By default the runner uses the "
            "nonconvex Phase 6 scope only."
        ),
    )
    parser.add_argument(
        "--per-instance-time-limit",
        type=float,
        default=300.0,
        help="Wall-clock time limit in seconds for each discopt or Alpine case.",
    )
    parser.add_argument(
        "--discopt-mode",
        choices=("amp", "default"),
        default="amp",
        help=(
            "Solve the translated MINLPTests cases with the AMP solver or the "
            "default solve path."
        ),
    )
    parser.add_argument(
        "--alpine-project",
        type=Path,
        default=REPO_ROOT.parent / "Alpine.jl",
        help="Path to the local Alpine.jl checkout.",
    )
    parser.add_argument(
        "--minlptests-path",
        type=Path,
        default=REPO_ROOT.parent / "MINLPTests.jl",
        help="Path to the local MINLPTests.jl checkout.",
    )
    parser.add_argument("--julia-bin", default="julia", help="Julia executable to use.")
    parser.add_argument(
        "--julia-channel",
        default="+release",
        help=(
            "Optional juliaup channel argument, e.g. '+release'. Use an empty "
            "string to disable it."
        ),
    )
    args = parser.parse_args()

    mod = load_test_module()
    cases = case_catalog(
        mod,
        include_convex=args.include_convex,
        per_instance_time_limit=args.per_instance_time_limit,
    )
    discopt_records = run_discopt_cases(mod, cases, solver_mode=args.discopt_mode)
    alpine_records: list[dict[str, Any]] = []

    if not args.skip_alpine:
        alpine_records = run_alpine_cases(
            cases,
            args.alpine_project.resolve(),
            args.minlptests_path.resolve(),
            args.julia_bin,
            args.julia_channel,
            args.per_instance_time_limit,
        )

    payload = {
        "discopt": discopt_records,
        "alpine": alpine_records,
        "discopt_summary": summarize(discopt_records),
        "alpine_summary": summarize(alpine_records),
        "comparison_summary": compare_outcomes(discopt_records, alpine_records),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            build_markdown(discopt_records, alpine_records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
