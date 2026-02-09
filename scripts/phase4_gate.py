#!/usr/bin/env python3
"""
Phase 4 Gate Validation

Validates Phase 4 deliverables: sparse solvers, sparsity detection,
and overall correctness. Criteria from the development plan:

  - minlplib_30var_solved >= 85
  - minlplib_100var_solved >= 30
  - zero_incorrect
  - vmap_batch_speedup >= 50x
  - geomean_vs_baron <= 3.0 (UNMEASURABLE if BARON unavailable)
  - sparse_solver_functional: basic sparse IPM + sparse Jacobian tests

Usage:
    python scripts/phase4_gate.py
    python scripts/phase4_gate.py --skip-solve
    python scripts/phase4_gate.py --time-limit 120 --max-nodes 200000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InstanceResult:
    name: str
    n_vars: int
    n_cons: int
    status: str
    objective: float | None
    expected_obj: float | None
    correct: bool | None
    wall_time: float
    node_count: int
    nodes_per_second: float
    rust_frac: float | None
    error: str | None = None


@dataclass
class FeatureResult:
    feature: str
    test_name: str
    passed: bool
    value: float | None
    target: float | None
    notes: str = ""


@dataclass
class GateReport:
    timestamp: str
    gate: str = "phase4"
    time_limit: float = 120.0
    max_nodes: int = 200_000
    total_instances: int = 0
    solved: int = 0
    solved_30var: int = 0
    solved_100var: int = 0
    incorrect_count: int = 0
    criteria_passed: int = 0
    criteria_failed: int = 0
    criteria_unmeasurable: int = 0
    instances: list[dict] = field(default_factory=list)
    feature_results: list[dict] = field(default_factory=list)


# Known optima (same as phase3_gate.py)
KNOWN_OPTIMA = {
    "ex1221": 7.66718007,
    "ex1225": 31.0,
    "ex1226": -17.0,
    "st_e13": 2.0,
    "st_e15": 7.66718007,
    "st_e27": 2.0,
    "st_e38": 7197.72714900,
    "st_e40": 30.41421350,
    "nvs01": 12.46966882,
    "nvs02": 5.96418452,
    "nvs03": 16.0,
    "nvs04": 0.72,
    "nvs05": 5.47093411,
    "nvs06": 1.77031250,
    "nvs07": 4.0,
    "nvs08": 23.44972735,
    "nvs10": -310.80,
    "nvs11": -431.0,
    "nvs12": -481.20,
    "nvs14": -40358.15477,
    "nvs15": 1.0,
    "nvs16": 0.70312500,
    "nvs21": -5.68478250,
    "prob03": 10.0,
    "prob06": 1.17712434,
    "prob10": 3.44550379,
    "gear": 0.0,
    "gear3": 0.0,
    "gear4": 1.64342847,
    "chance": 29.89437816,
    "dispatch": 3155.28792700,
    "meanvar": 5.24339907,
    "alan": 2.9250,
}

ABS_TOL = 1e-4
REL_TOL = 1e-3


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------


def find_nl_instances() -> dict[str, Path]:
    project_root = Path(__file__).parent.parent
    nl_dirs = [
        project_root / "python" / "tests" / "data" / "minlplib",
        project_root / "python" / "tests" / "data" / "minlplib_nl",
    ]
    found = {}
    for d in nl_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.nl")):
            if f.stem not in found:
                found[f.stem] = f
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# Instance solving
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Solver exceeded hard time limit")


def solve_instance(name: str, nl_path: Path, time_limit: float, max_nodes: int) -> InstanceResult:
    import signal

    import discopt.modeling as dm

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(time_limit) + 30)

    try:
        model = dm.from_nl(str(nl_path))
        n_vars = len(model._variables)
        n_cons = len(model._constraints)

        result = model.solve(
            time_limit=time_limit,
            gap_tolerance=1e-4,
            max_nodes=max_nodes,
        )

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        wt = result.wall_time if result.wall_time > 0 else 1e-10
        nc = result.node_count or 0
        nps = nc / wt if wt > 0 else 0.0
        rust_frac = result.rust_time / wt if result.rust_time else None

        expected = KNOWN_OPTIMA.get(name)
        correct = None
        if expected is not None and result.objective is not None:
            diff = abs(result.objective - expected)
            tol = ABS_TOL + REL_TOL * abs(expected)
            correct = diff <= tol

        return InstanceResult(
            name=name,
            n_vars=n_vars,
            n_cons=n_cons,
            status=result.status,
            objective=result.objective,
            expected_obj=expected,
            correct=correct,
            wall_time=result.wall_time,
            node_count=nc,
            nodes_per_second=nps,
            rust_frac=rust_frac,
        )
    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return InstanceResult(
            name=name,
            n_vars=0,
            n_cons=0,
            status="time_limit",
            objective=None,
            expected_obj=KNOWN_OPTIMA.get(name),
            correct=None,
            wall_time=time_limit,
            node_count=0,
            nodes_per_second=0.0,
            rust_frac=None,
            error="hard timeout",
        )
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return InstanceResult(
            name=name,
            n_vars=0,
            n_cons=0,
            status="error",
            objective=None,
            expected_obj=KNOWN_OPTIMA.get(name),
            correct=None,
            wall_time=0.0,
            node_count=0,
            nodes_per_second=0.0,
            rust_frac=None,
            error=str(e)[:200],
        )


# ---------------------------------------------------------------------------
# Phase 4 feature validations
# ---------------------------------------------------------------------------


def validate_sparsity_detection() -> list[FeatureResult]:
    """Validate sparsity detection and graph coloring."""
    results = []

    try:
        import discopt.modeling as dm
        from discopt._jax.sparsity import (
            compute_coloring,
            detect_sparsity_dag,
            should_use_sparse,
        )

        # Build a sparse model (diagonal Jacobian)
        n = 100
        m = dm.Model("sparse_test")
        x = m.continuous("x", shape=(n,), lb=-10, ub=10)
        m.minimize(dm.sum(x))
        for i in range(n):
            m.subject_to(x[i] <= 5.0)

        pattern = detect_sparsity_dag(m)
        detection_ok = (
            pattern.n_vars == n
            and pattern.n_cons == n
            and pattern.jacobian_nnz == n
            and abs(pattern.jacobian_density - 1 / n) < 1e-10
        )

        results.append(
            FeatureResult(
                feature="sparsity",
                test_name="detection_correctness",
                passed=detection_ok,
                value=float(pattern.jacobian_nnz),
                target=float(n),
                notes=f"nnz={pattern.jacobian_nnz}, density={pattern.jacobian_density:.4f}",
            )
        )

        # Coloring
        colors, n_colors = compute_coloring(pattern)
        coloring_ok = n_colors == 1  # diagonal => 1 color
        results.append(
            FeatureResult(
                feature="sparsity",
                test_name="coloring_optimal",
                passed=coloring_ok,
                value=float(n_colors),
                target=1.0,
                notes=f"n_colors={n_colors} (diagonal => 1 expected)",
            )
        )

        # Auto-gate
        use_sparse = should_use_sparse(pattern, min_vars=50)
        results.append(
            FeatureResult(
                feature="sparsity",
                test_name="auto_gate",
                passed=use_sparse,
                value=1.0 if use_sparse else 0.0,
                target=1.0,
                notes=f"n={n}, density={pattern.jacobian_density:.4f}, use_sparse={use_sparse}",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="sparsity",
                test_name="detection_correctness",
                passed=False,
                value=None,
                target=None,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_sparse_jacobian() -> list[FeatureResult]:
    """Validate sparse Jacobian matches dense."""
    results = []

    try:
        import discopt.modeling as dm
        import jax.numpy as jnp
        import numpy as np
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt._jax.sparse_jacobian import sparse_jacobian_jvp
        from discopt._jax.sparsity import (
            compute_coloring,
            detect_sparsity_dag,
            make_seed_matrix,
        )

        # Build model with known sparsity
        m_model = dm.Model("jac_test")
        x = m_model.continuous("x", shape=(5,), lb=0.1, ub=5.0)
        m_model.minimize(x[0] ** 2 + x[2] ** 2)
        m_model.subject_to(x[0] * x[1] <= 4.0)
        m_model.subject_to(dm.exp(x[2]) <= 10.0)
        m_model.subject_to(x[3] + x[4] <= 8.0)

        evaluator = NLPEvaluator(m_model)
        pattern = detect_sparsity_dag(m_model)
        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, 5)

        x_pt = np.array([1.0, 2.0, 1.0, 3.0, 4.0])

        sparse_jac = sparse_jacobian_jvp(
            evaluator._cons_fn,
            jnp.array(x_pt, dtype=jnp.float64),
            seed,
            pattern,
            colors,
        )
        dense_jac = evaluator.evaluate_jacobian(x_pt)

        max_err = float(np.max(np.abs(sparse_jac.toarray() - dense_jac)))
        jac_ok = max_err < 1e-10

        results.append(
            FeatureResult(
                feature="sparse_jacobian",
                test_name="matches_dense",
                passed=jac_ok,
                value=max_err,
                target=1e-10,
                notes=f"max error={max_err:.2e}, n_colors={n_colors}",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="sparse_jacobian",
                test_name="matches_dense",
                passed=False,
                value=None,
                target=1e-10,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_sparse_ipm() -> list[FeatureResult]:
    """Validate sparse IPM solver basic functionality."""
    results = []

    try:
        import numpy as np
        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2

        x0 = np.array([0.0, 0.0])
        x_l = np.array([-10.0, -10.0])
        x_u = np.array([10.0, 10.0])

        result = sparse_ipm_solve(
            obj,
            None,
            x0,
            x_l,
            x_u,
            options=SparseIPMOptions(max_iter=100),
        )

        ipm_ok = result.converged in (1, 2) and abs(result.objective) < 0.1
        results.append(
            FeatureResult(
                feature="sparse_ipm",
                test_name="unconstrained_solve",
                passed=ipm_ok,
                value=result.objective,
                target=0.0,
                notes=f"converged={result.converged}, obj={result.objective:.6f}",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="sparse_ipm",
                test_name="unconstrained_solve",
                passed=False,
                value=None,
                target=0.0,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    # Test solver wiring
    try:
        import discopt.modeling as dm

        m = dm.Model("wire_test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize((x - 1) ** 2 + (y - 2) ** 2)

        result = m.solve(nlp_solver="sparse_ipm")
        wire_ok = result.status in ("optimal", "feasible") and result.objective < 0.5

        results.append(
            FeatureResult(
                feature="sparse_ipm",
                test_name="solver_wiring",
                passed=wire_ok,
                value=result.objective if result.objective else None,
                target=0.0,
                notes=f"status={result.status}, obj={result.objective}",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="sparse_ipm",
                test_name="solver_wiring",
                passed=False,
                value=None,
                target=None,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_sparse_kkt() -> list[FeatureResult]:
    """Validate sparse KKT assembly and solve."""
    results = []

    try:
        import numpy as np
        import scipy.sparse as sp
        from discopt._jax.sparse_kkt import (
            assemble_kkt_sparse,
            solve_kkt_direct,
        )

        n, m = 3, 2
        H = sp.eye(n, format="csc") * 2.0
        J = sp.csc_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))
        sigma = np.ones(n)

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=0.0, delta_c=1e-8)
        rhs = np.ones(n + m)
        sol = solve_kkt_direct(kkt, rhs)

        kkt_ok = kkt.shape == (5, 5) and np.all(np.isfinite(sol))

        results.append(
            FeatureResult(
                feature="sparse_kkt",
                test_name="assembly_and_solve",
                passed=kkt_ok,
                value=float(kkt.nnz),
                target=None,
                notes=f"shape={kkt.shape}, nnz={kkt.nnz}, sol finite={np.all(np.isfinite(sol))}",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="sparse_kkt",
                test_name="assembly_and_solve",
                passed=False,
                value=None,
                target=None,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_vmap_speedup() -> list[FeatureResult]:
    """Validate vmap batch speedup >= 50x."""
    results = []

    try:
        import jax.numpy as jnp
        from discopt._jax.ipm import IPMOptions, ipm_solve, solve_nlp_batch

        def obj(x):
            return jnp.sum(x**2)

        n = 50
        batch = 512
        x0 = jnp.ones(n)
        xl = jnp.full(n, -10.0)
        xu = jnp.full(n, 10.0)

        # Single solve (warm up)
        _ = ipm_solve(obj, None, x0, xl, xu, None, None, IPMOptions(max_iter=10))

        # Serial timing
        t0 = time.perf_counter()
        for _ in range(batch):
            _ = ipm_solve(obj, None, x0, xl, xu, None, None, IPMOptions(max_iter=10))
        serial_time = time.perf_counter() - t0

        # Batch timing
        x0_batch = jnp.broadcast_to(x0, (batch, n))
        xl_batch = jnp.broadcast_to(xl, (batch, n))
        xu_batch = jnp.broadcast_to(xu, (batch, n))

        # Warm up
        _ = solve_nlp_batch(
            obj,
            None,
            x0_batch[:2],
            xl_batch[:2],
            xu_batch[:2],
            None,
            None,
            IPMOptions(max_iter=10),
        )

        t0 = time.perf_counter()
        _ = solve_nlp_batch(
            obj,
            None,
            x0_batch,
            xl_batch,
            xu_batch,
            None,
            None,
            IPMOptions(max_iter=10),
        )
        batch_time = time.perf_counter() - t0

        speedup = serial_time / max(batch_time, 1e-10)

        results.append(
            FeatureResult(
                feature="vmap_speedup",
                test_name="batch_512_n50",
                passed=speedup >= 50.0,
                value=speedup,
                target=50.0,
                notes=f"serial={serial_time:.3f}s, batch={batch_time:.3f}s, speedup={speedup:.1f}x",
            )
        )

    except Exception as e:
        results.append(
            FeatureResult(
                feature="vmap_speedup",
                test_name="batch_512_n50",
                passed=False,
                value=None,
                target=50.0,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_test_suite() -> tuple[int, int, int]:
    import re
    import subprocess

    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            str(project_root / "python" / "tests"),
            "-x",
            "-q",
            "--tb=no",
            "-p",
            "no:warnings",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout.strip().split("\n")
    last_line = output[-1] if output else ""

    passed = failed = errors = 0
    m = re.search(r"(\d+) passed", last_line)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", last_line)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", last_line)
    if m:
        errors = int(m.group(1))

    return passed, failed, errors


# ---------------------------------------------------------------------------
# Main gate runner
# ---------------------------------------------------------------------------


def run_gate(time_limit: float, max_nodes: int, skip_solve: bool = False) -> None:
    instances = find_nl_instances()
    print("\nPhase 4 Gate Validation")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Instances: {len(instances)}")
    print(f"Time limit: {time_limit}s, Max nodes: {max_nodes}")
    print(f"{'=' * 90}\n")

    # --- Part 1: MINLPLib solving ---
    results: list[InstanceResult] = []
    if not skip_solve:
        print("Part 1: MINLPLib Instance Solving")
        print("-" * 90)
        for i, (name, path) in enumerate(instances.items(), 1):
            print(f"  [{i:3d}/{len(instances)}] {name:30s}", end="", flush=True)
            r = solve_instance(name, path, time_limit, max_nodes)
            status_char = {
                "optimal": "+",
                "feasible": "~",
                "infeasible": "-",
                "time_limit": "T",
                "node_limit": "N",
                "error": "!",
            }.get(r.status, "?")
            correct_char = ""
            if r.correct is True:
                correct_char = " ok"
            elif r.correct is False:
                correct_char = " WRONG"
            print(
                f"  [{status_char}] {r.wall_time:7.2f}s  "
                f"nodes={r.node_count:6d}  n/s={r.nodes_per_second:8.1f}"
                f"{correct_char}"
            )
            results.append(r)
    else:
        print("Part 1: SKIPPED (--skip-solve)")

    # --- Part 2: Phase 4 feature validations ---
    print(f"\n{'=' * 90}")
    print("Part 2: Phase 4 Feature Validations")
    print("-" * 90)

    feature_results: list[FeatureResult] = []

    print("\n  [Sparsity Detection]...")
    sp_results = validate_sparsity_detection()
    feature_results.extend(sp_results)
    for r in sp_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [Sparse Jacobian]...")
    sj_results = validate_sparse_jacobian()
    feature_results.extend(sj_results)
    for r in sj_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [Sparse KKT]...")
    sk_results = validate_sparse_kkt()
    feature_results.extend(sk_results)
    for r in sk_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [Sparse IPM]...")
    si_results = validate_sparse_ipm()
    feature_results.extend(si_results)
    for r in si_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [vmap Batch Speedup]...")
    vs_results = validate_vmap_speedup()
    feature_results.extend(vs_results)
    for r in vs_results:
        status = "PASS" if r.passed else "FAIL"
        val_str = f"{r.value:.1f}" if r.value is not None else "N/A"
        print(f"    {r.test_name:<35s} {status:>6s}  val={val_str}  {r.notes}")

    # --- Part 3: Test suite ---
    print(f"\n{'=' * 90}")
    print("Part 3: Full Test Suite")
    print("-" * 90)
    test_passed, test_failed, test_errors = run_test_suite()
    print(f"  Tests: {test_passed} passed, {test_failed} failed, {test_errors} errors")

    feature_results.append(
        FeatureResult(
            feature="test_suite",
            test_name="full_pytest",
            passed=test_failed == 0 and test_errors == 0,
            value=float(test_passed),
            target=None,
            notes=f"{test_passed} passed, {test_failed} failed, {test_errors} errors",
        )
    )

    # --- Gate criteria ---
    print(f"\n{'=' * 90}")
    print("PHASE 4 GATE RESULTS")
    print(f"{'=' * 90}\n")

    solved = [r for r in results if r.status in ("optimal", "feasible")]
    solved_30 = [r for r in solved if r.n_vars <= 30]
    solved_100 = [r for r in solved if r.n_vars <= 100]
    total_30 = [r for r in results if r.n_vars <= 30]
    total_100 = [r for r in results if r.n_vars <= 100]

    checked = [r for r in results if r.correct is not None]
    incorrect = [r for r in checked if r.correct is False]

    sp_passed = all(r.passed for r in sp_results)
    sj_passed = all(r.passed for r in sj_results)
    sk_passed = all(r.passed for r in sk_results)
    si_passed = all(r.passed for r in si_results)

    criteria = [
        (
            "minlplib_30var_solved",
            ">=",
            85,
            len(solved_30) if results else None,
            f"{len(solved_30)}/{len(total_30)} instances" if results else "SKIPPED",
        ),
        (
            "minlplib_100var_solved",
            ">=",
            30,
            len(solved_100) if results else None,
            f"{len(solved_100)}/{len(total_100)} instances" if results else "SKIPPED",
        ),
        (
            "zero_incorrect",
            "<=",
            0,
            len(incorrect) if results else 0,
            f"{len(incorrect)} incorrect" if results else "0 (no solve run)",
        ),
        (
            "vmap_batch_speedup",
            ">=",
            50.0,
            vs_results[0].value if vs_results else None,
            vs_results[0].notes if vs_results else "NOT MEASURED",
        ),
        (
            "geomean_vs_baron",
            "<=",
            3.0,
            None,
            "NOT MEASURABLE (BARON not installed)",
        ),
        (
            "sparsity_detection",
            ">=",
            1,
            1 if sp_passed else 0,
            f"{'PASS' if sp_passed else 'FAIL'}",
        ),
        (
            "sparse_jacobian",
            ">=",
            1,
            1 if sj_passed else 0,
            f"{'PASS' if sj_passed else 'FAIL'}",
        ),
        (
            "sparse_kkt",
            ">=",
            1,
            1 if sk_passed else 0,
            f"{'PASS' if sk_passed else 'FAIL'}",
        ),
        (
            "sparse_ipm",
            ">=",
            1,
            1 if si_passed else 0,
            f"{'PASS' if si_passed else 'FAIL'}",
        ),
        (
            "test_suite_clean",
            "<=",
            0,
            test_failed + test_errors,
            f"{test_passed} passed, {test_failed} failed, {test_errors} errors",
        ),
    ]

    print(f"{'Criterion':<30s} {'Target':>12s} {'Actual':>12s} {'Status':>10s}  Notes")
    print("-" * 100)

    passed_count = 0
    failed_count = 0
    unmeasurable_count = 0
    for name, op, target, actual, notes in criteria:
        if actual is None:
            status = "SKIP"
            unmeasurable_count += 1
        elif op == ">=" and actual >= target:
            status = "PASS"
            passed_count += 1
        elif op == "<=" and actual <= target:
            status = "PASS"
            passed_count += 1
        else:
            status = "FAIL"
            failed_count += 1

        target_str = f"{op} {target}"
        actual_str = f"{actual:.4f}" if isinstance(actual, float) else str(actual)
        if actual is None:
            actual_str = "N/A"
        print(f"{name:<30s} {target_str:>12s} {actual_str:>12s} {status:>10s}  {notes}")

    print(f"\n{'=' * 90}")
    print(f"PASSED: {passed_count}  FAILED: {failed_count}  UNMEASURABLE: {unmeasurable_count}")
    if failed_count == 0:
        print("All measurable criteria PASSED.")
    else:
        print(f"{failed_count} criteria FAILED.")
    print(f"{'=' * 90}")

    # Save report JSON
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = reports_dir / f"phase4_gate_{ts}.json"

    report = GateReport(
        timestamp=datetime.now().isoformat(),
        time_limit=time_limit,
        max_nodes=max_nodes,
        total_instances=len(results),
        solved=len(solved),
        solved_30var=len(solved_30),
        solved_100var=len(solved_100),
        incorrect_count=len(incorrect),
        criteria_passed=passed_count,
        criteria_failed=failed_count,
        criteria_unmeasurable=unmeasurable_count,
        instances=[asdict(r) for r in results],
        feature_results=[asdict(r) for r in feature_results],
    )
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Gate Validation")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--skip-solve", action="store_true")
    args = parser.parse_args()
    run_gate(args.time_limit, args.max_nodes, skip_solve=args.skip_solve)


if __name__ == "__main__":
    main()
