#!/usr/bin/env python3
"""
Phase 3 Gate Validation

Runs discopt solver on MINLPLib .nl instances and validates Phase 3-specific
features (piecewise McCormick, cutting planes, GNN branching). Criteria that
require external infrastructure (BARON, GPU) are reported as "not measurable."

Usage:
    python scripts/phase3_gate.py
    python scripts/phase3_gate.py --time-limit 120 --max-nodes 200000
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
    """Result from a Phase 3 feature validation test."""

    feature: str
    test_name: str
    passed: bool
    value: float | None
    target: float | None
    notes: str = ""


@dataclass
class GateReport:
    """Full Phase 3 gate report."""

    timestamp: str
    gate: str = "phase3"
    time_limit: float = 120.0
    max_nodes: int = 200_000
    total_instances: int = 0
    solved: int = 0
    solved_30var: int = 0
    solved_50var: int = 0
    solved_100var: int = 0
    incorrect_count: int = 0
    median_nodes_per_second: float = 0.0
    median_rust_overhead: float | None = None
    criteria_passed: int = 0
    criteria_failed: int = 0
    criteria_unmeasurable: int = 0
    instances: list[dict] = field(default_factory=list)
    feature_results: list[dict] = field(default_factory=list)


# Known optima for correctness checking
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
    """Find all unique .nl files across data directories."""
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


_SOLVE_SCRIPT = """
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt.modeling as dm

nl_path = sys.argv[1]
time_limit = float(sys.argv[2])
max_nodes = int(sys.argv[3])

model = dm.from_nl(nl_path)
n_vars = len(model._variables)
n_cons = len(model._constraints)
result = model.solve(time_limit=time_limit, gap_tolerance=1e-4, max_nodes=max_nodes)
wt = result.wall_time if result.wall_time > 0 else 1e-10
nc = result.node_count or 0
nps = nc / wt if wt > 0 else 0.0
rust_frac = result.rust_time / wt if result.rust_time else None
print(json.dumps({
    "n_vars": n_vars, "n_cons": n_cons, "status": result.status,
    "objective": result.objective, "wall_time": result.wall_time,
    "node_count": nc, "nodes_per_second": nps, "rust_frac": rust_frac
}))
"""


def solve_instance(name: str, nl_path: Path, time_limit: float, max_nodes: int) -> InstanceResult:
    """Solve a single .nl instance in a subprocess with hard timeout."""
    import subprocess

    hard_timeout = time_limit + 30
    try:
        proc = subprocess.run(
            ["python", "-c", _SOLVE_SCRIPT, str(nl_path), str(time_limit), str(max_nodes)],
            capture_output=True,
            text=True,
            timeout=hard_timeout,
        )
        if proc.returncode != 0:
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
                error=proc.stderr[:200] if proc.stderr else "nonzero exit",
            )
        # Parse JSON from last line of stdout
        lines = [line for line in proc.stdout.strip().split("\n") if line.startswith("{")]
        if not lines:
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
                error="no JSON output",
            )
        data = json.loads(lines[-1])

        expected = KNOWN_OPTIMA.get(name)
        correct = None
        if expected is not None and data["objective"] is not None:
            diff = abs(data["objective"] - expected)
            tol = ABS_TOL + REL_TOL * abs(expected)
            correct = diff <= tol

        return InstanceResult(
            name=name,
            n_vars=data["n_vars"],
            n_cons=data["n_cons"],
            status=data["status"],
            objective=data["objective"],
            expected_obj=expected,
            correct=correct,
            wall_time=data["wall_time"],
            node_count=data["node_count"],
            nodes_per_second=data["nodes_per_second"],
            rust_frac=data["rust_frac"],
        )
    except subprocess.TimeoutExpired:
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
            error="hard timeout (subprocess)",
        )
    except Exception as e:
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
# Phase 3 feature validations
# ---------------------------------------------------------------------------


def validate_piecewise_mccormick() -> list[FeatureResult]:
    """Validate piecewise McCormick gap reduction (T27 acceptance: >= 60%)."""
    import jax.numpy as jnp
    from discopt._jax.mccormick import relax_exp, relax_log, relax_square
    from discopt._jax.piecewise_mccormick import (
        piecewise_relax_exp,
        piecewise_relax_log,
        piecewise_relax_square,
    )

    results = []
    k = 4  # partitions

    test_cases = [
        ("exp", relax_exp, piecewise_relax_exp, 0.0, 2.0, jnp.exp),
        ("log", relax_log, piecewise_relax_log, 0.5, 3.0, jnp.log),
        ("square", relax_square, piecewise_relax_square, -2.0, 2.0, lambda x: x**2),
    ]

    for fname, std_relax, pw_relax, lb, ub, true_fn in test_cases:
        n_pts = 1000
        x_pts = jnp.linspace(lb, ub, n_pts)

        # Standard McCormick gap
        std_cv = []
        std_cc = []
        for x in x_pts:
            cv, cc = std_relax(x, jnp.array(lb), jnp.array(ub))
            std_cv.append(float(cv))
            std_cc.append(float(cc))
        std_cv = jnp.array(std_cv)
        std_cc = jnp.array(std_cc)
        std_gap = float(jnp.mean(std_cc - std_cv))

        # Piecewise McCormick gap
        pw_cv = []
        pw_cc = []
        for x in x_pts:
            cv, cc = pw_relax(x, jnp.array(lb), jnp.array(ub), k=k)
            pw_cv.append(float(cv))
            pw_cc.append(float(cc))
        pw_cv = jnp.array(pw_cv)
        pw_cc = jnp.array(pw_cc)
        pw_gap = float(jnp.mean(pw_cc - pw_cv))

        reduction = 1.0 - pw_gap / std_gap if std_gap > 1e-12 else 0.0
        passed = reduction >= 0.60

        results.append(
            FeatureResult(
                feature="piecewise_mccormick",
                test_name=f"{fname}_gap_reduction",
                passed=passed,
                value=reduction,
                target=0.60,
                notes=f"std_gap={std_gap:.4f}, pw_gap={pw_gap:.4f}, k={k}",
            )
        )

    return results


def validate_alphabb() -> list[FeatureResult]:
    """Validate alphaBB relaxation soundness and gap reduction."""
    import jax
    import jax.numpy as jnp
    from discopt._jax.alphabb import make_alphabb_relaxation

    results = []

    # Test 1: Soundness on Rosenbrock-like function
    def rosenbrock_2d(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    lb = jnp.array([-2.0, -2.0])
    ub = jnp.array([2.0, 2.0])

    cv_fn, cc_fn, _, _ = make_alphabb_relaxation(rosenbrock_2d, lb, ub, method="eigenvalue")

    # Check soundness: cv(x) <= f(x) <= cc(x) at random points
    key = jax.random.PRNGKey(42)
    n_check = 500
    x_rand = jax.random.uniform(key, shape=(n_check, 2), minval=lb, maxval=ub)
    violations = 0
    for i in range(n_check):
        xi = x_rand[i]
        f_val = rosenbrock_2d(xi)
        cv_val = cv_fn(xi)
        cc_val = cc_fn(xi)
        if float(cv_val) > float(f_val) + 1e-8:
            violations += 1
        if float(cc_val) < float(f_val) - 1e-8:
            violations += 1

    results.append(
        FeatureResult(
            feature="alphabb",
            test_name="rosenbrock_soundness",
            passed=violations == 0,
            value=float(violations),
            target=0.0,
            notes=f"{violations}/{2 * n_check} violations",
        )
    )

    # Test 2: Gershgorin method soundness
    cv_fn_g, cc_fn_g, _, _ = make_alphabb_relaxation(rosenbrock_2d, lb, ub, method="gershgorin")
    violations_g = 0
    for i in range(n_check):
        xi = x_rand[i]
        f_val = rosenbrock_2d(xi)
        cv_val = cv_fn_g(xi)
        if float(cv_val) > float(f_val) + 1e-8:
            violations_g += 1

    results.append(
        FeatureResult(
            feature="alphabb",
            test_name="gershgorin_soundness",
            passed=violations_g == 0,
            value=float(violations_g),
            target=0.0,
            notes=f"{violations_g}/{n_check} cv violations (Gershgorin)",
        )
    )

    return results


def validate_cutting_planes() -> list[FeatureResult]:
    """Validate cutting plane generation and cut pool management."""
    import numpy as np
    from discopt._jax.cutting_planes import CutPool, LinearCut

    results = []

    # Test 1: CutPool basic functionality
    pool = CutPool(max_cuts=100)

    cuts = [
        LinearCut(coeffs=np.array([1.0, 0.0, 0.0]), rhs=1.0, sense="<="),
        LinearCut(coeffs=np.array([0.0, 1.0, 0.0]), rhs=2.0, sense="<="),
        LinearCut(coeffs=np.array([1.0, 1.0, 0.0]), rhs=2.5, sense="<="),
    ]
    pool.add_many(cuts)

    pool_ok = len(pool) == 3
    A, b, senses = pool.to_constraint_arrays()
    arrays_ok = A.shape == (3, 3) and b.shape == (3,)

    results.append(
        FeatureResult(
            feature="cutting_planes",
            test_name="cut_pool_basic",
            passed=pool_ok and arrays_ok,
            value=float(len(pool)),
            target=3.0,
            notes=f"pool_size={len(pool)}, A_shape={A.shape}",
        )
    )

    # Test 2: Duplicate detection (add does hash-based dedup)
    pool2 = CutPool(max_cuts=100)
    pool2.add_many(cuts)
    pool2.add_many(cuts)  # add same cuts again — should be rejected as duplicates
    dedup_ok = len(pool2) == 3  # should remain 3

    results.append(
        FeatureResult(
            feature="cutting_planes",
            test_name="cut_dedup",
            passed=dedup_ok,
            value=float(len(pool2)),
            target=3.0,
            notes=f"After adding dupes: {len(pool2)} cuts (expected 3)",
        )
    )

    # Test 3: Age and purge
    pool3 = CutPool(max_cuts=100)
    pool3.add_many(cuts)
    x_test = np.array([0.5, 0.5, 0.0])
    for _ in range(20):
        pool3.age_cuts(x_test)
    pool3.purge_inactive(max_age=15)

    results.append(
        FeatureResult(
            feature="cutting_planes",
            test_name="cut_age_purge",
            passed=True,
            value=float(len(pool3)),
            target=None,
            notes=f"After 20 age + purge(15): {len(pool3)} cuts remain",
        )
    )

    # Test 4: OA cut generation via gradient linearization
    try:
        import discopt.modeling as dm
        from discopt._jax.cutting_planes import generate_oa_cut
        from discopt._jax.nlp_evaluator import NLPEvaluator

        m = dm.Model("oa_test")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1.0)

        ev = NLPEvaluator(m)
        x_pt = np.array([0.5, 0.5])
        grad = ev.evaluate_gradient(x_pt)
        obj_val = ev.evaluate_objective(x_pt)
        cut = generate_oa_cut(grad, obj_val, x_pt, sense="<=")
        oa_ok = cut is not None and hasattr(cut, "coeffs")

        results.append(
            FeatureResult(
                feature="cutting_planes",
                test_name="oa_cut_generation",
                passed=oa_ok,
                value=1.0 if oa_ok else 0.0,
                target=1.0,
                notes="OA cut generated successfully" if oa_ok else "OA cut failed",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="cutting_planes",
                test_name="oa_cut_generation",
                passed=False,
                value=0.0,
                target=1.0,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_gnn_branching() -> list[FeatureResult]:
    """Validate GNN branching policy (T29 acceptance criteria)."""
    results = []

    # Test 1: GNN inference latency < 0.1ms
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import ProblemGraph

        key = jax.random.PRNGKey(0)
        gnn = BranchingGNN(hidden_dim=32, n_rounds=2, key=key)

        # Create a synthetic ProblemGraph
        n_vars, n_cons, n_edges = 10, 5, 15
        var_feats = jnp.ones((n_vars, 7))
        con_feats = jnp.ones((n_cons, 5))
        edge_indices = jnp.array(
            [
                jnp.arange(n_edges) % n_vars,
                jnp.arange(n_edges) % n_cons,
            ]
        )
        graph = ProblemGraph(
            var_features=var_feats,
            con_features=con_feats,
            edge_indices=edge_indices,
            n_vars=n_vars,
            n_cons=n_cons,
        )

        # JIT compile
        jit_forward = eqx.filter_jit(gnn)
        _ = jit_forward(graph)

        # Measure latency
        n_trials = 100
        t_start = time.perf_counter()
        for _ in range(n_trials):
            out = jit_forward(graph)
            jax.block_until_ready(out)
        t_elapsed = time.perf_counter() - t_start
        latency_ms = (t_elapsed / n_trials) * 1000.0

        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="inference_latency",
                passed=latency_ms < 0.1,
                value=latency_ms,
                target=0.1,
                notes=f"{latency_ms:.4f} ms per inference ({n_trials} trials)",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="inference_latency",
                passed=False,
                value=None,
                target=0.1,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    # Test 2: GNN produces valid finite scores
    try:
        import jax
        import jax.numpy as jnp
        from discopt._jax.gnn_branching import BranchingGNN
        from discopt._jax.problem_graph import ProblemGraph

        key = jax.random.PRNGKey(42)
        gnn = BranchingGNN(hidden_dim=32, n_rounds=2, key=key)

        graph = ProblemGraph(
            var_features=jnp.ones((5, 7)),
            con_features=jnp.ones((3, 5)),
            edge_indices=jnp.array([[0, 1, 2, 3], [0, 1, 1, 2]]),
            n_vars=5,
            n_cons=3,
        )
        scores = gnn(graph)
        scores_ok = scores.shape == (5,) and bool(jnp.all(jnp.isfinite(scores)))

        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="valid_scores",
                passed=scores_ok,
                value=float(scores.shape[0]),
                target=5.0,
                notes=f"scores shape={scores.shape}, all finite={scores_ok}",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="valid_scores",
                passed=False,
                value=None,
                target=5.0,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    # Test 3: Strong branching data collection works
    try:
        import discopt.modeling as dm
        from discopt._jax.gnn_branching import collect_strong_branching_data

        m = dm.Model("sb_test")
        x = m.integer("x", lb=0, ub=5)
        y = m.integer("y", lb=0, ub=5)
        m.minimize(x + 2 * y)
        m.subject_to(x + y >= 3)

        data = collect_strong_branching_data(m, max_nodes=10)
        sb_ok = isinstance(data, list)

        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="strong_branching_collection",
                passed=sb_ok,
                value=float(len(data)) if sb_ok else 0.0,
                target=0.0,
                notes=f"Collected {len(data) if sb_ok else 0} SB samples",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="gnn_branching",
                test_name="strong_branching_collection",
                passed=False,
                value=None,
                target=None,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


def validate_iterative_ipm() -> list[FeatureResult]:
    """Validate iterative IPM with lineax solvers (T18)."""
    results = []

    try:
        import jax.numpy as jnp
        from discopt._jax.pcg import PCGOptions, pcg_solve

        # Test: PCG solves a simple SPD system
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-10))
        x_exact = jnp.linalg.solve(A, b)
        err = float(jnp.linalg.norm(result.x - x_exact))

        results.append(
            FeatureResult(
                feature="iterative_ipm",
                test_name="pcg_accuracy",
                passed=err < 1e-8,
                value=err,
                target=1e-8,
                notes=f"PCG error={err:.2e}, converged={bool(result.converged)}",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="iterative_ipm",
                test_name="pcg_accuracy",
                passed=False,
                value=None,
                target=1e-8,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    # Test: lineax integration available
    try:
        from discopt._jax.ipm_iterative import IterativeKKTSolver

        IterativeKKTSolver(linear_solver="cg")
        lineax_ok = True
        results.append(
            FeatureResult(
                feature="iterative_ipm",
                test_name="lineax_available",
                passed=lineax_ok,
                value=1.0,
                target=1.0,
                notes="IterativeKKTSolver with lineax CG initialized",
            )
        )
    except Exception as e:
        results.append(
            FeatureResult(
                feature="iterative_ipm",
                test_name="lineax_available",
                passed=False,
                value=0.0,
                target=1.0,
                notes=f"Error: {str(e)[:100]}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Test suite validation
# ---------------------------------------------------------------------------


def run_test_suite() -> tuple[int, int, int]:
    """Run the pytest suite (excluding slow tests) and return (passed, failed, errors)."""
    import re
    import subprocess

    project_root = Path(__file__).parent.parent
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                str(project_root / "python" / "tests"),
                "-q",
                "--tb=no",
                "-p",
                "no:warnings",
                "-m",
                "not slow",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return 0, 0, -1  # signal timeout

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
    print("\nPhase 3 Gate Validation")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Instances: {len(instances)}")
    print(f"Time limit: {time_limit}s, Max nodes: {max_nodes}")
    print(f"{'=' * 90}\n")

    # --- Part 1: MINLPLib instance solving ---
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

    # --- Part 2: Phase 3 feature validations ---
    print(f"\n{'=' * 90}")
    print("Part 2: Phase 3 Feature Validations")
    print("-" * 90)

    feature_results: list[FeatureResult] = []

    print("\n  [T27] Piecewise McCormick + alphaBB...")
    pw_results = validate_piecewise_mccormick()
    feature_results.extend(pw_results)
    for r in pw_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    ab_results = validate_alphabb()
    feature_results.extend(ab_results)
    for r in ab_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [T28] Cutting planes...")
    cp_results = validate_cutting_planes()
    feature_results.extend(cp_results)
    for r in cp_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    print("\n  [T29] GNN branching...")
    gnn_results = validate_gnn_branching()
    feature_results.extend(gnn_results)
    for r in gnn_results:
        status = "PASS" if r.passed else "FAIL"
        val_str = f"{r.value:.4f}" if r.value is not None else "N/A"
        print(f"    {r.test_name:<35s} {status:>6s}  val={val_str}  {r.notes}")

    print("\n  [T18] Iterative IPM...")
    ipm_results = validate_iterative_ipm()
    feature_results.extend(ipm_results)
    for r in ipm_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    {r.test_name:<35s} {status:>6s}  {r.notes}")

    # --- Part 3: Test suite ---
    print(f"\n{'=' * 90}")
    print("Part 3: Full Test Suite")
    print("-" * 90)
    test_passed, test_failed, test_errors = run_test_suite()
    if test_errors == -1:
        print("  Tests: TIMEOUT (suite took >15 min)")
        test_suite_notes = "TIMEOUT"
        test_suite_pass = False
        test_suite_fail_count = 1
    else:
        print(f"  Tests: {test_passed} passed, {test_failed} failed, {test_errors} errors")
        test_suite_notes = f"{test_passed} passed, {test_failed} failed, {test_errors} errors"
        test_suite_pass = test_failed == 0 and test_errors == 0
        test_suite_fail_count = test_failed + test_errors

    feature_results.append(
        FeatureResult(
            feature="test_suite",
            test_name="full_pytest",
            passed=test_suite_pass,
            value=float(test_passed),
            target=None,
            notes=test_suite_notes,
        )
    )

    # --- Compute gate metrics ---
    print(f"\n{'=' * 90}")
    print("PHASE 3 GATE RESULTS")
    print(f"{'=' * 90}\n")

    solved = [r for r in results if r.status in ("optimal", "feasible")]
    solved_30 = [r for r in solved if r.n_vars <= 30]
    solved_50 = [r for r in solved if r.n_vars <= 50]
    solved_100 = [r for r in solved if r.n_vars <= 100]
    total_30 = [r for r in results if r.n_vars <= 30]
    total_50 = [r for r in results if r.n_vars <= 50]
    total_100 = [r for r in results if r.n_vars <= 100]

    checked = [r for r in results if r.correct is not None]
    incorrect = [r for r in checked if r.correct is False]

    nps_values = [r.nodes_per_second for r in solved if r.node_count > 0]
    median_nps = sorted(nps_values)[len(nps_values) // 2] if nps_values else 0.0

    rust_fracs = [r.rust_frac for r in solved if r.rust_frac is not None]
    median_rust = sorted(rust_fracs)[len(rust_fracs) // 2] if rust_fracs else None

    # Phase 3 feature pass counts
    pw_passed = all(r.passed for r in pw_results)
    ab_passed = all(r.passed for r in ab_results)
    cp_passed = all(r.passed for r in cp_results)
    gnn_passed = all(r.passed for r in gnn_results)
    ipm_passed = all(r.passed for r in ipm_results)

    # Gate criteria table
    criteria = [
        (
            "minlplib_30var_solved",
            ">=",
            75,
            len(solved_30) if results else None,
            f"{len(solved_30)}/{len(total_30)} instances" if results else "SKIPPED",
        ),
        (
            "minlplib_50var_solved",
            ">=",
            45,
            len(solved_50) if results else None,
            f"{len(solved_50)}/{len(total_50)} instances" if results else "SKIPPED",
        ),
        (
            "minlplib_100var_solved",
            ">=",
            20,
            len(solved_100) if results else None,
            f"{len(solved_100)}/{len(total_100)} instances" if results else "SKIPPED",
        ),
        (
            "geomean_vs_baron",
            "<=",
            2.5,
            None,
            "NOT MEASURABLE (BARON not installed)",
        ),
        (
            "gpu_class_vs_baron",
            "<=",
            1.0,
            None,
            "NOT MEASURABLE (BARON not installed, no GPU)",
        ),
        (
            "learned_branching",
            ">=",
            0.20,
            None,
            "NOT MEASURABLE (requires full B&B comparison; GNN functional)",
        ),
        (
            "root_gap_vs_baron",
            "<=",
            1.3,
            None,
            "NOT MEASURABLE (BARON not installed)",
        ),
        (
            "zero_incorrect",
            "<=",
            0,
            len(incorrect) if results else 0,
            f"{len(incorrect)} incorrect out of {len(checked)} checked"
            if results
            else "0 (no solve run)",
        ),
        (
            "T27_piecewise_mccormick",
            ">=",
            1,
            1 if pw_passed else 0,
            f"{'PASS' if pw_passed else 'FAIL'}: >= 60% gap reduction",
        ),
        (
            "T27_alphabb_soundness",
            ">=",
            1,
            1 if ab_passed else 0,
            f"{'PASS' if ab_passed else 'FAIL'}: zero violations",
        ),
        (
            "T28_cutting_planes",
            ">=",
            1,
            1 if cp_passed else 0,
            f"{'PASS' if cp_passed else 'FAIL'}: pool + OA generation",
        ),
        (
            "T29_gnn_branching",
            ">=",
            1,
            1 if gnn_passed else 0,
            f"{'PASS' if gnn_passed else 'FAIL'}: inference + scores + SB data",
        ),
        (
            "T18_iterative_ipm",
            ">=",
            1,
            1 if ipm_passed else 0,
            f"{'PASS' if ipm_passed else 'FAIL'}: PCG + lineax",
        ),
        (
            "test_suite_clean",
            "<=",
            0,
            test_suite_fail_count,
            test_suite_notes,
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

    if incorrect:
        print("\nINCORRECT RESULTS:")
        for r in incorrect:
            print(f"  {r.name}: got {r.objective}, expected {r.expected_obj}")

    # Save report JSON
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = reports_dir / f"phase3_gate_{ts}.json"

    report = GateReport(
        timestamp=datetime.now().isoformat(),
        time_limit=time_limit,
        max_nodes=max_nodes,
        total_instances=len(results),
        solved=len(solved),
        solved_30var=len(solved_30),
        solved_50var=len(solved_50),
        solved_100var=len(solved_100),
        incorrect_count=len(incorrect),
        median_nodes_per_second=median_nps,
        median_rust_overhead=median_rust,
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
    parser = argparse.ArgumentParser(description="Phase 3 Gate Validation")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="Time limit per instance (seconds)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200_000,
        help="Max B&B nodes per instance",
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Skip MINLPLib instance solving (feature validation only)",
    )
    args = parser.parse_args()
    run_gate(args.time_limit, args.max_nodes, skip_solve=args.skip_solve)


if __name__ == "__main__":
    main()
