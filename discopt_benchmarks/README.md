# discopt Testing & Benchmarking Framework

A comprehensive validation, verification, and performance benchmarking framework
for the discopt solver (Rust + JAX hybrid MINLP solver).

## Architecture

```
discopt_benchmarks/
├── config/
│   ├── benchmarks.toml          # Benchmark suite definitions and targets
│   └── solvers.toml             # Solver configurations and paths
├── benchmarks/
│   ├── runner.py                # Main benchmark orchestrator
│   ├── problem_loader.py        # MINLPLib / Netlib / CUTEst loader
│   ├── solver_interface.py      # Uniform interface for all solvers
│   └── metrics.py               # Performance metrics computation
├── tests/
│   ├── test_correctness.py      # Correctness V&V tests
│   ├── test_subsolver_lp.py     # Rust LP subsolver validation
│   ├── test_subsolver_nlp.py    # Hybrid NLP subsolver validation
│   ├── test_relaxation.py       # McCormick relaxation quality tests
│   ├── test_interop.py          # Rust↔JAX interface tests
│   └── test_regression.py       # Performance regression detection
├── utils/
│   ├── profiles.py              # Dolan-Moré performance profiles
│   ├── statistics.py            # Statistical analysis utilities
│   └── reporting.py             # Report generation (markdown + plots)
├── reports/                     # Generated benchmark reports
├── run_benchmarks.py            # CLI entry point
└── README.md
```

## Quick Start

```bash
# Run correctness tests
pytest tests/ -v

# Run full benchmark suite
python run_benchmarks.py --suite phase1

# Run comparison against BARON
python run_benchmarks.py --suite comparison --solvers discopt,baron

# Generate performance report
python run_benchmarks.py --suite phase2 --report
```

## AMP Custom Partition Heuristics

AMP benchmark runs can supply Alpine-style custom partition hooks directly to
`Model.solve`. This keeps heuristic experiments outside solver internals:

```python
from discopt._jax.discretization import add_adaptive_partition


def choose_partition_vars(ctx):
    # Start from any built-in rule, then customize the returned flat indices.
    selected = ctx["builtin_pick_partition_vars"]("adaptive_vertex_cover", ctx.get("distance"))
    return selected[:1] or ctx["partition_candidates"][:1]


def update_scaling(ctx):
    return min(40.0, ctx["current_scaling_factor"] * 1.25)


def refine_partitions(ctx):
    return add_adaptive_partition(
        ctx["disc_state"],
        ctx["solution"],
        ctx["var_indices"],
        ctx["lb"],
        ctx["ub"],
    )


result = model.solve(
    solver="amp",
    disc_var_pick=choose_partition_vars,
    partition_scaling_factor_update=update_scaling,
    disc_add_partition_method=refine_partitions,
    time_limit=300,
)
```

Category benchmarks default to the NLP-BB backends (`ipm`, `ripopt`, `ipopt`).
For an AMP-specific benchmark over the same problem categories, select AMP
explicitly:

```bash
python run_category_benchmarks.py --category minlp --level smoke --solvers amp --time-limit 60 --report --html
python run_category_benchmarks.py --category global_opt --level smoke --solvers amp --time-limit 60 --report --html
```

For local Alpine/incidence checks, use the repository's documented pixi plus uv
`.venv` workflow before running `make test-amp-integration`; avoid reusing an
uncontrolled local Python environment for those solver-dependent examples.

## Benchmark Suites

| Suite       | Purpose                              | Instances | Time Limit |
|-------------|--------------------------------------|-----------|------------|
| `smoke`     | Quick sanity check                   | 10        | 60s        |
| `phase1`    | Phase 1 milestone validation         | ~50       | 3600s      |
| `phase2`    | Phase 2 milestone validation         | ~150      | 3600s      |
| `phase3`    | Phase 3 milestone validation         | ~300      | 3600s      |
| `full`      | Complete MINLPLib                     | ~1700     | 3600s      |
| `comparison`| Head-to-head solver comparison       | ~200      | 3600s      |
| `subsolver` | LP/NLP subsolver-only benchmarks     | varies    | 300s       |
| `nightly`   | CI regression suite                  | ~100      | 1800s      |

## Phase Gate Criteria

Each phase has automated pass/fail criteria defined in `config/benchmarks.toml`.
Run `python run_benchmarks.py --gate phaseN` to check.
