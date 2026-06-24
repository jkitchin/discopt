"""Performance observability harness + regression gate (perf plan Stage 0).

See ``docs/dev/performance-plan.md``. The point of this package is to make the
cost centers the plan identified *measurable and gated*: it records the metrics
that the general benchmark harness did not (XLA compile count/seconds and
time-to-first-incumbent), runs a fixed vendored panel, and fails CI on a
correctness regression or a deterministic perf regression vs a committed baseline.
"""

from discopt_benchmarks.perf.measure import count_xla_compiles, measure_solve

__all__ = ["count_xla_compiles", "measure_solve"]
