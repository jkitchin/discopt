# Pinned baselines

This directory holds checked-in `BenchmarkResults` JSONs that serve as the
regression reference for the `--gate <suite>` check.

## Pinning a baseline

```
python run_benchmarks.py --suite full --pin-baseline
```

That writes `discopt_benchmarks/baselines/full.json` from the just-completed
run. Commit it deliberately — it's the contract for what "no regression" means
on the next PR.

## Gate check

```
python run_benchmarks.py --gate full
```

Loads the most recent `reports/full_*.json`, compares to `baselines/full.json`,
and exits non-zero if:

  * solved-count drops by more than the tolerance (default 0),
  * any new instance has an incorrect objective vs. the MINLPLib reference.

Time regressions are reported but informational by default. Pass `--strict-time`
to make a >1.5× slowdown on any commonly-solved instance also fail the gate.

## Cadence

Re-pin a baseline when an algorithmic change intentionally moves the numbers
(e.g. a new presolve pass that solves +12 more instances but also reorders
node-counts). Don't re-pin to mask regressions — investigate first.
