# QPLIB corpus

QPLIB is a library of 453 quadratic programming instances, **390 of them
nonconvex**, with reference solution *points* rather than only objective values.
Ingested under issue #830. Upstream: <https://qplib.zib.de/>, CC-BY 4.0, Furini
et al., *Math. Prog. Comp.* 11(2), 2019, doi:10.1007/s12532-018-0147-4.

## Why it is here

The in-repo QCQP coverage is `discopt_benchmarks/benchmarks/problems/qcqp_problems.py`
— 11 synthetic families of at most 6 variables whose optima we compute
ourselves by vertex enumeration and multistart. That oracle is only as good as
our own code, which is the wrong property for a correctness corpus.

QPLIB is independent, adversarial, and does not overlap what we already have:
zero `qplib_*` names appear in the 4831-file MINLPLib snapshot. It also ships
solution **vectors**, so an incumbent can be feasibility-verified directly
rather than compared to a published number.

## Location and access

The corpus lives at `~/Dropbox/projects/discopt-minlp-benchmark/qplib/`
(~1.1 GB) alongside the MINLPLib snapshot. Its `README.md` is the reference for
layout, refetching, and the format's undocumented conventions. Override the
location with `DISCOPT_QPLIB_DIR`.

Read instances with `discopt.interfaces.qplib` — the native `.qplib` format is
parsed directly, with no AMPL or GAMS conversion step:

```python
from discopt.interfaces import qplib

model = qplib.from_qplib(".../qplib/QPLIB_3814.qplib")   # -> discopt Model

inst = qplib.read_qplib(".../qplib/QPLIB_3814.qplib")    # or in two steps
x, objvar = qplib.read_solution(".../sol/QPLIB_3814.sol", inst)
inst.evaluate_objective(x)   # reproduces objvar
inst.max_violation(x)        # feasibility-verify the reference point
```

## Selecting instances

Filter `qplib_manifest.csv` in the corpus directory; do not hardcode instance
names. It records, per instance, the structural facts plus whether this reader
reproduces the published objective and how far the reference point violates the
parsed model. Regenerate with:

```bash
cd discopt_benchmarks && python -m benchmarks.qplib_manifest
```

`usable_oracle` is the column to gate on — a best-known value exists, the reader
reproduces it, and the reference point is feasible. **421 of 453 qualify**; 27
ship no reference point, 4 are upstream-defective, 1 is marginally infeasible.
The four defective instances are identified by a general rule (metadata declares
objective variables, file stores none), not a name list, so a future QPLIB
release that fixes them needs no code change — only the expected count in
`test_qplib_corpus.py`.

## Verification

All 453 instances parse. 4448 assertions against QPLIB's own `instancedata.csv`
and its reference points pass, with failures confined to the 5 instances the
corpus README documents.

- `python/tests/test_qplib_reader.py` — 9 vendored fixtures (~260 KB,
  `python/tests/data/qplib/`), chosen to cover every conditional layout branch
  except `C=B`/`C=D`/`O=C`, whose smallest instances are too large to vendor.
  Runs in CI, <1 s.
- `python/tests/test_qplib_corpus.py` — all 453 instances, marked `slow`,
  skipped when the corpus is absent. ~29 s. Covers the three remaining branches.

## Format hazards

The `.qplib` format has no section markers and its layout is **conditional on the
three-character `probtype` code**. Getting that conditioning wrong does not
raise — it shifts the token stream and yields a well-formed but wrong model, so
every check here is against QPLIB's own metadata or its reference points rather
than against our expectations. Two conventions in particular are undocumented
upstream and were established empirically:

- every stored quadratic entry carries a factor of 1/2, **off-diagonal as well
  as diagonal** (fitted by least squares over the corpus: exactly (0.5, 0.5),
  rank 2, residual 4.5e-12);
- `.sol` record names are GAMS variable numbers counting `objvar` as variable 1,
  so index `k - 2` (offset 0 gives 93 out-of-range and 34 type-mismatched
  records over the corpus; the adopted offset gives zero of each).

The corpus README has the full list.

## Status: not yet a benchmark suite

QPLIB is ingested and verified, but **not wired into a benchmark suite**. A
measurement in the corpus README (2026-08-02) shows discopt finds no incumbent
on the instances tried — the solver does not get past the root NLP seed, at
roughly 9 s per node on a 17-variable problem, because a QCQP falls through to
the JAX evaluator without a Gurobi licence and there is no in-house `solve_qcp`.
Adding a suite should wait on an analytic QCQP path, or be scoped via the
manifest to instances the solver can make progress on.
