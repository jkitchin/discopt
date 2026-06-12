# Vendored MINLPLib `.nl` instances

A small curated subset of [MINLPLib](https://www.minlplib.org) instances in
AMPL `.nl` format, vendored here so the test suite is self-contained and runs in
CI without a network fetch. These are public benchmark instances distributed by
MINLPLib for solver evaluation; each retains its upstream name.

The full library (~1700 instances, ~50 MB) is **not** vendored. To fetch the
full set or additional instances into a local cache
(`~/.cache/discopt/minlplib`, or `$DISCOPT_MINLPLIB_CACHE`):

```bash
python -m discopt_benchmarks.scripts.fetch_minlplib --instances ex1224,gkocis,super1
```

## Which tests use these

- `test_minlptests.py` and other solver tests use the QP/MIQP/`nvs*`/`ex12*`
  families.
- `test_convexity_minlplib_suspect.py` (issue #40) uses the SUSPECT-regression
  subset: convexity-detector soundness and completeness coverage on instances
  from the SUSPECT regression list (cog-imperial/suspect#11). Ground-truth
  convexity labels are taken from MINLPLib's `instancedata.csv` `convex` column
  and pinned directly in that test.

A few large SUSPECT-regression instances (`super1`..`super3t`) are intentionally
**not** vendored — they classify in tens of seconds and are exercised only when
present in the local MINLPLib cache, skipped otherwise.
