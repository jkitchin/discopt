### comment 0 — jkitchin 2026-08-16T04:43:12Z

## Retraction: bucket A is not an inert opt-out (CLAUDE.md §11)

This issue's bucket A says the `DISCOPT_RELAX_ROW_FILTER` opt-out "no longer
disables anything", and draws the §5 conclusion that "the escape hatch promised
when these flags graduated does not exist, and any future A/B measurement using
the flag silently compares a path against itself".

**Measured, and that is wrong.** The flag is consulted; the mechanism it gates is
simply dormant on `hda`.

The gate at `python/discopt/_relax/mccormick_lp.py:1615` is

```python
if _tuning().relax_row_filter and not (
    res.status == "optimal"
    or (res.status == "infeasible" and getattr(res, "farkas_certified", False))
):
```

`_tuning()` is `solver_tuning.current()`, which resolves the env per solve via a
ContextVar — not an import-time singleton, so this is *not* the #1035 shape.

Counting how often the filter actually runs, per arm, on `hda.nl` (`time_limit=60`):

```
FLAG=1: bound=-64473.44239643685 status=time_limit filter_invocations=0 rows_dropped=0
FLAG=0: bound=-64473.44239643685 status=time_limit filter_invocations=0 rows_dropped=0
```

Zero in **both** arms. A zero from a probe is exactly the CLAUDE.md §6 hazard, so
a control run tallied the node-LP statuses that gate the branch, plus a positive
control:

```
node LP solves observed: 4
  status tally: {'optimal': 4}
  non-certified (would OPEN the filter branch): 0
  filter invocations: 0

positive control: non-certified tally moved by 1 (must be 1)
```

Every node LP now returns a certified `optimal`, so the failure-triggered branch
never opens — with the flag ON or OFF. The two arms are identical because the
mechanism is dormant, not because the opt-out is broken.

### What this changes

* The §5 "keep the `=0` opt-out" promise is **intact** for this flag. An A/B on an
  instance whose LP does break down would still measure two different paths.
* What is actually stale is the three tests' *premise*: they assert that with the
  filter off, `hda`'s ill-conditioned root LP false-fails and the bound falls back
  to candidate A's loose floor (`< -1e7`). `hda`'s root LP no longer false-fails,
  so that loose floor is unreachable in either arm and the tight `-64473.44` is
  the honest answer both times.
* The tight bound is sound — it satisfies the same tests' `bound <= _HDA_OPT`
  assertion.

Most likely cause of the numerics improving: the feral `0.16.0` upgrade in #1008
(threshold-Markowitz sparse LU), which targeted exactly this fill/conditioning
class. Not verified here, and it does not change the disposition.

### Revised disposition for bucket A

Not a contract defect. Three tests encode a precondition (`hda` false-fails
without the filter) that the code no longer satisfies. They need to be re-pointed
at an instance that still provokes an uncertified node LP — otherwise they assert
nothing about the flag — or retired with the measurement recorded. Re-pointing is
the better option: it restores a real opt-out test instead of deleting coverage.

This does **not** move bucket A onto the release's critical path; the earlier
framing did, which is why it is corrected here rather than left to stand.

Probes: `probe_rowfilter.py`, `probe_rowfilter2.py` (both print executed counts
and exit non-zero at zero; neither swallows an exception).
