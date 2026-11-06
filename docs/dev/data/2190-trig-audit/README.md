# #2190 trig enclosure audit

`audit.py` checks every sin/cos/tan interval enclosure, curvature verdict, pole
test and periodic window in the tree against mpmath at 60 digits. It runs on
11,705 boxes: points, one-ulp boxes and ±0.3 boxes at every rounded `kπ/2` for
|k| ≤ 600, boxes within 1e-3 of `kπ/2` at magnitudes 1e6, 1e9 and 1e12, and
3,000 random boxes with |x| up to 1e13. A subsample of 1,500 boxes goes through
Rust FBBT. The run takes about 10 minutes.

`audit.summary.log` is the output at the PR head: **157,477 checks, 0 misses.**

## Before the fix

These are miss counts from the same script run on `7303ebe` (after the
`interval.sin`/`cos` fix, before the other sites were fixed). That run's
periodic-window check tested the raw `lo + _TWO_PI`; the helpers it now checks
did not exist yet.

| Site | Misses before |
|---|---|
| `operator_relaxations.trig_range(sin / cos / tan)` | 9,059 / 9,132 / 5,585 of 11,705 |
| `uniform_relax._curv_tan` | 1,945 of 11,705 |
| periodic window `lo + _TWO_PI` | 33 of 207 |

A separate 104,016-check sweep of `interval.sin`/`cos` alone (|k| ≤ 2000, plus
20,000 random boxes up to about 3e15) found 62,718 misses on the original
`_sinusoid` and 0 after. That sweep is a subset of what `audit.py` covers, so it
is not committed separately.

`solvers/scip._tan_pole_possible` is not in `audit.py`. It was checked with a
2,390-box probe near `π/2 + kπ` for |k| up to 1e9: 2 missed poles before, 0
after. Both boxes are now fixed cases in
`python/tests/test_2190_sinusoid_enclosure.py::test_scip_tan_pole_test_sees_pole_near_large_k`.
