# #2142 Part B: AffinePow reverse step (performance-plan §113) — killed at the entry check

`PREREG.md` is the pre-registered check. The flag `DISCOPT_AFFINE_POW_REVERSE` existed at
`2d7d9c54`..`8b552939` and was removed in `b4c69aa4`. Box files are in the session scratchpad;
regenerate them with `b1_capture.py OUT.npz 1000 10`.

| file | what |
| --- | --- |
| `b1_capture.py`, `b1_capture.log` | cstr hull node boxes (every 10th of the first 1000 deterministic nodes; 94 boxes) |
| `b2_kernel.py`, `b2_{K,KR}.log` | kernel propagation per box at 15 and 100000 rounds, reverse OFF / ON |
| `b3_scip.py`, `b3_S.log`, `b3_ablations.log` | SCIP 6.3 root propagation per box (and with one propagator off) |
| `b4_compare.py`, `b4_compare.log` | the pre-registered closure: 0.000 at both round caps |
| `b5_probe.log` | why: 0 of 1,880 (term, box) pairs where `1/w_hi` exceeds the base's lower bound |
| `b6_scip_sanity.py`, `b6_scip_sanity.log` | SCIP calls captured boxes empty even with no cutoff (unexplained, recorded) |

> **Raw logs removed (#2142 follow-up).** The `.log` / `.log.gz` run traces cited in this directory were
> removed to keep checkouts small; the results they support are recorded in these notes. The originals
> remain in git history: `git show e39975e1:docs/dev/data/2142-affinepow-reverse/<name>.log.gz | gunzip` (or `.log`, no gunzip).
