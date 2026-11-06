# #2142: mixed hull (perspective for proven-convex rows, big-M otherwise) — entry kill (performance-plan §111)

The flag `DISCOPT_HULL_MIXED_BIGM` (commits `8292ded9`, `d3d59a0f`) was retired at the
entry experiment in commit `bcdf1e61`. That commit restores `gdp_reformulate.py` and
`solver_tuning.py` to `d963f813` byte for byte.

- `e0_classify.py` / `.log`: GDPlib cstr has 75 disjunct rows, 70 of them linear. All 5
  nonlinear rows are bilinear equalities, and the detector proves 0 of them convex. The
  lowering check confirms the branch fired: big-M on 5, perspective on 0. With the flag
  OFF, nothing changes.
- `e1_arms.py ARM MAX_NODES`: one deterministic cstr solve. ARM is `hull`, `mixed` or
  `bigm`. The script prints an executed-assertion count.
  `arm_<arm>_<nodes>.log` holds each result.
- `draft_test_2142_hull_mixed_bigm.py.txt` is the soundness test drafted for step 2.
  It has three differential-battery shapes, fixed-selector optima and an
  unbounded-M refusal. It was never run, because the kill came first. The shapes can be
  reused.

Environment: the extension was built from `57743608`. Its Rust is source-equivalent to
`d963f813`, apart from one added test. `PYTHONHASHSEED=0`, deterministic,
`time_limit=7200` as a backstop. Up to 4 concurrent arms ran on 4 cores.
The big-M 10k arm matches §108's baseline bit for bit (0.5568486578).

> **Raw logs removed (#2142 follow-up).** The `.log` / `.log.gz` run traces cited in this directory were
> removed to keep checkouts small; the results they support are recorded in these notes. The originals
> remain in git history: `git show e39975e1:docs/dev/data/2142-hull-mixed/<name>.log.gz | gunzip` (or `.log`, no gunzip).
