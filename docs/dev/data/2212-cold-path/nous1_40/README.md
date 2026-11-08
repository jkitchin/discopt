# The §121 nous1 capture, re-taken on Linux

`capture_nous1.py --seed 5` (retry OFF, flag OFF, 60 s): 10,800 node LPs observed, **561 undecided,
all warm-started** (§121 on macOS: 590). The first 40 are replayed by `replay_nous1.py` from their
inherited basis, as the kernel solves them, against HiGHS (the `throughput_panel.py` oracle):

| file | arm | decided |
|---|---|---|
| `replay_off.jsonl` | flag OFF | 0 / 40 |
| `replay_on_before_warm_hook.jsonl` | flag ON, `c9d6e1c`..`30c9b84` | 27 / 40 (13 never reached the cascade: singular inherited basis, `solve_lp_cols_warm`'s own cold path) |
| `replay_on.jsonl` | flag ON, `210f400` | **39 / 40**, 0 wrong (#28: an attempt the outer original-data audit refused) |

#2212's bar: >= 36 / 40 without a caller-side retry. The 40 captured LPs with their inherited bases
are `nous1_undecided.npz` (432 KB), so the replay runs without re-capturing:
`DISCOPT_LP_LOGICAL_DUAL_COLD=1 python -u ../replay_nous1.py nous1_undecided.npz`.
