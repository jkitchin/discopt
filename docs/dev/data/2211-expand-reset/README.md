# #2211: the `DISCOPT_PRIMAL_EXPAND_RESET` §5 panel

The #956 T2′ EXPAND-free re-solve of an undecided cold primal had been default OFF
"pending the plan's §5 panel" since 2026-08-09, and the panel had never run. This is that
panel. The flag was **retired** on it (`docs/dev/flag-retirement-audit.md`).

| file | what |
| --- | --- |
| `corpus.jsonl`, `corpus.summary.log` | `issue2115_graduation_panel.py --flag DISCOPT_PRIMAL_EXPAND_RESET --deterministic --max-nodes 2000 --time-limit 120 --assert-unfired-identical --jobs 2` over the 74-instance in-repo corpus (oracle `cert-optima.json`), on the release build of base `288e200` (before this PR touched Rust; every row asserts `discopt` loaded from this checkout), each arm in its own process, order alternating |
| `control_cvxnonsep_r{1,2,3}.*` | the same panel on `cvxnonsep_nsig30` alone, three replicates run concurrently: the control for the one cert-clean violation |

**Result.** Fired: 230 rescues on 14 instances (the anti-vacuity check). Certified 59 → 59,
0 bounds tighter, 1 looser (`4stufen`: both arms hit the 120 s backstop, ON spent it on 1357
retries for 6 rescues and reached 15 nodes against OFF's 31). Nodes 6849 → 6795, wall
2263 → 2241 s (one replicate; directional only, CLAUDE.md §9). Bit-identical where it did
not fire: 53 checked, 7 skipped at the wall backstop.

**The one violation is not the flag.** `cvxnonsep_nsig30` reported "mechanism did not run but
objective differs" (130.62871265397885 vs 130.6287126437295, same bound, same 165 nodes, 0
retries in both arms). In the control, replicates 1 and 3 are identical across arms and
replicate 2 shows the same two values **in the opposite arms** to the corpus run. Each value
occurs under each setting, so this is run-to-run non-determinism of the incumbent at
`deterministic=True` and 1e-8 relative, independent of the flag (which, with 0 retries,
executes the same pivots as OFF). Recorded here and in the PR; not investigated further.
