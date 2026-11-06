# #2170: the root pump skip needs a closed gap (2026-11-05)

The CLAUDE.md §5 panel for changing `DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT` (graduated in
#2143). Under the old rule both root pumps were skipped whenever an incumbent existed.
Under the new rule (`solver.py` `_root_pump_skip_decision`) they are skipped only when the
incumbent meets the solve's own gap test (`_gap_values_converged`, `gap_tolerance` /
`abs_gap_tol`) against the tree's global lower bound. That bound includes a
`seed_root_bound`, such as the LOA proposer's.

## Arms

| arm | code |
|---|---|
| `old` | PR #2173 head `e7d1a772` (worktree on `PYTHONPATH`, the branch's compiled `_rust`) |
| `new` | `69c2747a` (this branch) |

Every row records the `discopt` it imported and whether it has `_root_pump_skip_decision`.
The report refuses a row whose marker does not match its arm (CLAUDE.md §8): 0 of 96 did
not match.

- `gdp_parity` 24 instances x {big-M, hull}, 120 s. The two arms ran back to back per
  cell, in alternating order, in fresh `gdp_baseline.worker` subprocesses; 2 cells ran at
  a time on a 4-core host. Load 1-min was 1-5 (other probes ran beside it), so walls are
  single-rep and noisy.
- `mod_hens` and `kaibel` builders call an `ipopt` executable this host lacks. They were
  re-run in both arms with `shim_sitecustomize.py` on `PYTHONPATH`, which routes
  `SolverFactory("ipopt")` to the discopt Pyomo plugin for the builders' setup solves
  only. It reproduces #2144's Cafaro coefficients (k = 14.915304, b = 0.0305925). These
  rows carry `"shim": true`.

`panel.py`, `worker.py`, `report.py` are the drivers as run (paths point at the session
scratchpad). `report.txt` is `report.py gdp_parity_2arm.jsonl`.

## Result

**Cert-clean**: 0 incorrect, 0 dual bounds past an oracle, 24 -> 24 solved, 76
incumbents re-checked in Pyomo, all verified.

**One objective changed, the target:** methanol hull **645.51 -> -1793.43** (BARON's
best known). The pumps now run at 55 s ("open against the bound -4699.67"), and the
NLP-relaxation pump finds -1793.43 at ~97 s, as on `main` before #2144.

**Where the gate now differs from the old rule** (pumps run, old skipped): 18 cells
besides methanol hull. On every one the objective is unchanged.

- Certified in both arms (9 cells: modprodnet big-M, eight_process x2,
  eight_process_logical x2, ex_633 x2, nonconvex_hen x2): identical objective, bound and
  node count. Wall cost of the pumps: up to +4.4 s (nonconvex_hen hull 13.5 -> 17.9 s,
  eight_process hull 15.8 -> 18.7 s).
- Not certified in either arm (9 cells: cstr hull, gdp_col big-M, hda hull, methanol
  big-M, modprodnet hull, spectralog x2, mod_hens x2): same incumbent. The tree gets fewer
  nodes within 120 s where the pumps took time (e.g. mod_hens hull 61 -> 3, gdp_col big-M
  15 -> 7, methanol big-M 155 -> 95), and the time-limited bound moves with it, in both
  directions (methanol big-M -4614.16 -> -4622.43, mod_hens big-M 34146.10 -> 33728.03,
  spectralog hull 10.10 -> 11.74). mod_hens big-M, the #2150/#2144 kill cell, stays at
  106766.53.

Over the 24 cells both arms certify, the SGM wall ratio new/old is **1.039** (sd of log
0.111, single rep). Total panel wall: old 2725 s, new 2716 s.

**syngas hull**, the cell the flag was graduated on: it has no incumbent at 120 s on this
host in either arm, because the root reaches the LOA proposer at ~60 s and LOA needs ~45 s
more. So it was run separately at 240 s, three arms side by side:

| arm | gate | status | wall |
|---|---|---|---|
| new | skipped: 4669.023485 meets the gap test against 4668.808968 | optimal | 119.5 s |
| old | skipped: an incumbent exists | optimal | 118.8 s |
| `=0` | pumps run | optimal | 242.3 s |

The #2143 gain is kept. That case is the closed-gap case: LOA's bound is installed before
the pump stage.

## Reading

Of the old rule's skips, the only ones that saved anything on this panel were closed-gap
(syngas). The open-gap skips saved at most a few seconds on certified cells, and on
methanol hull they cost the incumbent. The new gate keeps the closed-gap skip and returns
open-gap cells to the legacy pump behaviour.
