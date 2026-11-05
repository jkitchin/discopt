# #2143: panels with the four #2143 flags on (2026-11-04/05)

All runs are on the owner's machine (`kitchin.lan.local.cmu.edu`). Each row records
`discopt.__file__` and `git_head`.

## `DISCOPT_GDP_LOA_PROPOSER` (`main` 8f4b34e6)

- `gdp_loa_proposer.jsonl` / `.log`: the GDP parity panel. Command:
  ```
  DISCOPT_SUBNLP_REDUCED=1 DISCOPT_HEUR_FIXING_SCREEN=1 DISCOPT_ROOT_WARMSTART_FIRST=1 \
  DISCOPT_GDP_LOA_PROPOSER=1 python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \
      --flag DISCOPT_GDP_LOA_PROPOSER --phase solve --all --reps 2 --time-limit 120
  ```
  - The script toggles only the proposer, so the other three flags are on in both arms.
  - It ran as 4 `--include` groups in parallel, at load 10-20.
  - 192 solves: incorrect 0, bound crossings 0, solved-gained 0, SGM wall 31.48 -> 31.39 s.
  - One loss is reported: syngas big-M, 1/2 OFF vs 0/2 ON. That is load noise; see
    `syngas_lowload.jsonl` below.
- `corpus_bundle.jsonl` / `.log`: the in-repo corpus with all four flags as one bundle.
  - Command: `flag_onoff_corpus_panel.py`, 60 s, 2 jobs, 2 reps.
  - 148 rows: certified 118 -> 118, 272 incumbent feasibility checks, 0 errors.
- `syngas_lowload.jsonl`: syngas re-run one solve at a time, OFF/ON alternating, load 1.9-4.

  | method | proposer OFF | proposer ON |
  |---|---|---|
  | big-M | optimal, 139 nodes, 99.9 / 98.7 s | optimal, 139 nodes, 115.2 / 111.8 s |
  | hull | no incumbent, bound 4664.27, 31 nodes | incumbent 4669.0235, bound 4347.80, 3 / 7 nodes |

## Stage 4 panels (branch `fix/2143-root-budget`, 2026-11-05)

In each panel the flag under test toggles, and the other #2143 flags are ON in both arms.
GDP: `issue1917_gdp_flag_panel.py --phase solve --all --reps 2 --time-limit 120`, 4
`--include` groups in parallel. Corpus: `flag_onoff_corpus_panel.py --time-limit 60
--jobs 2 --reps 2`.

| files | flag | result |
|---|---|---|
| `pump_skip_gdp.*`, `pump_skip_corpus.*`, `pump_skip_spectralog_lowload.txt` | `DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT` (corpus on the default config) | cert-clean; spectralog hull loss was load noise (4/4 at low load) |
| `loa_bound_gdp.*` | `DISCOPT_GDP_LOA_PROPOSER_BOUND` | cert-clean; syngas hull 0/2 -> 2/2 |
| `screen_gdp.*`, `screen_corpus.*` | `DISCOPT_HEUR_FIXING_SCREEN` | cert-clean |
| `warmstart_corpus.*` | `DISCOPT_ROOT_WARMSTART_FIRST`, `--from-start` | cert-clean, neutral |

Summary and verdicts: `docs/dev/performance-plan.md` §106.4 and
`docs/dev/flag-retirement-audit.md`.
