# #2184 review finding 1: short time limits and the shrunk root share

The review asked whether `DISCOPT_GDP_CONFIG_ROOT_SKI` can delay the first constructor
incumbent past a short time limit. A pre-constructor phase of a few ms would give the root
call a few ms (no sub-NLP), with the rest held back about 9-15 s.

| file | what |
|---|---|
| `gdp_tl10.jsonl`, `gdp_tl20.jsonl` | `issue1917_gdp_flag_panel.py --phase solve --all --reps 1` at 10 s and 20 s, OFF/ON (`run.sh`) |
| `score.txt`, `score.py` | per cell, OFF vs ON: incumbent lost or gained, objective better or worse, solved changed |
| `cfg_on.jsonl` | `../probes/cfg_probe.py` with the flag ON (the default) on the 12 cells where the share shrinks: the root call's grant, sub-NLPs and feasible points |

Result: **0 incumbents lost** at 10 s and at 20 s (48 cells each); no objective worse, no
certification changed, 0 incorrect. On the 12 cells where the share shrinks, the root
grant is 0.071-0.876 s (lowering, presolve and the root LP always precede the
constructor), and every root call runs at least 3 sub-NLPs. 11 of 12 find a feasible point
at the root, the last improvement by 0.18 s. clay0203 hull finds none at the root in either
arm (0 of 7 with the 1.5 s grant in `../cfg_probe.jsonl`) and certifies anyway.
