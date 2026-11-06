# #2182 task 2: the big-M / hull per-solve fixed cost and `DISCOPT_GDP_CONFIG_ROOT_SKI`

The write-up is in `docs/dev/gdp-parity-baseline.md` ("#2182 task 2"). Host: a 4-core Linux
VM, main `63910b2`, Python 3.12, Pyomo 6.10.1. Each probe pinned to the core the G6.1 gate run
was not using.

| file | what |
|---|---|
| `root_split.jsonl` | `probes/root_split.py`: each solve split at `SolveResult.root_time` (10 instances × {big-m, hull}) |
| `cfg_probe.jsonl` | `probes/cfg_probe.py`: the GDP-config constructor per solve (grant, spend, sub-NLPs, distinct configurations, time of the last improvement). H1's class measurement |
| `entry.jsonl`, `entry.summary.txt` | H2 entry A/B (`probes/h2_drive.sh`, `probes/h2_ab.py`, scored by `probes/h2_score.py`): 15 instances × 2 methods × 3 reps, OFF/ON interleaved |
| `prereg.md`, `verdict.py` | the §5 panel's rule and scorer, committed (`24329ce`) before the first panel row |
| `gdp_parity_120s.jsonl` | panel A: `issue1917_gdp_flag_panel.py --phase solve --all --reps 2 --time-limit 120` |
| `corpus_60s_2rep.jsonl` | panel B: `flag_onoff_corpus_panel.py --time-limit 60 --reps 2 --jobs 1`, oracle `www.minlplib.org/minlplib.solu` fetched for the run |
| `verdict.txt` | `python verdict.py .` on the two panels: **GRADUATE** |
| `medterm_highs_version.txt`, `probes/medterm_highs_version.py` | the side finding: med_term_purchasing big-M certifies with highspy 1.11.0, not with 1.13.0-1.15.1 |

`probes/phase_probe.py` and `probes/children.py` are the cProfile attribution of
eight_process hull and ex_633 big-M quoted in the write-up. Their `.prof` dumps are not
kept.
