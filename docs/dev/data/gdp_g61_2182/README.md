# #2182 task 1: G6.1 re-run on main `63910b2` (2026-11-06)

The write-up is in `docs/dev/gdp-parity-baseline.md` ("#2182 task 1"). Score:

    python discopt_benchmarks/scripts/gdp_g61_score.py docs/dev/data/gdp_g61_2182

| file | what |
|---|---|
| `rep1.jsonl` | repeat 1: 24 `gdp_parity` instances × 14 arms, `--jobs 3`, 300 s + 120 s grace |
| `timing_rep2.jsonl`, `timing_rep3.jsonl` | the 172 (instance, family) pairs both sides solved in rep 1, re-run |
| `serial.jsonl` | the jobs=1 control: two pairs per family |
| `pairs.txt` | the pairs repeats 2-3 re-ran |
| `score.txt` | the scorer's output on this directory |
| `control_pyomo_scipexe.jsonl` | Pyomo big-m / hull re-run on the SCIP executable (`GDP_BASELINE_GLOBAL=scip`, #2149's setting) |
| `old_vs_new_same_host.jsonl`, `.summary.txt` | `old_new_ab.py`: the #2149 build `1f16c936` (rebuilt here) vs main, every cell main solved, interleaved, one run per pinned core; `old_new_score.py` scores it |
| `run_g61.sh`, `run_scipexe_control.sh`, `run_tail.log` | the drivers as run (they hard-code the session's scratch paths) and the tail of the run log |
