# #1917 §5 flag panels: partial preview (2026-10-31, main `4517c7b`)

**Superseded** by the full 24-instance panels in `../1917-flag-panel/` (2026-11-01),
which graduated `DISCOPT_GDP_BOUND_PRETRANSFORM` and retired
`DISCOPT_MBIGM_CONDITIONED`. Kept as the record of what the preview showed.

**Not a graduation or retirement panel.** The container had only the PyPI GDPlib
(20.6.2): 13 of the 24 `gdp_parity` instances do not exist in it, and the 4 that
do (`biofuel`, `gdp_col`, `stranded_gas`, `syngas`) fail to build (missing data
files / `AttributeError`). Installing the git GDPlib was not permitted. What ran
is the 7 Pyomo GDPopt models, via `issue1917_gdp_flag_panel.py` (fire, then solve:
3 reps, OFF/ON interleaved, 120 s). **Wall times are contaminated**: other test runs
shared the 4-core machine (load1 up to 8.7), so only solved counts, verdicts and
node counts are evidence here.

| flag | cells fired / lowerable | cert-clean | solved OFF -> ON | nodes OFF -> ON (median) |
|---|---|---|---|---|
| `DISCOPT_GDP_BOUND_PRETRANSFORM` | 6 / 14 | 0 incorrect, 0 bound crossings, 0 lost | 18/18 -> 18/18 | eight_process 33 -> 27, eight_process_logical 25 -> 17 (big-m) / 19 (mbigm), nonconvex_hen 3 -> 5 |
| `DISCOPT_HULL_CONIC_QUADRATIC` | 1 / 7 | clean | 2/3 -> 3/3 (`constrained_layout_clay0203`) | 387 -> 109 |
| `DISCOPT_MBIGM_CONDITIONED` | 7 / 7 | clean | 21/21 -> 21/21 | identical on 5; clay0203 135 -> 161, nonconvex_hen 3 -> 5 |

Reproduce the report: `python discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py
--flag <FLAG> --report --out docs/dev/data/1917-flag-panel-preview/<FLAG>.jsonl`.
The conic row agrees with its full 2026-10-22 panel (fires on 1 of 19, OFF unsolved,
ON solved). The full panel needs `pip install "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"`.
