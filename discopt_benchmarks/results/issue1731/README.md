# #1731 `DISCOPT_SPATIAL_OBJ_LATTICE` graduation panel (2026-10-17)

The record behind the flag's retirement (`docs/dev/flag-retirement-audit.md`).
The scripts ran against `f37dd291` (the flag and `_objective_lattice` present);
the flag is deleted, so they no longer run on main and are kept as the record.

- `filter.py` -- run `_objective_lattice` over the corpus candidates -> `population.csv` (108).
- `panel.py` + `one.py` -- OFF vs ON, 60 s, one subprocess per arm, interleaved -> `panel_60s.jsonl`.
- `analyze.py` -> `analysis.txt` (both §5 bars).
