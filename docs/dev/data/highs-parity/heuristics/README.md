# Gap 1 entry experiment: who finds HiGHS's first incumbent

Rows: `noinc.txt`, the 31 population rows (120-row MIPLIB population, 20 s, 1 thread)
where HiGHS 1.15.1 ends with an incumbent and discopt (`89671b22`) ends with none.

- `highs_log.py <instance> <log>` runs HiGHS with logging (20 s, 1 thread, rel gap 1e-4).
- `parse_hl.py <logs...>` reads the source letter and time of every new incumbent from
  the B&B display lines (`J` feasibility jump, `z`/`u` trivial zero/upper, `R`
  randomized rounding, `L` sub-MIP, `C` central rounding, `S` LP solve, `T` tree node,
  `H` heuristic) and prints the first one per row plus the counts.

Result (`highs_first_incumbent.summary.log`): 29 of the 31 rows get a first incumbent in
20 s; feasibility jump finds it on 12, a trivial point on 6 (5 `z`, 1 `u`), randomized
rounding on 4, sub-MIP on 5, central rounding / LP solve on 2. discopt had neither
feasibility jump nor the trivial heuristics; both are ported in
`crates/discopt-core/src/bnb/feas_jump.rs`.
