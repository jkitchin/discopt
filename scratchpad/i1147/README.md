# #1147 — complementarity provenance: the measurements

* `repro_1147.py` — the entry experiment. Prints the pair count before/after each
  rebuilding pass. On `main`: `after=0` for GDP (big-m/hull/mbigm),
  `expand_integer_products` and `factorable_reformulate`; `binary_multilinear`
  abstains. After the fix: `after=1` on all of them.
* `panel_1147.py` — the bound-neutral panel over the 66-instance in-repo `.nl`
  corpus. Asserts the loaded module and a version marker (CLAUDE.md §8), prints
  per-instance progress (§10) and an executed-solve count (§6).
  `panel_before.json` / `panel_after.json` are its two arms at a 10 s limit
  (`marker: false` / `marker: true`). Diff: status and objective identical
  66/66; every recorded field identical on all 44 converged instances.
* `probe_two.py` — the two instances whose `node_count` differed on the panel
  (`bchoco07`, `tls2`). Both are unconverged (`time_limit` / `feasible`) runs,
  and both values reproduce *within* an arm across interleaved repeats
  (2 rounds x 3 reps per arm), so the panel difference is a wall-clock artifact,
  not a bound change (CLAUDE.md §9).

Run from the repository root.
