# #1917 F1: the §5 graduate/retire panel for `DISCOPT_VUB_CMIR_ROOT`

This README is committed **before** the panel runs. The rule below is fixed, and the
verdict comes from `discopt_benchmarks/scripts/issue1917_vub_cmir_verdict.py`
applied to the two `.jsonl` files in this directory. Re-derive the verdict with:

```bash
python discopt_benchmarks/scripts/issue1917_vub_cmir_verdict.py \
    --corpus docs/dev/data/1917-vub-cmir-panel/corpus.jsonl \
    --gdp docs/dev/data/1917-vub-cmir-panel/gdp.jsonl
```

## Code under test

* The tip of `fix/1917-finish` at `6d3e2f79`. That includes the #1153 budget hoist
  `e91cd8f3` and the other merged #1917 work, so the OFF arm is the integration
  branch as it stands.
* The Rust extension is built with `maturin build --release` and its `.so` is
  unzipped into the worktree.
* Before launch, both `discopt.__file__` and `discopt._rust.__file__` are asserted to
  load from the worktree, and the audit marker is checked to read 51.

## Panels

All arms are interleaved: per instance and rep, OFF and ON run back to back, and the
order alternates. Every solve is a fresh subprocess. The load gate is `--max-load 12`:
a solve starts only while the 1-minute load is at most 12, and each row records the
load it started at. Background load from other agents is about 10-11.

* **Corpus:** `flag_onoff_corpus_panel.py --flag DISCOPT_VUB_CMIR_ROOT --reps 2
  --jobs 3 --time-limit 30 --max-load 12`, over the 74 in-repo `.nl`
  (`python/tests/data/minlplib_nl/`), with `minlplib.solu` as the oracle. ON
  incumbents are feasibility-checked with `check_feasibility`.
* **GDP:** `issue1917_gdp_flag_panel.py --flag DISCOPT_VUB_CMIR_ROOT --phase solve
  --all --reps 2 --time-limit 120 --max-load 12` over the 24 `gdp_parity` instances
  under `discopt-big-m`, run serially. Incumbents are re-checked in Pyomo, and
  `gb.score` gives each row its verdict.

That is at most 4 concurrent solver processes.

## Pre-registered rule

Fired instance:
* An instance has **fired** when the ON arm installed at least one root row
  (`vub_cmir/rows > 0`) in at least one rep.
* An instance with 0 rows installed runs the same mathematics in both arms; only the
  separator's wall time differs.

**GRADUATE** (default ON, `=0` opt-out, legacy path kept) only if all four hold:

1. **Cert-clean**, on both panels:
   * Corpus: no instance in any rep with a `problems` entry. That means no certified
     objective off the oracle, no infeasibility claim against a known solution, no
     dual bound past the MINLPLib best-known point (`=opt=`/`=best=`), and no
     infeasible incumbent.
   * GDP: no `incorrect` verdict, and no bound past the proven oracle.
2. **No certification regression.** No (instance, rep) where OFF certified and ON did
   not. Corpus "certified" is `gap_certified`; GDP "certified" is verdict `solved`.
3. **Node SGM ratio ON/OFF ≤ 0.95**:
   * Shift 10 nodes.
   * Computed over fired instances where both arms finished before the time limit
     (`optimal` or `infeasible`) in that rep, then per rep, then the mean over reps;
     the sd is reported.
   * Instances that hit the time limit are excluded from this bar. At a time limit
     the node count measures node throughput, not search effort, so a slower ON
     node would read as an improvement. The ratio including them is reported as
     secondary only.
4. **Wall SGM ratio ON/OFF ≤ 1.05**:
   * Shift 1 s.
   * Computed over every (instance, rep) pair on both panels with a finite wall in
     both arms, fired or not, so the separator's overhead counts. Per rep, then the
     mean over reps; the sd is reported. Per-panel ratios are reported too.

Pairs with an error in either arm are listed and excluded from the ratios. An error
that appears ON and not OFF is a cert-clean failure.

**Otherwise:**
* **RETIRE:** delete the flag, `vub_cmir.py`'s entry points, `set_root_rows` and the
  `solver.py` install hook in one commit, keep the test fixtures, and record this
  measurement in `docs/dev/flag-retirement-audit.md`.
* Or, only if a specific non-default use is named, **keep it as a documented
  opt-out** whose docstring says why it is not the default and what would change that.

## Result (run 2026-11-01 19:35 to 22:21, verdict applied 2026-11-02)

Data: `corpus.jsonl` (148 rows = 74 × 2 reps) and `gdp.jsonl` (96 rows = 24 × 2
arms × 2 reps). The rule above was applied unchanged, by the verdict script with
exit 1.

* **Coverage:** 196 (instance, rep) pairs and 0 errors. Rows were installed on 21
  instances: 13 corpus, and 8 GDP (gdp_col, methanol, mod_hens, modprodnet,
  eight_process, eight_process_logical, nonconvex_hen, syngas).
* **Load:** every solve started at load1 ≤ 12.

| bar | result | |
|---|---|---|
| 1 cert-clean | 0 pairs with a problem; 267 corpus incumbent feasibility checks; certified 116 → 116 corpus rows | PASS |
| 2 no certification regression | 0 lost; 1 gained (syngas rep 0) | PASS |
| 3 node SGM ON/OFF ≤ 0.95 (fired, both finished) | **1.0132**, sd 0.0250 (rep 0: 1.0309, n=10; rep 1: 0.9955, n=11) | **FAIL** |
| 4 wall SGM ON/OFF ≤ 1.05 (all pairs) | 1.0061, sd 0.0091 (corpus 0.9967 / 1.0105; GDP 1.0065 / 1.0191) | PASS |

Secondary, not a bar: the node ratio including time-limited runs is 0.8380, sd
0.0369. At a time limit that ratio measures node throughput, which is why the rule
excludes those runs.

Per instance (rep 0 / rep 1), where the rows changed something:
* **Helped:**
  * syngas: certified in both reps ON; OFF certified only in rep 1.
    Nodes 153 → 91 and 139 → 113.
  * heatexch_gen3: an incumbent where OFF had none, bound 43888 → 45304 / 45313.
  * beuster: bound 6352 → 6358.
  * ex1225 and nonconvex_hen: 5 → 3 nodes.
  * methanol: bound −4611.8 / −4612.3 → −4611.3.
* **Hurt:**
  * nvs05: 89 → 107 nodes in both reps.
  * eight_process: 27 → 39 nodes in both reps.
  * gdp_col: 57 → 63 and 63 → 89 nodes.
  * mod_hens: bound 48297 / 47623 → 46181 / 45319 at the time limit.
  * 4stufen rep 1: bound 21177 → 19713.
  * heatexch_gen1 rep 1: bound 95491 → 60599 at the time limit, though rep 0 went
    43877 → 60599.

**Verdict: NOT GRADUATE → RETIRE.**
* The cuts are sound everywhere they ran, but not net-positive.
* The only certification gained is syngas. A gain confined to one instance is
  CLAUDE.md §2's named-instance case, the same reading that retired
  `DISCOPT_GDP_DISJUNCT_FBBT`.
* No non-default use was named, so no opt-out is kept.
