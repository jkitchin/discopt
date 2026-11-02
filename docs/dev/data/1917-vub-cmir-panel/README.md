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
