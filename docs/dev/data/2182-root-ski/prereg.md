# Pre-registration: §5 panel for `DISCOPT_GDP_CONFIG_ROOT_SKI` (#2182)

Written 2026-11-06, **before any panel row exists** (the entry A/B in `entry.jsonl` is
the hypothesis test that motivated the flag, and is not part of this panel). The
committed version of this file is the rule. If a rule proves unworkable, add a dated
amendment below the original text; do not edit the text.

## What the flag does, and why §5 calls it bound-changing

Under `DISCOPT_GDP_CONFIG_DEFER` (default ON) the root call of the GDP-config
constructor gets a fixed 1.5 s and the rest of its grant is held back until the search
has run as long as that rest. ON, the root share is `min(1.5 s, time the solve spent
before the constructor)`; the hold-back is unchanged. No bound, cut or relaxation
changes. Which incumbents are found, and when, does change, and through pruning so do
node counts and the dual bound at a time limit. §5 therefore treats it as
bound-changing. Its intended benefit is **wall time** on small GDPs, so unlike the
incumbent-heuristic panels the net-positive bar below gates on wall.

## Arms, panels, reps

* **Arms:** `DISCOPT_GDP_CONFIG_ROOT_SKI=1` (ON) against `=0` (OFF); nothing else
  differs. Fresh subprocess per (instance, arm) solve, arms interleaved per instance,
  order alternated by instance and rep (the panel scripts' own scheduling).
* **Panel A, GDP:** the 24-instance `gdp_parity` panel × `gdp_method ∈ {big-m, hull}`
  = 48 cells, all solved (`--all`): the flag acts at solve time, so a lowering
  fingerprint cannot rule a cell out. **2 reps, 120 s.**
* **Panel B, corpus:** the 74 `.nl` files in `python/tests/data/minlplib_nl/` with
  the MINLPLib oracle (`www.minlplib.org/minlplib.solu`, fetched for the run).
  **2 reps, 60 s.**
* **Placement:** each panel runs as one process pinned to one core (`taskset`), with
  `--jobs 1`. The host is shared with the #2182 G6.1 gate run on the other cores; the
  interleaved arms see the same neighbours.

## Commands (from the worktree root, `PYTHONPATH` pinned to it)

```bash
python -c "import discopt, os, discopt.solver as S; \
  assert discopt.__file__.startswith(os.getcwd()), discopt.__file__; \
  assert hasattr(S, '_gdp_config_root_ski_enabled')"
taskset -c 3 python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \
    --flag DISCOPT_GDP_CONFIG_ROOT_SKI --phase solve --all --reps 2 \
    --time-limit 120 --out docs/dev/data/2182-root-ski/gdp_parity_120s.jsonl
taskset -c 3 python -u discopt_benchmarks/scripts/flag_onoff_corpus_panel.py \
    --flag DISCOPT_GDP_CONFIG_ROOT_SKI --solu <dir>/minlplib.solu \
    --time-limit 60 --jobs 1 --reps 2 \
    --out docs/dev/data/2182-root-ski/corpus_60s_2rep.jsonl
python docs/dev/data/2182-root-ski/verdict.py docs/dev/data/2182-root-ski
```

`verdict.py` is committed with this file, before any data, and applies exactly the
bars below. A *cell* is an (instance, method) pair in A, an instance in B.

## Panel validity (V). A failure voids the run; it is not a verdict.

* V1. Every row's `discopt` / `discopt_file` path lies in the worktree under test (§8).
* V2. Executed solves: 48 × 2 arms × 2 reps = 192 in A; 74 × 2 × 2 = 296 in B. A
  missing row voids the run (§6).
* V3. At least one ON incumbent is independently feasibility-checked in each panel
  (A: the Pyomo re-check; B: `check_feasibility`).

## Cert-clean (C). Any violation, in any rep, and the flag does not graduate.

* C1. A: 0 `incorrect` verdicts (`gdp_baseline.score`). B: 0 rows with `problems`.
* C2. A: 0 dual bounds past a proven oracle.
* C3. No cell `solved` (A) / `cert` (B) OFF in a rep and not ON in the same rep,
  unless it also flips in the OFF arm across reps (a cell OFF itself does not hold
  is time-limit noise, and is reported by name, never silently dropped).
* C4. No cell where ON errors in both reps and OFF in neither.

## Net-positive (N). All must hold.

* N1. **Wall, where it acts.** Over A cells certified by both arms in both reps, the
  SGM wall (shift 1 s) ratio ON/OFF is ≤ 0.95 **in each rep separately**.
* N2. **No incumbents lost:** 0 cells (A or B) where OFF has an incumbent in both reps
  and ON has none in either.
* N3. **Primal quality not worse** on A cells uncertified in both arms and both reps
  with incumbents in all four runs: cells with ON better in both reps (rel 1e-4,
  objective sense) ≥ cells with ON worse in both reps.
* N4. **No tree inflation:** geometric mean of `(nodes_ON + 1) / (nodes_OFF + 1)`
  (per-arm median over reps) over cells certified by both arms in both reps is ≤ 1.05,
  in A and in B separately.
* N5. **Neutral elsewhere:** B's SGM wall (shift 1 s) ratio ON/OFF ≤ 1.05 in each rep.

## Outcomes, decided in advance

* **Graduate** (default ON, keep `=0` and the fixed-share path): V, C and N hold.
* **Retire:** C holds and N fails. Delete the flag and `_gdp_config_root_ski_enabled`
  in one PR recording this panel; keep `test_2182_gdp_config_root_ski.py`'s rule tests
  only where they still apply.
* **K1:** any C violation: the flag stays OFF; root-cause it before anything else.
* **K2:** any V failure: the run is void; fix the instrument and rerun both panels.
  Never patch a partial run.
