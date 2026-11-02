# Pre-registration: §5 graduation panel for `DISCOPT_MBIGM_LP_COMPONENTS` (#1917)

Written 2026-11-01, **before any panel data exists**. The committed version of this
file is the rule; nothing in it is revised after the first panel row is written.
If a rule turns out to be unworkable, record the change as a dated amendment below
the original text, and do not edit the text itself.

The M differential and lowering walls in the audit row
(`docs/dev/flag-retirement-audit.md`) are the entry experiment. They motivated this
panel, and none of them is reused as panel data: every number the verdict reads
comes from the runs below.

## What the flag does, and why it is §5 bound-changing

ON solves each mbigm big-M LP (`_compute_big_m_lp`) over the connected components
of the global linear relaxation's column-row graph that the row's body touches
(`_component_sub_lp`). When those components hold every column, the full LP is
solved unchanged. Dropping rows relaxes the LP, so in exact arithmetic an M can
only stay or grow. In floating point the two arms are different LPs, and the
in-house simplex can succeed on one and fail on the other: on `batch_processing`
it reports `numerical` on 448 of 450 full LPs and on 402 of the sub-LPs, so 36 M's
are smaller ON, because OFF falls back to the interval M there. A flag that can make
an M smaller changes the lowering, and so §5 treats it as bound-changing.

**Where it acts.** Only on a GDP lowered by mbigm: `gdp_method="mbigm"`, or
`"auto"` when the advisor routes a disjunction to mbigm. `Model.solve` defaults to
`big-m`, and `_compute_big_m_lp`'s only callers are the mbigm branches of
`reformulate_gdp` (`_relax/gdp_reformulate.py`). `gdp/user_big_m.py` calls
`_precompute_lp_relaxation` but solves its own LPs. The in-repo `.nl` corpus
(`python/tests/data/minlplib_nl/`) has no disjunctions, so the flag cannot touch any
of its solves, and it is **not** a panel here. That is the opposite of
`DISCOPT_SUBNLP_IMPLIED_BOX`, which acts on every route.

## Arms, panels, reps

* **Arms:** `DISCOPT_MBIGM_LP_COMPONENTS=1` (ON) against `=0` (OFF). Nothing else
  differs. Every (instance, arm) run is a fresh subprocess. Arms are interleaved
  per instance, and their order alternates by instance and by rep.
* **Instances:** the 24-instance `gdp_parity` panel (`gdplib_runner.GDP_PARITY`),
  method `mbigm`. All 24 are run in every panel (`--all` for S). A lowering
  fingerprint cannot rule an instance out, because on 23 of 24 the flag changes
  only the lowering wall, which is its intended effect.
* **Panel L, lowering** (`issue1917_mbigm_components_lowering.py --phase lower`):
  * `create_using("gdp.mbigm")` per (instance, arm, rep), 2 reps, `--timeout 900`;
  * each row records the lowering wall, every `_compute_big_m_lp` return (exact,
    as float hex) and the LP statuses.
  * OFF `biofuel` is expected to be `killed`: its entry lowering took 3814 s.
* **Panel A, soundness audit** (the same script, `--phase audit`):
  * one ON lowering per instance;
  * each OPTIMAL component sub-LP that differs from the full LP is re-solved as the
    full LP by HiGHS, up to `--audit-cap 2000` per instance in call order (the
    rest are counted, not dropped).
  * The sub-LP is a relaxation, so its minimum must not exceed the full minimum.
* **Panel S, solve** (`issue1917_gdp_flag_panel.py --phase solve --all`):
  * `gdp_baseline.py --worker <inst> discopt-mbigm`, 2 reps, `--time-limit 120
    --grace 600`;
  * the worker re-checks every incumbent in Pyomo.
  * Lowering runs inside `Model.solve`'s clock (`solver.py`, after `_solve_t0`) but
    cannot be interrupted. So a lowering that overruns 120 s ends the solve late
    rather than early, and the 600 s grace lets ON `biofuel` (146 s entry lowering)
    finish. OFF `biofuel` is expected to be `killed` at 720 s.
* **Machine:** every thread env set to 1, and one process at a time: L, then A,
  then S, serially. Load gate `--max-load 12` on all three, and each row records
  its `load1`. Not to start before team-lead's "machine free".

## Commands (run from the worktree root, with PYTHONPATH pinned to it)

```bash
# §8 preflight: the loaded discopt is this tree, and the flag code is present.
python -c "import discopt, os; from discopt._relax import gdp_reformulate as G; \
  assert discopt.__file__.startswith(os.getcwd()), discopt.__file__; \
  assert hasattr(G, '_component_sub_lp') and hasattr(G, '_lp_box_memo')"

D=docs/dev/data/1917-flag-panel
python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \
    --phase lower --reps 2 --timeout 900 --max-load 12 \
    --out $D/DISCOPT_MBIGM_LP_COMPONENTS.lower.jsonl
python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \
    --phase audit --timeout 7200 --audit-cap 2000 --max-load 12 \
    --out $D/DISCOPT_MBIGM_LP_COMPONENTS.audit.jsonl
python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \
    --flag DISCOPT_MBIGM_LP_COMPONENTS --phase solve --all --reps 2 \
    --time-limit 120 --grace 600 --max-load 12 \
    --out $D/DISCOPT_MBIGM_LP_COMPONENTS.jsonl

python discopt_benchmarks/scripts/issue1917_mbigm_components_verdict.py \
    --lower $D/DISCOPT_MBIGM_LP_COMPONENTS.lower.jsonl \
    --audit $D/DISCOPT_MBIGM_LP_COMPONENTS.audit.jsonl \
    --solve $D/DISCOPT_MBIGM_LP_COMPONENTS.jsonl
```

The verdict script is committed along with this file, before any data, and it
applies exactly the bars below. `--self-test` runs it on 12 synthetic panels
(clean, void, each C and N failure, K3), and each one must reach its expected exit
code. A "cell" is an instance.

## Panel validity (V). A failure voids the run; it is not a verdict.

* V1. Every row that ran to completion records a `discopt` path inside the worktree
  under test (§8). `killed` rows and S `error` rows record none and are exempt;
  an `error` row is judged by C4 / C6 instead.
* V2. Row counts: L = 24 × 2 × 2 = 96, A = 24, S = 24 × 2 × 2 = 96, and no instance
  is missing a rep. A missing row is a void run, never a silent skip (§6).
* V3. At least one ON incumbent in S is Pyomo-checked (`in_pyomo` recorded).
* V4. A re-solved at least one sub-LP over the whole panel, and no A row was
  `killed`.
* V5. No OFF row in L errored. OFF is today's default path, and if it crashes the
  instrument is broken.
* V6. Within each arm, the M's in L are identical between the two reps. A lowering
  that is not deterministic makes the per-row comparison meaningless.

## Cert-clean bar (C). Any violation, in any rep, means the flag does not graduate.

* C1. 0 `incorrect` verdicts in S (`gdp_baseline.score`): a refuted optimum claim,
  an incumbent better than the oracle, a false infeasibility claim, or an incumbent
  that violates the original Pyomo model.
* C2. 0 S dual bounds past a proven oracle (`_bound_crosses`).
* C3. No instance `solved` OFF in a rep and not `solved` ON in the same rep.
* C4. No instance where ON errors in both reps of S and OFF errors in neither.
* C5. 0 audit violations in A: no OPTIMAL sub-LP minimum above the HiGHS full-LP
  minimum by more than 1e-9 relative. That would be an M below the exact full-LP M,
  which is an unsound M.
* C6. No ON lowering error, in L or in A.
* C7. Where both arms lowered in the same rep of L, the number of LP M's is the
  same. The flag changes how each LP is solved, never which rows get one.

## Net-positive bar (N). All five must hold.

Only effects that repeat in **both** reps count. An instance that flips between
reps counts as neither a gain nor a loss.

* N1. **Lowering faster, broadly:** in at least 2 distinct instances, ON lowers in
  both reps and in each rep either OFF was `killed` or ON took at most 0.8 × OFF's
  lowering wall. Lowerings under 1 s in both arms are below timing resolution on a
  shared machine and do not count.
* N2. **Lowering not slower anywhere:** there is 0 instance where, in both reps,
  ON's lowering wall is at least 1.25 × OFF's (the longer one at least 1 s), or ON
  was `killed` while OFF lowered.
* N3. **M's not looser:** on no instance are more M's larger ON than smaller ON.
  Compare the first rep in which both arms lowered, row by row.
* N4. **No result lost:**
  * no instance with an S incumbent in both reps OFF and in neither rep ON;
  * the number of instances `solved` in both reps ON is at least the number OFF.
* N5. **No tree inflation:** over instances `solved` in both reps by both arms, the
  geometric mean of `(nodes_ON + 1) / (nodes_OFF + 1)` (per-arm median over reps)
  is at most 1.05.

Solve wall is reported, not gated. The exception is K3. A flag whose benefit is
the lowering does not have to show a faster solve to graduate. N1 measures that
benefit directly, and S guards against harm.

## Outcomes, decided in advance

* **Graduate** (default ON, keep `=0` and the OFF path): V, C and N all hold.
* **Retire:** C holds and N fails, or K3. In one PR that records this panel:
  * delete the flag, `_LPComponents`, `_component_sub_lp`, `_LPData.components`
    and the `by_component` memo-key field;
  * keep `_lp_box_memo` and the `_LPData.optima` memo, both bound-neutral and
    unflagged;
  * keep this panel's rows and the reproducer script;
  * keep the tests as regression fixtures where they still apply.

## Kill conditions

* K1. Any C violation. The flag stays OFF. Root-cause the violation before anything
  else, and retire the flag if it causes the violation.
* K2. A V failure. The run is void; fix the instrument and rerun the whole panel.
  Never patch a partial run.
* K3. The ON/OFF ratio of S's SGM solve `elapsed` (shift 10 s, capped at 720 s)
  exceeds 1.25 in both reps. Retire even if N holds.
