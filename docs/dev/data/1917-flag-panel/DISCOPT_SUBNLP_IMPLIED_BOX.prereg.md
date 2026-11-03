# Pre-registration: §5 graduation panel for `DISCOPT_SUBNLP_IMPLIED_BOX` (#1917)

Written 2026-11-01, **before any panel data exists**. The committed version of this
file is the rule; nothing in it is revised after the first panel row is written.
If a rule turns out to be unworkable, record the change as a dated amendment below
the original text, and do not edit the text itself.

## What the flag does, and why it is §5 bound-changing

ON solves each primal-heuristic sub-NLP on the box that node FBBT implies
(`_tighten_node_bounds_with_status`, the same routine B&B node NLPs use). The
dual-bound machinery does not change. The incumbents the heuristics find do change,
and through pruning so do node counts and the dual bound at the time limit. §5
therefore treats it as bound-changing. Entry experiment (6d3e2f79, audit row):
`Invalid_Number_Detected` went from 164 to 6 on `ex_633_trespalacios`/hull, and
hull `hda` and `gdp_col` find an incumbent ON but none OFF.

## Arms, panel, reps

* **Arms:** `DISCOPT_SUBNLP_IMPLIED_BOX=1` (ON) against `=0` (OFF). Nothing else
  differs. Each (instance, arm) solve runs in a fresh subprocess. Arms are
  interleaved per instance, and their order alternates by instance and by rep.
* **Panel A, GDP:** the 24-instance `gdp_parity` panel (`gdplib_runner.GDP_PARITY`)
  × `gdp_method ∈ {hull, big-m}` = 48 cells. Every cell is solved (`--all`). The
  flag acts at solve time, so a lowering fingerprint cannot rule any cell out.
* **Panel B, in-repo corpus:** every `.nl` file in `python/tests/data/minlplib_nl/`
  (74 instances), with the MINLPLib `minlplib.solu` oracle. Heuristics run on every
  route, not only on GDP lowerings, which is why this panel is included.
* **Reps:** 2 per arm. **Time limit:** 120 s.
* **Threads:** every thread env set to 1. At most 2 processes: panel A runs
  serially, alongside panel B with `--jobs 1`. Load gate `--max-load 12` on panel A.

## Commands (run from the worktree root, with PYTHONPATH pinned to it)

```bash
# §8 preflight: the loaded discopt is this tree, and the flag code is present.
python -c "import discopt, os; from discopt._relax import primal_heuristics as P; \
  assert discopt.__file__.startswith(os.getcwd()), discopt.__file__; \
  assert hasattr(P, '_subnlp_implied_box_enabled') and hasattr(P, '_DeadlineCallbacks')"

python -u discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py \
    --flag DISCOPT_SUBNLP_IMPLIED_BOX --phase solve --all --reps 2 \
    --time-limit 120 --max-load 12 --out docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.jsonl
for rep in 0 1; do
  python -u discopt_benchmarks/scripts/flag_onoff_corpus_panel.py \
      --flag DISCOPT_SUBNLP_IMPLIED_BOX --solu <corpus>/minlplib.solu \
      --time-limit 120 --jobs 1 \
      --out docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.corpus.rep$rep.jsonl
done

python discopt_benchmarks/scripts/issue1917_implied_box_verdict.py \
    --gdp docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.jsonl \
    --corpus docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.corpus.rep0.jsonl \
             docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.corpus.rep1.jsonl
```

The verdict script is committed along with this file, before any data, and it
applies exactly the bars below. A "cell" is an (instance, method) pair in panel A,
or an instance in panel B.

## Panel validity (V). A failure voids the run; it is not a verdict.

* V1. Every row's recorded `discopt` / `discopt_file` path lies in the worktree
  under test (§8).
* V2. The executed-solve count is 48 × 2 × 2 = 192 for panel A and 74 × 2 × 2 = 296
  for panel B. A missing row is a void run, never a silent skip (§6).
* V3. At least one ON incumbent is feasibility-checked: Pyomo re-check in A,
  `check_feasibility` in B.

## Cert-clean bar (C). Any violation, in any rep, means the flag does not graduate.

* C1. 0 `incorrect` verdicts in A (`gdp_baseline.score`): a refuted optimum claim,
  an incumbent better than the oracle, a false infeasibility claim, or an incumbent
  that violates the original Pyomo model.
* C2. 0 dual bounds past a proven oracle in A. In B, 0 `problems` rows: a certified
  objective different from `=opt=`, `infeasible` claimed where MINLPLib has a
  solution, a bound beyond the oracle, an infeasible incumbent, or certification
  lost ON.
* C3. No cell that is `solved` (A) or `gap_certified` (B) in a rep OFF is unsolved
  ON in the same rep.
* C4. No cell where ON errors in both reps and OFF errors in neither.

## Net-positive bar (N). All four must hold.

Only effects that repeat in **both** reps count. A cell that flips between reps is
noise and counts as neither a gain nor a loss.

* N1. **Incumbents gained, broadly:** at least 3 cells, over at least 2 distinct
  instances, where ON has an incumbent in both reps and OFF has none in either.
* N2. **No incumbents lost:** 0 cells where OFF has an incumbent in both reps and
  ON has none in either.
* N3. **Primal quality not worse:** among cells where both arms have incumbents in
  both reps and neither certifies in either rep, the number of cells where ON is
  better in both reps (rel 1e-4 in the objective sense) is at least the number
  where it is worse in both reps.
* N4. **No tree inflation:** over cells certified by both arms in both reps, the
  geometric mean of `(nodes_ON + 1) / (nodes_OFF + 1)` (per-arm median over reps)
  is at most 1.05.

**Wall time is reported, not gated,** because the machine is shared and loaded.
The one exception is kill condition K3.

## Outcomes, decided in advance

* **Graduate** (default ON, keep `=0` and the OFF path): V, C and N all hold.
* **Retire:** C holds and N fails. Delete the flag and the code that only it uses
  (the `_solve_sub_nlp` implied-box branch and `_subnlp_implied_box_enabled`), in
  one PR that records this panel. Keep `_implied_box`, which the default-on FBBT
  refusal in `subnlp` (c5f98e36) uses. Keep the tests as regression fixtures where
  they still apply.

## Kill conditions

* K1. Any C violation. The flag stays OFF. Root-cause the violation before anything
  else; if the flag causes it, retire.
* K2. A V failure. The run is void; fix the instrument and rerun the whole panel.
  Never patch a partial run.
* K3. The ON/OFF ratio of the SGM wall (shift 10 s, capped at 120 s) exceeds 1.25
  in both reps. The per-sub-NLP FBBT then costs more than it buys, so the flag is
  retired even if N holds.
