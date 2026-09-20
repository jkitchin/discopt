# #1370 Part B entry experiment — vectorising identical blocks

Date: 2026-09-20. Tree: `835e822` (Part A landed). Harnesses:
`scratchpad/issue1370/block_eval_entry.py`, `scratchpad/issue1370/eval_share.py`.
Machine: this container, 1-min load average 0.42–0.55 at every run, arms
interleaved, medians over 5–15 reps with the spread reported (CLAUDE.md §9).

## Hypothesis and kill criterion, stated before the run

> Identical blocks share one sparsity pattern and one coloring, so they admit a
> single vectorised pass.

**KILL:** if one vectorised pass over K identical blocks is not at least 2x the
default evaluator's Jacobian + Lagrangian-Hessian cost at K = 64, Part B does
not ship.

## What was compared, and what was verified first

Model class, not instance: K structurally identical dynamic blocks (one
discretized trajectory each) coupled only through a shared parameter vector —
the shape `dae.fit.fit_trajectories`, `stochastic.extensive_form` and the
issue's SCOPF case all produce.

* **Baseline** — the default evaluator (POUNCE's Rust AD tape) on the whole
  model. NOT the `.nl`/ASL evaluator the issue's numbers were taken against;
  that substitution is what decides whether a win is *structure* or *engine*.
* **Candidate (compressed)** — C directional derivatives / Hessian-vector
  products over all K blocks at once, C from a greedy coloring of the block's
  pattern **taken from the emitted NLP**, not assumed from the model source.
* **Candidate (dense)** — a vmapped `jacfwd`/`hessian` per block, reported as
  the upper bound on the candidate's work.

Before any timing, the template was checked against the tape: block 0's
Jacobian and Lagrangian-Hessian entries reconstructed from
`jacobian_structure`/`hessian_structure` and compared elementwise. **348–3560
entries compared, max |diff| 2.2e-16 to 4.4e-16.** The harness exits non-zero
if that comparison count is zero or the difference is real — a candidate that
is fast because it computes something else is not a candidate.

## Result 1 — the mechanism holds

120 columns per block (steps=20, dim=6), 15 reps, 2 Jacobian / 1 Hessian colors:

| K | n | Jacobian | Lagrangian Hessian | combined |
|---|---|---|---|---|
| 2 | 246 | 0.51x | 1.36x | below 1 |
| 4 | 486 | 0.74x | 2.15x | 1.4x |
| 16 | 1,926 | 2.42x | 4.65x | 3.4x |
| 64 | 7,686 | 3.13x | 7.80x | **5.24x** (5.67x on the 15-rep repeat) |

1200 columns per block (steps=60, dim=20), 9 reps — block sizes near the
issue's 2,969-variable SCOPF block:

| K | n | Jacobian | Lagrangian Hessian | combined |
|---|---|---|---|---|
| 8 | 9,620 | 3.58x (1.429 → 0.400 ms, sd 0.541/0.063) | 10.96x (3.252 → 0.297 ms, sd 0.900/0.028) | 8.0x |
| 32 | 38,420 | 8.31x (7.470 → 0.898 ms, sd 1.695/0.076) | 30.19x (20.521 → 0.680 ms, sd 0.066) | **17.74x** |

**VERDICT: PROCEED** on the kill criterion — 5.24x against a 2x bar at K=64,
17.74x at realistic block sizes.

Two things the table says beyond the verdict:

* The win **is** the structure, not just the engine: the *dense* per-block arm,
  same engine and same blocks but no compression, runs 0.49x–1.77x — it loses
  to the tape almost everywhere. Compression over a shared pattern is what
  produces the gap.
* The win **needs** K. At K=2–4 the vectorised pass is at or below parity; it
  crosses 2x combined somewhere around K=8 and grows from there.

## Result 2 — the falsification: evaluation is 20–24% of the solve here, not 48%

The issue states "**Evaluation is 48% of that solve**", measured on its `.nl`
path. On discopt's path — POUNCE's Rust tape driven by POUNCE's IPM — it is
not. Measured on the same models with `discopt._timing` (the `rust` bucket is
the tape evaluator's own; `pounce` is the enclosing IPM's), full solves to
OPTIMAL:

| K | n | wall | iterations | `rust` (evaluation) | share |
|---|---|---|---|---|---|
| 8 | 9,620 | 0.709 s | 15 | 0.140 s | 19.7% |
| 32 | 38,420 | 4.217 s | 20 | 0.961 s | 22.8% |
| 64 | 76,820 | 6.735 s | 16 | 1.598 s | 23.7% |

A second, independent instrument (per-call costs measured directly × iteration
count) puts it at 11.6–20.8%, i.e. the same order from below — the two
instruments bracket rather than agree exactly, which is what they are for.

**Consequence for Part B's value.** Even a perfect evaluator — evaluation cost
driven to zero — caps the end-to-end gain at **1.31x** at K=64 on this class.
At the measured 10–17x on the derivatives themselves the gain is ~1.27x. That
is a real win, and it is additive with Part A's factorization win, but it is
not "the larger share" the issue describes:

> A without B delivers the factorization win only … because factorization is
> roughly a third of these solves. Evaluation is the other half and is the
> larger one.

On this class, at this scale, on this evaluator, evaluation is **not** the
larger share. Taking the issue's own "factorization is roughly a third" and
pounce#955's measured 3.8–4.8x on it, the arithmetic after Part A lands is:
factorization ~2.2 s of the 6.735 s K=64 solve → ~0.55 s, wall ~5.1 s, at which
point evaluation's 1.6 s **is** 31% and Part B is worth ~1.45x on top of Part
A. So the two halves do compound — roughly 1.9x together rather than the ~3x a
reader of the issue would expect.

Recorded here rather than discovered later (CLAUDE.md §4, §11): the mechanism
is confirmed, the premise about its share is corrected, and the scope decision
for the Part B build should be taken against 1.3–1.45x, not against 48%.

## What a Part B implementation would have to do

The harness hand-writes the block template as a JAX function. A shipping
implementation has to *derive* it from the model and prove it derived the right
one:

1. **Template identity.** Blocks must be proven structurally identical, not
   assumed — a canonical fingerprint over each block's expression DAG with
   columns renamed to block-local slots, plus pattern equality, plus agreement
   with the tape at random points. Assuming identity computes silently wrong
   derivatives, which is a §1 failure, not a performance one.
2. **The compact domain problem.** `_relax/dag_compiler` compiles to a function
   of the whole flat `x`. Evaluating K blocks by scattering K variants of a
   length-n vector is O(K·n) memory per call — 107 MB at SCOPF scale — so the
   template needs a compact `(block columns, shared columns)` domain, which
   means either a genuine sub-model extraction or a scatter that XLA can fuse
   away. This is unmeasured and is the main implementation risk.
3. **Scatter maps** for J and H, precomputed once, including the accumulation on
   the shared×shared corner (it sums over blocks).
4. **A default-OFF gate** (`DISCOPT_BLOCK_VECTOR_EVAL`) with its row in
   `docs/dev/flag-retirement-audit.md`, a differential panel against the default
   evaluator at the tape evaluator's own graduation bars (1e-10 grad/J, 1e-8 H),
   and a graduation attempt resolved into one of §5's three states.
5. **JAX on an opt-in solve path.** It is a core dependency but deliberately off
   the default path; the docstring must say exactly when it loads.
