# #2142 SCIP hull ablation — pre-registration of the discopt entry experiment

Written 2026-11-05 before any discopt arm past 1 node was run (CLAUDE.md §4).

## Evidence so far (SCIP 6.3.0 on `2142-exact-hull/scip/cstr_homog_spec.npz`, node limit 100k)

Single-component ablations (this directory, `*.log`): default certifies in 7,470 nodes;
presolve off 48,521 (6.5x, the largest single factor); conflict off 29,538 (4.0x);
presolve+conflict off 82,343; heuristics off 12,112; genvbounds off 9,118; separators off
9,082; perspective nlhdlr off 8,358; OBBT off 8,040; pseudoobj off 6,980; reformbinprods
off 7,470 (no-op); `nonlinear` propagation off NOT certified at 100k (2.57); all propagation
off 0.20 at 100k.

What SCIP's presolve does to this spec (`presolve_dump.py`, `root_box.py`): 285 cols / 779
rows -> 110 vars / 174 conss. It fixes 51 columns (49 continuous at 0, 2 binaries) — discopt's
own root FBBT (`propagate_spec_py`, 15 rounds) fixes **exactly the same 51** — and
aggregates 124 columns away (116 single-column: 100 continuous identities `x_j = x_k`, 7
aux identities, 9 negated binaries `y' = 1 - y`; 8 multi-column). Its remaining bound
tightenings are ~0.01-sized. So presolve's 6.5x is the **aggregation**, not tightening:
discopt hands its kernel 124 columns that are affine images of other columns, each a
separate branching candidate / propagation hop / envelope column.

## Hypothesis

H1: aggregating the spec's doubleton-equality columns (derived from the spec alone by
`aggr_derive.py`: root FBBT fixings, then parallel `<=` row pairs of support 2 to closure;
111 aggregations) before the kernel runs raises discopt's deterministic tree bound on the
homog hull at equal node counts. Mechanism expected: the kernel branches on one column per
equivalence class (no wasted splits on a copy whose box the linear link only transfers), its
McCormick envelopes are over the representative's box, and FBBT no longer needs a round per
identity hop.

## Experiment

`arm_aggr.py` (= `2142-exact-hull/arm.py` with a spy that rewrites the kernel spec: every
aggregated column substituted out of `c`, the fixed rows, the term/BLF forms; its bounds
mapped onto the representative; the column itself fixed to 0 and removed from
`obbt_candidates`; `incumbent_x` and the `incumbent_value` / `primal_hook` arguments
expanded back). cstr, `homog` lowering, deterministic, `PYTHONHASHSEED=0`,
`DISCOPT_PROFILE=1`, 10k and 20k nodes. Control: `homog` base at 10k re-run (must
reproduce 1.6108 bit-for-bit, §8). Baselines (`2142-exact-hull/proposal.md` §5): homog
1.6108 @10k, 2.1886 @20k; big-M default 2.2486 @10k, 2.7875 @20k.

## Kill criterion (pre-registered)

- **Kill** if the aggregated arm's 20k-node bound does not exceed **2.7875** (big-M default
  at 20k). Secondary record: the 10k bound against 1.6108 and 2.2486.
- Soundness gates (any failure = the emulation is wrong, not a result): bound <= oracle
  3.0620073 + 1e-6; incumbent >= oracle - 1e-5; every aggregated point verified on the full
  spec (`arm_aggr.py` asserts the expanded incumbent satisfies every original fixed row
  and term to 1e-6).
- The emulation is a *proxy* for an in-kernel aggregation presolve; a pass here justifies
  building that (proposal in `findings.md`), it is not the §5 panel.
