# Why node-local GMI does not move the bound on neos-3610051-istra / neos-3610040-iskar

Scripts and logs are in `/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/expert2/`:
`scip_arms.py` (arms1.log), `pysep.py` (arms2-5 logs, iskar arms printed inline below), `getopt.py` (optsols.json).
SCIP 10 / pyscipopt 6.3, presolve off, gap 1e-4, one solve at a time, load < 1.5.

## Answer

**Weak cuts are not the reason. Where the cuts live is.** SCIP's in-tree Gomory cuts are
**globally valid**, and they leave the subtree that generated them through two channels:

1. the **global cut pool**, which re-adds violated cuts at other nodes;
2. **conflict analysis**, which builds dual proofs over global rows.

discopt's B1' keeps every row strictly inside the subtree that made it, and discopt has
neither channel. When I run the *same* GMI code in SCIP with both channels shut, SCIP stalls
at **48.31 on istra**. That matches discopt's 48.32.

## Experiment 1: what kind of rows SCIP's tree Gomory cuts are (arms1.log)

An event handler counted the rows SCIP added to the LP:
- With `gomory`+`gomorymi` tree-only, the rows were **28,786 global and 0 local** on istra,
  and 10,186 global and 0 local on iskar.
- With `maxlocalbounddist=1`, they were still 14,196 global and 231 local.

The reason is that SCIP's default `separating/maxlocalbounddist=0` allows local bounds only
at the current best node, and MIR complements against global bounds. Branched binaries sit
at 0 or 1, which are also their global bounds, so the cut is global anyway.

Turning off pool separation (`separating/poolfreq=-1`) stops tree-only Gomory dead: istra
stalls at 47.00 and iskar at 35.46. In that mode SCIP's gomory sends global cuts only to the
pool, so this arm also turns most of the cuts off. That is why I ran Experiment 2.

Side note: in SCIP 10, `gomory` generates nothing when the dummy separators `gomorymi` and
`strongcg` are both off. The earlier result "gomory + X alone stalls"
(b1_scip_pairs.py) is an artifact of that: those arms had no Gomory cuts at all. The family
that matters is GMI (`gomory`+`gomorymi` certifies).

## Experiment 2: one Python GMI separator, three lifetimes (pysep.py)

The cut code is identical in every arm: textbook GMI, the same formula as `gomory.rs`, read
from SCIP's basis with my own B⁻¹. All SCIP separators are off, with freq 10, 50 cuts and
5 rounds unless noted. Only two things vary:
- the bounds used for complementation: L = node-local, G = global;
- the row's lifetime: sub = local row, subtree only; pool = global row plus `addPoolCut`.

Validity probe: every G cut, and every L cut whose box contains the optimum, was checked
against a known optimal solution. **0 violations in 17,000+ checks.** There were 0 basis
mismatches against SCIP's primal.

| arm (60 s) | iskar | istra |
|---|---|---|
| G-pool | **optimal**, 8,105 nodes | **optimal**, 10,641 nodes |
| G-pool, 20 cuts × 3 rounds | **optimal**, 14,706 nodes | **optimal**, 12,090 nodes |
| G-sub | 35.90 (time limit) | 47.63 (time limit) |
| L-sub (B1') | 36.14 (time limit) | 47.56 (time limit) |
| L-sub, freq 1, 20 × 3 (closest to discopt) | 35.79, 42,722 cuts | 47.49, 25,560 cuts |
| G-pool, poolfreq=-1 (conflict on) | **optimal**, 19,120 nodes | **optimal**, 27,409 nodes |
| G-pool, conflict off (pool on) | **optimal**, 21,105 nodes | **optimal**, 14,287 nodes |
| G-pool, poolfreq=-1 **and** conflict off | 36.49 (time limit) | **48.31 (time limit)** |

What the rows say:
- **Cut strength is not the lever.** The same cuts certify when they are shared and stall
  when they are subtree-only.
- **Separating more is not the lever.** L-sub at every node (42k cuts) is worse than
  G-pool at about 200 nodes.
- **Either channel alone suffices.** The pool alone certifies, and conflict analysis alone
  certifies. This also explains why the earlier SCIP leave-one-out found "conflict
  unnecessary": the pool was still on.

## The candidate explanations

1. **"Our GMI is weaker than SCIP's": falsified as the cause.**
   - My separator (the same formula as ours) certifies once its cuts are shared.
   - SCIP's own MIR-based gomory needs about 5x fewer nodes (1.6k against 8k), so it is
     better, but that is not the line between certifying and not.
   - Slack integrality is moot here: every row has a continuous variable, and the
     objective row's z is continuous.
2. **"Frontier nodes never get rounds": irrelevant in this form.** Separating at *every*
   node with subtree lifetime still stalls. I did not instrument the Rust frontier, because
   Experiment 2 makes it moot.
3. **"An interaction with something else in the tree": confirmed, and the interaction is
   identified.** It is the pool and conflict dual proofs, i.e. global sharing.
   Branching was not varied, since the stall reproduces under SCIP's own branching.
4. **Objective integrality (intobj): not needed.** Every arm above runs without intobj, and
   G-pool certifies.

## Recommended change, with entry experiment and kill criterion

**Change (flag value `DISCOPT_MILP_NODE_GMI=pool`, default OFF):**

1. **Separate at nodes against global bounds.** Use the node's tableau, but complement
   each nonbasic column at its *global* bound, and recompute β with
   `x̃* = |x* − bound|`.
   - The local-bound version is valid only in the node's box.
   - The global-bound version is valid everywhere, and for these rows it is the same cut,
     because branched binaries sit at global bounds.
   - Refuse a cut that references the slack of a node-local row.
2. **Put the selected cuts into a bounded, deduplicated, aging global pool.** Do not fold
   them into the shared matrix; the B1 fold showed the per-node cost of that.
3. **Pool-separate at every node, or every k-th node.** After the plain LP, append the pool
   cuts the node's x* violates (efficacy ≥ ε, top-K) to the node-local copy through the
   existing `node_local_gmi_bound` augment path. Re-solve warm and NS-verify as today.
   This replaces "inherited rows" with "violated pool rows".

Conflict dual proofs are the alternative channel. They are a larger build, so the pool
should go first.

**Entry experiment.**
- Direct driver (`b1p_node_gmi_inherit_direct` setup), 20 s, unrounded bound:
  istra, iskar and kaihu, flag OFF, `=1` and `=pool`.
- Soundness probe: assert every pool cut against the MIPLIB/SCIP optimal point (as in
  `pysep.py`), print the count of cuts checked, and require 0 violations.
- Print pool size, violated pool cuts added per node, and lifted nodes.

**Kill criterion.** Kill it if, at 20 s, `=pool` raises the unrounded bound by less than
**+0.3 over OFF on both istra and iskar** (istra ≥ 48.62, iskar ≥ 36.49) and certifies
neither. SCIP-sim G-pool reaches 49 / 37 at ≤ 15k nodes, which is less than discopt's 20 s
node count.
