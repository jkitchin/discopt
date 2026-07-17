# G-convexity transformation cut — graduation panel (#181, 2026-07-17)

Flag: `DISCOPT_G_CONVEX_CUTS` (bound-changing, default-OFF). Capability:
`inject_g_convex_cuts` recognizes constraint bodies certified **G-convex** on
the declared box and injects rigorously valid linear transformation cuts
(`exp(ρ·body) ≤ 1` supported at `x₀` with an interval-safe intercept). See
`python/discopt/_jax/convexity/g_convex_inject.py`.

## Verdict: **does NOT graduate — cert-clean but inert (net-neutral).**

The soundness bar passes; the net-positive bar fails. The flag stays default-OFF.
This is the `DISCOPT_CUT_INHERIT` outcome (sound ≠ helpful), recorded per
CLAUDE.md §5.

## Evidence

**Entry probe (dry run, flag ON).** Loaded 46 of the 49 cert-baseline instances
and counted cuts `inject_g_convex_cuts` would add at root presolve:

```
loaded 46 instances; total cuts injected: 0
```

The constant-ρ interval-Gershgorin detector (`certify_g_convex`) abstains on
every instance's **wide declared root box** — the augmented-Hessian PSD
enclosure is too loose there (the same box-width conservatism documented for the
ordinary convexity certificate). So the injector is a no-op at root on the whole
corpus.

**Neutrality sub-panel (OFF vs ON, tl=20s, 8 instances).** Confirms the flag
path is genuinely inert — no accidental perturbation even where 0 cuts fire:

| instance | baseline obj | OFF obj | ON obj | cuts | status OFF→ON |
|---|---|---|---|---|---|
| alan | 2.925000 | 2.925000 | 2.925000 | 0 | optimal→optimal |
| gbd | 2.200000 | 2.200000 | 2.200000 | 0 | optimal→optimal |
| ex1221 | 7.667180 | 7.667180 | 7.667180 | 0 | optimal→optimal |
| ex1222 | 1.076543 | 1.076543 | 1.076543 | 0 | optimal→optimal |
| nvs01 | 12.469669 | 12.469669 | 12.469669 | 0 | optimal→optimal |
| nvs03 | 16.000000 | 16.000000 | 16.000000 | 0 | optimal→optimal |
| st_test1 | -0.000000 | -0.000000 | -0.000000 | 0 | optimal→optimal |
| st_miqp1 | 281.000000 | 281.000000 | 281.000000 | 0 | optimal→optimal |

`SUMMARY: 8 instances, 0 soundness/neutrality violations.` Objectives match the
committed baseline within 1e-4; OFF == ON byte-identical; no optimal lost.

**Cut soundness (independent of the corpus).** The cut itself is proven sound by
`test_g_convex_inject.py`: 60k+ random **feasible** points across a `≤`
(`log(x²+y²)`) and a `≥` (`exp(-(x²+y²))`, G-concave body) constraint, **0 cut
violations** (worst feasible residual −0.0), and the cut is non-vacuous
(separates 13k+ infeasible points) on the tight boxes where it does fire.

## Why it's inert, and what would make it net-positive

The cut fires only where a constraint body is certified G-convex, which the
constant-ρ detector can only do on **tight** boxes. Those occur deep in the B&B
tree, not at root presolve where the injector currently runs. The full corpus
benefit arm (net-positive over ~4,800 MINLPLib instances) additionally could not
run in this container — no `~/Dropbox/projects/discopt-minlp-benchmark` corpus.

**Follow-up to revisit graduation (out of scope here, both bound-changing):**

1. **Per-node injection.** Move the cut from root presolve into the B&B node
   loop so it fires on the tightened FBBT/branch boxes the detector can certify.
   This is where the KMS gap-reduction gains would materialize.
2. **Tighter detector.** Reduce the augmented-Hessian outer-product interval
   slack (e.g. a single-pass AD of `∇²φ + ρ∇φ∇φᵀ` capturing the cancellation) so
   G-convexity certifies on wider boxes, including some roots.

Until one of these lands, the flag is sound but inert and correctly stays OFF.

## Update — per-node injection lands the benefit (2026-07-17)

Follow-up #1 above is implemented: `MccormickLPRelaxer._separate_g_convex` now
runs the transformation cut **per B&B node**, on the tightened node box, where
the constant-ρ detector can certify G-convexity. The cut is **box-local** —
valid only on the node box — so it is generated fresh per node and appended to
that node's LP only, and it runs **only when `out_cuts is None`** (a regular
node solve, never a pool-capture solve). That gate guarantees it can never enter
the inheritable root pool and be replayed on a sub-box — the exact C-43 / nvs22
false-optimum hazard.

**Soundness (oracle).** Re-ran the neutrality sub-panel with the per-node path
active over **18 instances** (OFF vs ON, tl=25s): **0 soundness/neutrality
violations** — every certified objective identical to the committed baseline,
every `optimal` preserved. The cut construction is rigorously valid on the node
box (interval-safe intercept), and the never-pooled gate keeps it from leaking
onto sub-boxes.

**Benefit.** On a branching G-convex model (`maximize x+y+z` s.t.
`log(x²+y²) ≤ 1.6`, integer `z`), flag OFF solves in **53 nodes**, flag ON in
**39 nodes** — a 26% node reduction, same certified optimum (4.14739). The cut
fires at the deeper nodes whose boxes have tightened enough to certify, exactly
as predicted.

Regression tests: `python/tests/test_g_convex_per_node.py` (flag preserves the
certified optimum; node count never regresses; the node LP bound only tightens).
The default path is unchanged — with the flag OFF `_separate_g_convex` is never
called.

The flag remains default-OFF pending the full corpus differential panel (the
~4,800-instance benefit arm, which needs the MINLPLib corpus absent from this
container). But the keystone result now holds: the capability is **sound and
net-positive where it fires**, no longer inert.
