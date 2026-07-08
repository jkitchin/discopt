# C-44 (#567) — column-identity-safe cut-pool inheritance:
# the C-43 false-fathom fixed AT THE SOURCE; #568 re-verify now inert. Sound —
# but the broad default-ON flip stays KILLED (flag-ON regresses several
# pool-firing instances >10%). Cut-inheritance stays OPT-IN. (2026-07-08)

**Status.** The soundness fix SHIPS. The nvs22 column-remapping false-fathom
class (C-43, #564) is fixed **at the source**: the inherited pool is remapped by
lifted-column *identity*, so no node is falsely Farkas-fathomed and the C-43/#568
runtime re-verify goes **inert** on nvs22 (`pool/dropped_nodes: 21 → 0`). The
default (force-off) path is **byte-identical** to `origin/main` (cert-neutrality
41/41, `|Δobj|=0`, node counts unchanged). The broad **default-ON graduation is
KILLED** by measurement (CLAUDE.md §4): the *sound* flag-ON path regresses several
pool-firing instances >10% on wall time (dispatch, nvs18, nvs08), which fails the
graduation gate "no instance >10% slower." `DISCOPT_CUT_INHERIT` / `cut_inherit`
stays **opt-in** (default force-off).

> **Method.** Apple M-series arm64, Python 3.12, release build
> (`maturin develop --release`; pounce `_pounce.abi3.so` = 4.73 MB, not debug),
> `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, corpus
> `~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`. A/B wall
> pairs run **SERIALLY** (fresh interpreter per solve, one solve at a time), per
> the #552/CUT-INHERIT-GRAD lesson (the TL=30s parallel artifact).

---

## Part 1 — the column-remap mechanism (nailed down, file:line)

A root cut-pool row is stated by **column position** over the ROOT build's lifted
column layout (`solver.py:5165`/`:5238` root-pool capture via
`solve_at_node(..., out_cuts=...)`). Each lifted aux column has a **stable
structural identity** — the term key that created it, available in
`build_milp_relaxation`'s returned `varmap` (`milp_relaxation.py:8423`):
`bilinear (i,j)→col`, `monomial (i,p)→col`, `trilinear`/`multilinear` tuple→col,
`fractional_power (i,p)→col`, `univariate_square (base_col, 2)→col`. Original
variables occupy the fixed first `n_orig` columns.

Per node the relaxation is re-built (`mccormick_lp.py:934` cold build) and/or
re-lifted by lifted-FBBT (`:987`–`:995`, which **swaps `milp, varmap` together**).
This can produce the **same column count with different column semantics**: a
tightened/degenerated box drops or re-orders lifted terms, so a position that was
`x_2·x_5·x_8` at the root becomes `x_3·x_4` (or an unrelated aux) at the node.
The pre-C-44 gate checked only the *count* (`_sparse_cols(_ia) == n_total`,
`mccormick_lp.py`, and `a_rows.shape[1] == inc.ncol` on the incremental path), so
it appended a root row by position onto the **wrong lifted variables** → an
invalid cut → a node whose box contains the true optimum can be **falsely
Farkas-fathomed** (the C-43 nvs22 mechanism; the #568 runtime re-verify catches it
but pays the recovery and, on the numerical class, costs a certificate).

**Measured on nvs22 (flag-ON), instrumenting `build_milp_relaxation` per node:**
the root layout has 69 columns; node builds recur with the **same 69-column count
but 16–24 columns remapped** (e.g. root col 26 = `trilinear (2,5,8)`; at a node
col 26 = `bilinear (3,4)` or `opaque`). 21 such nodes were being false-fathomed
(recovered by #568's re-verify: `pool/dropped_nodes = 21`).

## Part 2 — the fix: remap by column identity, skip if unmappable

`column_identities(varmap, n_total, n_orig)` (`mccormick_lp.py`) returns a
per-column identity tuple: `("orig", k)` for originals (always stable),
`(mapname, term_key)` for structurally-keyed aux (bilinear/monomial/trilinear/
multilinear/fractional_power), `("univariate_square", (base_identity, 2))` with
the base resolved *recursively* to its own identity, and `("opaque", k)` for any
unclaimed aux (position-locked — never remaps).

`_remap_pool_rows(a_rows, b_rows, root_idents, node_idents, ncol)` remaps each
pool row from root-column positions → the node position carrying the SAME
identity. **A row that references (with a nonzero coefficient) a column whose
identity is absent at the node is SKIPPED** — never appended over the wrong
columns (skipping an optional cut is always sound; fewer cuts only loosen).

Capture side: `solve_at_node`'s `out_cuts` chunk is now
`(A, b, col_idents)` (`mccormick_lp.py`, the capture tags the SAME `varmap` the
separated rows are stated over, post-FBBT-rebuild if it fired); `_root_cut_pool`
carries the identities (`solver.py`, three capture sites). Consume side:

* **Cold path** (`mccormick_lp.py` `_solve_at_node_impl`): build the node
  identities from the node `varmap` and remap, gated by the *same* equal-count
  precondition the legacy gate used (`_sparse_cols(_ia) == n_total`) so C-44 only
  *remaps* at equal count — exactly where the false-fathom lived — and is
  behaviour-neutral on the layouts the legacy gate rejected.
* **Incremental path** (`_try_incremental_node`, node identities =
  `inc.col_identities`, fixed across nodes because `_patch` never rebuilds
  columns): when the pool's column count does not match the incremental structure
  (`a_rows.shape[1] != inc.ncol`, e.g. dispatch: pool 23 cols vs incremental 10),
  **decline the incremental path (`return None`)** so the cold build — which
  re-lifts to the pool's own layout — inherits it there. This reproduces the
  pre-C-44 routing exactly (which relied incidentally on a sparse-`np.asarray`
  exception to decline) and is required for default-path byte-identity (dispatch:
  3 nodes; without it, 12 623 nodes — a bound-neutrality violation).

The #568 pool-infeasible re-verify is **kept as defense-in-depth** (per the task:
do not remove the soundness guarantee); it is now a **no-op on the remapping
class** — the appended cuts are valid, so no false fathom occurs.

## Part 3 — soundness (hard gates, CLAUDE.md §1)

### 3a. nvs22 — false-fathom removed at the source

| arm | status | objective | bound | `pool/dropped_nodes` |
|---|---|---:|---:|---:|
| default (force-off) | optimal | 6.05822 (oracle) | 6.05822 | — |
| flag-ON, `origin/main` (#568 re-verify) | optimal | 6.05822 | 6.05822 | **21** |
| **flag-ON, C-44** | **optimal** | **6.05822** | 6.05822 | **0** |

The remap is genuinely load-bearing: of 55 node remap calls, **44 had a reordered
node layout** (columns at different positions than root) — the exact cases that
were false-fathomed; **0 rows skipped** (every referenced lifted term existed at
each node, just at a different position, so the remap recovered the correct cut).
Direct anti-C43 probe: **12 opt-containing node solves, 0 falsely infeasible**
(pre-fix: 1 of 3 opt-containing nodes went falsely `infeasible`).

### 3b. HiGHS/oracle battery — 0 false-optima, 0 bound-crosses-oracle

Flag-ON, sense-aware oracle cross-check (min: bound ≤ opt; max: bound ≥ opt),
pool-firing + broad held-out (25 instances incl. nvs17/19/22/23/24, nvs06,
knp3-12, dispatch, tspn05, kall_circles_c6a–c8a, st_e05/e07, ex1223a, alkyl,
st_miqp1/2, nvs02/03/08/14/18): **VIOLATIONS: 0.** Every certified `optimal`
equals the oracle; every dual bound is a valid bound. A separate 24-instance
held-out slice (kall_circles skip 134–2635 rows, st_e07 skips 12) also
**0 violations** — the skip branch is exercised and sound.

### 3c. Feasible-point sampling of the REMAPPED cuts — no valid point cut

Reconstructing the exact lifted vector (originals + every aux synthesized from its
identity: bilinear→`z_i·z_j`, monomial→`z_i**p`, trilinear/multilinear→product,
usq→base², recursively) at random integer-feasible points inside each node box,
and checking `A_remapped @ z ≤ b + tol` on **every appended remapped row**:
across nvs22/nvs19/nvs17/nvs24, **max violation 0.0 over ~1.8 M row-checks**,
`unsynth = 0` (no opaque column ever leaks into an appended cut). No remapped cut
removes a feasible point.

### 3d. default-path cert-neutrality — byte-identical

`check_cert_neutrality.py` (default force-off, 41-instance panel): **41/41
byte-identical, `|Δobj| = 0.00e+00` on every row, node counts unchanged
(dispatch 3→3, nvs13 49→49, tspn05 39→39), exit 0.** No rebaseline (the gate does
not fire on the shipped default). The identity remap runs on the default path too
(the pre-existing cert:T1.3 inheritance) and is verified inert there.

## Part 4 — the empirical decision: SOUND, but no broad net speedup → KILL the flip

Serial A/B, force-off vs flag-ON, per the graduation criteria (material sound
speedup AND certs preserved-or-gained AND 0 false-optima AND no instance >10%
slower):

| instance | OFF | ON | verdict |
|---|---|---|---|
| nvs17 (TL30) | optimal 23.9 s, 173 n | optimal **17.3 s**, 225 n | **1.4× faster cert** (win) |
| nvs13 (TL30) | optimal 1.91 s, 49 n | optimal 1.58 s, 53 n | faster (win) |
| nvs24 (TL40) | feasible 9 n | feasible **31 n** (3.4×), same bnd | node win (TL-bound) |
| knp3-12 (TL40) | feasible 31 n | feasible **159 n** (5×), same bnd | node win (TL-bound) |
| nvs19 (TL90) | feasible @ 90 s (bnd −1098.96) | **optimal 63.3 s** (−1098.4=opt) | **cert GAINED** (sound tightening; the `dropped` nodes are the C-38 *numerical* class, recovered soundly — NOT remapping) |
| st_e07 (TL30) | optimal 2.24 s | optimal 2.20 s | neutral |
| nvs08 (TL30) | optimal 0.89 s | optimal 1.01 s | **+13% slower** |
| nvs18 (TL40) | optimal 6.23 s | optimal 7.37 s | **+18% slower** |
| dispatch (TL40) | optimal 0.47 s, 3 n | optimal 0.88 s, 23 n | **+87% slower** (cert preserved) |

**The wins are real and sound** on the dense-integer-QP subclass (nvs24 3.4×,
knp3-12 5×, nvs17 1.4×; nvs19 gains its cert at TL≥~63 s). **But flag-ON regresses
wall time >10% on several pool-firing instances** (dispatch +87%, nvs18 +18%,
nvs08 +13%) where the per-node square/PSD skip widens the tree more than the
inherited pool tightens it. Per the graduation kill criterion "no instance >10%
slower," a **broad default-ON flip is not justified**. The flip is **KILLED**; the
flag stays **opt-in** with C-44's soundness fix shipped.

### Note on the #567 nvs19 hypothesis (falsified — CLAUDE.md §4)

#567 predicted the source fix would restore nvs19's flag-ON certificate by
removing false fathoms. **Measurement:** on nvs19 the remap is a *no-op*
(root↔node columns are positionally identical — 0 reorders across 51 node remap
calls; the pool references 44 nonzero columns, 0 opaque). nvs19's `dropped` nodes
are therefore **not** column-remapping false-fathoms — they are the **C-38
numerical class** (a *valid* cut destabilizes the warm simplex, same family as
nvs06's `dropped=1`), which C-44 neither introduces nor can remove. nvs19 is a
**cert at the ~60 s budget edge**: force-OFF times out at 90 s while flag-ON
certifies at ~63 s (a sound ~1.4× convergence win), so the flag is
certs-preserved-or-gained on nvs19 — but that does not rescue the broad flip,
which the >10% regressions above kill.

## Cert-baseline disposition

No rebaseline. The shipped default is byte-identical to `origin/main`
(cert-neutrality 41/41, `|Δobj|=0`, node counts unchanged), so the
"sanctioned bound-changing rebaseline" clause is moot (the flip is not made).

## Gates

| gate | result |
|---|---|
| `pytest -m smoke` (python/tests) | **633 passed, 12 skipped** |
| `pytest -m slow test_adversarial_recent_fixes.py` | **10 passed** |
| `pytest test_c42_cut_inherit_coldpath.py` (slow+smoke) | **7 passed** (nvs22 now asserts `dropped_nodes==0`; nvs06/tspn05 hold) |
| `pytest test_c44_column_identity_inherit.py` | **5 passed** (remap-moves-coeff, skip-unmappable, naive-would-cut-feasible-point, usq-identity, tag-orig/aux) |
| `pytest test_cut_inherit_pool.py` | **passed** (3-tuple chunk fixup) |
| `test_amp.py` (AMP coverage set) | **135 passed** |
| `check_cert_neutrality.py` (default force-off, 41-panel) | **41/41 byte-identical**, `|Δobj|=0`, node counts unchanged, exit 0 |
| HiGHS/oracle battery + broad held-out (flag-ON) | **0 false-optima, 0 bound-crosses-oracle**; feasible-point sampling of remapped cuts: **max violation 0.0** |
| `ruff check` / `ruff format --check` (v0.14.6) | clean |
| pre-commit `mypy` (v2.1.0, whole package) | **Passed** |
| `cargo test -p discopt-core` | n/a — no Rust touched |

## Follow-ups (recorded, not shipped)

1. The >10% flag-ON regressions on dispatch/nvs18/nvs08 are the per-node
   square/PSD *skip* widening the tree, not the inheritance itself. A cost-aware
   skip (skip only when the inherited pool measurably reproduces the node's
   separation gain) could recover those and re-open the flip. #396 backlog.
2. nvs19's flag-ON `dropped` nodes are the C-38 numerical class (valid cut
   destabilizes the warm simplex); a conditioning-robust warm re-solve on the
   pool-augmented system would retire the last `dropped` recoveries. #396 backlog.
