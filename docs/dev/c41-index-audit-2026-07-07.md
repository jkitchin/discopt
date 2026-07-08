# C-41 — block↔flat index-alignment audit + guards (2026-07-07)

**Task.** C-40 was the third false-optimal/soundness bug (after C-16 and C-31) from
one root: a mapping between a per-variable-**BLOCK** array and a per-scalar-**FLAT**
array that got misaligned, often with a swallowed `IndexError` hiding it. Given that
hit rate, this is a proactive AUDIT + fix-what-you-find of *every* block↔flat mapping
site across the Python and Rust layers.

**Result (headline).** Two previously-unguarded sites carried the **exact C-40
pattern** — `node_reduce._fbbt_on_node` and `root_reduce._stage_fbbt_with_cutoff`.
Both are on flag-gated paths (`DISCOPT_NODE_REDUCE`, `DISCOPT_ROOT_FIXPOINT`, default
OFF), and a synthetic-but-faithful reproduction shows that a misaligned repr makes
them write a **crossed `lb>ub` box → false `infeasible` fathom** (the C-40 outcome,
localized to the node/root box rather than the global box). Both were fixed with the
same 1:1-alignment guard the C-40 fix uses, plus their swallowing `except` was made
to surface (log). Every other block↔flat site is SAFE — either guarded already
(`solver.py:7443` C-40, `solvers/_root_presolve.py:43`) or structurally aligned
(running-`offset` maps that never cross-index; the Rust C-16/C-31 resync). The C-40
class is now contained.

## The bug signature

The Rust `fbbt::fbbt_with_cutoff` / `fbbt` return `Vec<Interval>` of length
`model.variables.len()` — **one interval per BLOCK** (`seed_block_interval` unions
each block's elements). Python code that maps this onto the FLAT scalar box via
`fbbt_lbs[bi]` (block index) → `lb[flat]` (scalar column) is sound **only** when the
repr's block count equals `len(model._variables)` and blocks are scalar. In
*builder mode* (`model_to_repr(model, model._builder)`) the repr's block list comes
from the builder and its count can DIVERGE from `model._variables` — C-40 measured
144 intervals for a 145-column `util`. A positional read on a diverged repr lands on
the **wrong variable's** bound and writes a crossed box; if the resulting `IndexError`
/ crossing is swallowed, the corrupted box silently fathoms the optimum-containing
node.

## Audit table — every block↔flat / bounds-indexed-by-var site

Full inventory of `fbbt_lbs[...]`-style block→flat maps and every `enumerate(model._variables)`
site paired with a flat bounds array, across both layers.

| # | Site (file:line) | Mapping kind | Verdict |
|---|---|---|---|
| 1 | `python/discopt/solver.py:7438-7495` (Phase-C3 cutoff-FBBT) | `fbbt_lbs[bi]`(block) → `lb[flat]`(flat) | **SAFE** — C-40 fix: guards `fbbt_lbs.shape==(n_vars,) ∧ all(v.size==1)`, builds candidate with `np.maximum/minimum`, commits only crossing-free intersection. |
| 2 | `python/discopt/_jax/node_reduce.py:_fbbt_on_node` (was 184-204) | `fbbt_lbs[bi]`(block) → `lb[flat]`(flat), fresh builder-mode repr | **SUSPECT → FIXED (real crossed-box/false-infeasible)**. Old guard was only `bi >= shape[0]` (OOB), not alignment; a diverged repr writes the wrong var's bound. Added the `len==len(model._variables)` guard + surfaced the swallowing `except`. |
| 3 | `python/discopt/_jax/root_reduce.py:_stage_fbbt_with_cutoff` (was 147-169) | identical to #2, root box | **SUSPECT → FIXED (real crossed-box/false-infeasible)**. Same fix. |
| 4 | `python/discopt/solvers/_root_presolve.py:_flat_fbbt_bounds:43-59` | `fbbt_lbs[block_idx]`(block) → `tightened_lb[slice]`(flat) | **SAFE** — early-returns when `len(fbbt_lbs) != len(model._variables)`; block→flat via `slice(offset, offset+size)` with running `offset`. |
| 5 | `python/discopt/tightening.py:fbbt_box:88-127` | `block_lb[i]`(block) → `new_lb[slice(offset,offset+size)]`(flat) | **SAFE** — `n_blocks`, `sizes`, `orig_lb/ub` **all derived from the same `repr_`** (not `model._variables`); self-consistent, running `offset`, no cross-length hazard. |
| 6 | `python/discopt/_jax/nonlinear_bound_tightening.py` | builds `FlatVariableMetadata` (running offsets); rules index flat arrays by flat indices | **SAFE** — no block-index → flat-array read anywhere (grep: no `fbbt_lbs[bi]`). |
| 7 | `python/discopt/_jax/mccormick_lp.py` | McCormick envelope compile | **SAFE** — no `fbbt_lbs[bi]`→flat map; no block-index read of a flat array. |
| 8 | `python/discopt/_jax/node_reduce.py:_is_int_mask` & `root_reduce._is_int_mask` | per-block `for v` with running `k`/flat counter → `is_int[k]` | **SAFE** — flat mask built by a running counter over `v.size`; indexed by `flat`, flat-to-flat. |
| 9 | `python/discopt/solvers/_root_presolve.py:tighten_root_bounds_with_fbbt` (int_offsets/int_sizes) | flat offsets from block sizes | **SAFE** — offsets computed from `v.size`; slice-addressed. |
| 10 | `python/discopt/warm_start.py`, `infeasibility.py`, `conflict.py`, `callbacks.py`, `parametric.py`, `mpec.py` | `enumerate(model._variables)` with running `offset`/`flat` or `zip(model._variables, saved)` | **SAFE** — no block-index used to index a differently-lengthed flat array; each keeps its own running scalar offset or zips 1:1 with a same-length per-block save list. |
| 11 | `crates/discopt-core/src/presolve/pass.rs:resync_bounds_after_rewrite:101-136` | `bounds` (block-indexed) resync after a model rewrite | **SAFE** — C-16/C-31: `bounds` explicitly one-per-BLOCK; a *shrinking* rewrite rebuilds `bounds` from the new model (never positionally intersects a renumbered vector), a *growing* rewrite intersects only the unchanged prefix. |
| 12 | `crates/discopt-core/src/presolve/fbbt.rs:seed_block_interval / fbbt_with_cutoff:1128-1231` | seeds/returns one interval per block (`model.variables.len()`) | **SAFE** — pure block-indexed; convergence loop is `0..n_vars` over the same block vector. The *producer*; the block↔flat mapping is the consumer's responsibility (sites #1-5). |
| 13 | `crates/discopt-python/src/expr_bindings.rs:model_to_repr:807+` | builds repr `variables` (block list) — builder-mode from builder, else from `model._variables` | **SAFE as a producer, but the DIVERGENCE SOURCE**: in builder mode the block list is the builder's, whose count can differ from `model._variables` (the C-40 144-vs-145). This is *why* the consumer guards (#1-3) are load-bearing. |

**Swallowed-exception smell (CLAUDE.md §3):** the two SUSPECT sites (#2, #3) each wrapped
the repr build + FBBT in a bare `except Exception: return …` with no log — the exact
compounding smell behind C-40 (misalign → corrupt → eat the `IndexError`). Both were
changed to `except Exception as exc: logger.debug(…); return …` so a genuine failure
is surfaced, not hidden.

## Reproduction — the two SUSPECT sites are real bugs

Regression tests in `python/tests/test_r2_branch_and_reduce.py`
(`test_node_reduce_misaligned_repr_forgoes_tightening`,
`test_root_reduce_misaligned_repr_forgoes_tightening`) inject a deliberately
misaligned repr (`_MisalignedRepr`: `fbbt_with_cutoff` returns `n_blocks-1`
intervals whose positional values would cross a variable's box) and assert the
tightening is forgone (box unchanged, no false fathom).

**Pre-fix (guard removed), node_reduce returns:**

```
NodeReduceResult(lb=array([1.e+09, 0.e+00]),
                 ub=array([-1.00000000e+09, 5.00004005e-01]),
                 infeasible=True, n_tightened=3)
```

`lb[0]=1e9 > ub[0]=-1e9` — a **crossed box → false `infeasible`** (the C-40 mechanism).
**Post-fix:** the misaligned repr is forgone; box stays `[0,4]×[0,4]`, `infeasible=False`,
the optimum retained. Verified failing before / passing after in both directions
(`git stash` of the two source files).

Note the fresh-repr divergence does **not** reproduce on the current corpus for
these two sites (a 250-instance MINLPLib scan found the fresh
`model_to_repr(model, builder)` always returns `len == len(model._variables)`;
`util`'s node_reduce/root_reduce reprs are aligned at 145). The divergence C-40 proved
(144 vs 145) is carried by the **persistent** `_model_repr` — confirmed still live:
solving `util` with logging shows `solver.py:7443` firing "returned 144 intervals,
flat n_vars=145". The node/root sites rebuild a fresh repr that happens to align
today, but their mapping had **no structural guarantee** of it — so the guard is
correctness insurance for exactly the divergence condition C-40 established, and the
synthetic repro shows the failure it prevents.

## The fix (general, mirrors C-40)

`python/discopt/_jax/node_reduce.py` and `python/discopt/_jax/root_reduce.py`, one
hunk each, no name/shape special-casing:

1. After materializing `fbbt_lbs`/`fbbt_ubs`, **forgo** the tightening unless
   `fbbt_lbs.shape[0] == fbbt_ubs.shape[0] == len(model._variables)` (the same 1:1
   alignment invariant as `solver.py:7443` and `solvers/_root_presolve.py:43`). A
   misaligned repr keeps a valid, looser box (CLAUDE.md §3 — forgo an *optional*
   tightening, never a valid bound). The old `bi >= shape[0]` OOB skip is removed —
   the alignment guard makes it dead.
2. The bare swallowing `except Exception:` now `logger.debug(...)`s the failure
   before returning the box unchanged.

Both sites already end with a `lb > ub` → `infeasible`/fathom check on the tightened
box; with the alignment guard that check now only ever sees an aligned (correctly
mapped) box.

## Gates

- **Regression tests** (new, `test_r2_branch_and_reduce.py`): 2 tests, FAIL before /
  PASS after (verified both directions via `git stash`). Full file: 8 passed.
- `pytest python/tests/test_c40_util_false_optimal.py` — 1 passed (C-40 still green;
  `util` bound 999.28 ≤ 999.58).
- `pytest -m smoke python/tests/` — **624 passed, 14 skipped** (includes the 2 new
  C-41 smoke tests).
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — **10 passed**.
- `ruff check` + `ruff format --check` (v0.14.6) on all changed files — clean.
- **mypy** (pre-commit mirror v2.1.0 → mypy 1.15.0) on the two changed modules —
  `Success: no issues found`; zero new errors. (Whole-package mypy in this venv
  crashes on the numpy stubs' `type` statement — a Python-3.12/numpy-version env
  mismatch, pre-existing and unrelated; CI pins a compatible numpy.)
- **cert-baseline** (`check_cert_neutrality.py`, 41-panel): all 41 objectives
  **byte-identical** (`|Δobj|=0.00e+00`, `nodes X->X` on every row). The one reported
  "violation" is `nvs13 node_count 19 -> 49` — **pre-existing drift on `origin/main`**,
  reproduced on the clean tree with both changed source files reverted (nvs13 →
  `nodes=49`, still `optimal`, obj `-585.2` correct). Identical to the drift the C-40
  doc records; NOT introduced by C-41 (the change only touches the `DISCOPT_NODE_REDUCE`
  / `DISCOPT_ROOT_FIXPOINT` paths, default OFF, and the guards are no-ops on aligned
  reprs, so the default-path certificate is unchanged). No rebaseline needed.
- `cargo test -p discopt-core` — **not run / N/A**: no Rust touched (the Rust resync
  #11 was audited SAFE, not modified).

## Follow-up (tracked, unchanged from C-40)

The persistent `_model_repr` 144-vs-145 divergence (builder-mode block count diverging
from `model._variables`) is still latent — the guards forgo the *optional* tightening
on those models rather than fixing the layout. Restoring the tightening by exposing
the repr's flat-column mapping is a performance follow-up, not a soundness item; the
current behavior is correct, just weaker on diverged models. (C-40 doc §Follow-up.)
