# ENGINE-2 — the zero FT pivot is a bump-elimination cancellation on a *nonsingular* basis, not a redundant lifted column (2026-07-08, #557)

**Task.** ENGINE-1 (#570, `engine1-ft-update-2026-07-08.md`) proved the ~13× BARON
wall gap on the dense-QP class is the in-house simplex refactorizing on nearly
every pivot because feral's Forrest–Tomlin (FT) LU update returns
`RefactorCause::TinyPivot` with pivot magnitude **exactly 0.0**, ~98–100 % of the
time — an accuracy-necessary refactor. ENGINE-2 determines the *source* of that
zero pivot and takes the one plausibly-byte-identical shot: **if the zero traces
to duplicate / linearly-dependent lifted (McCormick / square / moment) columns,
dedup them bound-neutrally** to kill the refactor storm.

**Verdict: KILL (option b — GENUINE / structural).** The zero pivot is **not** a
rank deficiency and **not** a removable redundant column. Across **120 captured
FT-fail events** (60 on nvs24, 60 on st_e36), the pre-update basis `B`, the
augmented `[B | a_q]`, and the post-swap basis `B'` are **all full column rank and
well-conditioned** (median cond(B′) 1.8e4 / 6.0e5; max ≤ 3.4e6; σ_min ≥ 1.5e-4).
The entering replacement column `a_q` is in `span(B)` only in the trivial sense
that *every* vector is in the span of a full-rank `m×m` basis — its `B⁻¹a_q`
spike has 6–104 nonzeros and a **nonzero** leading pivot `ρ[slot] ∈ [5e-3, 5.2]`.
The exactly-zero pivot arises **later**, inside the FT/Reid **bump elimination**:
replacing basis column `slot` makes the upper factor upper-Hessenberg over the
bump `[slot..m-1]`, and the fixed-pivot-order sub-diagonal elimination hits a
diagonal that cancels to **0.0** — reproduced offline for **60/60** events on both
instances on a matrix that is provably nonsingular. There is **no duplicate
structural identity and no linearly-dependent lifted column to remove**, so a
build-time dedup is inapplicable (nothing to dedup) and would be unsound to force
(dropping a full-rank column changes the feasible region). The fix is a
**pivot-searching LU update (Bartels–Golub / Suhl–Suhl)** or a HiGHS node backend,
not a relaxation-layer change. Nothing bound-changing or bound-neutral ships from
this task; the PR is the evidence doc.

Build: release (`maturin develop --release`). Env: `PYTHONPATH=<wt>/python
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, dense-QP regime `DISCOPT_SQUARE_SEPARATE=0`
(matching ENGINE-1/THRU-5, to isolate the base LP solve). Corpus
`~/Dropbox/projects/discopt-minlp-benchmark/`. `feral = 0.12.0` (external
crates.io crate; its FT update is not editable from this repo).

## Phase 0 — instrumentation used (all TEMPORARY, stripped from the PR)

To capture the *exact* zero-pivot state, a TEMP env-gated (`DISCOPT_E2_DUMP`) dump
was added at the FT-fail site in `crates/discopt-core/src/lp/simplex/primal.rs`
(the same site as #570's `RefacFtTinyPivot` split): on each of the first `k`
FT-fails it serialized, as JSON-lines, the **pre-update** basis `B` (reconstructed
by restoring the leaving column `self.basis[slot] ← leaving`, since the pivot code
overwrites `basis[slot]=q` *before* the FT update), the entering replacement
column `a_q`, and the indices `slot`/`q`/`leaving`. This dump was **removed** after
the analysis (the branch is byte-identical to `origin/main` in code — the PR ships
only this doc). The offline analysis scripts (numpy/scipy) are archived in the
session scratchpad (`analyze_dump.py`, `analyze_Bprime.py`, `ft_mechanism.py`,
`ft_bump.py`, `probe_struct.py`).

## Phase 1 — the redundancy tests (all NEGATIVE)

### 1a. Root relaxation has NO duplicate / dependent lifted columns

Built the lifted McCormick relaxation (`build_milp_relaxation`) at the root box and
computed `column_identities` (the stable structural-identity map already used for
cut-pool remapping: orig `(i)`, bilinear `(i,j)`, monomial `(i,p)`, trilinear,
univariate_square `((base_id),2)`, …):

| instance | n_orig | lifted cols | rows | duplicate structural identities | cross-class same-monomial groups | matrix_rank(A) |
|---|---:|---:|---:|---:|---:|---:|
| nvs24  | 10 | 65 | 220 | **0** | **0** | **65/65 (full)** |
| st_e36 |  2 | 82 | 485 | **0** | **0** | **82/82 (full)** |

No `x_i²` lifted twice (square vs bilinear(i,i)); no moment/square/bilinear column
that is an exact linear combination of others; the constraint matrix has **full
column rank**. So the hypothesized build-time redundancy (task Phase 1.3) **does
not exist** in the lift.

### 1b. At every captured FT-fail, the basis is FULL-RANK and well-conditioned

Dense linear-algebra on the captured pre-update basis `B` (m = 46 st_e36, m = 220
nvs24), aggregated over 60 events each:

| instance | events | singular `B` | singular `B'` (post-swap) | `a_q ∈ span(B)` | cond(B′) med / max |
|---|---:|---:|---:|:--:|---:|
| st_e36 | 60 | **0** | **0** | yes (residual ≈ 0) | 1.8e4 / 3.3e5 |
| nvs24  | 60 | **0** | **0** | yes (residual ≈ 0) | 6.0e5 / 3.4e6 |

`#tiny_singular_value(B) = 0` in all events. The updated basis `B'` (with column
`slot` replaced by `a_q`) is **nonsingular** in all 120 events — i.e. the pivot
step is legitimate; a from-scratch factorization of `B'` succeeds (which is exactly
what the refactor does).

### 1c. The FTRAN pivot is NOT the zero

`ρ = B⁻¹ a_q` (the FT spike). Its leaving-row entry `ρ[slot]` — the *first* FT
pivot — is **nonzero** everywhere: st_e36 `ρ[slot] ∈ {0.13 … 5.16}`, nvs24
`ρ[slot] ∈ {5e-3 … 2.30}`; the spike has 6–104 nonzeros. So the zero is not the
initial spike pivot.

### 1d. The zero pivot is an FT *bump-elimination* cancellation (reproduced 120/120)

Reproducing feral's column-replacement update offline: `B = P L U`; spike in the
U-frame `y = L⁻¹Pᵀa_q`; overwrite `U[:,slot] ← y` and cyclic-shift columns
`slot..m-1` to expose the upper-Hessenberg **bump**; eliminate the sub-diagonal
row-by-row in the **fixed** order using the running diagonal as pivot (the
FT/Reid scheme — no row/column pivot search). Result:

| instance | events | FT-bump min\|pivot\| = 0.0 | at bump-row |
|---|---:|---:|:--:|
| st_e36 | 60 | **60/60** | = `slot` (varies per event) |
| nvs24  | 60 | **60/60** | = `slot` (varies per event) |

The fixed-order bump elimination divides by an **exactly-zero** diagonal on a
matrix (`B'`) that is nonsingular and well-conditioned — precisely feral's
`sparse_update.rs` `TinyPivot` at `mag = 0.0` (lines 454/460/498 per ENGINE-1).
A **partial-pivoting refactor** re-selects a stable pivot order (which is why the
from-scratch refactor always succeeds), and a **pivot-searching update**
(Bartels–Golub / Suhl–Suhl) would permute the bump to a nonzero pivot without a
full refactor.

### 1e. Independent corroboration of ENGINE-1 on the clean build

`DISCOPT_PROFILE` on st_e36 (dense-QP regime, 8 s), whole-run aggregate:
`Refactorizations = 11715`, `RefacFtFail = 11714`, **`RefacFtTinyPivot = 11714`
(100 %)**, `RefacFtGrowth = 0`, `RefacFtSingular = 0`, `RefacCap = 1`. Confirms the
#570 cause-split and that 100 % of the storm is the zero-pivot path characterized
above.

## Phase 2 — the decision: (b) GENUINE, dedup inapplicable → KILL

The task's fork:

* **(a) REDUNDANT** — the zero traces to duplicate / linearly-dependent lifted
  columns → dedup bound-neutrally. **Falsified.** No duplicate identities (1a),
  full column rank at root (1a) and in every captured basis (1b), and the zero is
  a bump-elimination artifact on a *nonsingular* basis (1d). There is no redundant
  column to merge or drop; forcing a drop of a full-rank column would change the
  feasible region (unsound, CLAUDE.md §1).
* **(b) GENUINE** — the replacement is dependent for a numerical/structural reason
  not tied to a removable column. **Confirmed.** The dependency is *not* in the
  columns at all; it is the FT update's fixed pivot order failing to find the
  nonzero pivot that provably exists (`B'` nonsingular). Dedup can't fix it.

Nothing ships from this task except this evidence doc: there is no bound-neutral
column to remove, and the real fix is an LP-kernel change (below), out of scope
for a byte-identical PR.

## Phase 3 — ranked next option (the real fix)

1. **Pivot-searching LU basis update (Bartels–Golub / Suhl–Suhl).** The measured
   root cause: FT's *fixed-order* bump elimination hits a structural-zero diagonal
   on a nonsingular basis. BG/Suhl–Suhl searches the bump for a nonzero (and
   numerically-stable) pivot, avoiding the from-scratch refactor. This is the
   direct, minimal-scope fix. `feral = 0.12.0` is an external crate whose FT update
   (`sparse_update.rs`) is not editable here — so this is **upstream feral work or
   a vendored fork**; a version bump is not byte-identical. The discopt-side
   reproducer is the #570 `RefacFtTinyPivot` counter plus the capture harness in
   this doc (the JSON-lines dump of `B`/`a_q`/`slot` at the FT-fail).
2. **A production sparse-LP node backend (HiGHS) for the dense-QP node LP**, keeping
   the in-house simplex where it wins. HiGHS's update kernel handles these bases
   (it is BARON's order of magnitude). Largest scope; only if (1) stalls.
3. **(Struck)** Sparser / column-localized lift (ENGINE-1 §4 #1). ENGINE-1 ranked
   this first on the assumption the dense columns *produce* a rank-order-dependent
   spike. ENGINE-2's measurement refines that: the spike is legitimate and the
   basis is full-rank/well-conditioned; shrinking the bump would reduce the
   *frequency* of the fixed-order collision but does not address the mechanism, and
   it is bound-changing (loosens the relaxation) for a speed effect that (1)
   delivers soundly. De-prioritized behind the LU-kernel fix.

Roadmap update for #557: the highest-leverage lever is now **(1) a pivot-searching
LU update**, with the zero-pivot mechanism pinned to the FT bump elimination
(not the relaxation columns).

## §5 — what shipped

**Nothing in code.** The branch is byte-identical to `origin/main` (the #570
merge, which carries the `RefacFtTinyPivot` cause-split and the pre-existing
`milp_driver.rs` clippy fix). The temporary `DISCOPT_E2_DUMP` capture used for the
analysis was removed. This PR ships only this evidence doc.

## §6 — gates (byte-identical + accuracy + suites)

* **Byte-identical certs:** trivially guaranteed — `git diff origin/main` touches
  only `docs/dev/engine2-zero-pivot-2026-07-08.md`; there is **no code delta**, so
  `node_count` and certified `objective` on the cert panel and the dense-QP set
  (nvs19/24, st_e36, knp3-12) are identical to `origin/main` by construction.
* **Accuracy (C-39 soundness battery):** `test_alan_soundness`,
  `test_cert_finite_bound_soundness`, `test_nonrigorous_fathom_decertifies_optimal`
  — **14 passed, 0 false-infeasibles / 0 false-optima.**
* `cargo test -p discopt-core`: **428 + 4 + 1 passed, 0 failed.**
* `pytest -m smoke python/tests/`: PASS (see PR).
* `pytest -m slow test_adversarial_recent_fixes.py`: PASS (see PR).
* ruff / ruff-format / mypy: no code touched (doc-only).

## §7 — nodes/s before/after

No speedup is claimed or possible from this task: it ships a characterized KILL,
not a per-node change. Dense-QP nodes/s (st_e36 ≈ 9 nps, nvs24 ≈ 0.3 nps in the
cap) is unchanged and remains gated by the zero-pivot refactor storm, whose fix is
Phase-3 option (1). The measurement that moves it — the FT bump-pivot search — is
now root-caused with a reproducer.
