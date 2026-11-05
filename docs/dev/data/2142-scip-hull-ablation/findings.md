# #2142: why SCIP certifies cstr's division-free (homog) hull in 7,470 nodes — SCIP ablation and the discopt gap

Spec: `../2142-exact-hull/scip/cstr_homog_spec.npz` (285 cols, 20 binary, 779 `<=` rows, 5
monomial terms, 29 BLF products; the kernel spec discopt hands its spatial tree under the homog
lowering). Runner: `../2142-scip-expert/run.py`, SCIP 6.3.0 (PySCIPOpt), 1 thread, seed shift 0,
node limit 100,000, 600 s backstop; `ablate.sh NAME ARM [--param k=v ...]` -> `NAME.log`.
Reference: discopt on the same model, deterministic (proposal §5): 1.6108 @10k, 2.1886 @20k,
not certified at 300 s (30,246 nodes, bound 1.58); big-M default 2.2486 @10k, 2.7875 @20k.

## 1. Single-component ablations (nodes to certify; bound at the limit when not certified)

| arm | SCIP params | nodes | bound | wall s | branchings bin / cont / aux |
|---|---|---|---|---|---|
| default | — | **7,470** | opt 3.0620008 | 13 | 402 / 25,561 / 3,386 |
| pseudoobj off | `propagating/pseudoobj/freq=-1` | 6,980 | opt | 20 | |
| reformbinprods off | `constraints/nonlinear/reformbinprods=false` | 7,470 | opt (bit-identical: no-op) | | |
| multi-aggregation off | `presolving/donotmultaggr=true` | 7,493 | opt | | |
| OBBT off | `propagating/obbt/freq=-1` | 8,040 | opt | | |
| perspective nlhdlr off | `nlhdlr/perspective/enabled=false` | 8,358 | opt | | |
| separators off | `separating/maxrounds[root]=0` | 9,082 | opt | | |
| genvbounds off | `propagating/genvbounds/freq=-1` | 9,118 | opt | | |
| heuristics off | `setHeuristics(OFF)` | 12,112 | opt | | |
| **conflict off** | `conflict/enable=false` | **29,538** (4.0x) | opt | 36 | 382 / 102,442 / 10,176 |
| **presolve off** | `presolving/maxrounds=0` | **48,521** (6.5x) | opt | 145 | 131 / 48,472 / **0** |
| **aggregation off** (presolve on) | `presolving/donotaggr=true` | **50,269** (6.7x) | opt | 133 | 476 / 52,651 / 0 |
| presolve + conflict off | | 82,343 | opt | 201 | |
| "emu": presolve, conflict, separators, OBBT, genvbounds, redcost, rootredcost, vbounds, probing, symmetry off | | 93,734 | opt | 219 | 90 / 93,704 / 0 |
| `linear` cons propagation off | `constraints/linear/propfreq=-1` | >100k | 1.70 | | |
| `nonlinear` cons propagation off | `constraints/nonlinear/propfreq=-1` | >100k | 2.57 | | |
| all propagation off | `propagating/maxrounds[root]=0` | >100k | 0.20 | | |

**`emu` at 20k nodes: bound 1.6757** (`emu_20k.log`) — BELOW discopt's 2.1886 at 20k on the same
model. With presolve (aggregation), conflict and the extra propagators off, SCIP's base loop
(LP + cons_nonlinear/linear propagation + ENFOLP cuts + spatial branching) is no better than
discopt's tree with its dual proofs; the whole SCIP advantage is in the switched-off
components, of which aggregation (6.5x) and conflict (4x, which discopt already has) dominate.

Pending at the time of writing (queued, logs land in this directory):
`emu_nonlinprop_off_20k`, `nopre_noconf_20k`, `nopresolve_20k`, `noconflict_20k`,
`donotaggr_20k` (the 20k-node bounds of the big arms, for comparison with discopt's 2.1886),
and the conflict sub-arms `conf_graphonly` (`useinflp=c`), `conf_dualonly` (`useinflp=d`),
`conf_nolocalrows`.

Default-run statistics (`default_stats.log`): presolve `linear` 120 AggrVars, 635 DelCons, 28
ChgBounds; `trivial` 51 FixedVars; probing 4 aggr / 22 bound changes; in tree `nonlinear`
453k domreds / 401 cutoffs / 57k ENFOLP cuts, `linear` 151k domreds, genvbounds 22k,
pseudoobj 212 cutoffs; conflict: 995 infeasible-LP calls -> 93 dual rays (17 applied
globally, 77 locally), bound-exceeding LP 1,600 calls -> 1; branching: cons_nonlinear
enforcement 7,486 children, relpscost 28 children (so ~all branching is spatial, on continuous
columns); 68 Ipopt sub-NLPs; incumbent 3.0620008 found at node 7,414 (first solution 3.1302 at
the root). No restarts (nruns = 1).

## 2. What SCIP's presolve does to this spec, and what discopt already does

`presolve_dump.py` (presolve only, transformed problem written as `presolved_homog.cip`):
285 cols / 779 rows -> **110 vars / 174 conss** (55 linear, 78 varbound, 8 setppc, 33
nonlinear). Removed columns: **51 fixed** (49 continuous at 0, binaries x0 = 1, x1 = 0) and
**124 aggregated**: 100 continuous identities `x_j = x_k`, 7 aux identities, 9 negated
binaries `y' = 1 - y`, 8 multi-column (e.g. `x259 = x32 - x189 - x42`, `x274 = -2 x284`).

`root_box.py`: discopt's own per-node propagator (`propagate_spec_py`, 15 rounds, the tree's
cap) on the spec's global box fixes **exactly the same 51 columns**; SCIP's remaining extra
bound changes on surviving columns are ~0.01-sized (18 columns, e.g. lb 0 -> 0.00999). So
presolve's 6.5x is **not** bound tightening, which discopt matches at the root, and the
`donotaggr` arm confirms it directly: with presolve on but aggregation off SCIP needs 50,269
nodes, the same as with no presolve at all. **The mechanism is the column aggregation.**

`aggr_derive.py` derives the aggregations from the spec with no SCIP input: root FBBT
fixings, then (to closure) every pair of parallel `<=` rows of support 2 that forms a
doubleton equality `a x_j + b x_k = c`; 111 aggregations in 2 rounds (SCIP: 116
single-column; the sets differ only in which member of each class is the representative and
in chains). Saved as `aggr.npz` (`x_j = a x_k + b`).

Why aggregation moves a *spatial* tree this much (SCIP's census shows it): without presolve
SCIP never branches on an aux column and spends 2x the branchings on continuous originals;
each equivalence class of copies (`x`, its disjunct copy `x_1` with the other copy fixed at
0 by the root fixings) is otherwise two branching candidates, two envelope columns, two
pseudocost histories and an FBBT hop. discopt's kernel sees all 124 duplicate columns.

## 3. Conflict analysis

Conflict off is 4.0x (29,538). discopt already has the counterpart (learned dual-proof rows,
graduated §112: on homog 81 proofs learned / 159 fathoms / 323k tightenings in 20k nodes,
bound 2.19 with them vs 0.75 without). SCIP's 93 dual rays are of the same order, so the
conflict component is not where discopt is missing a mechanism; the sub-arms (pending) will
say how much of SCIP's value is the local-row proofs (77 of 94 applied locally) that
discopt's global-rows-only construction drops.

## 4. discopt entry experiment (pre-registered in `PREREG.md`)

`arm_aggr.py`: the `2142-exact-hull/arm.py` driver with a spy on `solve_spatial_tree_py`
that rewrites the kernel spec with the 111 derived aggregations (objective, fixed rows,
term/BLF forms, bounds mapped to the representative, aggregated column fixed at 0 and dropped
from `obbt_candidates`; `incumbent_x` and the `incumbent_value` / `primal_hook` arguments
expanded back; every expanded incumbent asserted feasible on the ORIGINAL spec to 1e-6).
Arms: `aggr` 10k, `aggr` 20k, `base` 10k (bit-identity control against 1.6107840610272945).
Kill: 20k bound <= 2.7875. Logs: `arm_aggr_10000.log`, `arm_aggr_20000.log`,
`arm_base_10000.log` (lines `RESULT ...`, `rewrite stats`, `counters`). **Results pending at
the time of writing** (launched 14:09, ~4 processes on 4 cores; the base 10k alone takes
~3 min unloaded, 20k ~7 min).

## 5. Implementation proposal (for the agent that builds it)

**Mechanism:** a spec-level aggregation presolve for the spatial kernel — SCIP's
`cons_linear` doubleton-equality aggregation (`SCIPaggregateVars`), applied once to the
`SpatialKernelSpec` after root FBBT and before the tree, with the result expanded back.

- **Flag:** `DISCOPT_KERNEL_AGGREGATE` (default OFF until the §5 panel; `solver_tuning.py`
  field `kernel_aggregate`, `_env_flag(..., default=False)`), read in `solver.py` next to the
  `dual_proof_propagation` kwarg. Add the row to `docs/dev/flag-retirement-audit.md`.
- **Where:** Python, `python/discopt/_relax/spatial_aggregate.py` (new):
  `aggregate_spec(spec) -> (spec2, expand)` = `aggr_derive.py` + `arm_aggr.rewrite` made
  production code: (1) `_rust.propagate_spec_py` on the global box (15 rounds) -> fixed
  columns; (2) reduce rows by the fixings, find parallel support-2 row pairs (scaled
  negatives) -> doubleton equalities -> `x_j := a x_k + b`, iterate to closure, resolve
  chains; (3) a column is aggregatable only when the encoding can express it (identity for
  `term_out`/`blf_w` outputs and for kind-0/1 term inputs; any affine map for kind-2/3/4
  inputs and BLF forms; skip when `c_j != 0 and b != 0`, or move `b c_j` into
  `meta_obj_offset`); representative = the lower-index column, integer over continuous;
  (4) substitute into `c`, fixed rows (drop `0 <= b` rows, assert `b >= 0`), terms, BLF
  forms, `obbt_candidates`; map `x_j`'s bounds and integrality onto `x_k`; fix `x_j` at 0.
  Called in `solver.py` right before `_rust.solve_spatial_tree_py` (the §4 spy, made the
  default path under the flag): wrap `incumbent_value` and `primal_hook` with `expand`,
  expand `res["incumbent_x"]`. Alternative: Rust, at the spec intake in
  `crates/discopt-core/src/bnb/spatial_tree.rs` (`solve_spatial_tree`), expanding
  `incumbent_x` and the callback points there — same algorithm, no Python round trip; prefer
  Python first (observable, testable without a rebuild).
- **Soundness:** every substitution is an equality implied by two globally valid rows, so
  the reduced problem is equivalent on the global box; the only numerical step is the
  `(c - b x_k)/a` arithmetic (reject pairs with `|a| < 1e-9` or coefficient ratio outside
  `[1e-6, 1e6]`); the expanded point is re-verified by the existing
  `_native_kernel_verify_point`. Bound-neutral in the sense of §5? No — it changes branching
  candidates and envelopes, so it is bound-*changing*: differential panel (flag ON vs OFF over
  the in-repo corpus + GDP parity + tanksize/nous1), cert-clean and net-positive, before
  default ON.
- **Tests** (`python/tests/test_2142_kernel_aggregate.py`): (a) unit: a 3-column spec with
  `x0 = x1` encoded as two `<=` rows aggregates to 2 columns, `expand` restores `x0`;
  (b) a negated-binary pair and an affine `x_j = 2 x_k + 1` into a kind-2 square and a BLF
  form; (c) a term-output / kind-1 input non-identity pair is *not* aggregated;
  (d) regression on cstr homog (`DISCOPT_HULL_EXACT_PERSPECTIVE=1`): 111 aggregations, the
  expanded incumbent feasible on the unaggregated spec, 10k-node deterministic bound equal
  to the recorded `arm_aggr_10000.log` value; (e) every point handed to `incumbent_value` /
  `primal_hook` has `len == n_cols` and satisfies the aggregation equalities.
- **Counters:** `KernelAggregated` (columns), `KernelAggrRowsDropped`.
