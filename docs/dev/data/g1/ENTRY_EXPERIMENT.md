# G1 Entry Experiment record (nvs05, 20s budget)

Scripts (committed): `g1_entry_experiment.py` (census + naive fused),
`g1_fused_probe2.py` (tuple/device_get fusion), `g1_micro.py` (component
micro-bench), `g1_conv.py` (conversion variants), `g1_concat_probe.py`
(concat-fusion on realistic access pattern).

## Part 1 — callback census (ONE nvs05 solve, time_limit=20, threads=1)

Baseline solve: objective=5.47093412640568, bound=5.351553883096131,
node_count=411, status=feasible, wall=20.15s.

Top-level POUNCE callbacks (per-quantity; jacobian_values wraps jacobian_dense,
hessian_values wraps lagrangian_hessian — nested times overlap):

| quantity          | calls   | total µs   | µs/call |
|-------------------|---------|------------|---------|
| objective         | 380,558 | 3,253,690  | 8.55    |
| constraints       | 382,898 | 2,205,603  | 5.76    |
| gradient          | 42,958  | 239,434    | 5.57    |
| jacobian_values   | 45,967  | 1,665,888  | 36.24   |
| (jacobian_dense)  | 47,258  | 1,527,566  | 32.32   |
| hessian_values    | 44,445  | 1,976,964  | 44.48   |
| (lagrangian_hess) | 44,445  | 1,775,814  | 39.96   |

Distinct iterates: 375,131; total callbacks 988,529.

Iterate quantity co-occurrence (key finding):
- **objective+constraints ONLY: 329,280 iterates (87.8%)** — line-search trial
  points; gradient/Jacobian NOT needed there.
- full (obj+grad+cons+jac+hess): ~39,390 + 3,826 ≈ 43k iterates (11.5%).

## Part 2 — prototype verdicts

- **Naive fuse (f,g,c,J) into one jitted tuple + `jax.device_get`: KILLED
  (0.76–0.90x, slower).** Two reasons: (a) computing the expensive gradient +
  Jacobian at the 88% obj/cons-only trial points is wasted work; (b)
  `jax.device_get` on a pytree is ~5.8µs vs ~0.35µs for a plain `np.asarray` —
  10x slower.
- Micro-bench (already-realized jax arrays): `float(y)`=1.20µs,
  `np.asarray(y)`=0.28µs, `jax.device_get((y,c))`=5.83µs. Dispatch-only for a
  scalar kernel = 2.69µs. → the removable cost is the *dispatch* (fuse it) and
  the fix is to convert with `np.asarray` on ONE concatenated array (never
  `device_get`, never per-scalar `float()` on a fresh dispatch).

## Winning design — concat-fusion, memoized per iterate

Fuse *co-occurring* quantities into ONE jitted function returning ONE
**concatenated array** (not a tuple), convert with ONE `np.asarray`, slice:
- group FC = concat([f, c])  (objective + constraints)
- group GJ = concat([g, J.ravel()])  (gradient + Jacobian)  — dense path only
- Hessian stays separate (called ~10x less; 45,710 vs 402,669).
- iterate memo keyed on `x.tobytes()` (+ params token when parameters exist);
  first quantity in a group computes+caches, the rest slice numpy from the memo.

### Result (realistic 88/12 access pattern replay, median of 3)

| path                     | µs/iter | speedup |
|--------------------------|---------|---------|
| per-quantity (baseline)  | 13.09   | 1.00x   |
| concat-fused memoized    | 5.35    | **2.45x** |

max_abs_diff between paths: f=0, c=0, g=0, J=0 → **byte-identical**.

**Verdict: PASS** (≥2x callback-overhead reduction, zero value deviation).
Proceed to implementation (concat-fusion, NOT the plan's literal tuple form).
