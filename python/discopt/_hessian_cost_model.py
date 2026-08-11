"""Hessian first-compile cost model, with no JAX import.

Pure arithmetic over model-size features, split out of ``_relax/nlp_evaluator`` so a
caller can consult it without loading JAX. ``solver._objective_is_convex_quadratic``
imports ``estimate_dense_obj_hessian_compile_s`` on every solve; while it lived in
the evaluator module that one import pulled the whole JAX stack onto an otherwise
JAX-free path (#75) — measured: a pure-bilinear QCQP still loaded 210 jax modules
for a cost estimate that never touches JAX.

The estimates model the *JAX* evaluator's XLA compile. The tape backend has no
compile step and reports 0.0 directly.
"""

from __future__ import annotations

# --- First-time Lagrangian-Hessian XLA compile-cost model (F4) ------------------
# The first evaluate_hessian_values call forces an uninterruptible first-time XLA
# compile. The root-heuristic budget gate (solver.py) needs an a-priori estimate
# of that cost, from cheap model-size features available before the compile runs.
#
# Measured (docs/dev/perf-followup-plan-2026-07-05.md F4 entry experiment; M-series
# arm64, JAX 0.10.2, pounce 0.7.0). First-compile wall vs model size, per path:
#
#   path    instance         n_vars  hess_nnz   compile_s
#   dense   tls2                 37        16       0.15
#   dense   fac2                 66       972       0.15
#   sparse  heatexch_gen1       112       164       ~1.0
#   sparse  hda                 722      1094       2.5–8.3   (noisy across runs)
#   sparse  casctanks           500       820       3.7–5.0
#   sparse  heatexch_gen3       580      1020      46–49
#   sparse  contvar             296      1168     186
#
# FALSIFICATION (§0.6): the plan hypothesized a clean compile ~ f(n_vars, nnz)
# curve. It does not exist. Regressing log(compile) on n_vars gives R^2 = 0.002
# (essentially zero — contvar at n=296 compiles ~74x slower than hda at n=722),
# and the same instance's compile varies 2.5s->8.3s run to run. The cost is
# governed by the *shape/depth* of the lifted DAG (contvar's deep nested
# log/exp/division chains), not by any cheap size scalar, and is not reliably
# predictable in advance.
#
# Consequently the estimate is deliberately CONSERVATIVE, not a point predictor:
#   * DENSE path: bounded and always cheap in the measured range -> small constant.
#   * SPARSE (compressed-HVP) path with the kernel not yet compiled: the compile
#     is potentially very large (up to ~3x a 60s budget) and unpredictable, so the
#     estimate returns a large floor. The gate then only enters when there is
#     ample budget headroom, and skips (soundly — it is a primal heuristic) when
#     there is not. The conservative floor is what preserves the time_limit
#     contract; a precise number is neither available nor needed, because being
#     wrong high only skips a heuristic (never affects the dual bound).
_HESSIAN_COMPILE_DENSE_S = 0.5
# Floor for the risky first sparse compile. The compile can be arbitrarily large
# (measured 1s->186s cold, R^2~0 vs any cheap size feature) and cannot be polled
# once entered, so this is a *risk headroom*, not a point estimate: the gate only
# starts a first sparse compile when at least this much budget remains. Chosen so
# the in-solve first heuristic still runs on a normal (tens-of-seconds) budget —
# the measured in-solve first compile is a few seconds — while a tight budget
# (e.g. time_limit=5) refuses to gamble the whole contract on an unbounded
# compile. A single policy constant on the whole no-relaxation class, not tuned
# per instance. Over-estimating only skips a primal heuristic (sound).
_HESSIAN_COMPILE_SPARSE_FLOOR_S = 15.0


def estimate_hessian_compile_s(n_vars: int, hessian_nnz: int, use_sparse: bool) -> float:
    """Conservative estimate of the first-time Hessian XLA compile wall (seconds).

    See the module-level comment for the measured basis and why this is a
    conservative floor rather than a point predictor (the compile is not reliably
    predictable from cheap size features; R^2 ~ 0 vs n_vars). Used only to gate
    entry into *primal-heuristic* NLPs, so an over-estimate merely skips a
    heuristic (always sound) and never touches the dual bound.
    """
    if not use_sparse:
        # Dense ``jacfwd∘jacfwd`` path. The flat constant held only because the F4
        # measurements covered dense on TINY objectives (n<=66). A large dense
        # Hessian (a big quadratic form like qap, hessian_nnz~43k) compiles for
        # tens of seconds — super-linear in the term count — so size the dense
        # estimate too (#654). ``hessian_nnz`` counts matrix nonzeros (off-diagonal
        # entries appear twice); halve it to the distinct-quadratic-term count the
        # dense-objective model is calibrated on.
        return estimate_dense_obj_hessian_compile_s(hessian_nnz // 2)
    # Sparse compressed-HVP path: unpredictable and potentially budget-dwarfing.
    return _HESSIAN_COMPILE_SPARSE_FLOOR_S


# --- Dense OBJECTIVE-Hessian compile-cost model (#654 qap overrun) ---------------
# ``estimate_hessian_compile_s`` above models the LAGRANGIAN Hessian, whose dense
# branch is a flat small constant because it was fit only on tiny-objective
# instances (tls2 n=37, fac2 n=66 obj). It is BLIND to the dense OBJECTIVE Hessian
# ``jax.jacfwd(jax.jacfwd(obj_fn))`` that ``_objective_is_convex_quadratic`` forces:
# that kernel's XLA codegen is super-linear in the number of quadratic cross-terms
# of the objective, and on a large quadratic form it dwarfs the whole time budget
# uninterruptibly.
#
# Measured (M-series arm64, JAX 0.10.2), first dense-objective-Hessian compile vs
# the objective's quadratic nnz (distinct x_i·x_j / x_i^2 terms):
#     instance   obj_quad_nnz   compile_s
#     fac2              972        0.15
#     qap            21 424       48+     (gradient compile alone > 150s)
# 22x more nnz -> ~320x compile (empirical exponent ~1.87): unmistakably
# super-linear. As with the sparse floor this is deliberately CONSERVATIVE, not a
# point predictor — it only gates entry into the convex-objective node bound (a
# bound *tightening*, never a validity source), so over-estimating merely falls
# back to the McCormick relaxation (sound) and never touches the dual bound.
_DENSE_OBJ_HESS_CHEAP_NNZ = 1500  # below this the dense obj-Hessian compile is trivially cheap
_DENSE_OBJ_HESS_REF_NNZ = 1500  # ratio anchor for the super-linear growth term


def estimate_dense_obj_hessian_compile_s(obj_quad_nnz: int) -> float:
    """Conservative estimate of the first dense OBJECTIVE-Hessian XLA compile (s).

    ``obj_quad_nnz`` is the count of distinct quadratic terms in the objective
    (bilinear cross terms + squares); the dense ``jacfwd∘jacfwd`` second-derivative
    graph — and hence its XLA codegen time — grows super-linearly in that count.
    Below :data:`_DENSE_OBJ_HESS_CHEAP_NNZ` the compile is trivially cheap
    (constant). Above it, a quadratic-in-nnz growth term is used; the exponent 2.0
    slightly over-estimates the measured ~1.87 on purpose, so the value is an upper
    bound the budget gate can trust. Over-estimating only skips the convex-objective
    *tightening* (sound); it never affects the dual bound or the returned optimum.
    """
    if obj_quad_nnz <= _DENSE_OBJ_HESS_CHEAP_NNZ:
        return _HESSIAN_COMPILE_DENSE_S
    ratio = float(obj_quad_nnz) / float(_DENSE_OBJ_HESS_REF_NNZ)
    return _HESSIAN_COMPILE_DENSE_S * ratio * ratio
