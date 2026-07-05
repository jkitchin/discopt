//! Batched LP solving over a shared constraint matrix.
//!
//! A great deal of the simplex's per-solve cost is *setup* that depends only on
//! the constraint matrix `A`, not on the rhs or bounds: equilibration scaling
//! ([`Scaling`]) and the dense→scaled matrix copy. In the contexts that dominate
//! the solver — a spatial/MILP B&B re-solving thousands of nodes that all share
//! the same `A`, strong-branching probes, and parametric/scenario sweeps — that
//! setup is otherwise redone on every solve even though it is identical.
//!
//! [`solve_lp_batch`] amortizes it: the scaling and the scaled matrix/objective
//! are computed once, then each instance (its own rhs and bounds) is solved
//! independently and, under the `parallel` feature, concurrently. Results are
//! returned in input order, each already mapped back to the original (unscaled)
//! space, so a batch solve is observationally identical to solving each LP on its
//! own with [`solve_lp`](super::solve_lp) — only faster.
//!
//! [`solve_lp_multi_rhs`] is the common special case: one LP (`A`, `c`, bounds)
//! solved for several right-hand sides.

use super::primal::solve_lp_scaled;
use super::scaling::Scaling;
use super::{LpSolve, SimplexOptions};
use crate::lp::crossover::LpView;

/// One LP in a batch: the right-hand side and variable bounds. The constraint
/// matrix `A` and objective `c` are shared by the whole batch (see
/// [`solve_lp_batch`]).
#[derive(Debug, Clone)]
pub struct LpInstance {
    /// Right-hand side `b`, length `m`.
    pub b: Vec<f64>,
    /// Lower bounds `l`, length `n`.
    pub l: Vec<f64>,
    /// Upper bounds `u`, length `n`.
    pub u: Vec<f64>,
}

/// Solve a batch of LPs `min cᵀx s.t. A x = bₖ, lₖ ≤ x ≤ uₖ` that share the
/// row-major `m × n` matrix `a` and objective `c`. The equilibration scaling and
/// scaled matrix are computed once and reused for every instance; instances are
/// solved independently (concurrently under the `parallel` feature) and returned
/// in order, with each solution mapped back to the original space.
pub fn solve_lp_batch(
    a: &[f64],
    m: usize,
    n: usize,
    c: &[f64],
    instances: &[LpInstance],
    opts: &SimplexOptions,
) -> Vec<LpSolve> {
    // Shared setup: scale once if the matrix warrants it, else solve raw.
    match Scaling::from_matrix(a, m, n) {
        Some(scaling) => {
            let a_s = scaling.scale_matrix(a);
            let c_s = scaling.scale_c(c);
            let solve_one = |inst: &LpInstance| -> LpSolve {
                let b_s = scaling.scale_b(&inst.b);
                let l_s = scaling.scale_lower(&inst.l);
                let u_s = scaling.scale_upper(&inst.u);
                let view = LpView {
                    a: &a_s,
                    m,
                    n,
                    c: &c_s,
                    l: &l_s,
                    u: &u_s,
                };
                let mut sol = solve_lp_scaled(&view, &b_s, opts);
                scaling.unscale_x(&mut sol.x);
                // Map the certificate vectors back to the original space too, so a
                // batch solve is observationally identical to `solve_lp` (which
                // unscales all three). A scaled dual/ray checked against the
                // unscaled A/b would make the Neumaier–Shcherbina safe bound or the
                // Farkas verification spuriously fail — exactly on the ill-scaled
                // LPs where the certificate matters most (Rust-1).
                scaling.unscale_dual(&mut sol.dual);
                scaling.unscale_ray(&mut sol.ray);
                sol
            };
            map_instances(instances, solve_one)
        }
        None => {
            let solve_one = |inst: &LpInstance| -> LpSolve {
                let view = LpView {
                    a,
                    m,
                    n,
                    c,
                    l: &inst.l,
                    u: &inst.u,
                };
                solve_lp_scaled(&view, &inst.b, opts)
            };
            map_instances(instances, solve_one)
        }
    }
}

/// Solve one LP `min cᵀx s.t. A x = bₖ, l ≤ x ≤ u` for several right-hand sides
/// `b_list`, sharing the matrix, objective, and bounds across all of them.
pub fn solve_lp_multi_rhs(
    lp: &LpView<'_>,
    b_list: &[Vec<f64>],
    opts: &SimplexOptions,
) -> Vec<LpSolve> {
    let instances: Vec<LpInstance> = b_list
        .iter()
        .map(|b| LpInstance {
            b: b.clone(),
            l: lp.l.to_vec(),
            u: lp.u.to_vec(),
        })
        .collect();
    solve_lp_batch(lp.a, lp.m, lp.n, lp.c, &instances, opts)
}

/// Map `solve_one` over the instances, in parallel when the `parallel` feature is
/// on and the batch is large enough to amortize task-spawn overhead. Each solve
/// is independent (its own factorization) and reads only the shared, immutable
/// scaled matrix, so the result order is deterministic regardless of scheduling.
fn map_instances<F>(instances: &[LpInstance], solve_one: F) -> Vec<LpSolve>
where
    F: Fn(&LpInstance) -> LpSolve + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const PAR_MIN_BATCH: usize = 4;
        if instances.len() >= PAR_MIN_BATCH {
            return instances.par_iter().map(&solve_one).collect();
        }
    }
    instances.iter().map(&solve_one).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::{solve_lp, LpStatus};

    const INF: f64 = 1e20;

    fn opts() -> SimplexOptions {
        SimplexOptions::default()
    }

    // A batched solve must return exactly what solving each LP on its own does.
    #[test]
    fn batch_matches_individual_solves() {
        // min -x0 - 2 x1 s.t. x0+x1+s0=B0, x0+3x1+s1=B1, x∈[0,inf], s∈[0,inf].
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = vec![0.0; 4];
        let u = vec![INF; 4];
        let instances: Vec<LpInstance> = [(4.0, 6.0), (5.0, 6.0), (2.0, 9.0), (10.0, 1.0)]
            .iter()
            .map(|&(b0, b1)| LpInstance {
                b: vec![b0, b1],
                l: l.clone(),
                u: u.clone(),
            })
            .collect();

        let batched = solve_lp_batch(&a, m, n, &c, &instances, &opts());
        assert_eq!(batched.len(), instances.len());
        for (inst, got) in instances.iter().zip(&batched) {
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &inst.l,
                u: &inst.u,
            };
            let single = solve_lp(&lp, &inst.b, &opts());
            assert_eq!(got.status, single.status);
            assert!(
                (got.obj - single.obj).abs() < 1e-9,
                "batch {} vs single {}",
                got.obj,
                single.obj
            );
        }
    }

    // The same, but on an ill-scaled matrix so the shared-scaling path is taken.
    #[test]
    fn batch_matches_individual_on_ill_scaled() {
        // Wide-range coefficients trigger equilibration; the batch shares it.
        let a = [1e8, 1.0, 1.0, 1e8, 1.0, 0.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let c = [1.0, 1.0, 0.0, 0.0];
        let l = vec![0.0; 4];
        let u = vec![1e6, 1e6, INF, INF];
        let instances: Vec<LpInstance> = [(2e8, 1.0), (1e8, 2.0)]
            .iter()
            .map(|&(b0, b1)| LpInstance {
                b: vec![b0, b1],
                l: l.clone(),
                u: u.clone(),
            })
            .collect();

        let batched = solve_lp_batch(&a, m, n, &c, &instances, &opts());
        for (inst, got) in instances.iter().zip(&batched) {
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &inst.l,
                u: &inst.u,
            };
            let single = solve_lp(&lp, &inst.b, &opts());
            assert_eq!(got.status, single.status);
            if single.status == LpStatus::Optimal {
                assert!((got.obj - single.obj).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn multi_rhs_matches_individual() {
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = vec![0.0; 4];
        let u = vec![INF; 4];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let b_list = vec![vec![4.0, 6.0], vec![3.0, 3.0], vec![8.0, 2.0]];
        let got = solve_lp_multi_rhs(&lp, &b_list, &opts());
        for (b, sol) in b_list.iter().zip(&got) {
            let single = solve_lp(&lp, b, &opts());
            assert_eq!(sol.status, single.status);
            assert!((sol.obj - single.obj).abs() < 1e-9);
        }
    }

    // Rust-1 regression: on the shared-scaling path the batch must return the
    // dual/ray in the *original* (unscaled) space — bit-identical to what the
    // single-solve `solve_lp` returns, which unscales them. Before the fix the
    // batch left `dual`/`ray` in scaled space, so a caller verifying the
    // Neumaier–Shcherbina safe bound or the Farkas ray against the unscaled A/b
    // would silently see a wrong certificate.
    #[test]
    fn batch_unscales_dual_like_single_solve() {
        // Wide-range coefficients force equilibration (the scaled branch). Bounded
        // feasible instances → Optimal, so `dual` (the row multipliers) is nonempty.
        let a = [1e8, 1.0, 1.0, 1e8, 1.0, 0.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let c = [1.0, 1.0, 0.0, 0.0];
        let l = vec![0.0; 4];
        let u = vec![1e6, 1e6, INF, INF];
        let instances: Vec<LpInstance> = [(2e8, 1.0), (1e8, 2.0)]
            .iter()
            .map(|&(b0, b1)| LpInstance {
                b: vec![b0, b1],
                l: l.clone(),
                u: u.clone(),
            })
            .collect();

        let batched = solve_lp_batch(&a, m, n, &c, &instances, &opts());
        // The scaled branch must actually be taken for this to test anything.
        assert!(Scaling::from_matrix(&a, m, n).is_some());

        for (inst, got) in instances.iter().zip(&batched) {
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &inst.l,
                u: &inst.u,
            };
            let single = solve_lp(&lp, &inst.b, &opts());
            assert_eq!(got.status, single.status);
            assert_eq!(
                got.dual.len(),
                single.dual.len(),
                "dual length mismatch batch vs single"
            );
            for (gb, gs) in got.dual.iter().zip(&single.dual) {
                assert!(
                    (gb - gs).abs() < 1e-9,
                    "batch dual {gb} != single (unscaled) dual {gs}"
                );
            }
        }
    }

    // Rust-1 regression, ray arm: an infeasible ill-scaled instance yields a
    // Farkas ray; the batch must unscale it to match the single solve.
    #[test]
    fn batch_unscales_ray_like_single_solve() {
        // x0 - x1 = b0 and x0 - x1 = b1 with b0 != b1 (and equal columns via the
        // ill-scaled block) is contradictory → Infeasible with a Farkas ray.
        // Use a wide-range matrix so the scaled branch runs.
        let a = [1e8, -1e8, 1.0, -1.0];
        let (m, n) = (2, 2);
        let c = [0.0, 0.0];
        let l = vec![0.0, 0.0];
        let u = vec![INF, INF];
        // Row0: 1e8 x0 - 1e8 x1 = 1e8  → x0 - x1 = 1.
        // Row1:     x0 -     x1 = 5     → x0 - x1 = 5.  Contradiction.
        let instances = vec![LpInstance {
            b: vec![1e8, 5.0],
            l: l.clone(),
            u: u.clone(),
        }];

        let batched = solve_lp_batch(&a, m, n, &c, &instances, &opts());
        assert!(Scaling::from_matrix(&a, m, n).is_some());

        let inst = &instances[0];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &inst.l,
            u: &inst.u,
        };
        let single = solve_lp(&lp, &inst.b, &opts());
        let got = &batched[0];
        assert_eq!(got.status, single.status);
        // Whichever certificate vector the status populates, batch must match the
        // unscaled single solve element-for-element.
        assert_eq!(got.dual.len(), single.dual.len());
        assert_eq!(got.ray.len(), single.ray.len());
        for (gb, gs) in got.dual.iter().zip(&single.dual) {
            assert!((gb - gs).abs() < 1e-9, "batch dual {gb} != single {gs}");
        }
        for (gb, gs) in got.ray.iter().zip(&single.ray) {
            assert!((gb - gs).abs() < 1e-9, "batch ray {gb} != single {gs}");
        }
    }

    #[test]
    fn empty_batch_is_empty() {
        let a = [1.0, 1.0];
        let got = solve_lp_batch(&a, 1, 2, &[1.0, 0.0], &[], &opts());
        assert!(got.is_empty());
    }
}
