//! One LP matrix equilibrated once and solved many times, with every `Optimal`
//! verified against the ORIGINAL data before it leaves this module (#1945).
//!
//! The auto-scaling entry points ([`super::solve_lp`], [`super::solve_lp_cols_scaled`],
//! [`super::solve_lp_warm`], [`super::solve_lp_warm_csc`], [`super::solve_lp_batch`])
//! equilibrate internally and, since #1942, re-audit a scaled `Optimal` against the
//! original LP: the simplex's own final audit runs on the equilibrated system with a
//! floor of `1e-6` in SCALED units, so a column scaled by `~1e6` admits `~0.1` of
//! violation in original units. A caller that scales the matrix itself and calls the
//! solve core on scaled data bypassed that check, because the core never sees the
//! original units. The MILP driver did exactly that (it equilibrates once per batch
//! and shares the factors across the batch's node LPs), and on real instances it
//! returned `Optimal` node LPs whose unscaled point missed a bound or row by up to
//! `1e-3` relative, some of them integral vertices it adopted as incumbents (#1945:
//! 314 of 18,930 driver `Optimal`s over the in-repo corpus on the AMP route).
//!
//! [`ScaledProblem`] is the way to scale once and solve many times without losing
//! that guarantee. It owns the original matrix, the [`Scaling`] and the scaled
//! matrix; its solves take ORIGINAL bounds and return results in SOLVE coordinates
//! (scaled when the matrix is scaled -- exactly what the caller's existing
//! unscale-after-solve code expects), and they all pass through [`ScaledProblem::verify`]:
//!
//! * an `Optimal` whose unscaled point fails the original-space audit is re-solved on
//!   the original data (#649's recovery, warm from the same start when there is one);
//! * a retry that decides the LP is mapped back into solve coordinates -- exactly,
//!   since every factor is a power of two -- so a Farkas ray or row duals verify and
//!   unscale exactly as a scaled solve's would;
//! * a retry that decides nothing turns the verdict into `Numerical` with no
//!   certificate, which callers treat as an undecided LP (branch, never fathom).
//!
//! The check can only refuse or re-derive an `Optimal`, never create one. Outside
//! `lp/simplex` no code builds a [`Scaling`] or scales a matrix; the structural test
//! below fails if one does, so a new caller cannot reopen the bypass silently.

use super::primal::{downgrade_false_optimal, point_fails_original_audit, retry_decided};
use super::scaling::Scaling;
use super::sparse::SparseCols;
use super::{solve_lp_cols, solve_lp_warm_scaled_csc, LpSolve, LpStatus, SimplexOptions};
use crate::lp::basis::Basis;
use crate::lp::crossover::LpView;

/// An LP matrix with its objective and rhs, equilibrated once when ill-conditioned.
/// See the module documentation.
pub struct ScaledProblem {
    m: usize,
    n: usize,
    scaling: Option<Scaling>,
    original: SparseCols,
    /// The equilibrated matrix; `None` when the matrix is well-conditioned (solves
    /// then run on `original`, and the simplex's own audit is in original units).
    scaled: Option<SparseCols>,
    c: Vec<f64>,
    b: Vec<f64>,
    c_s: Vec<f64>,
    b_s: Vec<f64>,
}

impl ScaledProblem {
    /// Equilibrate `original` (an `m × n` CSC) when its dynamic range warrants it
    /// ([`Scaling::from_sparse`]); otherwise keep it as is.
    pub fn new(original: SparseCols, m: usize, n: usize, c: &[f64], b: &[f64]) -> Self {
        let scaling = Scaling::from_sparse(&original, m, n);
        let (scaled, c_s, b_s) = match &scaling {
            Some(s) => {
                let mut a = original.clone();
                s.scale_cols(&mut a);
                (Some(a), s.scale_c(c), s.scale_b(b))
            }
            None => (None, Vec::new(), Vec::new()),
        };
        ScaledProblem {
            m,
            n,
            scaling,
            original,
            scaled,
            c: c.to_vec(),
            b: b.to_vec(),
            c_s,
            b_s,
        }
    }

    /// The equilibration, or `None` when solves run on the original data.
    pub fn scaling(&self) -> Option<&Scaling> {
        self.scaling.as_ref()
    }

    /// The matrix solves run on (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_matrix(&self) -> &SparseCols {
        self.scaled.as_ref().unwrap_or(&self.original)
    }

    /// The original, unscaled matrix.
    pub fn original_matrix(&self) -> &SparseCols {
        &self.original
    }

    /// Solve-space objective (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_c(&self) -> &[f64] {
        if self.scaling.is_some() {
            &self.c_s
        } else {
            &self.c
        }
    }

    /// Solve-space rhs (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_b(&self) -> &[f64] {
        if self.scaling.is_some() {
            &self.b_s
        } else {
            &self.b
        }
    }

    /// Solve-space copies of ORIGINAL bounds `l`, `u` (infinite sides preserved).
    pub fn solve_bounds(&self, l: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match &self.scaling {
            Some(s) => (s.scale_lower(l), s.scale_upper(u)),
            None => (l.to_vec(), u.to_vec()),
        }
    }

    /// Warm re-solve from `start` over ORIGINAL bounds `l`, `u` (dual simplex, cold
    /// fallback on any difficulty). The result is in solve coordinates and verified
    /// ([`Self::verify`]).
    pub fn solve_warm(
        &self,
        l: &[f64],
        u: &[f64],
        start: &Basis,
        opts: &SimplexOptions,
    ) -> LpSolve {
        let (sl, su) = self.solve_bounds(l, u);
        let view = self.view(self.solve_c(), &sl, &su);
        let sol = solve_lp_warm_scaled_csc(&view, self.solve_b(), start, opts, self.solve_matrix());
        self.verify(sol, l, u, Some(start), opts)
    }

    /// Cold solve over ORIGINAL bounds `l`, `u`. The result is in solve coordinates
    /// and verified ([`Self::verify`]).
    pub fn solve_cold(&self, l: &[f64], u: &[f64], opts: &SimplexOptions) -> LpSolve {
        let (sl, su) = self.solve_bounds(l, u);
        let sol = solve_lp_cols(
            self.solve_matrix().clone(),
            self.m,
            self.n,
            self.solve_c(),
            &sl,
            &su,
            self.solve_b(),
            opts,
        );
        self.verify(sol, l, u, None, opts)
    }

    /// The choke point: check a solve-space result `sol` of this problem over
    /// ORIGINAL bounds `l`, `u` against the original data. Returns `sol` unchanged
    /// unless it is an `Optimal` of a scaled solve whose unscaled point fails the
    /// original-space audit; then the LP is re-solved on the original data (warm
    /// from `start` when given, else cold) and a decided retry is returned in solve
    /// coordinates, or the verdict becomes `Numerical` with no certificate.
    ///
    /// For a caller that runs its own solve on [`Self::solve_matrix`] (e.g. a
    /// prepared dual re-optimization); [`Self::solve_warm`] / [`Self::solve_cold`]
    /// call it already.
    pub fn verify(
        &self,
        sol: LpSolve,
        l: &[f64],
        u: &[f64],
        start: Option<&Basis>,
        opts: &SimplexOptions,
    ) -> LpSolve {
        // Unscaled: the simplex's own final audit already ran in original units.
        let Some(s) = &self.scaling else {
            return sol;
        };
        if sol.status != LpStatus::Optimal {
            return sol;
        }
        let mut x = sol.x.clone();
        s.unscale_x(&mut x);
        if !point_fails_original_audit(&x, &self.original, self.m, self.n, &self.b, l, u) {
            return sol;
        }
        let view = self.view(&self.c, l, u);
        let mut alt = match start {
            Some(basis) => solve_lp_warm_scaled_csc(&view, &self.b, basis, opts, &self.original),
            None => solve_lp_cols(
                self.original.clone(),
                self.m,
                self.n,
                &self.c,
                l,
                u,
                &self.b,
                opts,
            ),
        };
        if !retry_decided(&alt) {
            return downgrade_false_optimal(sol);
        }
        // The retry ran on the original data; express it in solve coordinates, as
        // every other result of this problem is. Exact: power-of-two factors.
        s.scale_x(&mut alt.x);
        s.scale_dual(&mut alt.dual);
        s.scale_ray(&mut alt.ray);
        alt
    }

    fn view<'a>(&'a self, c: &'a [f64], l: &'a [f64], u: &'a [f64]) -> LpView<'a> {
        LpView {
            a: &[], // the CSC solvers read the matrix from their CSC argument
            m: self.m,
            n: self.n,
            c,
            l,
            u,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::{
        solve_lp, solve_lp_batch, solve_lp_cols_scaled, solve_lp_warm, solve_lp_warm_csc,
        LpInstance,
    };

    const INF: f64 = 1e20;
    const FIXTURE: &str = include_str!("testdata/adhya1stp_tiny_coef_false_optimal_lp.json");

    struct Lp {
        m: usize,
        n: usize,
        sp: SparseCols,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
    }

    /// The fixture's LP and its captured basis (the flat-JSON format of
    /// `dual.rs`'s `parse_stall_fixture`).
    fn fixture() -> (Lp, Basis) {
        fn field<'a>(json: &'a str, key: &str) -> &'a str {
            let k = format!("\"{key}\":");
            &json[json.find(&k).expect("key present") + k.len()..]
        }
        fn scalar(json: &str, key: &str) -> usize {
            let s = field(json, key);
            s[..s.find([',', '}']).unwrap()].trim().parse().unwrap()
        }
        fn arr(json: &str, key: &str) -> Vec<f64> {
            let s = field(json, key);
            let open = s.find('[').unwrap() + 1;
            let close = s[open..].find(']').unwrap() + open;
            s[open..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse().unwrap())
                .collect()
        }
        let us = |k: &str| {
            arr(FIXTURE, k)
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>()
        };
        let lp = Lp {
            m: scalar(FIXTURE, "m"),
            n: scalar(FIXTURE, "n"),
            sp: SparseCols::from_csc(us("col_ptr"), us("row_idx"), arr(FIXTURE, "vals")),
            c: arr(FIXTURE, "c"),
            l: arr(FIXTURE, "l"),
            u: arr(FIXTURE, "u"),
            b: arr(FIXTURE, "b"),
        };
        let basis = Basis {
            col_status: arr(FIXTURE, "col_status")
                .into_iter()
                .map(|v| v as i8)
                .collect(),
            basic_vars: us("basic_vars"),
        };
        (lp, basis)
    }

    /// Max relative violation of `l <= x <= u` and `A x = b` in ORIGINAL units.
    fn original_violation(lp: &Lp, x: &[f64]) -> f64 {
        let mut worst = 0.0f64;
        for j in 0..lp.n {
            if lp.l[j] > -INF {
                worst = worst.max((lp.l[j] - x[j]) / (1.0 + lp.l[j].abs()));
            }
            if lp.u[j] < INF {
                worst = worst.max((x[j] - lp.u[j]) / (1.0 + lp.u[j].abs()));
            }
        }
        let mut ax = vec![0.0; lp.m];
        for j in 0..lp.n {
            let (rows, vals) = lp.sp.col(j);
            for (&i, &a) in rows.iter().zip(vals) {
                ax[i] += a * x[j];
            }
        }
        (0..lp.m).fold(worst, |w, i| {
            w.max((ax[i] - lp.b[i]).abs() / (1.0 + lp.b[i].abs()))
        })
    }

    /// #1945 regression. The MILP driver's old recipe -- equilibrate once, solve the
    /// SCALED matrix with the core solvers, unscale `x` -- returns `Optimal` on this
    /// infeasible LP with column 0 off its bound by 0.0179 (asserted first, so the
    /// fixture provably still exercises the mechanism). Through `ScaledProblem` the
    /// same solves never report an `Optimal` that fails the original LP.
    #[test]
    fn scaled_problem_refuses_the_false_optimal_the_old_driver_recipe_returned() {
        let (lp, basis) = fixture();
        let opts = SimplexOptions::default();
        let mut checked = 0usize;

        // The old recipe, cold and warm, on the driver's scaled data.
        let scaling = Scaling::from_sparse(&lp.sp, lp.m, lp.n).expect("fixture is ill-scaled");
        let mut a_s = lp.sp.clone();
        scaling.scale_cols(&mut a_s);
        let (c_s, b_s) = (scaling.scale_c(&lp.c), scaling.scale_b(&lp.b));
        let (l_s, u_s) = (scaling.scale_lower(&lp.l), scaling.scale_upper(&lp.u));
        let view = LpView {
            a: &[],
            m: lp.m,
            n: lp.n,
            c: &c_s,
            l: &l_s,
            u: &u_s,
        };
        let old = [
            solve_lp_cols(a_s.clone(), lp.m, lp.n, &c_s, &l_s, &u_s, &b_s, &opts),
            solve_lp_warm_scaled_csc(&view, &b_s, &basis, &opts, &a_s),
        ];
        for mut sol in old {
            assert_eq!(
                sol.status,
                LpStatus::Optimal,
                "the fixture no longer reproduces #1945"
            );
            scaling.unscale_x(&mut sol.x);
            let v = original_violation(&lp, &sol.x);
            assert!(
                v > 1e-3,
                "old recipe's Optimal was feasible after all ({v:e})"
            );
            checked += 2;
        }

        // The same solves through the choke point.
        let problem = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
        assert!(problem.scaling().is_some());
        let _guard = crate::profile::test_guard();
        for warm in [false, true] {
            crate::profile::reset_totals();
            crate::profile::set_enabled(true);
            let mut sol = if warm {
                problem.solve_warm(&lp.l, &lp.u, &basis, &opts)
            } else {
                problem.solve_cold(&lp.l, &lp.u, &opts)
            };
            let refused = crate::profile::counter_snapshot()
                .into_iter()
                .find(|(k, _)| *k == "ScaledOptimalOriginalAuditFail")
                .map_or(0, |(_, v)| v);
            crate::profile::set_enabled(false);
            assert!(
                refused >= 1,
                "warm={warm}: the original-space audit never fired"
            );
            checked += 1;
            if sol.status == LpStatus::Optimal {
                problem.scaling().unwrap().unscale_x(&mut sol.x);
                let v = original_violation(&lp, &sol.x);
                assert!(
                    v <= 1e-6,
                    "warm={warm}: Optimal still violates the original LP ({v:e})"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 8, "probe did not run every assertion");
    }

    /// Deterministic generator (no `rand` dependency): xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random standard-form LP `[A | I] x = b` whose columns mix entries of
    /// ~1e-14..1e-9 with O(1) ones (the #1942/#1945 class), feasible by construction
    /// (`b = A x0` for a point `x0` inside the box) unless `perturb_infeasible`.
    fn random_ill_scaled_lp(rng: &mut Rng, m: usize, k: usize, perturb_infeasible: bool) -> Lp {
        let n = k + m;
        let mut dense = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..k {
                let r = rng.next();
                if r < 0.15 {
                    dense[i * n + j] = 10f64.powf(-14.0 + 5.0 * rng.next()); // tiny
                } else if r < 0.6 {
                    dense[i * n + j] = 4.0 * rng.next() - 2.0;
                }
            }
            dense[i * n + k + i] = 1.0; // slack
        }
        let mut l = vec![0.0; n];
        let mut u = vec![0.0; n];
        let mut x0 = vec![0.0; n];
        for j in 0..k {
            l[j] = 10.0 * rng.next() - 5.0;
            u[j] = l[j] + 0.5 + 10.0 * rng.next();
            x0[j] = l[j] + (u[j] - l[j]) * rng.next();
        }
        for j in k..n {
            l[j] = 0.0;
            u[j] = INF;
            x0[j] = 3.0 * rng.next();
        }
        let mut b: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| dense[i * n + j] * x0[j]).sum())
            .collect();
        if perturb_infeasible {
            b[0] -= 1e3; // slack 0 would have to go negative
        }
        let c = (0..n)
            .map(|j| if j < k { 2.0 * rng.next() - 1.0 } else { 0.0 })
            .collect();
        Lp {
            m,
            n,
            sp: SparseCols::from_dense(&dense, m, n),
            c,
            l,
            u,
            b,
        }
    }

    fn dense(lp: &Lp) -> Vec<f64> {
        let mut a = vec![0.0; lp.m * lp.n];
        for j in 0..lp.n {
            let (rows, vals) = lp.sp.col(j);
            for (&i, &v) in rows.iter().zip(vals) {
                a[i * lp.n + j] = v;
            }
        }
        a
    }

    /// #1945 invariant: whichever public entry point solves an LP, an `Optimal` holds
    /// on the ORIGINAL data (relative 1e-6 plus roundoff slack). Covers the
    /// auto-scaling entry points #1942 patched and both `ScaledProblem` solves, over
    /// the fixture and a family of random ill-scaled LPs.
    #[test]
    fn every_public_solve_returns_only_original_feasible_optima() {
        let opts = SimplexOptions::default();
        let mut rng = Rng(0x1945_1942_dead_beef);
        let mut lps: Vec<Lp> = vec![fixture().0];
        for t in 0..60 {
            let (m, k) = (3 + t % 7, 4 + (t * 5) % 9);
            lps.push(random_ill_scaled_lp(&mut rng, m, k, t % 10 == 9));
        }
        let (mut optima, mut solves) = (0usize, 0usize);
        for lp in &lps {
            let a = dense(lp);
            let view = LpView {
                a: &a,
                m: lp.m,
                n: lp.n,
                c: &lp.c,
                l: &lp.l,
                u: &lp.u,
            };
            // Slack basis start for the warm entry points (the last m columns of the
            // generated LPs; for the fixture any shape-valid start exercises the path).
            let start = {
                let mut col_status = vec![crate::lp::basis::AT_LOWER; lp.n];
                let basic_vars: Vec<usize> = (lp.n - lp.m..lp.n).collect();
                for &j in &basic_vars {
                    col_status[j] = crate::lp::basis::BASIC;
                }
                Basis {
                    col_status,
                    basic_vars,
                }
            };
            let problem = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
            let mut results: Vec<(&str, LpSolve)> = vec![
                ("solve_lp", solve_lp(&view, &lp.b, &opts)),
                (
                    "solve_lp_cols_scaled",
                    solve_lp_cols_scaled(
                        lp.sp.clone(),
                        lp.m,
                        lp.n,
                        &lp.c,
                        &lp.l,
                        &lp.u,
                        &lp.b,
                        &opts,
                    ),
                ),
                ("solve_lp_warm", solve_lp_warm(&view, &lp.b, &start, &opts)),
                (
                    "solve_lp_warm_csc",
                    solve_lp_warm_csc(
                        lp.sp.clone(),
                        lp.m,
                        lp.n,
                        &lp.c,
                        &lp.l,
                        &lp.u,
                        &lp.b,
                        None,
                        &opts,
                    ),
                ),
            ];
            let inst = [LpInstance {
                b: lp.b.clone(),
                l: lp.l.clone(),
                u: lp.u.clone(),
            }];
            results.push((
                "solve_lp_batch",
                solve_lp_batch(&a, lp.m, lp.n, &lp.c, &inst, &opts).remove(0),
            ));
            // `ScaledProblem` returns solve coordinates: unscale before checking.
            for (name, mut sol) in [
                (
                    "ScaledProblem::solve_cold",
                    problem.solve_cold(&lp.l, &lp.u, &opts),
                ),
                (
                    "ScaledProblem::solve_warm",
                    problem.solve_warm(&lp.l, &lp.u, &start, &opts),
                ),
            ] {
                if let Some(s) = problem.scaling() {
                    s.unscale_x(&mut sol.x);
                }
                results.push((name, sol));
            }
            for (name, sol) in results {
                solves += 1;
                if sol.status == LpStatus::Optimal {
                    optima += 1;
                    let v = original_violation(lp, &sol.x);
                    assert!(
                        v <= 1e-5,
                        "{name}: Optimal violates the original LP by {v:e}"
                    );
                }
            }
        }
        assert_eq!(solves, lps.len() * 7);
        assert!(
            optima >= 200,
            "too few Optimal verdicts checked ({optima}) -- generator degenerate?"
        );
    }

    /// #1945 structure: outside `lp/simplex` no code may build a `Scaling` or scale a
    /// matrix -- that is how a caller ends up solving scaled data with no
    /// original-space audit. Scale once through `ScaledProblem`, or call an
    /// auto-scaling entry point. Test modules (after `#[cfg(test)]`) are skipped.
    #[test]
    fn nothing_outside_lp_simplex_scales_a_matrix_itself() {
        // A measurement binary, not a solve path: it times the scaled warm solve.
        const ALLOW: &[&str] = &["bin/e0_warm_bench.rs"];
        const FORBIDDEN: &[&str] = &[
            "Scaling::from_sparse(",
            "Scaling::from_matrix(",
            ".scale_cols(",
            "ScaledLp::maybe_new(",
        ];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src.clone()];
        let (mut files, mut offenders) = (0usize, Vec::new());
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().is_none_or(|e| e != "rs") {
                    continue;
                }
                let rel = p
                    .strip_prefix(&src)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.starts_with("lp/simplex/") || ALLOW.contains(&rel.as_str()) {
                    continue;
                }
                files += 1;
                let text = std::fs::read_to_string(&p).unwrap();
                let prod = text.split("#[cfg(test)]").next().unwrap_or("");
                for pat in FORBIDDEN {
                    if prod.contains(pat) {
                        offenders.push(format!("{rel}: {pat}"));
                    }
                }
            }
        }
        assert!(
            files > 20,
            "scanned only {files} files -- wrong source root?"
        );
        assert!(
            offenders.is_empty(),
            "scales outside lp/simplex (use ScaledProblem): {offenders:?}"
        );
    }
}
