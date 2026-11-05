//! Native separation of Anderson et al.'s ideal ReLU inequalities (#2157).
//!
//! For one unstable neuron `y = max(0, w·x + b)` encoded with an indicator `q`
//! (big-M or partition formulation, #2134) over the box `l <= x <= u`, Theorem 1
//! of Anderson, Huchette, Ma, Tjandraatmadja and Vielma (Math. Prog. 2020,
//! arXiv:1811.08359) gives the convex hull as `y >= 0`, `y >= w·x + b`,
//! `0 <= q <= 1` and, for every subset `I` of the inputs,
//!
//! ```text
//! y <= sum_{i in I} w_i (x_i - Lc_i (1 - q)) + (b + sum_{i not in I} w_i Uc_i) q
//! ```
//!
//! with `Lc_i = l_i`, `Uc_i = u_i` when `w_i >= 0` and swapped otherwise. The
//! most violated member at a point is found term by term (`i in I` iff
//! `w_i (x_i - Lc_i (1 - q)) < w_i Uc_i q`), so separation is `O(n_in)` per
//! neuron. This is the Rust twin of
//! `discopt.ml.formulations.relu_partition.AndersonCutSeparator`; running it as a
//! [`MilpNodeHook`] keeps a big-M network on the monolithic Rust MILP engine
//! instead of moving it to the Python spatial tree, which is what a Python
//! `cut_callback` does.
//!
//! **Validity (why these are global cuts).** Every subset `I` gives a valid
//! inequality for every *integer* point of the encoding whose `x` lies in the box:
//! at `q = 0` the encoding forces `y = 0` and the right-hand side is
//! `sum_I w_i (x_i - Lc_i) >= 0`; at `q = 1` it forces `y = w·x + b` and the
//! right-hand side is `w·x + b + sum_{not I} w_i (Uc_i - x_i) >= y`. The box is
//! the one the caller passes, which must be the input columns' own bounds, so a
//! cut never depends on a node's local box -- the [`MilpNodeHook`] contract.
//!
//! **Tiny weights.** An input may always be moved out of `I` (any subset is
//! valid), which removes its column from the row. Inputs whose whole swing
//! `|w_i| (u_i - l_i)` is below [`TINY_SWING`] are kept out of `I`, so a row never
//! carries a coefficient many orders of magnitude below its others. This only
//! weakens a cut by at most that swing; it never makes one invalid.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bnb::milp_driver::{MilpNodeHook, MilpNodeVerdict};
use crate::lp::gomory::GomoryCut;

/// Inputs whose total contribution range `|w_i| (u_i - l_i)` is below this are
/// never placed in `I` (see the module docs: sound, at most this much weaker).
pub const TINY_SWING: f64 = 1e-9;

/// One unstable neuron: column indices are structural columns of the MILP.
#[derive(Debug, Clone)]
struct Neuron {
    y: usize,
    q: usize,
    bias: f64,
    x: Vec<usize>,
    w: Vec<f64>,
    lc: Vec<f64>,
    uc: Vec<f64>,
}

/// Exact separator for the ideal ReLU inequalities of a set of neurons.
#[derive(Debug)]
pub struct ReluIdealSeparator {
    neurons: Vec<Neuron>,
    n_struct: usize,
    tol: f64,
    calls: AtomicUsize,
    cuts: AtomicUsize,
}

impl ReluIdealSeparator {
    /// Build from a CSR-style neuron list.
    ///
    /// Neuron `k` has output column `y_cols[k]`, indicator column `q_cols[k]`,
    /// bias `bias[k]` and inputs `ptr[k]..ptr[k+1]` of `x_cols` / `w` / `x_lb` /
    /// `x_ub` (the box of each input column). Refuses -- rather than repairs --
    /// any malformed input: an out-of-range column, a non-finite weight or bias,
    /// or a non-finite / inverted box on a nonzero weight (no hull row exists
    /// over an unbounded input, and a silently dropped neuron would read as "no
    /// violated cut").
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_struct: usize,
        y_cols: &[usize],
        q_cols: &[usize],
        bias: &[f64],
        ptr: &[usize],
        x_cols: &[usize],
        w: &[f64],
        x_lb: &[f64],
        x_ub: &[f64],
        tol: f64,
    ) -> Result<Self, String> {
        let k = y_cols.len();
        if q_cols.len() != k || bias.len() != k {
            return Err(format!(
                "relu separator: y_cols/q_cols/bias lengths differ ({k}/{}/{})",
                q_cols.len(),
                bias.len()
            ));
        }
        if ptr.len() != k + 1 || ptr.first() != Some(&0) {
            return Err("relu separator: ptr must have length n_neurons + 1 and start at 0".into());
        }
        let nnz = *ptr.last().unwrap();
        if x_cols.len() != nnz || w.len() != nnz || x_lb.len() != nnz || x_ub.len() != nnz {
            return Err(format!(
                "relu separator: x_cols/w/x_lb/x_ub must all have length ptr[-1] = {nnz}"
            ));
        }
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(format!(
                "relu separator: tol must be finite and >= 0, got {tol}"
            ));
        }
        let mut neurons = Vec::with_capacity(k);
        for j in 0..k {
            let (s, e) = (ptr[j], ptr[j + 1]);
            if e < s {
                return Err("relu separator: ptr must be non-decreasing".into());
            }
            let (y, q) = (y_cols[j], q_cols[j]);
            if y >= n_struct || q >= n_struct || y == q {
                return Err(format!(
                    "relu separator: neuron {j} has y={y}, q={q} over {n_struct} structural columns"
                ));
            }
            if !bias[j].is_finite() {
                return Err(format!("relu separator: neuron {j} has a non-finite bias"));
            }
            let mut nx = Vec::with_capacity(e - s);
            let mut nw = Vec::with_capacity(e - s);
            let mut lc = Vec::with_capacity(e - s);
            let mut uc = Vec::with_capacity(e - s);
            for t in s..e {
                let (c, wi, l, u) = (x_cols[t], w[t], x_lb[t], x_ub[t]);
                if c >= n_struct || c == y || c == q {
                    return Err(format!(
                        "relu separator: neuron {j} input column {c} is out of range or \
                         aliases its output/indicator"
                    ));
                }
                if !wi.is_finite() {
                    return Err(format!(
                        "relu separator: neuron {j} has a non-finite weight"
                    ));
                }
                if wi == 0.0 {
                    continue;
                }
                if !(l.is_finite() && u.is_finite() && l <= u) {
                    return Err(format!(
                        "relu separator: neuron {j} input column {c} has box [{l}, {u}]; the \
                         ideal inequalities need a finite box on every weighted input"
                    ));
                }
                nx.push(c);
                nw.push(wi);
                if wi > 0.0 {
                    lc.push(l);
                    uc.push(u);
                } else {
                    lc.push(u);
                    uc.push(l);
                }
            }
            neurons.push(Neuron {
                y,
                q,
                bias: bias[j],
                x: nx,
                w: nw,
                lc,
                uc,
            });
        }
        Ok(Self {
            neurons,
            n_struct,
            tol,
            calls: AtomicUsize::new(0),
            cuts: AtomicUsize::new(0),
        })
    }

    /// Structural width the separator addresses.
    pub fn n_struct(&self) -> usize {
        self.n_struct
    }

    /// Number of neurons the separator holds.
    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Executed separation calls (CLAUDE.md §6: 0 means "never ran", not "no cut").
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// Cuts returned over all calls (before the driver's de-duplication).
    pub fn cuts_returned(&self) -> usize {
        self.cuts.load(Ordering::Relaxed)
    }

    /// The most violated ideal inequality of every neuron that has one at `x`,
    /// in the driver's `coeffs · x >= rhs` form (dense over structural columns).
    pub fn separate(&self, x: &[f64]) -> Vec<GomoryCut> {
        let mut out = Vec::new();
        if x.len() < self.n_struct {
            return out;
        }
        for nr in &self.neurons {
            let qs = x[nr.q];
            let ys = x[nr.y];
            // rhs(x*, q*) of the most violated member, and its I.
            let mut rhs_val = nr.bias * qs;
            let mut in_i = Vec::with_capacity(nr.x.len());
            for t in 0..nr.x.len() {
                let wi = nr.w[t];
                let in_term = wi * (x[nr.x[t]] - nr.lc[t] * (1.0 - qs));
                let out_term = wi * nr.uc[t] * qs;
                let tiny = (wi * (nr.uc[t] - nr.lc[t])).abs() < TINY_SWING;
                let take = !tiny && in_term < out_term;
                in_i.push(take);
                rhs_val += if take { in_term } else { out_term };
            }
            if ys - rhs_val <= self.tol * rhs_val.abs().max(1.0) {
                continue;
            }
            // y <= sum_I w (x - Lc) + q (b + sum_I w Lc + sum_~I w Uc), i.e.
            // sum_I w x - y + q (b + sum_I w Lc + sum_~I w Uc) >= sum_I w Lc.
            let mut coeffs = vec![0.0; self.n_struct];
            let mut q_coef = nr.bias;
            let mut rhs = 0.0;
            for t in 0..nr.x.len() {
                let wi = nr.w[t];
                if in_i[t] {
                    coeffs[nr.x[t]] += wi;
                    q_coef += wi * nr.lc[t];
                    rhs += wi * nr.lc[t];
                } else {
                    q_coef += wi * nr.uc[t];
                }
            }
            coeffs[nr.y] -= 1.0;
            coeffs[nr.q] += q_coef;
            if !rhs.is_finite() || coeffs.iter().any(|v| !v.is_finite()) {
                continue; // cannot happen with finite, validated data; never emit one
            }
            out.push(GomoryCut { coeffs, rhs });
        }
        out
    }
}

impl MilpNodeHook for ReluIdealSeparator {
    fn separate_node(&self, x: &[f64]) -> MilpNodeVerdict {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let cuts = self.separate(x);
        self.cuts.fetch_add(cuts.len(), Ordering::Relaxed);
        if cuts.is_empty() {
            MilpNodeVerdict::None
        } else {
            MilpNodeVerdict::Cuts(cuts)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One neuron `y = max(0, x0 - x1 + 0.5)` over `[-1,1]^2`; columns
    /// `x0=0, x1=1, y=2, q=3`.
    fn sep() -> ReluIdealSeparator {
        ReluIdealSeparator::new(
            4,
            &[2],
            &[3],
            &[0.5],
            &[0, 2],
            &[0, 1],
            &[1.0, -1.0],
            &[-1.0, -1.0],
            &[1.0, 1.0],
            1e-9,
        )
        .unwrap()
    }

    fn relu_point(x0: f64, x1: f64) -> [f64; 4] {
        let pre = x0 - x1 + 0.5;
        [x0, x1, pre.max(0.0), if pre > 0.0 { 1.0 } else { 0.0 }]
    }

    #[test]
    fn no_cut_at_integer_feasible_points() {
        let s = sep();
        let mut checked = 0;
        for i in 0..=10 {
            for j in 0..=10 {
                let p = relu_point(-1.0 + 0.2 * i as f64, -1.0 + 0.2 * j as f64);
                assert!(s.separate(&p).is_empty(), "cut at feasible point {p:?}");
                checked += 1;
            }
        }
        assert_eq!(checked, 121);
    }

    #[test]
    fn every_returned_cut_is_valid_and_separates() {
        let s = sep();
        // A point the big-M rows admit (U = 2.5, L = -1.5: y <= min(U q,
        // pre - L (1 - q)) = 1.25) but the hull does not: at x0 = u0 = 1 the q = 0
        // branch is empty, and the I = {1} member caps y at 0.25.
        let p = [1.0, 1.0, 1.0, 0.5];
        let cuts = s.separate(&p);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        let act = |v: &[f64]| -> f64 { cut.coeffs.iter().zip(v).map(|(a, b)| a * b).sum() };
        assert!(act(&p) < cut.rhs - 1e-6, "cut does not separate");
        let mut checked = 0;
        for i in 0..=20 {
            for j in 0..=20 {
                let v = relu_point(-1.0 + 0.1 * i as f64, -1.0 + 0.1 * j as f64);
                assert!(act(&v) >= cut.rhs - 1e-12, "cut removes feasible {v:?}");
                checked += 1;
            }
        }
        assert_eq!(checked, 441);
    }

    #[test]
    fn refuses_unbounded_input() {
        let r = ReluIdealSeparator::new(
            4,
            &[2],
            &[3],
            &[0.0],
            &[0, 1],
            &[0],
            &[1.0],
            &[f64::NEG_INFINITY],
            &[1.0],
            1e-9,
        );
        assert!(r.is_err());
    }

    #[test]
    fn tiny_weights_stay_out_of_the_row() {
        let s = ReluIdealSeparator::new(
            4,
            &[2],
            &[3],
            &[0.5],
            &[0, 2],
            &[0, 1],
            &[1.0, -1e-12],
            &[-1.0, -1.0],
            &[1.0, 1.0],
            1e-9,
        )
        .unwrap();
        let cuts = s.separate(&[1.0, 1.0, 1.0, 0.5]);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].coeffs[1], 0.0);
    }

    #[test]
    fn hook_counts_calls() {
        let s = sep();
        let _ = s.separate_node(&[1.0, 1.0, 1.0, 0.5]);
        let _ = s.separate_node(&relu_point(0.3, 0.1));
        assert_eq!(s.calls(), 2);
        assert_eq!(s.cuts_returned(), 1);
    }
}
