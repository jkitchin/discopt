//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `lp/highs_cutgen/mod.rs` for why
//! HiGHS is ported rather than linked.
//!
//! `HighsMipSolverData::moreHeuristicsAllowed` (`mip/HighsMipSolverData.cpp`): the
//! budget that keeps the primal heuristics' LP iterations at `mip_heuristic_effort` of
//! the search's, and the sub-MIP charge of `HighsPrimalHeuristics::solveSubMip`
//! (its iterations scaled by the share of columns the fixings left free).
//!
//! discopt runs no tree restarts, so HiGHS's `*_before_run` counters are all zero here.

/// HiGHS's `mip_heuristic_effort` default.
pub const HEURISTIC_EFFORT: f64 = 0.05;

/// The search state `moreHeuristicsAllowed` reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EffortState {
    /// `mipsolver.submip`: this solve is itself a heuristic sub-MIP.
    pub submip: bool,
    /// `heuristic_lp_iterations`.
    pub heuristic_lp_iterations: i64,
    /// `total_lp_iterations` (node, strong-branch, separation and heuristic LPs).
    pub total_lp_iterations: i64,
    /// `sb_lp_iterations`.
    pub sb_lp_iterations: i64,
    /// `pruned_treeweight`: the fraction of the tree closed.
    pub pruned_treeweight: f64,
    /// `num_leaves`.
    pub num_leaves: i64,
    /// `num_nodes`.
    pub num_nodes: i64,
}

/// `HighsMipSolverData::moreHeuristicsAllowed` with `heuristic_effort` = `effort`.
pub fn more_heuristics_allowed(s: &EffortState, effort: f64) -> bool {
    let heur = s.heuristic_lp_iterations as f64;
    let total = s.total_lp_iterations as f64;
    if s.submip {
        // In a sub-MIP only what is proportional to the effort spent so far.
        return heur < total * effort;
    }
    if s.pruned_treeweight < 1e-3 && s.num_leaves < 10 && s.num_nodes < 1000 {
        // The beginning of the main search: an initial offset of 10,000 iterations.
        if heur < total * effort + 10000.0 {
            return true;
        }
    } else if s.heuristic_lp_iterations
        < 100000 + ((s.total_lp_iterations - s.heuristic_lp_iterations - s.sb_lp_iterations) >> 1)
    {
        // Node LP iterations of this run (no restarts: the whole search), projected to
        // the whole tree by the pruned weight; everything else is an offset.
        let node_iters =
            (s.total_lp_iterations - s.heuristic_lp_iterations - s.sb_lp_iterations) as f64;
        let estim = heur / ((total - node_iters) + node_iters / s.pruned_treeweight.max(0.01));
        // Spend the heuristics' share in the first 80 % of the tree, and the share of
        // the first 30 % as early as possible.
        if estim < (0.3f64 / 0.8).max(s.pruned_treeweight.min(0.8) / 0.8) * effort {
            return true;
        }
    }
    false
}

/// `solveSubMip`'s charge: the sub-MIP's LP iterations scaled by the share of the
/// columns its fixings left free (`adjustmentfactor`), truncated as HiGHS's `size_t`
/// cast truncates.
pub fn sub_mip_charge(sub_iters: i64, fixing_rate: f64, n_integral: usize, n_cont: usize) -> i64 {
    let unfixed = (n_integral + n_cont) as f64;
    let factor = ((1.0 - fixing_rate) * n_integral as f64 + n_cont as f64) / unfixed.max(1.0);
    (factor * sub_iters.max(0) as f64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(heur: i64, total: i64, sb: i64, w: f64, leaves: i64, nodes: i64) -> EffortState {
        EffortState {
            submip: false,
            heuristic_lp_iterations: heur,
            total_lp_iterations: total,
            sb_lp_iterations: sb,
            pruned_treeweight: w,
            num_leaves: leaves,
            num_nodes: nodes,
        }
    }

    /// Each branch of HiGHS's function, at and around its thresholds (values worked by
    /// hand from `HighsMipSolverData.cpp:670-730`). Counts what it checked (§6).
    #[test]
    fn more_heuristics_allowed_follows_highs_branch_by_branch() {
        let mut checked = 0;
        let mut check = |s: EffortState, want: bool| {
            assert_eq!(more_heuristics_allowed(&s, HEURISTIC_EFFORT), want, "{s:?}");
            checked += 1;
        };
        // Sub-MIP: heur < 5 % of total.
        let mut s = st(49, 1000, 0, 0.0, 0, 0);
        s.submip = true;
        check(s, true);
        s.heuristic_lp_iterations = 50;
        check(s, false);
        // Early main search: 10,000-iteration offset.
        check(st(10_049, 1000, 0, 0.0, 0, 0), true);
        check(st(10_050, 1000, 0, 0.0, 0, 0), false);
        // Leaving the early phase by leaves, nodes or weight.
        // heur 2000, total 100000, sb 0: node iters 98000, w = 0.5:
        // estim = 2000 / (2000 + 98000 / 0.5) = 2000 / 198000 = 0.0101
        // bound = max(0.375, 0.625) * 0.05 = 0.03125 -> allowed.
        check(st(2000, 100_000, 0, 0.5, 10, 0), true);
        // heur 20000, total 100000, w = 0.1: node 80000, estim = 20000 / (20000 + 800000)
        // = 0.0244, bound = max(0.375, 0.125) * 0.05 = 0.01875 -> refused.
        check(st(20_000, 100_000, 0, 0.1, 0, 1000), false);
        // The hard cap: heur >= 100000 + (total - heur - sb) / 2 refuses outright.
        check(st(150_000, 200_000, 0, 0.9, 0, 1000), false);
        // Weight floored at 0.01: heur 100, total 1e6, w 0 -> node 999900,
        // estim = 100 / (100 + 99_990_000) ~ 1e-6 -> allowed.
        check(st(100, 1_000_000, 0, 0.0, 50, 5000), true);
        assert_eq!(checked, 8);
    }

    #[test]
    fn sub_mip_charge_scales_by_the_unfixed_share() {
        assert_eq!(sub_mip_charge(1000, 0.0, 10, 0), 1000);
        assert_eq!(sub_mip_charge(1000, 0.75, 10, 0), 250);
        // 10 integral at 50 % fixed + 10 continuous: (5 + 10) / 20 = 0.75.
        assert_eq!(sub_mip_charge(1000, 0.5, 10, 10), 750);
        assert_eq!(sub_mip_charge(1000, 0.5, 0, 0), 0);
    }
}
