//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `lp/highs_cutgen/mod.rs` for why
//! HiGHS is ported rather than linked: deep integration with, and behaviour compatible
//! with, discopt's own search, without a C/C++ compiler or a native library to ship, for
//! easy distribution.
//!
//! `HighsPseudocost` (`mip/HighsPseudocost.{h,cpp}`, stage 5 of
//! `docs/dev/highs-mip-replacement-plan.md`): per-column up/down pseudocosts with sample
//! counts and the running average over all columns (`cost_total`), inference and cutoff
//! observations, conflict scores with HiGHS's growing conflict weight and its rescaling,
//! the reliability tests, the branching scores (`getScore`, `getScoreUp`,
//! `getScoreDown`), the delta buffer a worker flushes into the master pseudocost
//! (`markChanged`, `flushPseudoCost`, `syncPseudoCost`, `removeChanged`), and
//! `HighsPseudocostInitialization`, which seeds a later solve (a restart, or a sub-MIP)
//! from an earlier one, optionally through the presolve column map.
//!
//! Every formula keeps the C++ evaluation order, including integer-to-double
//! conversion points, and `std::max` is translated as [`cmax`] (C++'s `(a < b) ? b : a`,
//! which differs from [`f64::max`] on a NaN operand), so later replay against HiGHS can
//! be bit-exact. Exact replay additionally needs HiGHS compiled without FMA contraction
//! of `a * b + c` (`-ffp-contract=off`); Rust never contracts.
//!
//! Not translated, and why:
//! - The `HighsMipSolver` constructor's reads of `mipsolver.numCol()`,
//!   `options_mip_->mip_pscost_minreliable`, `pscostinit` and
//!   `mipdata_->postSolveStack.getOrigColIndex` are replaced by explicit arguments
//!   ([`Pseudocost::new`], [`Pseudocost::with_initialization`]); the arithmetic is the
//!   same. Likewise the `HighsPostsolveStack` overload of the initialization takes the
//!   original column count and the reduced-to-original column map.
//! - `HighsPseudocost() = default` (members left uninitialized) has no Rust counterpart.
//! - `deltas.reserve(min(256, numCol))` is a capacity hint only.
//! - The `HS_PSCOST_ONLY` environment switch in the vendored `getScore` is discopt's own
//!   debugging patch of the HiGHS tree, not upstream behaviour, and is left out.

// The code mirrors the C++ statement for statement (index loops over parallel arrays),
// so a reader can check it against HiGHS line by line.
#![allow(clippy::needless_range_loop)]

/// `minThreshold`: the floor on every factor of a product score.
pub const MIN_THRESHOLD: f64 = 1e-6;

/// The default of HiGHS's `mip_pscost_minreliable` option (`HighsOptions.h`).
pub const DEFAULT_MIN_RELIABLE: i32 = 8;

/// C++'s `std::max(a, b)` for doubles: `(a < b) ? b : a`.
#[inline]
pub fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// The `mapScore` lambda of `getScore`, `getScoreUp` and `getScoreDown`: maps
/// `[0, inf)` onto `[0, 1)`.
#[inline]
fn map_score(score: f64) -> f64 {
    1.0 - 1.0 / (1.0 + score)
}

/// `HighsPseudocostInitialization`: a snapshot of a pseudocost, with sample counts
/// capped, used to seed the pseudocost of a later solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PseudocostInitialization {
    /// `pseudocostup`.
    pub pseudocostup: Vec<f64>,
    /// `pseudocostdown`.
    pub pseudocostdown: Vec<f64>,
    /// `nsamplesup`.
    pub nsamplesup: Vec<i32>,
    /// `nsamplesdown`.
    pub nsamplesdown: Vec<i32>,
    /// `inferencesup`.
    pub inferencesup: Vec<f64>,
    /// `inferencesdown`.
    pub inferencesdown: Vec<f64>,
    /// `ninferencesup`.
    pub ninferencesup: Vec<i32>,
    /// `ninferencesdown`.
    pub ninferencesdown: Vec<i32>,
    /// `conflictscoreup`, divided by the source's conflict weight.
    pub conflictscoreup: Vec<f64>,
    /// `conflictscoredown`, divided by the source's conflict weight.
    pub conflictscoredown: Vec<f64>,
    /// `cost_total`.
    pub cost_total: f64,
    /// `inferences_total`.
    pub inferences_total: f64,
    /// `conflict_avg_score`, divided by `ncol * conflict_weight` (a per-column average).
    pub conflict_avg_score: f64,
    /// `nsamplestotal` (HiGHS stores `min(1, source count)`).
    pub nsamplestotal: i64,
    /// `ninferencestotal` (HiGHS stores `min(1, source count)`).
    pub ninferencestotal: i64,
}

impl PseudocostInitialization {
    /// `HighsPseudocostInitialization(const HighsPseudocost&, HighsInt maxCount)`
    /// (`HighsPseudocost.cpp`): same column space as `pscost`.
    pub fn new(pscost: &Pseudocost, max_count: i32) -> Self {
        let ncol = pscost.pseudocostup.len();
        let mut init = PseudocostInitialization {
            pseudocostup: pscost.pseudocostup.clone(),
            pseudocostdown: pscost.pseudocostdown.clone(),
            nsamplesup: pscost.nsamplesup.clone(),
            nsamplesdown: pscost.nsamplesdown.clone(),
            inferencesup: pscost.inferencesup.clone(),
            inferencesdown: pscost.inferencesdown.clone(),
            ninferencesup: pscost.ninferencesup.clone(),
            ninferencesdown: pscost.ninferencesdown.clone(),
            conflictscoreup: vec![0.0; pscost.conflictscoreup.len()],
            conflictscoredown: vec![0.0; pscost.conflictscoreup.len()],
            cost_total: pscost.cost_total,
            inferences_total: pscost.inferences_total,
            conflict_avg_score: pscost.conflict_avg_score,
            nsamplestotal: 1i64.min(pscost.nsamplestotal),
            ninferencestotal: 1i64.min(pscost.ninferencestotal),
        };
        // `conflict_avg_score /= ncol * pscost.conflict_weight;` (HighsInt * double).
        init.conflict_avg_score /= ncol as f64 * pscost.conflict_weight;
        for i in 0..ncol {
            init.nsamplesup[i] = init.nsamplesup[i].min(max_count);
            init.nsamplesdown[i] = init.nsamplesdown[i].min(max_count);
            init.ninferencesup[i] = init.ninferencesup[i].min(1);
            init.ninferencesdown[i] = init.ninferencesdown[i].min(1);
            init.conflictscoreup[i] = pscost.conflictscoreup[i] / pscost.conflict_weight;
            init.conflictscoredown[i] = pscost.conflictscoredown[i] / pscost.conflict_weight;
        }
        init
    }

    /// `HighsPseudocostInitialization(const HighsPseudocost&, HighsInt maxCount,
    /// const HighsPostsolveStack&)`: lifts a pseudocost on the presolved columns to the
    /// original column space. `orig_col_index[i]` is
    /// `postsolveStack.getOrigColIndex(i)` and `orig_num_col` is
    /// `postsolveStack.getOrigNumCol()`. Original columns with no presolved image keep
    /// zero statistics.
    ///
    /// Unlike [`PseudocostInitialization::new`], HiGHS sets `ninferencesup/down` to 1
    /// for every mapped column here, including columns that never had an inference
    /// observation; this is translated as written.
    pub fn with_col_map(
        pscost: &Pseudocost,
        max_count: i32,
        orig_num_col: usize,
        orig_col_index: &[usize],
    ) -> Self {
        let ncols = pscost.pseudocostup.len();
        assert_eq!(
            orig_col_index.len(),
            ncols,
            "column map must cover every presolved column"
        );
        let mut init = PseudocostInitialization {
            pseudocostup: vec![0.0; orig_num_col],
            pseudocostdown: vec![0.0; orig_num_col],
            nsamplesup: vec![0; orig_num_col],
            nsamplesdown: vec![0; orig_num_col],
            inferencesup: vec![0.0; orig_num_col],
            inferencesdown: vec![0.0; orig_num_col],
            ninferencesup: vec![0; orig_num_col],
            ninferencesdown: vec![0; orig_num_col],
            conflictscoreup: vec![0.0; orig_num_col],
            conflictscoredown: vec![0.0; orig_num_col],
            cost_total: pscost.cost_total,
            inferences_total: pscost.inferences_total,
            conflict_avg_score: pscost.conflict_avg_score,
            nsamplestotal: 1i64.min(pscost.nsamplestotal),
            ninferencestotal: 1i64.min(pscost.ninferencestotal),
        };
        init.conflict_avg_score /= ncols as f64 * pscost.conflict_weight;
        for i in 0..ncols {
            let o = orig_col_index[i];
            init.pseudocostup[o] = pscost.pseudocostup[i];
            init.pseudocostdown[o] = pscost.pseudocostdown[i];
            init.nsamplesup[o] = max_count.min(pscost.nsamplesup[i]);
            init.nsamplesdown[o] = max_count.min(pscost.nsamplesdown[i]);
            init.inferencesup[o] = pscost.inferencesup[i];
            init.inferencesdown[o] = pscost.inferencesdown[i];
            init.ninferencesup[o] = 1;
            init.ninferencesdown[o] = 1;
            init.conflictscoreup[o] = pscost.conflictscoreup[i] / pscost.conflict_weight;
            init.conflictscoredown[o] = pscost.conflictscoredown[i] / pscost.conflict_weight;
        }
        init
    }
}

/// `HighsPseudocostDelta`: the changes to one column since the last flush, as sums and
/// counts so they can be merged into another pseudocost's averages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PseudocostDelta {
    /// `col`.
    pub col: usize,
    /// `nsamplesup`.
    pub nsamplesup: i32,
    /// `nsamplesdown`.
    pub nsamplesdown: i32,
    /// `ninferencesup`.
    pub ninferencesup: i32,
    /// `ninferencesdown`.
    pub ninferencesdown: i32,
    /// `ncutoffsup`.
    pub ncutoffsup: i32,
    /// `ncutoffsdown`.
    pub ncutoffsdown: i32,
    /// `pseudocostup_sum`.
    pub pseudocostup_sum: f64,
    /// `pseudocostdown_sum`.
    pub pseudocostdown_sum: f64,
    /// `inferencesup_sum`.
    pub inferencesup_sum: f64,
    /// `inferencesdown_sum`.
    pub inferencesdown_sum: f64,
    /// `conflictscoreup_sum`.
    pub conflictscoreup_sum: f64,
    /// `conflictscoredown_sum`.
    pub conflictscoredown_sum: f64,
}

/// `HighsPseudocost`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pseudocost {
    pseudocostup: Vec<f64>,
    pseudocostdown: Vec<f64>,
    nsamplesup: Vec<i32>,
    nsamplesdown: Vec<i32>,
    inferencesup: Vec<f64>,
    inferencesdown: Vec<f64>,
    ninferencesup: Vec<i32>,
    ninferencesdown: Vec<i32>,
    ncutoffsup: Vec<i32>,
    ncutoffsdown: Vec<i32>,
    conflictscoreup: Vec<f64>,
    conflictscoredown: Vec<f64>,
    /// `changedpos` (`-1` in HiGHS is `None`).
    changedpos: Vec<Option<usize>>,
    deltas: Vec<PseudocostDelta>,

    conflict_weight: f64,
    conflict_avg_score: f64,
    cost_total: f64,
    inferences_total: f64,
    delta_cost_sum: f64,
    delta_inferences_sum: f64,
    nsamplestotal: i64,
    ninferencestotal: i64,
    ncutoffstotal: i64,
    delta_nsamplestotal: i64,
    delta_ninferencestotal: i64,
    minreliable: i32,
    degeneracy_factor: f64,
}

/// `addDeltaAverage` for `HighsInt` counts.
fn add_delta_average_i32(avg: &mut f64, count: &mut i32, delta_sum: f64, delta_count: i32) {
    if delta_count <= 0 {
        return;
    }
    *avg = (*avg * f64::from(*count) + delta_sum) / f64::from(*count + delta_count);
    *count += delta_count;
}

/// `addDeltaAverage` for `int64_t` counts.
fn add_delta_average_i64(avg: &mut f64, count: &mut i64, delta_sum: f64, delta_count: i64) {
    if delta_count <= 0 {
        return;
    }
    *avg = (*avg * *count as f64 + delta_sum) / (*count + delta_count) as f64;
    *count += delta_count;
}

impl Pseudocost {
    /// `HighsPseudocost(const HighsMipSolver&)` with `mipsolver.pscostinit == nullptr`:
    /// `num_col` is `mipsolver.numCol()`, `minreliable` is the
    /// `mip_pscost_minreliable` option ([`DEFAULT_MIN_RELIABLE`] by default).
    pub fn new(num_col: usize, minreliable: i32) -> Self {
        Pseudocost {
            pseudocostup: vec![0.0; num_col],
            pseudocostdown: vec![0.0; num_col],
            nsamplesup: vec![0; num_col],
            nsamplesdown: vec![0; num_col],
            inferencesup: vec![0.0; num_col],
            inferencesdown: vec![0.0; num_col],
            ninferencesup: vec![0; num_col],
            ninferencesdown: vec![0; num_col],
            ncutoffsup: vec![0; num_col],
            ncutoffsdown: vec![0; num_col],
            conflictscoreup: vec![0.0; num_col],
            conflictscoredown: vec![0.0; num_col],
            changedpos: vec![None; num_col],
            deltas: Vec::new(),
            conflict_weight: 1.0,
            conflict_avg_score: 0.0,
            cost_total: 0.0,
            inferences_total: 0.0,
            delta_cost_sum: 0.0,
            delta_inferences_sum: 0.0,
            nsamplestotal: 0,
            ninferencestotal: 0,
            ncutoffstotal: 0,
            delta_nsamplestotal: 0,
            delta_ninferencestotal: 0,
            minreliable,
            degeneracy_factor: 1.0,
        }
    }

    /// `HighsPseudocost(const HighsMipSolver&)` with `mipsolver.pscostinit` set:
    /// column `i` of this (presolved) problem takes the statistics of column
    /// `orig_col_index[i]` (`postSolveStack.getOrigColIndex(i)`) of `init`. Cutoff
    /// counts are not carried over (HiGHS's initialization has none).
    pub fn with_initialization(
        num_col: usize,
        minreliable: i32,
        init: &PseudocostInitialization,
        orig_col_index: &[usize],
    ) -> Self {
        assert_eq!(
            orig_col_index.len(),
            num_col,
            "column map must cover every column"
        );
        let mut p = Pseudocost::new(num_col, minreliable);
        p.cost_total = init.cost_total;
        p.inferences_total = init.inferences_total;
        p.nsamplestotal = init.nsamplestotal;
        p.ninferencestotal = init.ninferencestotal;
        // `conflict_avg_score = pscostinit->conflict_avg_score * mipsolver.numCol();`
        p.conflict_avg_score = init.conflict_avg_score * num_col as f64;
        for i in 0..num_col {
            let o = orig_col_index[i];
            p.pseudocostup[i] = init.pseudocostup[o];
            p.nsamplesup[i] = init.nsamplesup[o];
            p.pseudocostdown[i] = init.pseudocostdown[o];
            p.nsamplesdown[i] = init.nsamplesdown[o];
            p.inferencesup[i] = init.inferencesup[o];
            p.ninferencesup[i] = init.ninferencesup[o];
            p.inferencesdown[i] = init.inferencesdown[o];
            p.ninferencesdown[i] = init.ninferencesdown[o];
            p.conflictscoreup[i] = init.conflictscoreup[o];
            p.conflictscoredown[i] = init.conflictscoredown[o];
        }
        p
    }

    /// Number of columns.
    pub fn num_col(&self) -> usize {
        self.pseudocostup.len()
    }

    /// `markChanged`: the delta record of `col`, created on first change.
    fn mark_changed(&mut self, col: usize) -> &mut PseudocostDelta {
        debug_assert!(col < self.changedpos.len());
        let pos = match self.changedpos[col] {
            Some(p) => p,
            None => {
                let p = self.deltas.len();
                self.changedpos[col] = Some(p);
                self.deltas.push(PseudocostDelta {
                    col,
                    ..PseudocostDelta::default()
                });
                p
            }
        };
        &mut self.deltas[pos]
    }

    /// `increaseConflictWeight`: grows the weight of later conflict-score increments
    /// by 2% (so recent conflicts dominate), rescaling everything back to weight 1 once
    /// it exceeds 1000.
    pub fn increase_conflict_weight(&mut self) {
        self.conflict_weight *= 1.02;

        if self.conflict_weight > 1000.0 {
            let scale = 1.0 / self.conflict_weight;
            self.conflict_weight = 1.0;
            self.conflict_avg_score *= scale;

            for i in 0..self.conflictscoreup.len() {
                self.conflictscoreup[i] *= scale;
                self.conflictscoredown[i] *= scale;
            }
            for delta in &mut self.deltas {
                delta.conflictscoreup_sum *= scale;
                delta.conflictscoredown_sum *= scale;
            }
        }
    }

    /// `setDegeneracyFactor`.
    pub fn set_degeneracy_factor(&mut self, degeneracy_factor: f64) {
        debug_assert!(degeneracy_factor >= 1.0);
        self.degeneracy_factor = degeneracy_factor;
    }

    /// `increaseConflictScoreUp`.
    pub fn increase_conflict_score_up(&mut self, col: usize) {
        let w = self.conflict_weight;
        let delta = self.mark_changed(col);
        delta.conflictscoreup_sum += w;
        self.conflictscoreup[col] += w;
        self.conflict_avg_score += w;
    }

    /// `increaseConflictScoreDown`.
    pub fn increase_conflict_score_down(&mut self, col: usize) {
        let w = self.conflict_weight;
        let delta = self.mark_changed(col);
        delta.conflictscoredown_sum += w;
        self.conflictscoredown[col] += w;
        self.conflict_avg_score += w;
    }

    /// `setMinReliable`.
    pub fn set_min_reliable(&mut self, minreliable: i32) {
        self.minreliable = minreliable;
    }

    /// `getMinReliable`.
    pub fn get_min_reliable(&self) -> i32 {
        self.minreliable
    }

    /// `getNumObservations`.
    pub fn get_num_observations(&self, col: usize) -> i32 {
        self.nsamplesup[col] + self.nsamplesdown[col]
    }

    /// `getNumObservationsUp`.
    pub fn get_num_observations_up(&self, col: usize) -> i32 {
        self.nsamplesup[col]
    }

    /// `getNumObservationsDown`.
    pub fn get_num_observations_down(&self, col: usize) -> i32 {
        self.nsamplesdown[col]
    }

    /// `addCutoffObservation`: the `upbranch` child of a branching on `col` was cut off.
    pub fn add_cutoff_observation(&mut self, col: usize, upbranch: bool) {
        let delta = self.mark_changed(col);
        if upbranch {
            delta.ncutoffsup += 1;
        } else {
            delta.ncutoffsdown += 1;
        }
        self.ncutoffstotal += 1;
        if upbranch {
            self.ncutoffsup[col] += 1;
        } else {
            self.ncutoffsdown[col] += 1;
        }
    }

    /// `addObservation`: moving `col` by `delta` (signed, nonzero) raised the objective
    /// by `objdelta >= 0`. Updates the column's running mean of the unit gain in that
    /// direction and the overall mean `cost_total`.
    pub fn add_observation(&mut self, col: usize, delta: f64, objdelta: f64) {
        debug_assert!(delta != 0.0);
        debug_assert!(objdelta >= 0.0);
        if delta > 0.0 {
            let unit_gain = objdelta / delta;
            {
                let ps_delta = self.mark_changed(col);
                ps_delta.nsamplesup += 1;
                ps_delta.pseudocostup_sum += unit_gain;
            }
            let mut d = unit_gain - self.pseudocostup[col];
            self.nsamplesup[col] += 1;
            self.pseudocostup[col] += d / f64::from(self.nsamplesup[col]);

            d = unit_gain - self.cost_total;
            self.nsamplestotal += 1;
            self.cost_total += d / self.nsamplestotal as f64;
            self.delta_nsamplestotal += 1;
            self.delta_cost_sum += unit_gain;
        } else {
            let unit_gain = -objdelta / delta;
            {
                let ps_delta = self.mark_changed(col);
                ps_delta.nsamplesdown += 1;
                ps_delta.pseudocostdown_sum += unit_gain;
            }
            let mut d = unit_gain - self.pseudocostdown[col];
            self.nsamplesdown[col] += 1;
            self.pseudocostdown[col] += d / f64::from(self.nsamplesdown[col]);

            d = unit_gain - self.cost_total;
            self.nsamplestotal += 1;
            self.cost_total += d / self.nsamplestotal as f64;
            self.delta_nsamplestotal += 1;
            self.delta_cost_sum += unit_gain;
        }
    }

    /// `addInferenceObservation`: the `upbranch` child of a branching on `col`
    /// propagated `ninferences` bound changes.
    pub fn add_inference_observation(&mut self, col: usize, ninferences: i32, upbranch: bool) {
        let ninf = f64::from(ninferences);
        {
            let delta = self.mark_changed(col);
            if upbranch {
                delta.ninferencesup += 1;
                delta.inferencesup_sum += ninf;
            } else {
                delta.ninferencesdown += 1;
                delta.inferencesdown_sum += ninf;
            }
        }
        let mut d = ninf - self.inferences_total;
        self.ninferencestotal += 1;
        self.inferences_total += d / self.ninferencestotal as f64;
        self.delta_ninferencestotal += 1;
        self.delta_inferences_sum += ninf;
        if upbranch {
            d = ninf - self.inferencesup[col];
            self.ninferencesup[col] += 1;
            self.inferencesup[col] += d / f64::from(self.ninferencesup[col]);
        } else {
            d = ninf - self.inferencesdown[col];
            self.ninferencesdown[col] += 1;
            self.inferencesdown[col] += d / f64::from(self.ninferencesdown[col]);
        }
    }

    /// `isReliable`: both directions have at least `minreliable` samples.
    pub fn is_reliable(&self, col: usize) -> bool {
        self.nsamplesup[col].min(self.nsamplesdown[col]) >= self.minreliable
    }

    /// `isReliableUp`.
    pub fn is_reliable_up(&self, col: usize) -> bool {
        self.nsamplesup[col] >= self.minreliable
    }

    /// `isReliableDown`.
    pub fn is_reliable_down(&self, col: usize) -> bool {
        self.nsamplesdown[col] >= self.minreliable
    }

    /// `getAvgPseudocost`: `cost_total`.
    pub fn get_avg_pseudocost(&self) -> f64 {
        self.cost_total
    }

    /// `getPseudocostUp(col, frac, offset)`: below `minreliable` samples the column's
    /// pseudocost is blended with the average, weight `0.9 + 0.1 n / minreliable` on
    /// the column (0 with no sample).
    pub fn get_pseudocost_up_offset(&self, col: usize, frac: f64, offset: f64) -> f64 {
        let up = frac.ceil() - frac;
        let mut cost;
        let n = self.nsamplesup[col];
        if n == 0 || n < self.minreliable {
            let weight_ps = if n == 0 {
                0.0
            } else {
                0.9 + 0.1 * f64::from(n) / f64::from(self.minreliable)
            };
            cost = weight_ps * self.pseudocostup[col];
            cost += (1.0 - weight_ps) * self.get_avg_pseudocost();
        } else {
            cost = self.pseudocostup[col];
        }
        up * (offset + cost)
    }

    /// `getPseudocostDown(col, frac, offset)`; see [`Self::get_pseudocost_up_offset`].
    pub fn get_pseudocost_down_offset(&self, col: usize, frac: f64, offset: f64) -> f64 {
        let down = frac - frac.floor();
        let mut cost;
        let n = self.nsamplesdown[col];
        if n == 0 || n < self.minreliable {
            let weight_ps = if n == 0 {
                0.0
            } else {
                0.9 + 0.1 * f64::from(n) / f64::from(self.minreliable)
            };
            cost = weight_ps * self.pseudocostdown[col];
            cost += (1.0 - weight_ps) * self.get_avg_pseudocost();
        } else {
            cost = self.pseudocostdown[col];
        }
        down * (offset + cost)
    }

    /// `getPseudocostUp(col, frac)`: `(ceil(frac) - frac)` times the column's up
    /// pseudocost, or `cost_total` with no sample.
    pub fn get_pseudocost_up(&self, col: usize, frac: f64) -> f64 {
        let up = frac.ceil() - frac;
        if self.nsamplesup[col] == 0 {
            return up * self.cost_total;
        }
        up * self.pseudocostup[col]
    }

    /// `getPseudocostDown(col, frac)`.
    pub fn get_pseudocost_down(&self, col: usize, frac: f64) -> f64 {
        let down = frac - frac.floor();
        if self.nsamplesdown[col] == 0 {
            return down * self.cost_total;
        }
        down * self.pseudocostdown[col]
    }

    /// `getConflictScoreUp`.
    pub fn get_conflict_score_up(&self, col: usize) -> f64 {
        self.conflictscoreup[col] / self.conflict_weight
    }

    /// `getConflictScoreDown`.
    pub fn get_conflict_score_down(&self, col: usize) -> f64 {
        self.conflictscoredown[col] / self.conflict_weight
    }

    /// `ncutoffs / max(1, ncutoffs + nsamples)` (the `cutOffScoreUp/Down` terms).
    #[inline]
    fn cutoff_rate(ncutoffs: i32, nsamples: i32) -> f64 {
        f64::from(ncutoffs) / cmax(1.0, f64::from(ncutoffs) + f64::from(nsamples))
    }

    /// The `avgCutoffs` term.
    #[inline]
    fn avg_cutoffs(&self) -> f64 {
        self.ncutoffstotal as f64 / cmax(1.0, self.ncutoffstotal as f64 + self.nsamplestotal as f64)
    }

    /// `getScore(col, upcost, downcost)`: product scores for cost, inferences, cutoffs
    /// and conflicts, each normalized by the squared average and mapped to `[0, 1)`,
    /// combined with weights 1 (cost, divided by the degeneracy factor), `1e-2`
    /// (conflicts) and `1e-4` (cutoffs, inferences; both times the degeneracy factor).
    pub fn get_score_costs(&self, col: usize, upcost: f64, downcost: f64) -> f64 {
        let cost_score = cmax(upcost, MIN_THRESHOLD) * cmax(downcost, MIN_THRESHOLD)
            / cmax(MIN_THRESHOLD, self.cost_total * self.cost_total);
        let inference_score = cmax(self.inferencesup[col], MIN_THRESHOLD)
            * cmax(self.inferencesdown[col], MIN_THRESHOLD)
            / cmax(MIN_THRESHOLD, self.inferences_total * self.inferences_total);

        let cut_off_score_up = Self::cutoff_rate(self.ncutoffsup[col], self.nsamplesup[col]);
        let cut_off_score_down = Self::cutoff_rate(self.ncutoffsdown[col], self.nsamplesdown[col]);
        let avg_cutoffs = self.avg_cutoffs();

        let cutoff_score = cmax(cut_off_score_up, MIN_THRESHOLD)
            * cmax(cut_off_score_down, MIN_THRESHOLD)
            / cmax(MIN_THRESHOLD, avg_cutoffs * avg_cutoffs);

        let conflict_score_up = self.conflictscoreup[col] / self.conflict_weight;
        let conflict_score_down = self.conflictscoredown[col] / self.conflict_weight;
        let conflict_score_avg =
            self.conflict_avg_score / (self.conflict_weight * self.conflictscoreup.len() as f64);
        let conflict_score = cmax(conflict_score_up, MIN_THRESHOLD)
            * cmax(conflict_score_down, MIN_THRESHOLD)
            / cmax(MIN_THRESHOLD, conflict_score_avg * conflict_score_avg);

        map_score(cost_score) / self.degeneracy_factor
            + self.degeneracy_factor
                * (1e-2 * map_score(conflict_score)
                    + 1e-4 * (map_score(cutoff_score) + map_score(inference_score)))
    }

    /// `getScore(col, frac)`: [`Self::get_score_costs`] with the unblended
    /// pseudocosts of `frac`.
    pub fn get_score(&self, col: usize, frac: f64) -> f64 {
        let upcost = self.get_pseudocost_up(col, frac);
        let downcost = self.get_pseudocost_down(col, frac);

        self.get_score_costs(col, upcost, downcost)
    }

    /// `getScoreUp`: the one-sided score of the up branch (ratios to the averages,
    /// not products; no degeneracy factor).
    pub fn get_score_up(&self, col: usize, frac: f64) -> f64 {
        let cost_score = self.get_pseudocost_up(col, frac) / cmax(MIN_THRESHOLD, self.cost_total);
        let inference_score = self.inferencesup[col] / cmax(MIN_THRESHOLD, self.inferences_total);

        let cut_off_score_up = Self::cutoff_rate(self.ncutoffsup[col], self.nsamplesup[col]);
        let avg_cutoffs = self.avg_cutoffs();

        let cutoff_score = cut_off_score_up / cmax(MIN_THRESHOLD, avg_cutoffs);

        let conflict_score_up = self.conflictscoreup[col] / self.conflict_weight;
        let conflict_score_avg =
            self.conflict_avg_score / (self.conflict_weight * self.conflictscoreup.len() as f64);
        let conflict_score = conflict_score_up / cmax(MIN_THRESHOLD, conflict_score_avg);

        map_score(cost_score)
            + (1e-2 * map_score(conflict_score)
                + 1e-4 * (map_score(cutoff_score) + map_score(inference_score)))
    }

    /// `getScoreDown`; see [`Self::get_score_up`].
    pub fn get_score_down(&self, col: usize, frac: f64) -> f64 {
        let cost_score = self.get_pseudocost_down(col, frac) / cmax(MIN_THRESHOLD, self.cost_total);
        let inference_score = self.inferencesdown[col] / cmax(MIN_THRESHOLD, self.inferences_total);

        let cut_off_score_down = Self::cutoff_rate(self.ncutoffsdown[col], self.nsamplesdown[col]);
        let avg_cutoffs = self.avg_cutoffs();

        let cutoff_score = cut_off_score_down / cmax(MIN_THRESHOLD, avg_cutoffs);

        let conflict_score_down = self.conflictscoredown[col] / self.conflict_weight;
        let conflict_score_avg =
            self.conflict_avg_score / (self.conflict_weight * self.conflictscoredown.len() as f64);
        let conflict_score = conflict_score_down / cmax(MIN_THRESHOLD, conflict_score_avg);

        map_score(cost_score)
            + (1e-2 * map_score(conflict_score)
                + 1e-4 * (map_score(cutoff_score) + map_score(inference_score)))
    }

    /// `getAvgInferencesUp`.
    pub fn get_avg_inferences_up(&self, col: usize) -> f64 {
        self.inferencesup[col]
    }

    /// `getAvgInferencesDown`.
    pub fn get_avg_inferences_down(&self, col: usize) -> f64 {
        self.inferencesdown[col]
    }

    /// `flushPseudoCost`: merges `pseudocost`'s pending deltas (a worker's changes since
    /// its last flush or sync) into `self` (the master), then clears them.
    /// Conflict-score sums are rescaled from the worker's conflict weight to ours.
    pub fn flush_pseudo_cost(&mut self, pseudocost: &mut Pseudocost) {
        assert_eq!(pseudocost.ncutoffsup.len(), self.ncutoffsup.len());
        for delta in &pseudocost.deltas {
            let col = delta.col;
            debug_assert!(col < pseudocost.ncutoffsup.len());
            add_delta_average_i32(
                &mut self.pseudocostup[col],
                &mut self.nsamplesup[col],
                delta.pseudocostup_sum,
                delta.nsamplesup,
            );
            add_delta_average_i32(
                &mut self.pseudocostdown[col],
                &mut self.nsamplesdown[col],
                delta.pseudocostdown_sum,
                delta.nsamplesdown,
            );
            add_delta_average_i32(
                &mut self.inferencesup[col],
                &mut self.ninferencesup[col],
                delta.inferencesup_sum,
                delta.ninferencesup,
            );
            add_delta_average_i32(
                &mut self.inferencesdown[col],
                &mut self.ninferencesdown[col],
                delta.inferencesdown_sum,
                delta.ninferencesdown,
            );
            let conflict_scale = self.conflict_weight / pseudocost.conflict_weight;
            let conflict_delta_up = conflict_scale * delta.conflictscoreup_sum;
            let conflict_delta_down = conflict_scale * delta.conflictscoredown_sum;
            self.conflictscoreup[col] += conflict_delta_up;
            self.conflictscoredown[col] += conflict_delta_down;
            self.conflict_avg_score += conflict_delta_up + conflict_delta_down;
            self.ncutoffsup[col] += delta.ncutoffsup;
            self.ncutoffsdown[col] += delta.ncutoffsdown;
            self.ncutoffstotal += i64::from(delta.ncutoffsup + delta.ncutoffsdown);
        }
        add_delta_average_i64(
            &mut self.cost_total,
            &mut self.nsamplestotal,
            pseudocost.delta_cost_sum,
            pseudocost.delta_nsamplestotal,
        );
        add_delta_average_i64(
            &mut self.inferences_total,
            &mut self.ninferencestotal,
            pseudocost.delta_inferences_sum,
            pseudocost.delta_ninferencestotal,
        );
        pseudocost.remove_changed();
    }

    /// `syncPseudoCost`: overwrites `pseudocost`'s statistics with ours (master to
    /// worker) and clears its pending deltas. `minreliable` and the degeneracy factor
    /// are not copied, as in HiGHS.
    pub fn sync_pseudo_cost(&self, pseudocost: &mut Pseudocost) {
        let n = self.pseudocostup.len();
        assert!(
            pseudocost.pseudocostup.len() >= n,
            "std::copy target too short"
        );
        pseudocost.pseudocostup[..n].copy_from_slice(&self.pseudocostup);
        pseudocost.pseudocostdown[..n].copy_from_slice(&self.pseudocostdown);
        pseudocost.nsamplesup[..n].copy_from_slice(&self.nsamplesup);
        pseudocost.nsamplesdown[..n].copy_from_slice(&self.nsamplesdown);
        pseudocost.inferencesup[..n].copy_from_slice(&self.inferencesup);
        pseudocost.inferencesdown[..n].copy_from_slice(&self.inferencesdown);
        pseudocost.ninferencesup[..n].copy_from_slice(&self.ninferencesup);
        pseudocost.ninferencesdown[..n].copy_from_slice(&self.ninferencesdown);
        pseudocost.ncutoffsup[..n].copy_from_slice(&self.ncutoffsup);
        pseudocost.ncutoffsdown[..n].copy_from_slice(&self.ncutoffsdown);
        pseudocost.conflictscoreup[..n].copy_from_slice(&self.conflictscoreup);
        pseudocost.conflictscoredown[..n].copy_from_slice(&self.conflictscoredown);
        pseudocost.conflict_weight = self.conflict_weight;
        pseudocost.conflict_avg_score = self.conflict_avg_score;
        pseudocost.cost_total = self.cost_total;
        pseudocost.inferences_total = self.inferences_total;
        pseudocost.nsamplestotal = self.nsamplestotal;
        pseudocost.ninferencestotal = self.ninferencestotal;
        pseudocost.ncutoffstotal = self.ncutoffstotal;
        pseudocost.remove_changed();
    }

    /// `removeChanged`: drops the pending deltas.
    pub fn remove_changed(&mut self) {
        for delta in &self.deltas {
            self.changedpos[delta.col] = None;
        }
        self.deltas.clear();
        self.delta_cost_sum = 0.0;
        self.delta_inferences_sum = 0.0;
        self.delta_nsamplestotal = 0;
        self.delta_ninferencestotal = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts executed checks so a test that checked nothing fails (CLAUDE.md §6).
    struct Probe {
        checks: usize,
        name: &'static str,
    }

    impl Probe {
        fn new(name: &'static str) -> Self {
            Probe { checks: 0, name }
        }
        #[track_caller]
        fn bits(&mut self, got: f64, want: f64, what: &str) {
            self.checks += 1;
            assert!(
                got.to_bits() == want.to_bits(),
                "{}: {what}: got {got:e}, want {want:e}",
                self.name
            );
        }
        #[track_caller]
        fn close(&mut self, got: f64, want: f64, rel: f64, what: &str) {
            self.checks += 1;
            let tol = rel * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tol,
                "{}: {what}: got {got:e}, want {want:e}",
                self.name
            );
        }
        #[track_caller]
        fn truth(&mut self, cond: bool, what: &str) {
            self.checks += 1;
            assert!(cond, "{}: {what}", self.name);
        }
        fn finish(self, min: usize) -> usize {
            println!("{}: {} checks executed", self.name, self.checks);
            assert!(
                self.checks >= min,
                "{}: only {} checks executed (expected >= {min})",
                self.name,
                self.checks
            );
            self.checks
        }
    }

    /// `addObservation` (header lines of `addObservation`): running means per
    /// direction and overall, with the unit gain `objdelta / |delta|`.
    #[test]
    fn highs_pseudocost_add_observation_hand_computed() {
        let mut p = Probe::new("add_observation");
        let mut ps = Pseudocost::new(3, DEFAULT_MIN_RELIABLE);
        // Up: delta 0.5, objdelta 2 -> unit gain 4; first sample: pc = 0 + 4/1.
        ps.add_observation(0, 0.5, 2.0);
        p.bits(ps.pseudocostup[0], 4.0, "pcup after 1");
        p.bits(ps.cost_total, 4.0, "cost_total after 1");
        // Up: delta 0.25, objdelta 0.5 -> 2; d = 2 - 4 = -2; pc = 4 + -2/2 = 3.
        ps.add_observation(0, 0.25, 0.5);
        p.bits(ps.pseudocostup[0], 3.0, "pcup after 2");
        p.bits(ps.cost_total, 3.0, "cost_total after 2");
        // Down: delta -0.5, objdelta 3 -> unit gain -3 / -0.5 = 6; total 3 + 3/3 = 4.
        ps.add_observation(0, -0.5, 3.0);
        p.bits(ps.pseudocostdown[0], 6.0, "pcdown");
        p.bits(ps.cost_total, 4.0, "cost_total after 3");
        p.truth(ps.nsamplesup[0] == 2 && ps.nsamplesdown[0] == 1, "counts");
        p.truth(ps.get_num_observations(0) == 3, "getNumObservations");
        p.truth(ps.get_num_observations_up(0) == 2, "getNumObservationsUp");
        p.truth(
            ps.get_num_observations_down(0) == 1,
            "getNumObservationsDown",
        );
        p.truth(ps.nsamplestotal == 3, "nsamplestotal");
        // Delta buffer: one record for col 0 with the raw sums.
        p.truth(
            ps.deltas.len() == 1 && ps.changedpos[0] == Some(0),
            "one delta",
        );
        p.bits(ps.deltas[0].pseudocostup_sum, 6.0, "delta up sum 4+2");
        p.bits(ps.deltas[0].pseudocostdown_sum, 6.0, "delta down sum");
        p.bits(ps.delta_cost_sum, 12.0, "delta_cost_sum");
        p.truth(ps.delta_nsamplestotal == 3, "delta_nsamplestotal");
        p.bits(ps.get_avg_pseudocost(), 4.0, "getAvgPseudocost");

        // getPseudocostUp(col, frac): up = ceil(2.25) - 2.25 = 0.75; 0.75 * 3.
        p.bits(ps.get_pseudocost_up(0, 2.25), 2.25, "getPseudocostUp");
        // getPseudocostDown: down = 0.25; 0.25 * 6.
        p.bits(ps.get_pseudocost_down(0, 2.25), 1.5, "getPseudocostDown");
        // Column 1 has no samples: falls back to cost_total = 4.
        p.bits(
            ps.get_pseudocost_up(1, 0.5),
            2.0,
            "getPseudocostUp fallback",
        );
        p.bits(
            ps.get_pseudocost_down(1, 0.5),
            2.0,
            "getPseudocostDown fallback",
        );
        // An integral frac gives 0 both ways.
        p.bits(ps.get_pseudocost_up(0, 3.0), 0.0, "integral up");
        p.bits(ps.get_pseudocost_down(0, 3.0), 0.0, "integral down");

        // getPseudocostUp(col, frac, offset), minreliable 8, n = 2:
        // weightPs = 0.9 + 0.1*2/8 = 0.925; cost = 0.925*3 + 0.075*4 = 3.075;
        // up * (offset + cost) = 0.75 * (0.5 + 3.075) = 2.68125.
        let w = 0.9 + 0.1 * 2.0 / 8.0;
        let mut cost = w * 3.0;
        cost += (1.0 - w) * 4.0;
        p.bits(
            ps.get_pseudocost_up_offset(0, 2.25, 0.5),
            0.75 * (0.5 + cost),
            "pcup off",
        );
        p.close(
            ps.get_pseudocost_up_offset(0, 2.25, 0.5),
            2.68125,
            1e-15,
            "pcup off dec",
        );
        // Down, n = 1: weightPs = 0.9125; cost = 0.9125*6 + 0.0875*4 = 5.825;
        // 0.25 * (0.5 + 5.825) = 1.58125.
        let w = 0.9 + 0.1 * 1.0 / 8.0;
        let mut cost = w * 6.0;
        cost += (1.0 - w) * 4.0;
        p.bits(
            ps.get_pseudocost_down_offset(0, 2.25, 0.5),
            0.25 * (0.5 + cost),
            "pcd off",
        );
        p.close(
            ps.get_pseudocost_down_offset(0, 2.25, 0.5),
            1.58125,
            1e-15,
            "pcd off dec",
        );
        // No sample: weightPs = 0, pure average: 0.5 * (0 + 4) = 2.
        p.bits(
            ps.get_pseudocost_up_offset(2, 1.5, 0.0),
            2.0,
            "pcup off, n=0",
        );
        // Reliable (minreliable 2 <= n = 2): pure column value 0.75 * (0 + 3).
        ps.set_min_reliable(2);
        p.bits(
            ps.get_pseudocost_up_offset(0, 2.25, 0.0),
            2.25,
            "pcup off reliable",
        );
        // minreliable 0 with n = 0 still takes the blended branch (n == 0).
        ps.set_min_reliable(0);
        p.bits(
            ps.get_pseudocost_down_offset(1, 0.5, 1.0),
            0.5 * (1.0 + 4.0),
            "mr 0",
        );
        p.finish(30);
    }

    /// `isReliable` / `isReliableUp` / `isReliableDown` / `getMinReliable`.
    #[test]
    fn highs_pseudocost_reliability() {
        let mut p = Probe::new("reliability");
        let mut ps = Pseudocost::new(2, 3);
        p.truth(ps.get_min_reliable() == 3, "getMinReliable");
        for k in 0..3 {
            p.truth(!ps.is_reliable_up(0), "not yet reliable up");
            ps.add_observation(0, 1.0, k as f64);
        }
        p.truth(ps.is_reliable_up(0), "reliable up at 3");
        p.truth(!ps.is_reliable_down(0), "down not reliable");
        p.truth(!ps.is_reliable(0), "min(3, 0) < 3");
        for _ in 0..3 {
            ps.add_observation(0, -1.0, 1.0);
        }
        p.truth(ps.is_reliable(0), "min(3, 3) >= 3");
        ps.set_min_reliable(4);
        p.truth(!ps.is_reliable(0), "raised threshold");
        ps.set_min_reliable(0);
        p.truth(
            ps.is_reliable(1),
            "threshold 0: unsampled column is reliable",
        );
        p.finish(10);
    }

    /// `addInferenceObservation` and `getAvgInferencesUp/Down`.
    #[test]
    fn highs_pseudocost_inferences_hand_computed() {
        let mut p = Probe::new("inferences");
        let mut ps = Pseudocost::new(2, 8);
        ps.add_inference_observation(1, 3, true);
        ps.add_inference_observation(1, 5, true);
        ps.add_inference_observation(1, 1, false);
        // up mean (3+5)/2, down 1, overall (3+5+1)/3: 3 -> 3 + (5-3)/2 = 4 -> 4 + (1-4)/3.
        p.bits(ps.get_avg_inferences_up(1), 4.0, "avg inferences up");
        p.bits(ps.get_avg_inferences_down(1), 1.0, "avg inferences down");
        p.bits(ps.inferences_total, 3.0, "inferences_total");
        p.truth(ps.ninferencestotal == 3, "ninferencestotal");
        p.truth(
            ps.ninferencesup[1] == 2 && ps.ninferencesdown[1] == 1,
            "counts",
        );
        p.bits(ps.delta_inferences_sum, 9.0, "delta sum");
        p.bits(ps.deltas[0].inferencesup_sum, 8.0, "delta up sum");
        p.truth(ps.deltas[0].ninferencesdown == 1, "delta down count");
        p.finish(8);
    }

    /// `addCutoffObservation` and the cutoff terms of the scores.
    #[test]
    fn highs_pseudocost_cutoffs_hand_computed() {
        let mut p = Probe::new("cutoffs");
        let mut ps = Pseudocost::new(2, 8);
        ps.add_cutoff_observation(0, true);
        ps.add_cutoff_observation(0, true);
        ps.add_cutoff_observation(0, false);
        ps.add_observation(0, 1.0, 2.0);
        p.truth(ps.ncutoffsup[0] == 2 && ps.ncutoffsdown[0] == 1, "per col");
        p.truth(ps.ncutoffstotal == 3, "total");
        p.truth(
            ps.deltas[0].ncutoffsup == 2 && ps.deltas[0].ncutoffsdown == 1,
            "delta",
        );
        // cutOffScoreUp = 2 / max(1, 2 + 1) = 2/3; avgCutoffs = 3 / max(1, 3 + 1) = 0.75.
        p.bits(Pseudocost::cutoff_rate(2, 1), 2.0 / 3.0, "cutOffScoreUp");
        p.bits(Pseudocost::cutoff_rate(0, 0), 0.0, "0 / max(1, 0)");
        p.bits(ps.avg_cutoffs(), 0.75, "avgCutoffs");
        p.finish(6);
    }

    /// `getScore(col, upcost, downcost)`, `getScore(col, frac)`, `getScoreUp`,
    /// `getScoreDown`, worked out term by term from the header's formulas.
    #[test]
    fn highs_pseudocost_scores_hand_computed() {
        let mut p = Probe::new("scores");
        let mut ps = Pseudocost::new(2, 8);
        ps.add_observation(0, 1.0, 2.0); // up gain 2
        ps.add_observation(0, -1.0, 4.0); // down gain 4; cost_total = 3
        ps.add_inference_observation(0, 2, true);
        ps.add_inference_observation(0, 4, false); // inferences_total = 3
        ps.add_cutoff_observation(0, true); // ncutoffsup 1, total 1
        ps.increase_conflict_score_up(0); // weight 1: up 1, avg 1
        ps.increase_conflict_score_down(0); // down 1, avg 2
        p.bits(ps.cost_total, 3.0, "cost_total");
        p.bits(ps.inferences_total, 3.0, "inferences_total");
        p.bits(ps.conflict_avg_score, 2.0, "conflict_avg_score");

        let ms = |s: f64| 1.0 - 1.0 / (1.0 + s);
        // getScore(0, 2, 4): costScore = 2*4/9; inferenceScore = 2*4/9;
        // cutOffScoreUp = 1/max(1, 1+1) = 0.5, Down = 0 -> 1e-6;
        // avgCutoffs = 1/max(1, 1+2) = 1/3; cutoffScore = 0.5*1e-6/(1/9);
        // conflictScoreAvg = 2 / (1*2) = 1; conflictScore = 1*1/1 = 1.
        let cost = 2.0 * 4.0 / 9.0;
        let inf = 2.0 * 4.0 / 9.0;
        let avgc = 1.0 / 3.0;
        let cut = 0.5 * 1e-6 / (avgc * avgc);
        let want = ms(cost) / 1.0 + 1.0 * (1e-2 * ms(1.0) + 1e-4 * (ms(cut) + ms(inf)));
        p.bits(ps.get_score_costs(0, 2.0, 4.0), want, "getScore(costs)");
        // Decimal check: ms(8/9) = 8/17; ms(1) = 0.5; ms(4.5e-6) ~ 4.49998e-6.
        p.close(
            ps.get_score_costs(0, 2.0, 4.0),
            8.0 / 17.0 + 5e-3 + 1e-4 * (4.5e-6 / 1.0000045 + 8.0 / 17.0),
            1e-14,
            "getScore decimal",
        );
        // getScore(0, 0.5): upcost = 0.5*2 = 1, downcost = 0.5*4 = 2.
        p.bits(
            ps.get_score(0, 0.5),
            ps.get_score_costs(0, 1.0, 2.0),
            "getScore(frac)",
        );
        // Degeneracy factor 2 moves weight from cost to the tie-breakers.
        ps.set_degeneracy_factor(2.0);
        let want = ms(cost) / 2.0 + 2.0 * (1e-2 * ms(1.0) + 1e-4 * (ms(cut) + ms(inf)));
        p.bits(
            ps.get_score_costs(0, 2.0, 4.0),
            want,
            "getScore degeneracy 2",
        );
        // Costs below minThreshold are floored.
        let want0 =
            ms(1e-6 * 1e-6 / 9.0) / 2.0 + 2.0 * (1e-2 * ms(1.0) + 1e-4 * (ms(cut) + ms(inf)));
        p.bits(ps.get_score_costs(0, 0.0, -5.0), want0, "floored costs");

        // getScoreUp(0, 0.5): costScore = 1/3; inferenceScore = 2/3;
        // cutoffScore = 0.5 / (1/3) = 1.5; conflictScore = 1/1 = 1.
        let want_up = ms(1.0 / 3.0) + (1e-2 * ms(1.0) + 1e-4 * (ms(0.5 / avgc) + ms(2.0 / 3.0)));
        p.bits(ps.get_score_up(0, 0.5), want_up, "getScoreUp");
        p.close(
            ps.get_score_up(0, 0.5),
            0.25 + 5e-3 + 1e-4 * (0.6 + 0.4),
            1e-14,
            "getScoreUp decimal",
        );
        // getScoreDown(0, 0.5): costScore = 2/3; inferenceScore = 4/3; cutoff 0; conflict 1.
        let want_dn = ms(2.0 / 3.0) + (1e-2 * ms(1.0) + 1e-4 * (ms(0.0) + ms(4.0 / 3.0)));
        p.bits(ps.get_score_down(0, 0.5), want_dn, "getScoreDown");
        p.close(
            ps.get_score_down(0, 0.5),
            0.4 + 5e-3 + 1e-4 * (4.0 / 7.0),
            1e-14,
            "getScoreDown decimal",
        );
        // getScoreUp/Down carry no degeneracy factor: unchanged by it.
        ps.set_degeneracy_factor(1.0);
        p.bits(ps.get_score_up(0, 0.5), want_up, "getScoreUp df-free");

        // A fresh pseudocost: every term is floored; nothing is NaN (cost_total 0,
        // conflict avg 0 -> divide by max(1e-6, 0)).
        let fresh = Pseudocost::new(3, 8);
        let s = fresh.get_score(1, 0.5);
        p.truth(s.is_finite(), "fresh getScore finite");
        // Every product term is max(0, 1e-6) * max(0, 1e-6) / max(1e-6, 0), so all four
        // mapped terms are equal.
        let m = ms(1e-6 * 1e-6 / 1e-6);
        p.bits(
            s,
            m / 1.0 + 1.0 * (1e-2 * m + 1e-4 * (m + m)),
            "fresh getScore",
        );
        p.bits(fresh.get_score_up(1, 0.5), 0.0, "fresh getScoreUp");
        p.finish(14);
    }

    /// `increaseConflictWeight`, `increaseConflictScoreUp/Down`, and the rescaling at
    /// weight > 1000 (1.02^349 is the first power above 1000).
    #[test]
    fn highs_pseudocost_conflict_weight_rescaling() {
        let mut p = Probe::new("conflict_weight");
        let mut ps = Pseudocost::new(2, 8);
        ps.increase_conflict_score_up(0);
        ps.increase_conflict_weight();
        p.bits(ps.conflict_weight, 1.02, "weight 1.02");
        ps.increase_conflict_score_down(1);
        p.bits(ps.conflictscoredown[1], 1.02, "down += weight");
        p.bits(ps.conflict_avg_score, 2.02, "avg");
        p.bits(
            ps.get_conflict_score_up(0),
            1.0 / 1.02,
            "getConflictScoreUp",
        );
        p.bits(
            ps.get_conflict_score_down(1),
            1.02 / 1.02,
            "getConflictScoreDown",
        );
        p.bits(ps.deltas[1].conflictscoredown_sum, 1.02, "delta down sum");

        // Drive the weight past 1000 and track the C++ arithmetic alongside.
        let mut w = 1.02f64;
        let mut up0 = 1.0f64;
        let mut down1 = 1.02f64;
        let mut avg = 2.02f64;
        let mut d_up0 = 1.0f64;
        let mut calls = 1usize;
        let mut rescaled_at = None;
        while rescaled_at.is_none() {
            ps.increase_conflict_weight();
            calls += 1;
            w *= 1.02;
            if w > 1000.0 {
                let scale = 1.0 / w;
                w = 1.0;
                avg *= scale;
                up0 *= scale;
                down1 *= scale;
                d_up0 *= scale;
                rescaled_at = Some(calls);
            }
            p.bits(ps.conflict_weight, w, "weight sequence");
        }
        p.truth(rescaled_at == Some(349), "first rescale on the 349th call");
        p.bits(ps.conflict_weight, 1.0, "weight reset to 1");
        p.bits(ps.conflictscoreup[0], up0, "up rescaled");
        p.bits(ps.conflictscoredown[1], down1, "down rescaled");
        p.bits(ps.conflict_avg_score, avg, "avg rescaled");
        p.bits(ps.deltas[0].conflictscoreup_sum, d_up0, "delta rescaled");
        // The normalized score is (up to rounding) invariant under the rescale: one
        // increment at weight 1 against 1.02^349 now.
        p.close(
            ps.get_conflict_score_up(0),
            1.02f64.powi(-349),
            1e-12,
            "normalized",
        );
        p.finish(350);
    }

    /// `HighsPseudocostInitialization` (both overloads) and the
    /// `HighsPseudocost(mipsolver)` constructor's use of it.
    #[test]
    fn highs_pseudocost_initialization() {
        let mut p = Probe::new("initialization");
        let mut src = Pseudocost::new(3, 8);
        for _ in 0..5 {
            src.add_observation(0, 0.5, 1.0); // up gain 2, five samples
        }
        src.add_observation(1, -1.0, 3.0);
        src.add_inference_observation(0, 4, true);
        src.add_inference_observation(0, 6, true);
        src.increase_conflict_weight();
        src.increase_conflict_score_up(2);
        src.add_cutoff_observation(1, false);

        let init = PseudocostInitialization::new(&src, 3);
        p.truth(init.nsamplesup[0] == 3, "maxCount caps nsamplesup");
        p.truth(init.nsamplesdown[1] == 1, "below cap unchanged");
        p.truth(init.ninferencesup[0] == 1, "ninferences capped at 1");
        p.truth(init.ninferencesdown[0] == 0, "zero stays zero");
        p.truth(init.nsamplestotal == 1, "nsamplestotal = min(1, 6)");
        p.truth(init.ninferencestotal == 1, "ninferencestotal = min(1, 2)");
        p.bits(init.pseudocostup[0], 2.0, "pseudocost copied");
        p.bits(init.cost_total, src.cost_total, "cost_total copied");
        p.bits(init.conflictscoreup[2], 1.02 / 1.02, "conflict / weight");
        // avg = 1.02 / (3 * 1.02).
        p.bits(
            init.conflict_avg_score,
            1.02 / (3.0 * 1.02),
            "avg per column",
        );

        let ps = Pseudocost::with_initialization(3, 8, &init, &[0, 1, 2]);
        p.truth(
            ps.nsamplesup[0] == 3 && ps.nsamplestotal == 1,
            "counts carried",
        );
        p.bits(
            ps.conflict_avg_score,
            init.conflict_avg_score * 3.0,
            "avg * numCol",
        );
        p.bits(ps.conflict_weight, 1.0, "fresh weight");
        p.truth(
            ps.ncutoffsdown[1] == 0 && ps.ncutoffstotal == 0,
            "no cutoffs carried",
        );
        p.truth(ps.deltas.is_empty(), "no deltas");
        p.bits(ps.get_avg_inferences_up(0), 5.0, "inferences carried");

        // Through a column map: presolved cols 0,1,2 are original 4,0,2 of 5.
        let lifted = PseudocostInitialization::with_col_map(&src, 3, 5, &[4, 0, 2]);
        p.truth(lifted.pseudocostup.len() == 5, "original size");
        p.bits(lifted.pseudocostup[4], 2.0, "col 0 -> 4");
        p.bits(lifted.pseudocostdown[0], 3.0, "col 1 -> 0");
        p.truth(lifted.nsamplesup[4] == 3, "capped through map");
        p.truth(
            lifted.ninferencesup[0] == 1,
            "ninferences set to 1 unconditionally",
        );
        p.truth(lifted.ninferencesup[1] == 0, "unmapped col zero");
        p.bits(
            lifted.conflict_avg_score,
            1.02 / (3.0 * 1.02),
            "avg over presolved ncols",
        );
        // And back into a new presolve of 2 columns (orig 4 and 2).
        let ps2 = Pseudocost::with_initialization(2, 8, &lifted, &[4, 2]);
        p.bits(ps2.pseudocostup[0], 2.0, "round trip pc");
        p.bits(ps2.conflictscoreup[1], 1.0, "round trip conflict");
        p.bits(
            ps2.conflict_avg_score,
            lifted.conflict_avg_score * 2.0,
            "avg * new numCol",
        );
        p.finish(25);
    }

    /// A small deterministic generator (SplitMix64) for the randomized test.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Raw observations of one column, the ground truth the averages are checked against.
    #[derive(Default, Clone)]
    struct Raw {
        up: Vec<f64>,
        down: Vec<f64>,
        inf_up: Vec<f64>,
        inf_down: Vec<f64>,
        cut_up: i32,
        cut_down: i32,
        /// Conflict increments in absolute (never rescaled) weight units.
        conf_up: f64,
        conf_down: f64,
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Random observation sequences on a master and a worker that periodically flushes
    /// into the master and is re-synced from it. Checks after every step that the
    /// running means equal the means of the raw observations, counts and reliability
    /// match, conflict scores equal the absolute increments over the absolute weight,
    /// and every score lies in its range.
    #[test]
    fn highs_pseudocost_randomized_invariants() {
        let mut p = Probe::new("randomized");
        let ncol = 7usize;
        let mut rng = Rng(0x5eed_2026);
        let mut total_steps = 0usize;
        for round in 0..6i32 {
            let minreliable = (round % 4) + 1;
            let mut master = Pseudocost::new(ncol, minreliable);
            let mut worker = Pseudocost::new(ncol, minreliable);
            let df = 1.0 + round as f64 * 0.5;
            master.set_degeneracy_factor(df);
            let mut raw = vec![Raw::default(); ncol];
            // Absolute conflict weight 1.02^k (k = number of increaseConflictWeight
            // calls), kept as a log to avoid overflow over many rescales.
            let mut log_w = 0.0f64;
            for step in 0..400 {
                total_steps += 1;
                let col = rng.below(ncol as u64) as usize;
                let up = rng.below(2) == 0;
                let target: &mut Pseudocost = if rng.below(3) == 0 {
                    &mut master
                } else {
                    &mut worker
                };
                match rng.below(5) {
                    0 | 1 => {
                        let delta = (0.05 + 0.9 * rng.unit()) * if up { 1.0 } else { -1.0 };
                        let objdelta = 10.0 * rng.unit();
                        target.add_observation(col, delta, objdelta);
                        let g = objdelta / delta.abs();
                        if up {
                            raw[col].up.push(g);
                        } else {
                            raw[col].down.push(g);
                        }
                    }
                    2 => {
                        let n = rng.below(20) as i32;
                        target.add_inference_observation(col, n, up);
                        if up {
                            raw[col].inf_up.push(f64::from(n));
                        } else {
                            raw[col].inf_down.push(f64::from(n));
                        }
                    }
                    3 => {
                        target.add_cutoff_observation(col, up);
                        if up {
                            raw[col].cut_up += 1;
                        } else {
                            raw[col].cut_down += 1;
                        }
                    }
                    _ => {
                        // Conflicts only on the master, so its weight is the absolute one
                        // (a worker's conflicts are rescaled by the flush; checked below).
                        for _ in 0..rng.below(40) {
                            master.increase_conflict_weight();
                            log_w += 1.02f64.ln();
                        }
                        let abs = log_w.exp();
                        if up {
                            master.increase_conflict_score_up(col);
                            raw[col].conf_up += abs;
                        } else {
                            master.increase_conflict_score_down(col);
                            raw[col].conf_down += abs;
                        }
                    }
                }
                if step % 37 == 36 {
                    master.flush_pseudo_cost(&mut worker);
                    p.truth(worker.deltas.is_empty(), "flush clears worker deltas");
                    p.truth(
                        worker.changedpos.iter().all(|c| c.is_none()),
                        "changedpos reset",
                    );
                    master.sync_pseudo_cost(&mut worker);
                    p.truth(worker.cost_total == master.cost_total, "sync copies totals");
                    p.truth(
                        worker.pseudocostup == master.pseudocostup,
                        "sync copies arrays",
                    );
                    p.truth(
                        worker.conflict_weight == master.conflict_weight,
                        "sync weight",
                    );
                }
                if step % 37 != 36 && step % 11 != 0 {
                    continue;
                }
                // Check the master after merging pending worker changes into a copy.
                let mut merged = master.clone();
                let mut w2 = worker.clone();
                merged.flush_pseudo_cost(&mut w2);
                let ps = &merged;
                let all: Vec<f64> = raw
                    .iter()
                    .flat_map(|r| r.up.iter().chain(r.down.iter()))
                    .copied()
                    .collect();
                let all_inf: Vec<f64> = raw
                    .iter()
                    .flat_map(|r| r.inf_up.iter().chain(r.inf_down.iter()))
                    .copied()
                    .collect();
                p.truth(ps.nsamplestotal == all.len() as i64, "nsamplestotal");
                if !all.is_empty() {
                    p.close(ps.cost_total, mean(&all), 1e-11, "cost_total = mean");
                }
                if !all_inf.is_empty() {
                    p.close(
                        ps.inferences_total,
                        mean(&all_inf),
                        1e-11,
                        "inferences_total",
                    );
                }
                let ncut: i32 = raw.iter().map(|r| r.cut_up + r.cut_down).sum();
                p.truth(ps.ncutoffstotal == i64::from(ncut), "ncutoffstotal");
                let abs_w = log_w.exp();
                let conf_sum: f64 = raw.iter().map(|r| r.conf_up + r.conf_down).sum();
                p.close(
                    ps.conflict_avg_score / ps.conflict_weight,
                    conf_sum / abs_w,
                    1e-9,
                    "conflict_avg_score / weight",
                );
                for c in 0..ncol {
                    let r = &raw[c];
                    p.truth(ps.nsamplesup[c] == r.up.len() as i32, "nsamplesup");
                    p.truth(ps.nsamplesdown[c] == r.down.len() as i32, "nsamplesdown");
                    if !r.up.is_empty() {
                        p.close(ps.pseudocostup[c], mean(&r.up), 1e-11, "pcup = mean");
                    }
                    if !r.down.is_empty() {
                        p.close(ps.pseudocostdown[c], mean(&r.down), 1e-11, "pcdown = mean");
                    }
                    if !r.inf_up.is_empty() {
                        p.close(ps.get_avg_inferences_up(c), mean(&r.inf_up), 1e-11, "infup");
                    }
                    if !r.inf_down.is_empty() {
                        p.close(
                            ps.get_avg_inferences_down(c),
                            mean(&r.inf_down),
                            1e-11,
                            "infd",
                        );
                    }
                    p.truth(ps.ncutoffsup[c] == r.cut_up, "ncutoffsup");
                    p.truth(ps.ncutoffsdown[c] == r.cut_down, "ncutoffsdown");
                    p.close(
                        ps.get_conflict_score_up(c),
                        r.conf_up / abs_w,
                        1e-9,
                        "conf up",
                    );
                    p.close(
                        ps.get_conflict_score_down(c),
                        r.conf_down / abs_w,
                        1e-9,
                        "conf d",
                    );
                    let (nu, nd) = (r.up.len() as i32, r.down.len() as i32);
                    p.truth(ps.is_reliable_up(c) == (nu >= minreliable), "isReliableUp");
                    p.truth(
                        ps.is_reliable_down(c) == (nd >= minreliable),
                        "isReliableDown",
                    );
                    p.truth(
                        ps.is_reliable(c) == (nu.min(nd) >= minreliable),
                        "isReliable",
                    );
                    p.truth(ps.get_num_observations(c) == nu + nd, "getNumObservations");
                    let frac = c as f64 + 0.05 + 0.9 * rng.unit();
                    // getScore: each mapScore in [0, 1), so the total lies in
                    // [0, 1/df + df * (1e-2 + 2e-4)).
                    let s = ps.get_score(c, frac);
                    let hi = 1.0 / df + df * (1e-2 + 2e-4);
                    p.truth(s.is_finite() && (0.0..hi).contains(&s), "getScore range");
                    for s1 in [ps.get_score_up(c, frac), ps.get_score_down(c, frac)] {
                        p.truth(s1.is_finite() && (0.0..1.0102).contains(&s1), "one-sided");
                    }
                    // The blended pseudocost lies between the column value and the
                    // average (weights in [0, 1]) times the distance.
                    let upd = frac.ceil() - frac;
                    let b = ps.get_pseudocost_up_offset(c, frac, 0.0);
                    let (lo, hi) = if nu == 0 {
                        (ps.cost_total, ps.cost_total)
                    } else {
                        let a = ps.pseudocostup[c];
                        (a.min(ps.cost_total), a.max(ps.cost_total))
                    };
                    p.truth(
                        b >= upd * lo * (1.0 - 1e-12) - 1e-300
                            && b <= upd * hi * (1.0 + 1e-12) + 1e-300,
                        "blended pseudocost in [col, avg]",
                    );
                    // Unblended, reliable or not: the column value or cost_total.
                    let want = if nu == 0 {
                        ps.cost_total
                    } else {
                        ps.pseudocostup[c]
                    };
                    p.bits(ps.get_pseudocost_up(c, frac), upd * want, "getPseudocostUp");
                }
            }
        }
        p.truth(total_steps == 2400, "all steps ran");
        p.finish(5000);
    }

    /// `flushPseudoCost` rescales a worker's conflict increments from the worker's
    /// weight to the master's; `addDeltaAverage` skips empty deltas.
    #[test]
    fn highs_pseudocost_flush_conflict_scaling() {
        let mut p = Probe::new("flush_scaling");
        let mut master = Pseudocost::new(2, 8);
        let mut worker = Pseudocost::new(2, 8);
        for _ in 0..10 {
            master.increase_conflict_weight();
        }
        worker.increase_conflict_weight();
        worker.increase_conflict_score_up(1); // 1.02 in worker units
        let mw = master.conflict_weight;
        master.flush_pseudo_cost(&mut worker);
        let want = (mw / 1.02) * 1.02;
        p.bits(master.conflictscoreup[1], want, "scaled into master units");
        p.bits(master.conflict_avg_score, want + 0.0, "avg");
        p.truth(master.nsamplesup[1] == 0, "empty delta average skipped");
        p.bits(master.pseudocostup[1], 0.0, "unchanged average");
        // addDeltaAverage: avg = (avg * count + sum) / (count + delta_count).
        let mut avg = 3.0;
        let mut cnt = 2i32;
        add_delta_average_i32(&mut avg, &mut cnt, 9.0, 3);
        p.bits(avg, (3.0 * 2.0 + 9.0) / 5.0, "addDeltaAverage");
        p.truth(cnt == 5, "count");
        add_delta_average_i32(&mut avg, &mut cnt, 100.0, 0);
        p.truth(avg == 3.0 && cnt == 5, "delta_count 0 ignored");
        p.finish(7);
    }
}

/// Conflict analysis (`HighsDomain::ConflictSet`) updates the conflict scores.
impl crate::lp::highs_cutgen::domain::ConflictScores for Pseudocost {
    fn increase_conflict_weight(&mut self) {
        Pseudocost::increase_conflict_weight(self);
    }
    fn increase_conflict_score_up(&mut self, col: usize) {
        Pseudocost::increase_conflict_score_up(self, col);
    }
    fn increase_conflict_score_down(&mut self, col: usize) {
        Pseudocost::increase_conflict_score_down(self, col);
    }
}
