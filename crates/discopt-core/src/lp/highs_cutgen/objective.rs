//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! `HighsObjectiveFunction`: the objective's nonzero columns ordered integral first and
//! binaries first among those, with the optional clique partition of the objective
//! binaries that the domain's objective propagation uses.
//!
//! The clique partition itself (`HighsCliqueTable::cliquePartition`) belongs to the
//! clique table, which is not ported yet; [`ObjectiveFunction::setup_clique_partition`]
//! takes its output.

use super::random::hash_pair_i32;
use super::stdalgo::pdqsort;

/// In-place two-ended partition: the elements satisfying `pred` move to the front;
/// returns their count. Scans from the front for the first element failing `pred`
/// and from the back for the last one satisfying it, and swaps them, until the scans
/// meet. Independent implementation of the classic algorithm; HiGHS calls
/// `std::partition`, whose result order (libstdc++) it reproduces.
pub fn partition_two_ended<T, P: FnMut(&T) -> bool>(v: &mut [T], mut pred: P) -> usize {
    let (mut first, mut last) = (0usize, v.len());
    loop {
        loop {
            if first == last {
                return first;
            }
            if pred(&v[first]) {
                first += 1;
            } else {
                break;
            }
        }
        last -= 1;
        loop {
            if first == last {
                return first;
            }
            if !pred(&v[last]) {
                last -= 1;
            } else {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `HighsHashHelpers::hash(HighsInt)`.
fn hash_int(i: i32) -> u64 {
    hash_pair_i32(i, 0)
}

/// `HighsObjectiveFunction`.
#[derive(Clone, Debug)]
pub struct ObjectiveFunction {
    /// `model_->col_cost_`, one per column.
    pub cost: Vec<f64>,
    nonzeros: Vec<usize>,
    vals: Vec<f64>,
    num_integral: usize,
    num_binary: usize,
    partition_start: Vec<usize>,
    col_to_partition: Vec<i64>,
}

impl ObjectiveFunction {
    /// `HighsObjectiveFunction(mipsolver)`: `is_int`, `lb`, `ub` are the model's.
    pub fn new(cost: &[f64], is_int: &[bool], lb: &[f64], ub: &[f64]) -> Self {
        let n = cost.len();
        let mut nonzeros: Vec<usize> = (0..n).filter(|&i| cost[i] != 0.0).collect();
        let mut f = ObjectiveFunction {
            cost: cost.to_vec(),
            nonzeros: Vec::new(),
            vals: Vec::new(),
            num_integral: 0,
            num_binary: 0,
            partition_start: vec![0],
            col_to_partition: vec![-1; n],
        };
        if nonzeros.is_empty() {
            return f;
        }
        f.num_integral = partition_two_ended(&mut nonzeros, |&i| is_int[i]);
        if f.num_integral != 0 {
            f.num_binary = partition_two_ended(&mut nonzeros[..f.num_integral], |&i| {
                lb[i] == 0.0 && ub[i] == 1.0
            });
        }
        f.vals = nonzeros.iter().map(|&i| cost[i]).collect();
        f.nonzeros = nonzeros;
        f
    }

    /// `setupCliquePartition(globaldom, cliqueTable)`, given the output of
    /// `cliqueTable.cliquePartition(cost, clqvars, cliquePartitionStart)`: the
    /// objective binaries in partition order (`clqvars`) and the partition starts.
    pub fn setup_clique_partition(&mut self, clqvars: &[usize], starts: &[usize]) {
        if self.num_binary <= 1 {
            return;
        }
        assert_eq!(clqvars.len(), self.num_binary);
        let num_partitions = starts.len() - 1;
        if num_partitions == self.num_binary {
            self.partition_start = vec![0];
            return;
        }
        let mut ps = Vec::new();
        let mut k = 0usize;
        for i in 0..num_partitions {
            if starts[i + 1] - starts[i] == 1 {
                continue;
            }
            ps.push(k);
            for &col in &clqvars[starts[i]..starts[i + 1]] {
                self.col_to_partition[col] = k as i64;
                k += 1;
            }
        }
        ps.push(k);
        self.partition_start = ps;
        let ctp = &self.col_to_partition;
        let key = |i: usize| (ctp[i] as u32, hash_int(i as i32));
        pdqsort(&mut self.nonzeros[..self.num_binary], |&a, &b| {
            key(a) < key(b)
        });
        for i in 0..self.num_binary {
            self.vals[i] = self.cost[self.nonzeros[i]];
        }
    }

    /// `getObjectiveNonzeros()`.
    pub fn nonzeros(&self) -> &[usize] {
        &self.nonzeros
    }

    /// `getObjectiveValuesPacked()`.
    pub fn values_packed(&self) -> &[f64] {
        &self.vals
    }

    /// `getNumBinariesInObjective()`.
    pub fn num_binary(&self) -> usize {
        self.num_binary
    }

    /// Integral objective columns (binaries included).
    pub fn num_integral(&self) -> usize {
        self.num_integral
    }

    /// `getCliquePartitionStarts()`.
    pub fn partition_starts(&self) -> &[usize] {
        &self.partition_start
    }

    /// `getNumCliquePartitions()`.
    pub fn num_partitions(&self) -> usize {
        self.partition_start.len() - 1
    }

    /// `getColCliquePartition(col)`: the column's position in [`Self::nonzeros`] when
    /// it is in a clique partition, else `-1`.
    pub fn col_partition_pos(&self, col: usize) -> i64 {
        self.col_to_partition[col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_ended_partition_matches_the_reference_order() {
        // Reference outputs of std::partition (libstdc++) on these inputs.
        let mut v = vec![1, 2, 3, 4, 5, 6, 7];
        let k = partition_two_ended(&mut v, |&x| x % 2 == 0);
        assert_eq!(k, 3);
        assert_eq!(v, vec![6, 2, 4, 3, 5, 1, 7]);
    }

    #[test]
    fn partition_layout_puts_partition_members_first_at_their_positions() {
        // Columns 0..6: 0,1,2,3 binary; 4 integer; 5 continuous; 6 zero cost.
        let cost = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 0.0];
        let is_int = [true, true, true, true, true, false, true];
        let lb = [0.0; 7];
        let ub = [1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 1.0];
        let mut f = ObjectiveFunction::new(&cost, &is_int, &lb, &ub);
        assert_eq!(f.num_binary(), 4);
        assert_eq!(f.num_integral(), 5);
        assert_eq!(f.nonzeros().len(), 6);
        // Partitions {2,0} and {1} and {3}: only the first has more than one member.
        f.setup_clique_partition(&[2, 0, 1, 3], &[0, 2, 3, 4]);
        assert_eq!(f.partition_starts(), &[0, 2]);
        assert_eq!(f.col_partition_pos(2), 0);
        assert_eq!(f.col_partition_pos(0), 1);
        assert_eq!(f.col_partition_pos(1), -1);
        assert_eq!(&f.nonzeros()[..2], &[2, 0]);
        for (j, &c) in f.nonzeros().iter().enumerate() {
            assert_eq!(f.values_packed()[j], cost[c]);
        }
    }
}
