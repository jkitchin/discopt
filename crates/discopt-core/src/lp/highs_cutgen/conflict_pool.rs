//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! `HighsConflictPool` (`mip/HighsConflictPool.h/.cpp`, HiGHS 1.15.1): the store of
//! conflicts found by conflict analysis. A conflict is a set of bound literals
//! (`HighsDomainChange`s) that cannot all hold at once; conflicts live in one entry
//! vector, in ranges whose slots and indices are reused after deletion exactly as
//! HiGHS reuses them (free spaces by `(length, start)`, deleted indices last-in
//! first-out), so a conflict index replays HiGHS's. Conflicts age like cuts and are
//! removed past the age limit.
//!
//! As with [`super::pool::CutPool`], the domains that propagate the pool
//! (`propagationDomains`, HiGHS's `ConflictPoolPropagation` pointers) are not owned:
//! each mutating call takes a [`ConflictPoolObserver`] and notifies it where HiGHS
//! notifies its domains, handing over the conflict's entries. The domains report the
//! conflicts their propagation used back through [`ConflictPool::reset_age`]
//! (`Domain::take_conflict_age_resets`).
//!
//! The literals of a conflict cut come from a conflict-analysis frontier of a domain:
//! [`super::domain::Domain::conflict_cut_entries`] and
//! [`super::domain::Domain::reconvergence_cut_entries`] build them (the part of
//! `addConflictCut`/`addReconvergenceCut` that reads the domain), and
//! [`ConflictPool::add_conflict_cut`] stores them (the rest).

use super::domain::{Domain, DomainChange};
use std::collections::BTreeSet;

/// Told about pool changes (HiGHS's `ConflictPoolPropagation::conflictAdded` /
/// `conflictDeleted`, called for every domain in `propagationDomains`, in order).
pub trait ConflictPoolObserver {
    /// Conflict `conflict` was stored with literals `entries`. (HiGHS's observers
    /// read the entries from the pool; they are handed over because the pool is
    /// borrowed during the call.)
    fn conflict_added(&mut self, _conflict: usize, _entries: &[DomainChange]) {}
    /// Conflict `conflict` is being removed.
    fn conflict_deleted(&mut self, _conflict: usize) {}
}

/// A no-op observer.
pub struct NoConflictObserver;
impl ConflictPoolObserver for NoConflictObserver {}

/// The domains propagating one conflict pool, each with the index this pool has
/// among its conflict pools (`addConflictPool`); notified in order.
pub struct ConflictDomainObservers<'a>(pub Vec<(&'a mut Domain, usize)>);

impl ConflictPoolObserver for ConflictDomainObservers<'_> {
    fn conflict_added(&mut self, conflict: usize, entries: &[DomainChange]) {
        for (d, pool) in self.0.iter_mut() {
            d.conflict_added(*pool, conflict, entries);
        }
    }
    fn conflict_deleted(&mut self, conflict: usize) {
        for (d, pool) in self.0.iter_mut() {
            d.conflict_deleted(*pool, conflict);
        }
    }
}

/// `HighsConflictPool`.
#[derive(Clone, Debug)]
pub struct ConflictPool {
    /// `agelim_`.
    agelim: i32,
    /// `softlimit_`.
    softlimit: i32,
    /// `age_lock_`.
    age_lock: bool,
    /// `ageDistribution_`.
    age_distribution: Vec<i64>,
    /// `ages_` (-1 for a deleted conflict).
    ages: Vec<i16>,
    /// `modification_`.
    modification: Vec<u32>,
    /// `ageResetWhileLocked_` (atomics in HiGHS; one thread here).
    age_reset_while_locked: Vec<u8>,
    /// `conflictEntries_`.
    conflict_entries: Vec<DomainChange>,
    /// `conflictRanges_`: `(start, end)`, `(-1, -1)` for a deleted conflict.
    conflict_ranges: Vec<(i64, i64)>,
    /// `freeSpaces_`: `(length, start)`.
    free_spaces: BTreeSet<(i64, i64)>,
    /// `deletedConflicts_`.
    deleted_conflicts: Vec<usize>,
}

/// The placeholder HiGHS's `std::vector::resize` writes into new entry slots.
const EMPTY_ENTRY: DomainChange = DomainChange {
    boundval: 0.0,
    column: 0,
    boundtype: super::domain::BoundType::Lower,
};

impl ConflictPool {
    /// `HighsConflictPool(agelim, softlimit)`.
    pub fn new(agelim: i32, softlimit: i32) -> Self {
        ConflictPool {
            agelim,
            softlimit,
            age_lock: false,
            age_distribution: vec![0; agelim as usize + 1],
            ages: Vec::new(),
            modification: Vec::new(),
            age_reset_while_locked: Vec::new(),
            conflict_entries: Vec::new(),
            conflict_ranges: Vec::new(),
            free_spaces: BTreeSet::new(),
            deleted_conflicts: Vec::new(),
        }
    }

    /// The slot and index search shared by `addConflictCut`, `addReconvergenceCut`
    /// and `addConflictFromOtherPool`: returns `(conflictIndex, start)` with the
    /// age bookkeeping done.
    fn allocate(&mut self, conflict_len: i64) -> (usize, usize) {
        let start;
        let end;
        let slot = self.free_spaces.range((conflict_len, -1)..).next().copied();
        match slot {
            None => {
                start = self.conflict_entries.len() as i64;
                end = start + conflict_len;
                self.conflict_entries.resize(end as usize, EMPTY_ENTRY);
            }
            Some(freeslot) => {
                self.free_spaces.remove(&freeslot);
                start = freeslot.1;
                end = start + conflict_len;
                // The rest of a partly occupied space is registered again.
                if freeslot.0 > conflict_len {
                    self.free_spaces.insert((freeslot.0 - conflict_len, end));
                }
            }
        }
        let conflict_index = match self.deleted_conflicts.pop() {
            None => {
                self.conflict_ranges.push((start, end));
                let n = self.conflict_ranges.len();
                self.ages.resize(n, 0);
                self.modification.resize(n, 0);
                self.age_reset_while_locked.resize(n, 0);
                n - 1
            }
            Some(c) => {
                self.conflict_ranges[c] = (start, end);
                c
            }
        };
        self.age_reset_while_locked[conflict_index] = 0;
        self.modification[conflict_index] = self.modification[conflict_index].wrapping_add(1);
        self.ages[conflict_index] = 0;
        self.age_distribution[self.ages[conflict_index] as usize] += 1;
        (conflict_index, start as usize)
    }

    /// `addConflictCut(domain, reasonSideFrontier)`, with the frontier's literals
    /// already taken from the domain ([`Domain::conflict_cut_entries`]). Returns the
    /// conflict's index.
    pub fn add_conflict_cut(
        &mut self,
        entries: &[DomainChange],
        obs: &mut dyn ConflictPoolObserver,
    ) -> usize {
        self.store(entries, obs)
    }

    /// `addReconvergenceCut(domain, reconvergenceFrontier, reconvergenceDomchg)`,
    /// with the literals (the flipped reconvergence change first) already taken from
    /// the domain ([`Domain::reconvergence_cut_entries`]).
    pub fn add_reconvergence_cut(
        &mut self,
        entries: &[DomainChange],
        obs: &mut dyn ConflictPoolObserver,
    ) -> usize {
        self.store(entries, obs)
    }

    /// `addConflictFromOtherPool(conflictEntries, conflictLen)`.
    pub fn add_conflict_from_other_pool(
        &mut self,
        entries: &[DomainChange],
        obs: &mut dyn ConflictPoolObserver,
    ) -> usize {
        self.store(entries, obs)
    }

    fn store(&mut self, entries: &[DomainChange], obs: &mut dyn ConflictPoolObserver) -> usize {
        let (conflict_index, start) = self.allocate(entries.len() as i64);
        self.conflict_entries[start..start + entries.len()].copy_from_slice(entries);
        obs.conflict_added(conflict_index, entries);
        conflict_index
    }

    /// `removeConflict(conflict)`.
    pub fn remove_conflict(&mut self, conflict: usize, obs: &mut dyn ConflictPoolObserver) {
        obs.conflict_deleted(conflict);
        if self.ages[conflict] >= 0 {
            self.age_distribution[self.ages[conflict] as usize] -= 1;
            self.ages[conflict] = -1;
        }
        let (start, end) = self.conflict_ranges[conflict];
        // The space and the index are registered for reuse.
        self.deleted_conflicts.push(conflict);
        self.free_spaces.insert((end - start, start));
        self.conflict_ranges[conflict] = (-1, -1);
        self.modification[conflict] = self.modification[conflict].wrapping_add(1);
    }

    /// `performAging(thread_safe)`.
    pub fn perform_aging(&mut self, thread_safe: bool, obs: &mut dyn ConflictPoolObserver) {
        if self.age_lock {
            return;
        }
        let conflict_max_index = self.conflict_ranges.len();
        let mut agelim = self.agelim;
        let mut num_active_conflicts = self.num_conflicts() as i64;
        while agelim > 5 && num_active_conflicts > self.softlimit as i64 {
            num_active_conflicts -= self.age_distribution[agelim as usize];
            agelim -= 1;
        }
        for i in 0..conflict_max_index {
            if self.ages[i] < 0 {
                continue;
            }
            if thread_safe && self.age_reset_while_locked[i] == 1 {
                self.reset_age(i);
            }
            self.age_distribution[self.ages[i] as usize] -= 1;
            self.ages[i] += 1;
            self.age_reset_while_locked[i] = 0;
            if self.ages[i] as i32 > agelim {
                self.ages[i] = -1;
                self.remove_conflict(i, obs);
            } else {
                self.age_distribution[self.ages[i] as usize] += 1;
            }
        }
    }

    /// `syncConflictPool(syncpool)`: move every live conflict into `syncpool`
    /// (notifying `sync_obs`, the other pool's domains) and empty this pool
    /// (notifying `obs`).
    pub fn sync_conflict_pool(
        &mut self,
        syncpool: &mut ConflictPool,
        obs: &mut dyn ConflictPoolObserver,
        sync_obs: &mut dyn ConflictPoolObserver,
    ) {
        let conflict_max_index = self.conflict_ranges.len();
        for i in 0..conflict_max_index {
            if self.ages[i] < 0 {
                continue;
            }
            let (start, end) = self.conflict_ranges[i];
            debug_assert!(start >= 0 && end >= 0);
            let entries = self.conflict_entries[start as usize..end as usize].to_vec();
            syncpool.add_conflict_from_other_pool(&entries, sync_obs);
            self.remove_conflict(i, obs);
        }
        self.deleted_conflicts.clear();
        self.free_spaces.clear();
        self.conflict_ranges.clear();
        self.conflict_entries.clear();
        self.modification.clear();
        self.ages.clear();
        self.age_reset_while_locked.clear();
    }

    /// `resetAge(conflict)`.
    pub fn reset_age(&mut self, conflict: usize) {
        if self.ages[conflict] > 0 {
            if self.age_lock {
                self.age_reset_while_locked[conflict] = 1;
                return;
            }
            self.age_distribution[self.ages[conflict] as usize] -= 1;
            self.age_distribution[0] += 1;
            self.ages[conflict] = 0;
        }
    }

    /// `setAgeLimit(agelim)`.
    pub fn set_age_limit(&mut self, agelim: i32) {
        self.agelim = agelim;
        self.age_distribution.resize(agelim as usize + 1, 0);
    }

    /// `setAgeLock(ageLock)`.
    pub fn set_age_lock(&mut self, age_lock: bool) {
        self.age_lock = age_lock;
    }

    /// `getModificationCount(cut)`.
    pub fn modification_count(&self, conflict: usize) -> u32 {
        self.modification[conflict]
    }

    /// `getConflictEntryVector()`.
    pub fn conflict_entry_vector(&self) -> &[DomainChange] {
        &self.conflict_entries
    }

    /// `getConflictRanges()`.
    pub fn conflict_ranges(&self) -> &[(i64, i64)] {
        &self.conflict_ranges
    }

    /// `getNumConflicts()`.
    pub fn num_conflicts(&self) -> usize {
        self.conflict_ranges.len() - self.deleted_conflicts.len()
    }

    /// The literals of conflict `conflict`, `None` once it is deleted.
    pub fn conflict(&self, conflict: usize) -> Option<&[DomainChange]> {
        let (start, end) = self.conflict_ranges[conflict];
        if start < 0 {
            return None;
        }
        Some(&self.conflict_entries[start as usize..end as usize])
    }

    /// The age of a conflict (-1 once deleted).
    pub fn age(&self, conflict: usize) -> i16 {
        self.ages[conflict]
    }

    /// `ageDistribution_`, for checks.
    pub fn age_distribution(&self) -> &[i64] {
        &self.age_distribution
    }
}

#[cfg(test)]
mod tests {
    use super::super::domain::BoundType;
    use super::*;

    fn lit(col: usize, lower: bool, v: f64) -> DomainChange {
        DomainChange {
            boundval: v,
            column: col,
            boundtype: if lower {
                BoundType::Lower
            } else {
                BoundType::Upper
            },
        }
    }

    /// Records the notifications, in order.
    #[derive(Default)]
    struct Log(Vec<(char, usize, usize)>);
    impl ConflictPoolObserver for Log {
        fn conflict_added(&mut self, c: usize, e: &[DomainChange]) {
            self.0.push(('a', c, e.len()));
        }
        fn conflict_deleted(&mut self, c: usize) {
            self.0.push(('d', c, 0));
        }
    }

    fn check_distribution(p: &ConflictPool) {
        let mut dist = vec![0i64; p.age_distribution().len()];
        for c in 0..p.conflict_ranges().len() {
            if p.age(c) >= 0 {
                dist[p.age(c) as usize] += 1;
            }
        }
        assert_eq!(dist, p.age_distribution(), "age distribution out of sync");
    }

    #[test]
    fn slots_and_indices_are_reused_as_in_highs() {
        let mut p = ConflictPool::new(10, 100);
        let mut log = Log::default();
        let a = [lit(0, true, 1.0), lit(1, false, 0.0), lit(2, true, 1.0)];
        let b = [lit(3, true, 1.0)];
        let c0 = p.add_conflict_cut(&a, &mut log);
        let c1 = p.add_reconvergence_cut(&b, &mut log);
        assert_eq!((c0, c1), (0, 1));
        assert_eq!(p.conflict_ranges(), &[(0, 3), (3, 4)]);
        assert_eq!(p.num_conflicts(), 2);
        assert_eq!(p.modification_count(0), 1);
        p.remove_conflict(c0, &mut log);
        assert_eq!(p.conflict(c0), None);
        assert_eq!(p.num_conflicts(), 1);
        assert_eq!(p.modification_count(0), 2);
        // A shorter conflict takes index 0 back and the start of its space; the
        // remaining (1, 2) is free again.
        let c2 = p.add_conflict_cut(&[lit(4, true, 2.0), lit(5, false, 1.0)], &mut log);
        assert_eq!(c2, 0);
        assert_eq!(p.conflict_ranges()[0], (0, 2));
        assert_eq!(
            p.free_spaces.iter().copied().collect::<Vec<_>>(),
            vec![(1, 2)]
        );
        // A longer one appends.
        let c3 = p.add_conflict_cut(&a, &mut log);
        assert_eq!((c3, p.conflict_ranges()[c3]), (2, (4, 7)));
        assert_eq!(p.conflict(c3).unwrap(), &a);
        assert_eq!(
            log.0,
            vec![
                ('a', 0, 3),
                ('a', 1, 1),
                ('d', 0, 0),
                ('a', 0, 2),
                ('a', 2, 3)
            ]
        );
        check_distribution(&p);
    }

    #[test]
    fn aging_removes_old_conflicts_and_reset_keeps_them() {
        let mut p = ConflictPool::new(8, 1000);
        let mut log = Log::default();
        for k in 0..4 {
            p.add_conflict_cut(&[lit(k, true, 1.0)], &mut log);
        }
        let (mut removed, mut agings) = (0usize, 0usize);
        for round in 0..12 {
            // Conflict 0 is used by propagation every round; the others age.
            p.reset_age(0);
            p.perform_aging(false, &mut log);
            agings += 1;
            check_distribution(&p);
            if round < 7 {
                assert_eq!(p.num_conflicts(), 4, "round {round}");
            }
        }
        for &(k, c, _) in &log.0 {
            if k == 'd' {
                removed += 1;
                assert_ne!(c, 0, "the conflict in use was removed");
            }
        }
        assert_eq!(removed, 3);
        assert_eq!(p.num_conflicts(), 1);
        assert_eq!(p.age(0), 1);
        assert!(agings == 12);
        // Over the soft limit, the age limit drops (but not below 5) for one call.
        let mut q = ConflictPool::new(10, 2);
        for k in 0..6 {
            q.add_conflict_cut(&[lit(k, false, 0.0)], &mut log);
        }
        for _ in 0..6 {
            q.perform_aging(false, &mut log);
        }
        // Six calls take the conflicts to age 6 > 5: all removed.
        assert_eq!(q.num_conflicts(), 0);
        check_distribution(&q);
        // The age lock defers resets and freezes aging.
        let mut r = ConflictPool::new(10, 100);
        let c = r.add_conflict_cut(&[lit(0, true, 1.0)], &mut log);
        r.perform_aging(false, &mut log);
        r.set_age_lock(true);
        r.reset_age(c);
        assert_eq!(r.age(c), 1);
        r.perform_aging(false, &mut log);
        assert_eq!(r.age(c), 1);
        r.set_age_lock(false);
        r.perform_aging(true, &mut log);
        // The deferred reset applies first, then the conflict ages once.
        assert_eq!(r.age(c), 1);
        check_distribution(&r);
    }

    #[test]
    fn sync_moves_every_live_conflict() {
        let mut p = ConflictPool::new(10, 100);
        let mut q = ConflictPool::new(10, 100);
        let (mut lp, mut lq) = (Log::default(), Log::default());
        p.add_conflict_cut(&[lit(0, true, 1.0)], &mut lp);
        let c = p.add_conflict_cut(&[lit(1, true, 1.0), lit(2, false, 0.0)], &mut lp);
        p.add_conflict_cut(&[lit(3, true, 1.0)], &mut lp);
        p.remove_conflict(c, &mut lp);
        p.sync_conflict_pool(&mut q, &mut lp, &mut lq);
        assert_eq!(p.num_conflicts(), 0);
        assert!(p.conflict_ranges().is_empty());
        assert_eq!(q.num_conflicts(), 2);
        assert_eq!(q.conflict(0).unwrap(), &[lit(0, true, 1.0)]);
        assert_eq!(q.conflict(1).unwrap(), &[lit(3, true, 1.0)]);
        assert_eq!(lq.0, vec![('a', 0, 1), ('a', 1, 1)]);
    }
}
