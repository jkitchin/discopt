//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! `HighsCliqueTable` (`mip/HighsCliqueTable.{h,cpp}`, HiGHS 1.15.1): the conflict graph
//! over binary literals, stored as cliques. A literal is a [`CliqueVar`] `(col, val)`:
//! `val = 1` is `x_col`, `val = 0` is `1 - x_col`; a clique says at most one of its
//! literals is 1 at every feasible point (an *equality* clique: exactly one).
//!
//! Ported: storage (`cliqueentries`, the per-literal clique sets, the size-two edge
//! table, free slots and free spaces), `doAddClique`, `addClique` (fixings when a member
//! is fixed to 1, new-edge processing, covers, equality cliques of size two and the
//! substitutions they create), `removeClique`, `findCommonCliqueId`, `haveCommonClique`,
//! `findCommonClique`, `resolveSubstitution` (both), the neighbourhood queries,
//! `fixCol`, `processNewEdge`, `foundCover`, `processInfeasibleVertices`,
//! `propagateAndCleanup`, `vertexInfeasible`, `cleanupFixed`, `addImplications` (the
//! fixings `HighsDomain::changeBound` asks of the table when it fixes a binary),
//! `getNumImplications` (both), `cliquePartition` (both), `extractCliques` (model rows,
//! with `transformRows`), `extractCliquesFromCut`, `extractObjCliques`,
//! `runCliqueSubsumption`/`cliqueSubsumption`/`collectCliques`, `runCliqueMerging`
//! (both), `separateCliques` with `bronKerboschRecurse`, `computeMaximalCliques`,
//! `rebuild` and `buildFrom`. The VUB/VLB halves of the extraction call
//! [`super::vbound::VarBoundStore`], which already ports them.
//!
//! Interfaces instead of `HighsMipSolver`/`HighsDomain`: [`CliqueDomain`] (the global
//! or local domain: bounds, integrality, `changeBound`, `fixCol`, `propagate`, the
//! domain change stack) and [`CliqueRows`] (the model rows). [`GlobalDomainView`]
//! implements [`CliqueDomain`] over the translated [`super::domain::Domain`].
//!
//! Not ported, and why:
//! - `separateCliques(sol, globaldom, feastol)` (the three-argument overload): its body
//!   in HiGHS 1.15.1 is `exit(0)` followed by `#if 0` code.
//! - The node-pruning block inside `addClique`'s new-edge loop: it looks up
//!   `nodequeue.getUpNodes(col)` (a `std::set<std::pair<double, int64_t>>`) between
//!   `lower_bound({val, kHighsIInf})` and `upper_bound({val, kHighsIInf})`. Both return
//!   the first element after every `(val, node)` with `node < kHighsIInf`, so the range
//!   is empty for every node id below 2^31 - 1 and the block never prunes a node.
//! - The parallel branch of `queryNeighbourhood`: it collects the same indices and sorts
//!   them, so its result equals the sequential loop's (ported); `allowParallel` and
//!   `minEntriesForParallelism` are therefore not kept.
//! - `debugSolution.checkClique` (a debug-build check).
//!
//! Not faithful, recorded:
//! - Iteration order. HiGHS keeps each literal's cliques in a `HighsHashTree`, whose
//!   `for_each` and `find_common` orders are those of its hash layout. Here they are
//!   `BTreeMap`/`BTreeSet`s, iterated in ascending clique id, and `find_common` returns
//!   the smallest common clique id. This changes the order of fixings and removals in
//!   `processInfeasibleVertices`, `collectCliques` (hence subsumption removals),
//!   `addImplications`, `runCliqueMerging`'s candidate order, and which common clique
//!   `foundCover` processes first. The resulting table is a valid clique table either
//!   way; it is not bit-identical to HiGHS's.
//! - `extractCliques` merges a row's entries in a `HighsHashTable<HighsInt, double>`
//!   and iterates it in hash order; here the merged entries keep first-occurrence order.
//! - Re-entrancy. HiGHS's global domain calls `cliquetable.addImplications` from inside
//!   `changeBound` whenever it fixes a binary, including while the table is itself
//!   fixing columns (`fixCol`, `processInfeasibleVertices`). Here the table holds
//!   `&mut self` during those operations and the domain cannot call back into it.
//!   The table's own processing covers the same fixings for the columns it fixes
//!   (`processInfeasibleVertices` handles each one), and `propagateAndCleanup` those
//!   that propagation makes after it starts; fixings made by the domain's propagation
//!   inside an earlier `fixCol` are not revisited by the table, so it may imply fewer
//!   fixings than HiGHS. Fewer, never wrong: every fixing here is implied by a valid
//!   clique.
//! - `extractCliquesFromCut` computes `minact` before tightening integer bounds and
//!   uses that value for the VUB/VLB half. [`VarBoundStore::extract_from_cut`] recomputes
//!   it on the tightened bounds; tightening an upper bound of a positive-coefficient
//!   column (or a lower bound of a negative one) does not change the minimum activity,
//!   so the two agree unless the domain changes other bounds from inside `changeBound`.
//! - The C++ `std::partition` is reproduced with
//!   [`super::objective::partition_two_ended`] (libstdc++'s order); the VUB/VLB half
//!   keeps `vbound.rs`'s own partition, as before.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::cdouble::CDouble;
use super::domain::{BoundType, Domain, DomainChange, Reason};
use super::objective::partition_two_ended;
use super::random::HighsRandom;
use super::stdalgo::{pdqsort, pdqsort_branchless};
use super::vbound::{self, VarBoundStore};

/// `kHighsIInf` for 32-bit `HighsInt`.
pub const K_HIGHS_IINF: i32 = i32::MAX;
const INF: f64 = f64::INFINITY;

/// `HighsCliqueTable::CliqueVar`: literal `x_col` (`val = 1`) or `1 - x_col` (`val = 0`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CliqueVar {
    /// `col`.
    pub col: u32,
    /// `val`, 0 or 1.
    pub val: u8,
}

impl CliqueVar {
    /// `CliqueVar(col, val)`.
    pub fn new(col: usize, val: u8) -> Self {
        debug_assert!(val <= 1);
        CliqueVar {
            col: col as u32,
            val,
        }
    }
    /// The column as an index.
    #[inline]
    pub fn c(self) -> usize {
        self.col as usize
    }
    /// `index()`: `2 col + val`.
    #[inline]
    pub fn index(self) -> usize {
        2 * self.col as usize + self.val as usize
    }
    /// `weight(sol)`: the literal's value at `sol`.
    #[inline]
    pub fn weight(self, sol: &[f64]) -> f64 {
        if self.val == 1 {
            sol[self.c()]
        } else {
            1.0 - sol[self.c()]
        }
    }
    /// `complement()`.
    #[inline]
    pub fn complement(self) -> Self {
        CliqueVar {
            col: self.col,
            val: 1 - self.val,
        }
    }
    /// The literal's value at a point (`x[col]` rounded, complemented when `val = 0`).
    pub fn value_at(self, x: &[f64]) -> bool {
        (x[self.c()] > 0.5) == (self.val == 1)
    }
}

/// `HighsCliqueTable::Clique`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Clique {
    /// `start` (-1 when the slot is free).
    pub start: i32,
    /// `end`.
    pub end: i32,
    /// `origin`: the model row the clique came from, `-1` for a clique derived from
    /// model rows by merging or rebuilding, `kHighsIInf` for none.
    pub origin: i32,
    /// `numZeroFixed`.
    pub num_zero_fixed: i32,
    /// `equality`.
    pub equality: bool,
}

impl Clique {
    /// `numActive()`.
    pub fn num_active(&self) -> i32 {
        self.end - self.start - self.num_zero_fixed
    }
}

/// `HighsCliqueTable::Substitution`: column `substcol` equals literal `replace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Substitution {
    /// `substcol`.
    pub substcol: usize,
    /// `replace`.
    pub replace: CliqueVar,
}

/// The reason the table gives a domain for a bound change (`HighsDomain::Reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliqueReason {
    /// `Reason::unspecified()`.
    Unspecified,
    /// `Reason::cliqueTable(col, val)`: implied by literal `(col, val)` being 1;
    /// the payload is `2 col + val`, as in HiGHS.
    CliqueTable(usize),
}

/// What the clique table reads from and does to a `HighsDomain`.
pub trait CliqueDomain {
    /// `col_lower_`.
    fn col_lower(&self) -> &[f64];
    /// `col_upper_`.
    fn col_upper(&self) -> &[f64];
    /// `mipsolver->isColIntegral(col)` for every column.
    fn integrality(&self) -> &[bool];
    /// `infeasible()`.
    fn infeasible(&self) -> bool;
    /// `feastol()`.
    fn feastol(&self) -> f64;
    /// `changeBound({val, col, boundtype}, reason)`.
    fn change_bound(&mut self, boundtype: BoundType, col: usize, val: f64, reason: CliqueReason);
    /// `propagate()`.
    fn propagate(&mut self);
    /// `getDomainChangeStack().size()`.
    fn domain_change_stack_len(&self) -> usize;
    /// `getDomainChangeStack()[pos].column`.
    fn domain_change_col(&self, pos: usize) -> usize;
    /// `numModelNonzeros()`.
    fn num_model_nonzeros(&self) -> usize;

    /// `isBinary(col)`.
    fn is_binary(&self, col: usize) -> bool {
        self.integrality()[col] && self.col_lower()[col] == 0.0 && self.col_upper()[col] == 1.0
    }
    /// `isFixed(col)`.
    fn is_fixed(&self, col: usize) -> bool {
        self.col_lower()[col] == self.col_upper()[col]
    }
    /// `fixCol(col, val)` with `Reason::unspecified()`.
    fn fix_col(&mut self, col: usize, val: f64) {
        if self.col_lower()[col] < val {
            self.change_bound(BoundType::Lower, col, val, CliqueReason::Unspecified);
            if !self.infeasible() {
                self.propagate();
            }
        }
        if !self.infeasible() && self.col_upper()[col] > val {
            self.change_bound(BoundType::Upper, col, val, CliqueReason::Unspecified);
        }
    }
}

/// [`CliqueDomain`] over the translated `HighsDomain`; a clique-table reason becomes
/// the domain's `Reason::CliqueTable(col, val)`, which conflict analysis explains as
/// the fixing of `col` (an infeasibility from it stays unexplained, as in HiGHS).
pub struct GlobalDomainView<'a> {
    /// The domain.
    pub dom: &'a mut Domain,
    /// Column integrality (the domain's model).
    pub is_int: &'a [bool],
    /// `feastol()`.
    pub feastol: f64,
    /// `numModelNonzeros()`.
    pub num_model_nonzeros: usize,
}

impl CliqueDomain for GlobalDomainView<'_> {
    fn col_lower(&self) -> &[f64] {
        &self.dom.col_lower
    }
    fn col_upper(&self) -> &[f64] {
        &self.dom.col_upper
    }
    fn integrality(&self) -> &[bool] {
        self.is_int
    }
    fn infeasible(&self) -> bool {
        self.dom.infeasible()
    }
    fn feastol(&self) -> f64 {
        self.feastol
    }
    fn change_bound(&mut self, boundtype: BoundType, col: usize, val: f64, reason: CliqueReason) {
        let reason = match reason {
            CliqueReason::Unspecified => Reason::Unspecified,
            // HiGHS's `Reason::cliqueTable(col, val)` index `2 * col + val`.
            CliqueReason::CliqueTable(code) => Reason::CliqueTable(code >> 1, code & 1),
        };
        self.dom.change_bound(
            DomainChange {
                boundval: val,
                column: col,
                boundtype,
            },
            reason,
        );
    }
    fn propagate(&mut self) {
        self.dom.propagate();
    }
    fn domain_change_stack_len(&self) -> usize {
        self.dom.domain_change_stack().len()
    }
    fn domain_change_col(&self, pos: usize) -> usize {
        self.dom.domain_change_stack()[pos].column
    }
    fn num_model_nonzeros(&self) -> usize {
        self.num_model_nonzeros
    }
}

/// The model rows `extractCliques` reads (`mipdata_->ARstart_/ARindex_/ARvalue_`,
/// `rowLower`/`rowUpper`), with the tolerances it uses.
pub struct CliqueRows<'a> {
    /// Row starts, length `nrows + 1`.
    pub ar_start: &'a [usize],
    /// Column per entry.
    pub ar_index: &'a [usize],
    /// Value per entry.
    pub ar_value: &'a [f64],
    /// Row lower sides (`-inf` when absent).
    pub row_lower: &'a [f64],
    /// Row upper sides (`+inf` when absent).
    pub row_upper: &'a [f64],
    /// Rows of the original model: HiGHS stops at the first row whose original index
    /// (`postSolveStack.getOrigRowIndex`) is not below `orig_model_->num_row_`.
    pub num_orig_rows: usize,
    /// `mipdata_->feastol`.
    pub feastol: f64,
    /// `mipdata_->epsilon`.
    pub epsilon: f64,
}

/// A clique cut `sum vals x[inds] <= rhs` from [`CliqueTable::separate_cliques`], for
/// the caller to add to its cut pool (`cutpool.addCut(..., rhs, true, false, false)`).
#[derive(Clone, Debug, PartialEq)]
pub struct CliqueCut {
    /// Columns.
    pub inds: Vec<usize>,
    /// Coefficients (+1 or -1).
    pub vals: Vec<f64>,
    /// Right-hand side.
    pub rhs: f64,
}

/// `sortedEdge(v1, v2)` as a key of literal indices.
#[inline]
fn sorted_edge(v1: CliqueVar, v2: CliqueVar) -> (usize, usize) {
    if v1.col > v2.col {
        (v2.index(), v1.index())
    } else {
        (v1.index(), v2.index())
    }
}

/// `std::make_pair(a0, a1) > std::make_pair(b0, b1)` for a double first member.
#[inline]
fn pair_gt(a: (f64, usize), b: (f64, usize)) -> bool {
    b.0 < a.0 || (!(a.0 < b.0) && b.1 < a.1)
}

/// `activityContributionMin(val, lb, ub)`.
fn activity_contribution_min(val: f64, lb: f64, ub: f64) -> f64 {
    if val < 0.0 {
        if ub == INF {
            -INF
        } else {
            val * ub
        }
    } else if lb == -INF {
        -INF
    } else {
        val * lb
    }
}

/// `BronKerboschData`.
struct BkData<'a> {
    sol: &'a [f64],
    p: Vec<CliqueVar>,
    r: Vec<CliqueVar>,
    z: Vec<CliqueVar>,
    cliques: Vec<Vec<CliqueVar>>,
    neighbourhood_inds: Vec<usize>,
    w_r: f64,
    min_w: f64,
    feastol: f64,
    ncalls: i32,
    maxcalls: i32,
    maxcliques: i32,
    max_neighbourhood_queries: i64,
    num_neighbourhood_queries: i64,
}

impl<'a> BkData<'a> {
    fn new(sol: &'a [f64]) -> Self {
        BkData {
            sol,
            p: Vec::new(),
            r: Vec::new(),
            z: Vec::new(),
            cliques: Vec::new(),
            neighbourhood_inds: Vec::new(),
            w_r: 0.0,
            min_w: 1.05,
            feastol: 1e-6,
            ncalls: 0,
            maxcalls: 10000,
            maxcliques: 100,
            max_neighbourhood_queries: i64::MAX,
            num_neighbourhood_queries: 0,
        }
    }
    fn stop(&self) -> bool {
        self.maxcalls == self.ncalls
            || self.cliques.len() as i32 == self.maxcliques
            || self.num_neighbourhood_queries > self.max_neighbourhood_queries
    }
}

/// `HighsCliqueTable`.
#[derive(Clone, Debug)]
pub struct CliqueTable {
    cliqueentries: Vec<CliqueVar>,
    /// `invertedHashList`: per literal, clique id -> entry position (cliques of size
    /// three or more).
    inverted: Vec<BTreeMap<i32, i32>>,
    /// `invertedHashListSizeTwo`: per literal, its cliques of size two.
    inverted_size_two: Vec<BTreeSet<i32>>,
    /// `sizeTwoCliques`: sorted edge -> clique id.
    size_two_cliques: HashMap<(usize, usize), i32>,
    /// `freespaces`: (length, start) of reusable entry ranges.
    freespaces: BTreeSet<(i32, i32)>,
    freeslots: Vec<i32>,
    cliques: Vec<Clique>,
    numcliquesvar: Vec<i32>,
    infeasvertexstack: Vec<CliqueVar>,
    colsubstituted: Vec<i32>,
    substitutions: Vec<Substitution>,
    deletedrows: Vec<i32>,
    cliqueextensions: Vec<(i32, CliqueVar)>,
    iscandidate: Vec<bool>,
    col_deleted: Vec<bool>,
    cliquehits: Vec<u32>,
    cliquehitinds: Vec<i32>,
    randgen: HighsRandom,
    nfixings: i32,
    num_entries: i32,
    max_entries: i32,
    in_presolve: bool,
    /// `mipdata_->numCliqueEntriesAfterPresolve` (read by `extractCliquesFromCut`).
    num_entries_after_presolve: i32,
    /// `numNeighbourhoodQueries`.
    pub num_neighbourhood_queries: i64,
}

impl CliqueTable {
    /// `HighsCliqueTable(ncols)`.
    pub fn new(ncols: usize) -> Self {
        CliqueTable {
            cliqueentries: Vec::new(),
            inverted: vec![BTreeMap::new(); 2 * ncols],
            inverted_size_two: vec![BTreeSet::new(); 2 * ncols],
            size_two_cliques: HashMap::new(),
            freespaces: BTreeSet::new(),
            freeslots: Vec::new(),
            cliques: Vec::new(),
            numcliquesvar: vec![0; 2 * ncols],
            infeasvertexstack: Vec::new(),
            colsubstituted: vec![0; ncols],
            substitutions: Vec::new(),
            deletedrows: Vec::new(),
            cliqueextensions: Vec::new(),
            iscandidate: Vec::new(),
            col_deleted: vec![false; ncols],
            cliquehits: Vec::new(),
            cliquehitinds: Vec::new(),
            randgen: HighsRandom::new(0),
            nfixings: 0,
            num_entries: 0,
            max_entries: K_HIGHS_IINF,
            in_presolve: false,
            num_entries_after_presolve: 0,
            num_neighbourhood_queries: 0,
        }
    }

    // ---- small accessors (header inlines) ----

    /// `setPresolveFlag`.
    pub fn set_presolve_flag(&mut self, in_presolve: bool) {
        self.in_presolve = in_presolve;
    }
    /// `getPresolveFlag`.
    pub fn presolve_flag(&self) -> bool {
        self.in_presolve
    }
    /// `getNumEntries`.
    pub fn num_entries(&self) -> i32 {
        self.num_entries
    }
    /// `getRandgen`.
    pub fn randgen_mut(&mut self) -> &mut HighsRandom {
        &mut self.randgen
    }
    /// `setMaxEntries(numNz)`.
    pub fn set_max_entries(&mut self, num_nz: i32) {
        self.max_entries = 2_000_000 + 10 * num_nz;
    }
    /// Record `mipdata_->numCliqueEntriesAfterPresolve`.
    pub fn set_num_entries_after_presolve(&mut self, n: i32) {
        self.num_entries_after_presolve = n;
    }
    /// `isFull`.
    pub fn is_full(&self) -> bool {
        self.num_entries >= self.max_entries
    }
    /// `getNumFixings`.
    pub fn num_fixings(&self) -> i32 {
        self.nfixings
    }
    /// `numCliques()`.
    pub fn num_cliques(&self) -> usize {
        self.cliques.len() - self.freeslots.len()
    }
    /// `numCliques(v)`.
    pub fn num_cliques_of(&self, v: CliqueVar) -> i32 {
        self.numcliquesvar[v.index()]
    }
    /// `getDeletedRows`.
    pub fn deleted_rows(&self) -> &[i32] {
        &self.deletedrows
    }
    /// `getSubstitutions`.
    pub fn substitutions(&self) -> &[Substitution] {
        &self.substitutions
    }
    /// `getSubstitution(col)`.
    pub fn substitution(&self, col: usize) -> Option<&Substitution> {
        match self.colsubstituted[col] {
            0 => None,
            s => Some(&self.substitutions[s as usize - 1]),
        }
    }
    /// `getCliqueExtensions`.
    pub fn clique_extensions(&self) -> &[(i32, CliqueVar)] {
        &self.cliqueextensions
    }
    /// The live cliques: `(id, members, equality, origin)`, in slot order.
    pub fn cliques(&self) -> Vec<(usize, &[CliqueVar], bool, i32)> {
        self.cliques
            .iter()
            .enumerate()
            .filter(|(_, c)| c.start != -1)
            .map(|(i, c)| {
                (
                    i,
                    &self.cliqueentries[c.start as usize..c.end as usize],
                    c.equality,
                    c.origin,
                )
            })
            .collect()
    }

    // ---- storage ----

    /// `unlink(pos, cliqueid)`.
    fn unlink(&mut self, pos: usize, cliqueid: i32) {
        let v = self.cliqueentries[pos];
        self.numcliquesvar[v.index()] -= 1;
        let c = self.cliques[cliqueid as usize];
        if c.end - c.start == 2 {
            self.inverted_size_two[v.index()].remove(&cliqueid);
        } else {
            self.inverted[v.index()].remove(&cliqueid);
        }
    }

    /// `link(pos, cliqueid)`.
    fn link(&mut self, pos: usize, cliqueid: i32) {
        let v = self.cliqueentries[pos];
        debug_assert!(!self.col_deleted[v.c()]);
        self.numcliquesvar[v.index()] += 1;
        let c = self.cliques[cliqueid as usize];
        if c.end - c.start == 2 {
            self.inverted_size_two[v.index()].insert(cliqueid);
        } else {
            self.inverted[v.index()]
                .entry(cliqueid)
                .or_insert(pos as i32);
        }
    }

    /// `findCommonCliqueId(numQueries, v1, v2)`.
    pub fn find_common_clique_id_q(
        &self,
        num_queries: &mut i64,
        v1: CliqueVar,
        v2: CliqueVar,
    ) -> i32 {
        *num_queries += 1;
        if !self.inverted_size_two[v1.index()].is_empty()
            && !self.inverted_size_two[v2.index()].is_empty()
        {
            if let Some(&id) = self.size_two_cliques.get(&sorted_edge(v1, v2)) {
                return id;
            }
        }
        let h1 = &self.inverted[v1.index()];
        let h2 = &self.inverted[v2.index()];
        let (small, large) = if h1.len() <= h2.len() {
            (h1, h2)
        } else {
            (h2, h1)
        };
        for &k in small.keys() {
            if large.contains_key(&k) {
                return k;
            }
        }
        -1
    }

    /// `findCommonCliqueId(v1, v2)` (counts in `numNeighbourhoodQueries`).
    pub fn find_common_clique_id(&mut self, v1: CliqueVar, v2: CliqueVar) -> i32 {
        let mut q = self.num_neighbourhood_queries;
        let r = self.find_common_clique_id_q(&mut q, v1, v2);
        self.num_neighbourhood_queries = q;
        r
    }

    /// `haveCommonClique(v1, v2)`.
    pub fn have_common_clique(&mut self, v1: CliqueVar, v2: CliqueVar) -> bool {
        if v1.col == v2.col {
            return false;
        }
        self.find_common_clique_id(v1, v2) != -1
    }

    /// `haveCommonClique(numQueries, v1, v2)`.
    pub fn have_common_clique_q(
        &self,
        num_queries: &mut i64,
        v1: CliqueVar,
        v2: CliqueVar,
    ) -> bool {
        if v1.col == v2.col {
            return false;
        }
        self.find_common_clique_id_q(num_queries, v1, v2) != -1
    }

    /// `findCommonClique(v1, v2)`: the members of a clique holding both, if any.
    pub fn find_common_clique(&mut self, v1: CliqueVar, v2: CliqueVar) -> Option<&[CliqueVar]> {
        if v1 == v2 {
            return None;
        }
        let clq = self.find_common_clique_id(v1, v2);
        if clq == -1 {
            return None;
        }
        let c = self.cliques[clq as usize];
        Some(&self.cliqueentries[c.start as usize..c.end as usize])
    }

    /// `resolveSubstitution(CliqueVar&)`.
    pub fn resolve_substitution(&self, mut v: CliqueVar) -> CliqueVar {
        while self.colsubstituted[v.c()] != 0 {
            let subst = self.substitutions[self.colsubstituted[v.c()] as usize - 1];
            v = if v.val == 1 {
                subst.replace
            } else {
                subst.replace.complement()
            };
        }
        v
    }

    /// `resolveSubstitution(col, val, offset)`: rewrites `val * x_col` in terms of the
    /// replacing column, moving constants to `offset`.
    pub fn resolve_substitution_col(&self, col: &mut usize, val: &mut f64, offset: &mut f64) {
        while self.colsubstituted[*col] != 0 {
            let subst = self.substitutions[self.colsubstituted[*col] as usize - 1];
            if subst.replace.val == 0 {
                *offset += *val;
                *val = -*val;
            }
            *col = subst.replace.c();
        }
    }

    /// `doAddClique(cliquevars, numcliquevars, equality, origin)`.
    pub fn do_add_clique(&mut self, cliquevars: &[CliqueVar], equality: bool, origin: i32) {
        let numcliquevars = cliquevars.len() as i32;
        let cliqueid = match self.freeslots.pop() {
            None => {
                self.cliques.push(Clique::default());
                self.cliques.len() as i32 - 1
            }
            Some(id) => id,
        };
        let cid = cliqueid as usize;
        self.cliques[cid].equality = equality;
        self.cliques[cid].origin = origin;

        let max_end;
        let fs = self.freespaces.range((numcliquevars, -1)..).next().copied();
        match fs {
            None => {
                let start = self.cliqueentries.len() as i32;
                self.cliques[cid].start = start;
                self.cliques[cid].end = start + numcliquevars;
                max_end = self.cliques[cid].end;
                self.cliqueentries
                    .resize(max_end as usize, CliqueVar::default());
            }
            Some(freespace) => {
                self.freespaces.remove(&freespace);
                self.cliques[cid].start = freespace.1;
                self.cliques[cid].end = freespace.1 + numcliquevars;
                max_end = freespace.1 + freespace.0;
            }
        }
        self.cliques[cid].num_zero_fixed = 0;

        let start = self.cliques[cid].start;
        let mut fixtozero = false;
        let mut k = start;
        for &v0 in cliquevars {
            let v = self.resolve_substitution(v0);
            if fixtozero {
                self.infeasvertexstack.push(v);
                continue;
            }
            // x + ~x + ... <= 1 (after substitution): every other member is 0.
            let vc = v.complement().index();
            let clq_has_v_compl = if numcliquevars == 2 {
                self.inverted_size_two[vc].contains(&cliqueid)
            } else {
                self.inverted[vc].contains_key(&cliqueid)
            };
            if clq_has_v_compl {
                fixtozero = true;
                for j in start..k {
                    let e = self.cliqueentries[j as usize];
                    if e.col != v.col {
                        self.infeasvertexstack.push(e);
                    }
                    self.unlink(j as usize, cliqueid);
                }
                k = start;
                continue;
            }
            // x + x + ... <= 1: x = 0.
            let inserted = if numcliquevars == 2 {
                self.inverted_size_two[v.index()].insert(cliqueid)
            } else {
                match self.inverted[v.index()].entry(cliqueid) {
                    Entry::Vacant(e) => {
                        e.insert(k);
                        true
                    }
                    Entry::Occupied(_) => false,
                }
            };
            if !inserted {
                self.infeasvertexstack.push(v);
                continue;
            }
            self.cliqueentries[k as usize] = v;
            self.numcliquesvar[v.index()] += 1;
            k += 1;
        }

        if max_end > k {
            if self.cliqueentries.len() as i32 == max_end {
                self.cliqueentries.truncate(k as usize);
            } else {
                self.freespaces.insert((max_end - k, k));
            }
            if self.cliques[cid].end > k {
                match k - start {
                    0 => {
                        self.cliques[cid].start = -1;
                        self.cliques[cid].end = -1;
                        self.freeslots.push(cliqueid);
                        return;
                    }
                    1 => {
                        self.unlink(start as usize, cliqueid);
                        self.cliques[cid].start = -1;
                        self.cliques[cid].end = -1;
                        self.freeslots.push(cliqueid);
                        return;
                    }
                    2 => {
                        self.unlink(start as usize, cliqueid);
                        self.unlink(start as usize + 1, cliqueid);
                        self.cliques[cid].end = k;
                        self.link(start as usize, cliqueid);
                        self.link(start as usize + 1, cliqueid);
                    }
                    _ => self.cliques[cid].end = k,
                }
            }
        }

        let c = self.cliques[cid];
        let clique_len = c.end - c.start;
        self.num_entries += clique_len;
        if clique_len == 2 {
            let e = sorted_edge(
                self.cliqueentries[c.start as usize],
                self.cliqueentries[c.start as usize + 1],
            );
            self.size_two_cliques.entry(e).or_insert(cliqueid);
        }
    }

    /// `removeClique(cliqueid)`.
    pub fn remove_clique(&mut self, cliqueid: i32) {
        let c = self.cliques[cliqueid as usize];
        if c.origin != K_HIGHS_IINF && c.origin != -1 {
            self.deletedrows.push(c.origin);
        }
        let start = c.start;
        assert!(start != -1, "removeClique on a free slot");
        let end = c.end;
        let len = end - start;
        if len == 2 {
            let e = sorted_edge(
                self.cliqueentries[start as usize],
                self.cliqueentries[start as usize + 1],
            );
            self.size_two_cliques.remove(&e);
        }
        for i in start..end {
            self.unlink(i as usize, cliqueid);
        }
        self.freeslots.push(cliqueid);
        self.freespaces.insert((len, start));
        self.cliques[cliqueid as usize].start = -1;
        self.cliques[cliqueid as usize].end = -1;
        self.num_entries -= len;
    }

    // ---- neighbourhoods ----

    /// `queryNeighbourhood`: positions `i` in `q` whose literal shares a clique with `v`.
    fn query_neighbourhood(
        &self,
        neighbourhood_inds: &mut Vec<usize>,
        num_queries: &mut i64,
        v: CliqueVar,
        q: &[CliqueVar],
    ) {
        neighbourhood_inds.clear();
        if self.numcliquesvar[v.index()] == 0 {
            return;
        }
        for (i, &qi) in q.iter().enumerate() {
            if self.have_common_clique_q(num_queries, v, qi) {
                neighbourhood_inds.push(i);
            }
        }
    }

    /// `partitionNeighbourhood`: moves `v`'s neighbours in `q` to the front; returns
    /// their count.
    pub fn partition_neighbourhood(
        &self,
        neighbourhood_inds: &mut Vec<usize>,
        num_queries: &mut i64,
        v: CliqueVar,
        q: &mut [CliqueVar],
    ) -> usize {
        self.query_neighbourhood(neighbourhood_inds, num_queries, v, q);
        for i in 0..neighbourhood_inds.len() {
            q.swap(i, neighbourhood_inds[i]);
        }
        neighbourhood_inds.len()
    }

    /// `shrinkToNeighbourhood`: copies `v`'s neighbours in `q` to the front; returns
    /// their count.
    pub fn shrink_to_neighbourhood(
        &self,
        neighbourhood_inds: &mut Vec<usize>,
        num_queries: &mut i64,
        v: CliqueVar,
        q: &mut [CliqueVar],
    ) -> usize {
        self.query_neighbourhood(neighbourhood_inds, num_queries, v, q);
        for i in 0..neighbourhood_inds.len() {
            q[i] = q[neighbourhood_inds[i]];
        }
        neighbourhood_inds.len()
    }

    // ---- fixings and new edges ----

    /// `fixCol(globaldom, v, doProcessInfeasibleVertices)`: literal `v` is 0.
    pub fn fix_col<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        v: CliqueVar,
        do_process: bool,
    ) -> bool {
        let wasfixed = dom.is_fixed(v.c());
        dom.fix_col(v.c(), (1 - v.val) as f64);
        if dom.infeasible() {
            return false;
        }
        if !wasfixed {
            self.nfixings += 1;
            self.infeasvertexstack.push(v);
            if do_process {
                self.process_infeasible_vertices(dom);
            }
        }
        true
    }

    /// `replace` lambda of `processNewEdge`: every occurrence of `subst` becomes `repl`.
    fn replace_literal(&mut self, subst: CliqueVar, repl: CliqueVar) {
        let (si, ri) = (subst.index(), repl.index());
        self.numcliquesvar[ri] += self.numcliquesvar[si];
        self.numcliquesvar[si] = 0;
        let subst_list = std::mem::take(&mut self.inverted[si]);
        for (&cliqueid, &location) in &subst_list {
            self.inverted[ri].entry(cliqueid).or_insert(location);
            self.cliqueentries[location as usize] = repl;
        }
        let subst_list2 = std::mem::take(&mut self.inverted_size_two[si]);
        for &cliqueid in &subst_list2 {
            let mut pos = self.cliques[cliqueid as usize].start as usize;
            let mut other = pos + 1;
            if self.cliqueentries[other] == subst {
                std::mem::swap(&mut pos, &mut other);
            }
            self.inverted_size_two[ri].insert(cliqueid);
            self.cliqueentries[pos] = repl;
            let oe = self.cliqueentries[other];
            self.size_two_cliques.remove(&sorted_edge(subst, oe));
            self.size_two_cliques
                .entry(sorted_edge(repl, oe))
                .or_insert(cliqueid);
        }
    }

    /// `processNewEdge(globaldom, v1, v2)`: `v1 + v2 <= 1` is new. Returns whether it is
    /// a cover (`v1 + v2 = 1`), in which case one column was substituted by the other.
    pub fn process_new_edge<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        v1: CliqueVar,
        v2: CliqueVar,
    ) -> bool {
        if v1.col == v2.col {
            if v1.val == v2.val {
                self.fix_col(dom, v1, true);
                return false;
            }
            return true;
        }
        if self.have_common_clique(v1.complement(), v2) {
            self.fix_col(dom, v2, true);
            false
        } else if self.have_common_clique(v2.complement(), v1) {
            self.fix_col(dom, v1, true);
            false
        } else {
            if !self.found_cover(dom, v1.complement(), v2.complement()) {
                return false;
            }
            if dom.infeasible() {
                return true;
            }
            self.found_cover(dom, v1, v2);
            if dom.is_fixed(v1.c()) || dom.is_fixed(v2.c()) || dom.infeasible() {
                return true;
            }
            let (mut v1, mut v2) = (v1, v2);
            let substitution = if v2.col < v1.col {
                if v1.val == 1 {
                    v2 = v2.complement();
                }
                Substitution {
                    substcol: v1.c(),
                    replace: v2,
                }
            } else {
                if v2.val == 1 {
                    v1 = v1.complement();
                }
                Substitution {
                    substcol: v2.c(),
                    replace: v1,
                }
            };
            self.substitutions.push(substitution);
            self.colsubstituted[substitution.substcol] = self.substitutions.len() as i32;
            self.replace_literal(
                CliqueVar::new(substitution.substcol, 1),
                substitution.replace,
            );
            self.replace_literal(
                CliqueVar::new(substitution.substcol, 0),
                substitution.replace.complement(),
            );
            true
        }
    }

    /// `fixAllVarsInClique` lambda of `addClique`.
    fn fix_all_vars_in_clique<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        cv: &[CliqueVar],
        has_new_edge: &mut bool,
    ) -> bool {
        const MAX_NUM_CLIQUE_VARS: usize = 100;
        let n = cv.len();
        for i in 0..n {
            let c = cv[i].c();
            if !dom.is_fixed(c) || cv[i].val as f64 != dom.col_lower()[c] {
                continue;
            }
            // fixed to 1: every other member is 0
            for k in 0..n {
                if k == i {
                    continue;
                }
                if !self.fix_col(dom, cv[k], false) {
                    return false;
                }
            }
            self.process_infeasible_vertices(dom);
            return true;
        }
        if n > MAX_NUM_CLIQUE_VARS {
            return false;
        }
        for i in 0..n.saturating_sub(1) {
            if dom.is_fixed(cv[i].c()) {
                continue;
            }
            if self.num_cliques_of(cv[i]) == 0 && self.num_cliques_of(cv[i].complement()) == 0 {
                *has_new_edge = true;
                continue;
            }
            for j in i + 1..n {
                if dom.is_fixed(cv[j].c()) {
                    continue;
                }
                if self.have_common_clique(cv[i], cv[j]) {
                    continue;
                }
                *has_new_edge = true;
                let iscover = self.process_new_edge(dom, cv[i], cv[j]);
                if dom.infeasible() {
                    return false;
                }
                // (node-queue pruning: never prunes in HiGHS 1.15.1, see the header)
                if iscover {
                    for k in 0..n {
                        if k == i || k == j {
                            continue;
                        }
                        if !self.fix_col(dom, cv[k], false) {
                            return false;
                        }
                    }
                    self.process_infeasible_vertices(dom);
                    return true;
                }
            }
        }
        false
    }

    /// `addClique(mipsolver, cliquevars, numcliquevars, equality, origin)`.
    pub fn add_clique<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        cliquevars: &[CliqueVar],
        equality: bool,
        origin: i32,
    ) {
        let mut cv: Vec<CliqueVar> = cliquevars
            .iter()
            .map(|&v| self.resolve_substitution(v))
            .collect();
        let mut has_new_edge = false;
        // checkClique(hasNewEdge, false)
        if self.fix_all_vars_in_clique(dom, &cv, &mut has_new_edge) {
            return;
        }
        if dom.infeasible() {
            return;
        }
        if cv.len() == 2 && equality {
            // checkClique(hasNewEdge, true)
            let comp: Vec<CliqueVar> = cv.iter().map(|v| v.complement()).collect();
            if self.fix_all_vars_in_clique(dom, &comp, &mut has_new_edge) {
                return;
            }
            if dom.infeasible() {
                return;
            }
        }
        if !has_new_edge && origin == K_HIGHS_IINF {
            return;
        }
        cv.retain(|v| !dom.is_fixed(v.c()));
        if cv.len() < 2 {
            return;
        }
        self.do_add_clique(&cv, equality, origin);
        self.process_infeasible_vertices(dom);
    }

    /// `foundCover(globaldom, v1, v2)`: `v1 + v2 >= 1` holds too; every other member of
    /// a clique holding both is 0 and the clique is removed.
    pub fn found_cover<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        v1: CliqueVar,
        v2: CliqueVar,
    ) -> bool {
        let mut commonclique = self.find_common_clique_id(v1, v2);
        if commonclique == -1 {
            return false;
        }
        while commonclique != -1 {
            let c = self.cliques[commonclique as usize];
            for i in c.start..c.end {
                let e = self.cliqueentries[i as usize];
                if e == v1 || e == v2 {
                    continue;
                }
                if !self.fix_col(dom, e, false) {
                    return true;
                }
            }
            self.remove_clique(commonclique);
            commonclique = self.find_common_clique_id(v1, v2);
        }
        self.process_infeasible_vertices(dom);
        true
    }

    /// For each member of clique `cliqueid` other than column `col`: `fixCol`. Returns
    /// `true` on infeasibility (the `for_each` lambdas of `processInfeasibleVertices`).
    fn fix_others_in_clique<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        cliqueid: i32,
        col: u32,
    ) -> bool {
        let c = self.cliques[cliqueid as usize];
        for i in c.start..c.end {
            let e = self.cliqueentries[i as usize];
            if e.col == col {
                continue;
            }
            if !self.fix_col(dom, e, false) {
                return true;
            }
        }
        self.remove_clique(cliqueid);
        false
    }

    /// `processInfeasibleVertices(globaldom)`.
    pub fn process_infeasible_vertices<D: CliqueDomain>(&mut self, dom: &mut D) {
        while !self.infeasvertexstack.is_empty() && !dom.infeasible() {
            let v0 = self.infeasvertexstack.pop().unwrap().complement();
            let v = self.resolve_substitution(v0);
            let wasfixed = dom.is_fixed(v.c());
            dom.fix_col(v.c(), v.val as f64);
            if dom.infeasible() {
                return;
            }
            if !wasfixed {
                self.nfixings += 1;
            }
            if self.col_deleted[v.c()] {
                continue;
            }
            self.col_deleted[v.c()] = true;

            // literal v is 1: the other members of its cliques are 0
            let lists = std::mem::take(&mut self.inverted[v.index()]);
            let lists2 = std::mem::take(&mut self.inverted_size_two[v.index()]);
            for &cliqueid in lists.keys() {
                if self.fix_others_in_clique(dom, cliqueid, v.col) {
                    return;
                }
            }
            for &cliqueid in &lists2 {
                if self.fix_others_in_clique(dom, cliqueid, v.col) {
                    return;
                }
            }

            // its complement is 0 in each of its cliques
            let vc = v.complement().index();
            let lists = std::mem::take(&mut self.inverted[vc]);
            let lists2 = std::mem::take(&mut self.inverted_size_two[vc]);
            if self.in_presolve {
                for &cliqueid in lists.keys() {
                    self.cliques[cliqueid as usize].num_zero_fixed += 1;
                    if self.cliques[cliqueid as usize].num_active() <= 1 {
                        self.remove_clique(cliqueid);
                    }
                }
                continue;
            }
            for &cliqueid in &lists2 {
                self.remove_clique(cliqueid);
            }
            debug_assert!(self.cliquehitinds.is_empty());
            for &cliqueid in lists.keys() {
                let cid = cliqueid as usize;
                self.cliques[cid].num_zero_fixed += 1;
                let c = self.cliques[cid];
                if c.num_active() <= 1 {
                    self.remove_clique(cliqueid);
                } else if c.num_zero_fixed >= 10.max((c.end - c.start) >> 1) {
                    let mut clq: Vec<CliqueVar> =
                        self.cliqueentries[c.start as usize..c.end as usize].to_vec();
                    debug_assert_eq!(
                        clq.iter().filter(|x| self.col_deleted[x.c()]).count() as i32,
                        c.num_zero_fixed
                    );
                    self.remove_clique(cliqueid);
                    clq.retain(|x| {
                        !(dom.is_fixed(x.c()) && dom.col_lower()[x.c()] == (1 - x.val) as f64)
                    });
                    if clq.len() > 1 {
                        self.do_add_clique(&clq, false, K_HIGHS_IINF);
                    }
                }
            }
        }
        self.propagate_and_cleanup(dom);
    }

    /// `propagateAndCleanup(globaldom)`.
    fn propagate_and_cleanup<D: CliqueDomain>(&mut self, dom: &mut D) {
        let mut start = dom.domain_change_stack_len();
        dom.propagate();
        let mut end = dom.domain_change_stack_len();
        while !dom.infeasible() && start != end {
            for k in start..end {
                let col = dom.domain_change_col(k);
                if !dom.is_fixed(col) {
                    continue;
                }
                let lb = dom.col_lower()[col];
                if lb != 1.0 && lb != 0.0 {
                    continue;
                }
                let fixval = lb as u8;
                let v = CliqueVar::new(col, 1 - fixval);
                if self.num_cliques_of(v) != 0 {
                    self.vertex_infeasible(dom, col, 1 - fixval);
                    if dom.infeasible() {
                        return;
                    }
                }
            }
            start = dom.domain_change_stack_len();
            dom.propagate();
            end = dom.domain_change_stack_len();
        }
    }

    /// `vertexInfeasible(globaldom, col, val)`: literal `(col, val)` is 0.
    pub fn vertex_infeasible<D: CliqueDomain>(&mut self, dom: &mut D, col: usize, val: u8) {
        let wasfixed = dom.is_fixed(col);
        dom.fix_col(col, (1 - val) as f64);
        if dom.infeasible() {
            return;
        }
        if !wasfixed {
            self.nfixings += 1;
        }
        self.infeasvertexstack.push(CliqueVar::new(col, val));
        self.process_infeasible_vertices(dom);
    }

    /// `cleanupFixed(globaldom)`.
    pub fn cleanup_fixed<D: CliqueDomain>(&mut self, dom: &mut D) {
        let numcol = dom.col_upper().len();
        let oldnfixings = self.nfixings;
        for i in 0..numcol {
            if self.col_deleted[i] || !dom.is_fixed(i) {
                continue;
            }
            let lb = dom.col_lower()[i];
            if lb != 1.0 && lb != 0.0 {
                continue;
            }
            let fixval = lb as u8;
            self.vertex_infeasible(dom, i, 1 - fixval);
            if dom.infeasible() {
                return;
            }
        }
        if self.nfixings != oldnfixings {
            self.propagate_and_cleanup(dom);
        }
    }

    /// `addImplications(domain, col, val)`: the bound changes the domain makes when it
    /// fixes binary `col` to `val` (`HighsDomain::changeBound`'s clique-table hook).
    /// Substitutions are followed first, then every other member of each clique of the
    /// literal is set to 0, each with `Reason::cliqueTable(col, val)`.
    pub fn add_implications<D: CliqueDomain>(&self, dom: &mut D, col: usize, val: u8) {
        let reason = CliqueReason::CliqueTable(2 * col + val as usize);
        let mut v = CliqueVar::new(col, val);
        while self.colsubstituted[v.c()] != 0 {
            let subst = self.substitutions[self.colsubstituted[v.c()] as usize - 1];
            v = if v.val == 1 {
                subst.replace
            } else {
                subst.replace.complement()
            };
            if v.val == 1 {
                if dom.col_lower()[v.c()] == 1.0 {
                    continue;
                }
                dom.change_bound(BoundType::Lower, v.c(), 1.0, reason);
                if dom.infeasible() {
                    return;
                }
            } else {
                if dom.col_upper()[v.c()] == 0.0 {
                    continue;
                }
                dom.change_bound(BoundType::Upper, v.c(), 0.0, reason);
                if dom.infeasible() {
                    return;
                }
            }
        }
        let do_fixings = |dom: &mut D, cliqueid: i32| -> bool {
            let c = self.cliques[cliqueid as usize];
            for i in c.start..c.end {
                let e = self.cliqueentries[i as usize];
                if e.col == v.col {
                    continue;
                }
                if e.val == 1 {
                    if dom.col_upper()[e.c()] == 0.0 {
                        continue;
                    }
                    dom.change_bound(BoundType::Upper, e.c(), 0.0, reason);
                } else {
                    if dom.col_lower()[e.c()] == 1.0 {
                        continue;
                    }
                    dom.change_bound(BoundType::Lower, e.c(), 1.0, reason);
                }
                if dom.infeasible() {
                    return true;
                }
            }
            false
        };
        for &cliqueid in self.inverted[v.index()].keys() {
            if do_fixings(dom, cliqueid) {
                return;
            }
        }
        for &cliqueid in &self.inverted_size_two[v.index()] {
            if do_fixings(dom, cliqueid) {
                return;
            }
        }
    }

    /// `getNumImplications(col)`.
    pub fn num_implications(&self, col: usize) -> i32 {
        let i0 = CliqueVar::new(col, 0).index();
        let i1 = CliqueVar::new(col, 1).index();
        let mut numimplics = self.numcliquesvar[i0] + self.numcliquesvar[i1];
        for i in [i0, i1] {
            for &cliqueid in self.inverted[i].keys() {
                let c = self.cliques[cliqueid as usize];
                let nimplics = (c.end - c.start - 1) * (1 + c.equality as i32);
                numimplics += nimplics - 1;
            }
        }
        numimplics
    }

    /// `getNumImplications(col, val)`.
    pub fn num_implications_val(&self, col: usize, val: bool) -> i32 {
        let ival = CliqueVar::new(col, val as u8).index();
        let mut numimplics = self.numcliquesvar[ival];
        for &cliqueid in self.inverted[ival].keys() {
            let c = self.cliques[cliqueid as usize];
            let nimplics = (c.end - c.start - 1) * (1 + c.equality as i32);
            numimplics += nimplics - 1;
        }
        numimplics
    }

    // ---- partitions ----

    /// `cliquePartition(clqVars, partitionStart)`: shuffles `clq_vars` and reorders it
    /// into consecutive cliques; returns the partition starts (last = `len`).
    pub fn clique_partition(&mut self, clq_vars: &mut [CliqueVar]) -> Vec<usize> {
        self.randgen.shuffle(clq_vars);
        let mut neighbourhood_inds = Vec::with_capacity(clq_vars.len());
        let mut nq = self.num_neighbourhood_queries;
        let n = clq_vars.len();
        let mut partition_start = Vec::with_capacity(n + 1);
        let mut extension_end = n;
        partition_start.push(0);
        for i in 0..n {
            if i == extension_end {
                partition_start.push(i);
                extension_end = n;
            }
            let v = clq_vars[i];
            let extension_start = i + 1;
            extension_end = self.partition_neighbourhood(
                &mut neighbourhood_inds,
                &mut nq,
                v,
                &mut clq_vars[extension_start..extension_end],
            ) + extension_start;
        }
        partition_start.push(n);
        self.num_neighbourhood_queries = nq;
        partition_start
    }

    /// `cliquePartition(objective, clqVars, partitionStart)`: as
    /// [`Self::clique_partition`], greedily preferring literals with larger signed cost.
    pub fn clique_partition_objective(
        &mut self,
        objective: &[f64],
        clq_vars: &mut [CliqueVar],
    ) -> Vec<usize> {
        self.randgen.shuffle(clq_vars);
        // `(2 * v.val - 1)`: the 1-bit field promotes to int, so this is -1 or +1.
        let key = |v: &CliqueVar| (2.0 * v.val as f64 - 1.0) * objective[v.c()];
        pdqsort_branchless(clq_vars, |a, b| key(a) > key(b));
        let mut neighbourhood_inds = Vec::with_capacity(clq_vars.len());
        let mut nq = self.num_neighbourhood_queries;
        let n = clq_vars.len();
        let mut partition_start = Vec::with_capacity(n + 1);
        let mut extension_end = n;
        partition_start.push(0);
        let mut last_swapped_index = 0usize;
        for i in 0..n {
            if i == extension_end {
                partition_start.push(i);
                extension_end = n;
                if last_swapped_index >= i {
                    pdqsort_branchless(&mut clq_vars[i..last_swapped_index + 1], |a, b| {
                        key(a) > key(b)
                    });
                }
                last_swapped_index = 0;
            }
            let v = clq_vars[i];
            let extension_start = i + 1;
            extension_end = self.partition_neighbourhood(
                &mut neighbourhood_inds,
                &mut nq,
                v,
                &mut clq_vars[extension_start..extension_end],
            ) + extension_start;
            if let Some(&last) = neighbourhood_inds.last() {
                last_swapped_index = last_swapped_index.max(last + extension_start);
            }
        }
        partition_start.push(n);
        self.num_neighbourhood_queries = nq;
        partition_start
    }

    /// The clique-table half of `HighsObjectiveFunction::setupCliquePartition`: the
    /// objective binaries `binaries` (in `getObjectiveNonzeros()` order) as literals
    /// `(col, cost < 0)`, partitioned by [`Self::clique_partition_objective`]. Returns
    /// `(columns in partition order, partition starts)`, the input of
    /// [`super::objective::ObjectiveFunction::setup_clique_partition`].
    pub fn objective_clique_partition(
        &mut self,
        cost: &[f64],
        binaries: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut clqvars: Vec<CliqueVar> = binaries
            .iter()
            .map(|&c| CliqueVar::new(c, (cost[c] < 0.0) as u8))
            .collect();
        let starts = self.clique_partition_objective(cost, &mut clqvars);
        (clqvars.iter().map(|v| v.c()).collect(), starts)
    }

    // ---- extraction ----

    /// `extractCliques(mipsolver, inds, vals, complementation, rhs, nbin, perm, clique,
    /// feastol)`: one row `sum vals x <= rhs`, complemented to positive coefficients.
    #[allow(clippy::too_many_arguments)]
    fn extract_cliques_row<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        implics: &mut Option<&mut VarBoundStore>,
        inds: &[i32],
        vals: &[f64],
        complementation: &[i8],
        rhs: f64,
        feastol: f64,
    ) {
        let mut perm: Vec<usize> = (0..inds.len()).collect();
        let nbin = partition_two_ended(&mut perm, |&p| dom.is_binary(inds[p] as usize));
        let ntotal = perm.len();
        if nbin < ntotal {
            if let Some(vb) = implics.as_deref_mut() {
                let d = vbound::Domain {
                    lb: dom.col_lower(),
                    ub: dom.col_upper(),
                    is_int: dom.integrality(),
                };
                vb.extract_from_complemented(&d, inds, vals, complementation, rhs);
            }
        }
        if nbin <= 1 {
            return;
        }
        pdqsort(&mut perm[..nbin], |&p1, &p2| {
            pair_gt((vals[p1], p1), (vals[p2], p2))
        });
        if vals[perm[0]] + vals[perm[1]] <= rhs + feastol {
            return;
        }
        let lit = |pos: usize| -> CliqueVar {
            CliqueVar::new(inds[pos] as usize, (complementation[pos] != -1) as u8)
        };
        let mut clique: Vec<CliqueVar> = Vec::new();
        // set packing (all binary coefficients equal, rhs below twice that)
        if (vals[perm[0]] - vals[perm[nbin - 1]]).abs() <= feastol
            && rhs < 2.0 * vals[perm[nbin - 1]] - feastol
        {
            for &pos in &perm[..nbin] {
                clique.push(lit(pos));
            }
            self.add_clique(dom, &clique, false, K_HIGHS_IINF);
            return;
        }
        for k in (1..nbin).rev() {
            let mincliqueval = rhs - vals[perm[k]] + feastol;
            let cliqueend = perm[..k].partition_point(|&p| vals[p] > mincliqueval);
            if cliqueend == 0 {
                continue;
            }
            clique.clear();
            for &pos in &perm[..cliqueend] {
                clique.push(lit(pos));
            }
            clique.push(lit(perm[k]));
            if clique.len() >= 2 {
                self.add_clique(dom, &clique, false, K_HIGHS_IINF);
                if dom.infeasible() {
                    return;
                }
            }
            if cliqueend == k {
                return;
            }
        }
    }

    /// `extractCliques(mipsolver, transformRows)`: set-packing rows become cliques with
    /// their row as origin; with `transform_rows`, every other row is complemented to
    /// positive coefficients and its cliques (and, given `implics`, its variable
    /// bounds) extracted.
    pub fn extract_cliques<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        rows: &CliqueRows,
        mut implics: Option<&mut VarBoundStore>,
        transform_rows: bool,
    ) {
        let nrows = rows.ar_start.len() - 1;
        let mut clique: Vec<CliqueVar> = Vec::new();
        // `entries`: merged (col, val), first-occurrence order (see the header)
        let mut entries: Vec<(usize, f64)> = Vec::new();
        let mut entry_pos: HashMap<usize, usize> = HashMap::new();
        for i in 0..nrows {
            if i >= rows.num_orig_rows {
                break;
            }
            let (start, end) = (rows.ar_start[i], rows.ar_start[i + 1]);
            if rows.row_upper[i] == 1.0 {
                let mut issetppc = true;
                clique.clear();
                for j in start..end {
                    let col = rows.ar_index[j];
                    if dom.col_upper()[col] == 0.0 && dom.col_lower()[col] == 0.0 {
                        continue;
                    }
                    issetppc = dom.is_binary(col) && rows.ar_value[j] == 1.0;
                    if !issetppc {
                        break;
                    }
                    clique.push(CliqueVar::new(col, 1));
                }
                if issetppc {
                    self.add_clique(dom, &clique, rows.row_lower[i] == 1.0, i as i32);
                    if dom.infeasible() {
                        return;
                    }
                    continue;
                }
            }
            if !transform_rows || self.is_full() {
                continue;
            }
            let mut offset = 0.0;
            entries.clear();
            entry_pos.clear();
            for j in start..end {
                let mut col = rows.ar_index[j];
                let mut val = rows.ar_value[j];
                self.resolve_substitution_col(&mut col, &mut val, &mut offset);
                match entry_pos.get(&col) {
                    Some(&p) => entries[p].1 += val,
                    None => {
                        entry_pos.insert(col, entries.len());
                        entries.push((col, val));
                    }
                }
            }
            for (side, direction) in [(rows.row_upper[i], 1.0f64), (rows.row_lower[i], -1.0f64)] {
                // checkRow(rhs, direction)
                if direction * side == INF {
                    continue;
                }
                let mut rhs = direction * (side - offset);
                let (mut inds, mut vals, mut compl) = (Vec::new(), Vec::new(), Vec::new());
                let mut freevar = false;
                let mut nbin = 0usize;
                for &(col, v) in &entries {
                    let val = direction * v;
                    if val.abs() < rows.epsilon {
                        continue;
                    }
                    if dom.is_binary(col) {
                        nbin += 1;
                    }
                    if val < 0.0 {
                        freevar = dom.col_upper()[col] == INF;
                        if freevar {
                            break;
                        }
                        vals.push(-val);
                        inds.push(col as i32);
                        compl.push(-1i8);
                        rhs -= val * dom.col_upper()[col];
                    } else {
                        freevar = dom.col_lower()[col] == -INF;
                        if freevar {
                            break;
                        }
                        vals.push(val);
                        inds.push(col as i32);
                        compl.push(1i8);
                        rhs -= val * dom.col_lower()[col];
                    }
                }
                if !freevar && nbin != 0 {
                    self.extract_cliques_row(
                        dom,
                        &mut implics,
                        &inds,
                        &vals,
                        &compl,
                        rhs,
                        rows.feastol,
                    );
                    if dom.infeasible() {
                        return;
                    }
                }
            }
        }
    }

    /// `extractCliquesFromCut(mipsolver, inds, vals, len, rhs)` for a valid cut
    /// `sum vals x[inds] <= rhs`: tightens integer bounds, extracts variable bounds
    /// (given `implics`) and cliques. `feastol` is `mipdata_->feastol`.
    pub fn extract_cliques_from_cut<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        implics: Option<&mut VarBoundStore>,
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        feastol: f64,
    ) {
        if self.is_full() {
            return;
        }
        let len = inds.len();
        let mut minact = CDouble::new(0.0);
        let mut nbin = 0usize;
        for i in 0..len {
            let c = inds[i];
            if dom.is_binary(c) {
                nbin += 1;
            }
            if vals[i] > 0.0 {
                if dom.col_lower()[c] == -INF {
                    return;
                }
                minact += vals[i] * dom.col_lower()[c];
            } else {
                if dom.col_upper()[c] == INF {
                    return;
                }
                minact += vals[i] * dom.col_upper()[c];
            }
        }
        if rhs - minact < 0.0 {
            minact = CDouble::new(rhs);
        }
        for i in 0..len {
            let c = inds[i];
            if !dom.integrality()[c] {
                continue;
            }
            let bound_val = ((CDouble::new(rhs) - minact) / vals[i]).to_f64();
            if vals[i] > 0.0 {
                let b = (bound_val + dom.col_lower()[c] + dom.feastol()).floor();
                dom.change_bound(BoundType::Upper, c, b, CliqueReason::Unspecified);
            } else {
                let b = (bound_val + dom.col_upper()[c] - dom.feastol()).ceil();
                dom.change_bound(BoundType::Lower, c, b, CliqueReason::Unspecified);
            }
            if dom.infeasible() {
                return;
            }
        }
        if nbin <= 1 {
            return;
        }
        let mut perm: Vec<usize> = (0..len).collect();
        let nbin = partition_two_ended(&mut perm, |&p| dom.is_binary(inds[p]));
        if nbin < len {
            if let Some(vb) = implics {
                let d = vbound::Domain {
                    lb: dom.col_lower(),
                    ub: dom.col_upper(),
                    is_int: dom.integrality(),
                };
                let inds32: Vec<i32> = inds.iter().map(|&c| c as i32).collect();
                vb.extract_from_cut(&d, &inds32, vals, rhs);
            }
        }
        if nbin <= 1 {
            return;
        }
        pdqsort(&mut perm[..nbin], |&p1, &p2| {
            pair_gt((vals[p1].abs(), p1), (vals[p2].abs(), p2))
        });
        if vals[perm[0]].abs() + vals[perm[1]].abs()
            <= (CDouble::new(rhs) - minact + feastol).to_f64()
        {
            return;
        }
        let max_new_entries = (self.num_entries_after_presolve as i64
            + 100_000
            + 4 * dom.num_model_nonzeros() as i64)
            .min(self.num_entries as i64 + 10 * nbin as i64);
        let lit = |pos: usize| CliqueVar::new(inds[pos], (vals[pos] >= 0.0) as u8);
        let mut clique = Vec::with_capacity(nbin);
        let mut k = nbin - 1;
        while k != 0 && (self.num_entries as i64) < max_new_entries {
            let mincliqueval =
                (CDouble::new(rhs) - minact - vals[perm[k]].abs() + feastol).to_f64();
            let cliqueend = perm[..k].partition_point(|&p| vals[p].abs() > mincliqueval);
            if cliqueend != 0 {
                clique.clear();
                for &pos in &perm[..cliqueend] {
                    clique.push(lit(pos));
                }
                clique.push(lit(perm[k]));
                if clique.len() >= 2 {
                    self.add_clique(dom, &clique, false, K_HIGHS_IINF);
                    if dom.infeasible() || self.num_entries as i64 >= max_new_entries {
                        return;
                    }
                }
                if cliqueend == k {
                    return;
                }
            }
            k -= 1;
        }
    }

    /// `extractObjCliques(mipsolver)`, given the cutoff constraint
    /// `vals . x[inds] <= rhs` (`globaldom.getCutoffConstraint`) whose first
    /// `num_binary` entries are the objective binaries. The caller skips the call when
    /// the objective lower bound is `-inf`, as HiGHS does.
    pub fn extract_obj_cliques<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        num_binary: usize,
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        feastol: f64,
    ) {
        if num_binary <= 1 {
            return;
        }
        let mut perm: Vec<usize> = (0..num_binary).collect();
        let nbin = partition_two_ended(&mut perm, |&pos| {
            vals[pos] != 0.0 && !dom.is_fixed(inds[pos])
        });
        if nbin <= 1 {
            return;
        }
        pdqsort(&mut perm[..nbin], |&p1, &p2| {
            pair_gt((vals[p1].abs(), p1), (vals[p2].abs(), p2))
        });
        // globaldom.computeMinActivity(0, len, inds, vals, ninf, minact)
        let mut minact = CDouble::new(0.0);
        for (&c, &v) in inds.iter().zip(vals) {
            let contribution = activity_contribution_min(v, dom.col_lower()[c], dom.col_upper()[c]);
            if contribution != -INF {
                minact += contribution;
            }
        }
        if vals[perm[0]].abs() + vals[perm[1]].abs()
            <= (CDouble::new(rhs) - minact + feastol).to_f64()
        {
            return;
        }
        let lit = |pos: usize| CliqueVar::new(inds[pos], (vals[pos] >= 0.0) as u8);
        let mut clique = Vec::with_capacity(nbin);
        for k in (1..nbin).rev() {
            let mincliqueval =
                (CDouble::new(rhs) - minact - vals[perm[k]].abs() + feastol).to_f64();
            let cliqueend = perm[..k].partition_point(|&p| vals[p].abs() > mincliqueval);
            if cliqueend == 0 {
                continue;
            }
            clique.clear();
            for &pos in &perm[..cliqueend] {
                clique.push(lit(pos));
            }
            clique.push(lit(perm[k]));
            if clique.len() >= 2 {
                self.add_clique(dom, &clique, false, K_HIGHS_IINF);
                if dom.infeasible() {
                    return;
                }
            }
            if cliqueend == k {
                return;
            }
        }
    }

    // ---- subsumption and merging ----

    /// `collectCliques(clique)`.
    fn collect_cliques(&mut self, clique: &[CliqueVar]) {
        if self.cliquehits.len() < self.cliques.len() {
            self.cliquehits.resize(self.cliques.len(), 0);
        }
        for &v in clique {
            for &cliqueid in self.inverted[v.index()].keys() {
                if self.cliquehits[cliqueid as usize] == 0 {
                    self.cliquehitinds.push(cliqueid);
                }
                self.cliquehits[cliqueid as usize] += 1;
            }
            for &cliqueid in &self.inverted_size_two[v.index()] {
                if self.cliquehits[cliqueid as usize] == 0 {
                    self.cliquehitinds.push(cliqueid);
                }
                self.cliquehits[cliqueid as usize] += 1;
            }
        }
    }

    /// `cliqueSubsumption(clique, redundant, dominatingOrigin, removeCliqueCallback)`:
    /// returns `(redundant, dominatingOrigin, removed)`. `clear_origin` selects the
    /// callback of `runCliqueSubsumption` (origin reset before removal) over the one of
    /// `runCliqueMerging`.
    fn clique_subsumption(&mut self, clique: &[CliqueVar], clear_origin: bool) -> (bool, i32, i32) {
        self.collect_cliques(clique);
        let mut redundant = false;
        let mut dominating_origin = K_HIGHS_IINF;
        let mut nremoved = 0;
        let hitinds = std::mem::take(&mut self.cliquehitinds);
        for &cliqueid in &hitinds {
            let cid = cliqueid as usize;
            let hits = self.cliquehits[cid] as i32;
            self.cliquehits[cid] = 0;
            if hits == clique.len() as i32 {
                redundant = true;
                let o = self.cliques[cid].origin;
                if o != K_HIGHS_IINF && o != -1 {
                    dominating_origin = o;
                }
            } else if self.cliques[cid].num_active() == hits {
                if self.cliques[cid].equality {
                    // a subset is an equality clique: the remaining literals are 0
                    let size_two = self.cliques[cid].end - self.cliques[cid].start == 2;
                    for &v in clique {
                        let has = if size_two {
                            self.inverted_size_two[v.index()].contains(&cliqueid)
                        } else {
                            self.inverted[v.index()].contains_key(&cliqueid)
                        };
                        if !has {
                            self.infeasvertexstack.push(v);
                        }
                    }
                } else {
                    nremoved += 1;
                    if clear_origin {
                        self.cliques[cid].origin = K_HIGHS_IINF;
                    }
                    self.remove_clique(cliqueid);
                }
            }
        }
        let mut hitinds = hitinds;
        hitinds.clear();
        self.cliquehitinds = hitinds;
        (redundant, dominating_origin, nremoved)
    }

    /// `runCliqueSubsumption(globaldom, clique)`: removes the table's cliques that
    /// `clique` contains; empties `clique` when the table already contains it.
    pub fn run_clique_subsumption<D: CliqueDomain>(
        &mut self,
        dom: &D,
        clique: &mut Vec<CliqueVar>,
    ) -> i32 {
        clique.retain(|v| !self.col_deleted[v.c()]);
        if clique.len() <= 2 {
            return 0;
        }
        let (redundant, _, nremoved) = self.clique_subsumption(clique, true);
        if redundant {
            clique.clear();
        }
        if !self.infeasvertexstack.is_empty() {
            clique.retain(|v| !dom.is_fixed(v.c()));
        }
        nremoved
    }

    /// The candidate collection shared by both `runCliqueMerging` overloads: the
    /// unfixed, not-yet-candidate members of `extensionstart`'s cliques.
    fn add_merge_candidates<D: CliqueDomain>(
        &mut self,
        dom: &D,
        extensionstart: CliqueVar,
        out: &mut Vec<CliqueVar>,
    ) {
        let ids: Vec<i32> = self.inverted[extensionstart.index()]
            .keys()
            .copied()
            .chain(
                self.inverted_size_two[extensionstart.index()]
                    .iter()
                    .copied(),
            )
            .collect();
        for cliqueid in ids {
            let c = self.cliques[cliqueid as usize];
            for i in c.start..c.end {
                let e = self.cliqueentries[i as usize];
                if self.iscandidate[e.index()] || dom.is_fixed(e.c()) {
                    continue;
                }
                self.iscandidate[e.index()] = true;
                out.push(e);
            }
        }
    }

    /// `runCliqueMerging(globaldomain, clique, equation)`: extends `clique` by literals
    /// adjacent to all its members; for an equation those are fixed to 0 instead.
    pub fn run_clique_merging_clique<D: CliqueDomain>(
        &mut self,
        dom: &mut D,
        clique: &mut Vec<CliqueVar>,
        equation: bool,
    ) {
        let mut extensionstart = CliqueVar::default();
        let mut numcliques = K_HIGHS_IINF;
        self.iscandidate.resize(self.inverted.len(), false);
        let mut neighbourhood_inds = Vec::with_capacity(self.inverted.len());
        let initial = clique.len();
        for &v in clique.iter() {
            if dom.is_fixed(v.c()) {
                continue;
            }
            let this_num = self.num_cliques_of(v);
            if this_num < numcliques {
                numcliques = this_num;
                extensionstart = v;
            }
        }
        if numcliques == K_HIGHS_IINF {
            clique.clear();
            return;
        }
        for &v in clique.iter() {
            self.iscandidate[v.index()] = true;
        }
        self.add_merge_candidates(dom, extensionstart, clique);
        for &v in clique.iter() {
            self.iscandidate[v.index()] = false;
        }
        let mut nq = self.num_neighbourhood_queries;
        let mut i = 0;
        while i != initial && initial < clique.len() {
            if clique[i] != extensionstart {
                let ci = clique[i];
                let new_size = initial
                    + self.shrink_to_neighbourhood(
                        &mut neighbourhood_inds,
                        &mut nq,
                        ci,
                        &mut clique[initial..],
                    );
                clique.truncate(new_size);
            }
            i += 1;
        }
        if initial < clique.len() {
            self.randgen.shuffle(&mut clique[initial..]);
            let mut i = initial;
            while i < clique.len() {
                let extvar = clique[i];
                i += 1;
                let new_size = i + self.shrink_to_neighbourhood(
                    &mut neighbourhood_inds,
                    &mut nq,
                    extvar,
                    &mut clique[i..],
                );
                clique.truncate(new_size);
            }
        }
        self.num_neighbourhood_queries = nq;
        if equation {
            for i in initial..clique.len() {
                let v = clique[i];
                self.vertex_infeasible(dom, v.c(), v.val);
            }
        } else {
            self.run_clique_subsumption(dom, clique);
            if !clique.is_empty() {
                clique.retain(|v| {
                    !(dom.is_fixed(v.c()) && dom.col_lower()[v.c()] as i32 == (1 - v.val) as i32)
                });
            }
        }
        self.process_infeasible_vertices(dom);
    }

    /// `runCliqueMerging(globaldomain)`: extends every model-row clique (and every
    /// equality clique) by the literals adjacent to all its members, then re-merges the
    /// cliques derived from model rows (origin -1).
    pub fn run_clique_merging<D: CliqueDomain>(&mut self, dom: &mut D) {
        let mut extensionvars: Vec<CliqueVar> = Vec::new();
        self.iscandidate.resize(self.inverted.len(), false);
        let mut neighbourhood_inds = Vec::with_capacity(self.inverted.len());
        let numcliqueslots = self.cliques.len();
        let max_new_entries = self.num_entries as i64 + dom.num_model_nonzeros() as i64;
        let mut have_non_model = false;
        for k in 0..numcliqueslots {
            let ck = self.cliques[k];
            if ck.start == -1 {
                continue;
            }
            if !ck.equality && ck.origin == K_HIGHS_IINF {
                continue;
            }
            if ck.origin == -1 {
                have_non_model = true;
                continue;
            }
            let numclqvars = (ck.end - ck.start) as usize;
            if numclqvars == 0 {
                continue;
            }
            let clqvars: Vec<CliqueVar> =
                self.cliqueentries[ck.start as usize..ck.end as usize].to_vec();
            let mut extensionstart = clqvars[0];
            let mut numcliques = self.num_cliques_of(clqvars[0]);
            for &v in &clqvars[1..] {
                let n = self.num_cliques_of(v);
                if n < numcliques {
                    numcliques = n;
                    extensionstart = v;
                }
            }
            for &v in &clqvars {
                self.iscandidate[v.index()] = true;
            }
            self.add_merge_candidates(dom, extensionstart, &mut extensionvars);
            for &v in &clqvars {
                self.iscandidate[v.index()] = false;
            }
            for &v in &extensionvars {
                self.iscandidate[v.index()] = false;
            }
            let mut nq = self.num_neighbourhood_queries;
            for &cv in &clqvars {
                if extensionvars.is_empty() {
                    break;
                }
                if cv == extensionstart {
                    continue;
                }
                let new_size = self.shrink_to_neighbourhood(
                    &mut neighbourhood_inds,
                    &mut nq,
                    cv,
                    &mut extensionvars,
                );
                extensionvars.truncate(new_size);
            }
            if !extensionvars.is_empty() {
                self.randgen.shuffle(&mut extensionvars);
                let mut i = 0;
                while i < extensionvars.len() {
                    let extvar = extensionvars[i];
                    i += 1;
                    let new_size = i + self.shrink_to_neighbourhood(
                        &mut neighbourhood_inds,
                        &mut nq,
                        extvar,
                        &mut extensionvars[i..],
                    );
                    extensionvars.truncate(new_size);
                }
            }
            self.num_neighbourhood_queries = nq;

            if self.cliques[k].equality {
                for &v in &extensionvars.clone() {
                    self.vertex_infeasible(dom, v.c(), v.val);
                }
            } else {
                let originrow = self.cliques[k].origin;
                self.cliques[k].origin = K_HIGHS_IINF;
                let num_extensions = extensionvars.len();
                let c = self.cliques[k];
                extensionvars
                    .extend_from_slice(&self.cliqueentries[c.start as usize..c.end as usize]);
                let mut idx = 0usize;
                let col_deleted = &self.col_deleted;
                extensionvars.retain(|v| {
                    let keep = idx < num_extensions || !col_deleted[v.c()];
                    idx += 1;
                    keep
                });
                self.remove_clique(k as i32);
                let (redundant, dominating_origin, _) =
                    self.clique_subsumption(&extensionvars, false);
                if !redundant {
                    for &v in &extensionvars[..num_extensions] {
                        self.cliqueextensions.push((originrow, v));
                    }
                    extensionvars.retain(|v| {
                        !(dom.is_fixed(v.c())
                            && dom.col_lower()[v.c()] as i32 == (1 - v.val) as i32)
                    });
                    if extensionvars.len() > 1 {
                        self.do_add_clique(&extensionvars, false, originrow);
                    }
                } else if dominating_origin != K_HIGHS_IINF {
                    self.deletedrows.push(originrow);
                } else {
                    for &v in &extensionvars[..num_extensions] {
                        self.cliqueextensions.push((originrow, v));
                    }
                }
            }
            extensionvars.clear();
            self.process_infeasible_vertices(dom);
            if self.num_entries as i64 >= max_new_entries {
                break;
            }
        }
        if have_non_model {
            for k in 0..numcliqueslots {
                let ck = self.cliques[k];
                if ck.start == -1 || ck.origin != -1 {
                    continue;
                }
                extensionvars.clear();
                extensionvars
                    .extend_from_slice(&self.cliqueentries[ck.start as usize..ck.end as usize]);
                self.remove_clique(k as i32);
                self.run_clique_merging_clique(dom, &mut extensionvars, false);
                if extensionvars.len() > 1 {
                    self.do_add_clique(&extensionvars, false, K_HIGHS_IINF);
                }
            }
        }
    }

    // ---- separation ----

    /// `bronKerboschRecurse(data, Plen, X, Xlen)`.
    fn bron_kerbosch_recurse(&self, data: &mut BkData, plen: usize, x: &[CliqueVar]) {
        let sol = data.sol;
        let mut w = data.w_r;
        for i in 0..plen {
            w += data.p[i].weight(sol);
        }
        if w < data.min_w - data.feastol {
            return;
        }
        if plen == 0 && x.is_empty() {
            let clique = data.r.clone();
            if data.min_w < w - data.feastol {
                data.maxcliques -= data.cliques.len() as i32;
                data.cliques.clear();
                data.min_w = w;
            }
            data.cliques.push(clique);
            return;
        }
        data.ncalls += 1;
        if data.stop() {
            return;
        }
        let mut pivweight = -1.0;
        let mut pivot = CliqueVar::new(0, 0);
        for &xi in x {
            if xi.weight(sol) > pivweight {
                pivweight = xi.weight(sol);
                pivot = xi;
                if pivweight >= 1.0 - data.feastol {
                    break;
                }
            }
        }
        if pivweight < 1.0 - data.feastol {
            for i in 0..plen {
                if data.p[i].weight(sol) > pivweight {
                    pivweight = data.p[i].weight(sol);
                    pivot = data.p[i];
                    if pivweight >= 1.0 - data.feastol {
                        break;
                    }
                }
            }
        }
        let mut p_minus_nu = Vec::with_capacity(plen);
        self.query_neighbourhood(
            &mut data.neighbourhood_inds,
            &mut data.num_neighbourhood_queries,
            pivot,
            &data.p[..plen],
        );
        data.neighbourhood_inds.push(plen);
        let mut k = 0usize;
        for &i in &data.neighbourhood_inds {
            while k < i {
                p_minus_nu.push(data.p[k]);
                k += 1;
            }
            k += 1;
        }
        pdqsort(&mut p_minus_nu, |a, b| {
            pair_gt((a.weight(sol), a.index()), (b.weight(sol), b.index()))
        });
        let mut local_x: Vec<CliqueVar> = x.to_vec();
        let mut plen = plen;
        for v in p_minus_nu {
            let new_plen = self.partition_neighbourhood(
                &mut data.neighbourhood_inds,
                &mut data.num_neighbourhood_queries,
                v,
                &mut data.p[..plen],
            );
            let new_xlen = self.partition_neighbourhood(
                &mut data.neighbourhood_inds,
                &mut data.num_neighbourhood_queries,
                v,
                &mut local_x,
            );
            data.r.push(v);
            let wv = v.weight(sol);
            data.w_r += wv;
            self.bron_kerbosch_recurse(data, new_plen, &local_x[..new_xlen]);
            if data.stop() {
                return;
            }
            data.r.pop();
            data.w_r -= wv;
            w -= wv;
            if w < data.min_w {
                return;
            }
            let vpos = (new_plen..plen)
                .find(|&i| data.p[i] == v)
                .expect("bronKerboschRecurse: v not in P");
            plen -= 1;
            data.p.swap(vpos, plen);
            local_x.push(v);
        }
    }

    /// `separateCliques(mipsolver, sol, cutpool, feastol, randgen,
    /// localNumNeighbourhoodQueries)`: the most violated cliques at `sol` (weight
    /// above 1.05), each extended by zero-weight literals, returned as cuts for the
    /// caller's pool. `randgen = None` means the table's own generator, which (as
    /// HiGHS's `&randgen == &this->randgen`) also runs subsumption of the found
    /// cliques against the table. `num_nonzero` and `total_lp_iterations` bound the
    /// neighbourhood queries as in HiGHS.
    #[allow(clippy::too_many_arguments)]
    pub fn separate_cliques<D: CliqueDomain>(
        &mut self,
        dom: &D,
        sol: &[f64],
        integral_cols: &[usize],
        feastol: f64,
        num_nonzero: i64,
        total_lp_iterations: i64,
        randgen: Option<&mut HighsRandom>,
        local_num_neighbourhood_queries: &mut i64,
    ) -> Vec<CliqueCut> {
        let mut data = BkData::new(sol);
        data.feastol = feastol;
        data.max_neighbourhood_queries = 1_000_000 + 100 * num_nonzero + total_lp_iterations * 1000;
        if self.num_neighbourhood_queries > data.max_neighbourhood_queries {
            return Vec::new();
        }
        data.max_neighbourhood_queries -= self.num_neighbourhood_queries;
        for &i in integral_cols {
            if self.colsubstituted[i] != 0 || self.col_deleted[i] {
                continue;
            }
            for val in [0u8, 1] {
                let v = CliqueVar::new(i, val);
                if self.num_cliques_of(v) != 0 {
                    if v.weight(sol) > feastol {
                        data.p.push(v);
                    } else {
                        data.z.push(v);
                    }
                }
            }
        }
        let np = data.p.len();
        self.bron_kerbosch_recurse(&mut data, np, &[]);

        let own = randgen.is_none();
        let mut own_rg;
        let rg: &mut HighsRandom = match randgen {
            Some(r) => r,
            None => {
                own_rg = std::mem::replace(&mut self.randgen, HighsRandom::from_state(0));
                &mut own_rg
            }
        };
        let mut runcliquesubsumption = false;
        let mut cuts = Vec::with_capacity(data.cliques.len());
        for ci in 0..data.cliques.len() {
            let mut clique = std::mem::take(&mut data.cliques[ci]);
            let mut extensionend = data.z.len();
            for &v in &clique {
                extensionend = self.partition_neighbourhood(
                    &mut data.neighbourhood_inds,
                    &mut data.num_neighbourhood_queries,
                    v,
                    &mut data.z[..extensionend],
                );
                if extensionend == 0 {
                    break;
                }
            }
            if extensionend != 0 {
                rg.shuffle(&mut data.z[..extensionend]);
                let mut i = 0;
                while i < extensionend {
                    let k = i + 1;
                    let zi = data.z[i];
                    extensionend = k + self.partition_neighbourhood(
                        &mut data.neighbourhood_inds,
                        &mut data.num_neighbourhood_queries,
                        zi,
                        &mut data.z[k..extensionend],
                    );
                    i += 1;
                }
                clique.extend_from_slice(&data.z[..extensionend]);
            }
            let mut rhs = 1.0;
            // HiGHS tests the table's clique count here, not this clique's length.
            runcliquesubsumption = self.cliques.len() > 2;
            let mut inds = Vec::with_capacity(clique.len());
            let mut vals = Vec::with_capacity(clique.len());
            for &v in &clique {
                inds.push(v.c());
                if v.val == 0 {
                    vals.push(-1.0);
                    rhs -= 1.0;
                } else {
                    vals.push(1.0);
                }
            }
            cuts.push(CliqueCut {
                inds,
                vals,
                rhs: (rhs + 0.5f64).floor(),
            });
            data.cliques[ci] = clique;
        }
        if own {
            self.randgen = std::mem::replace(rg, HighsRandom::from_state(0));
        }
        *local_num_neighbourhood_queries += data.num_neighbourhood_queries;
        if runcliquesubsumption && own {
            for mut clique in std::mem::take(&mut data.cliques) {
                let nremoved = self.run_clique_subsumption(dom, &mut clique);
                if clique.is_empty() {
                    continue;
                }
                if nremoved != 0 {
                    self.do_add_clique(&clique, false, -1);
                }
            }
        }
        cuts
    }

    /// `computeMaximalCliques(vars, feastol)`: maximal cliques among `vars` treated as a
    /// 0/1 point.
    pub fn compute_maximal_cliques(&self, vars: &[CliqueVar], feastol: f64) -> Vec<Vec<CliqueVar>> {
        if vars.is_empty() {
            return Vec::new();
        }
        let maxcol = vars.iter().map(|v| v.c()).max().unwrap();
        let mut sol = vec![0.0; maxcol + 1];
        for v in vars {
            sol[v.c()] = v.val as f64;
        }
        let mut data = BkData::new(&sol);
        data.feastol = feastol;
        for &v in vars {
            if self.colsubstituted[v.c()] != 0 || self.col_deleted[v.c()] {
                continue;
            }
            if self.num_cliques_of(v) != 0 {
                data.p.push(v);
            }
        }
        let np = data.p.len();
        self.bron_kerbosch_recurse(&mut data, np, &[]);
        data.cliques
    }

    // ---- rebuild after presolve ----

    /// `rebuild(ncols, postSolveStack, globaldomain, cIndex, rIndex)`: the table over
    /// the reduced columns; `orig2reducedcol[c] = -1` for a removed column and
    /// `is_linearly_transformable(reduced col)` is `postSolveStack.isColLinearlyTransformable`.
    pub fn rebuild<D: CliqueDomain>(
        &mut self,
        ncols: usize,
        dom: &D,
        orig2reducedcol: &[i32],
        is_linearly_transformable: impl Fn(usize) -> bool,
    ) {
        let mut new_table = CliqueTable::new(ncols);
        new_table.set_presolve_flag(self.in_presolve);
        const DROPPED: u32 = K_HIGHS_IINF as u32;
        for i in 0..self.cliques.len() {
            let c = self.cliques[i];
            if c.start == -1 {
                continue;
            }
            let oldnumvars = c.end - c.start;
            for k in c.start as usize..c.end as usize {
                let col = orig2reducedcol[self.cliqueentries[k].c()];
                if col == -1
                    || !dom.is_binary(col as usize)
                    || !is_linearly_transformable(col as usize)
                {
                    self.cliqueentries[k].col = DROPPED;
                } else {
                    self.cliqueentries[k].col = col as u32;
                }
            }
            let kept: Vec<CliqueVar> = self.cliqueentries[c.start as usize..c.end as usize]
                .iter()
                .copied()
                .filter(|v| v.col != DROPPED)
                .collect();
            let numvars = kept.len() as i32;
            // dropped members are relaxed out, so an equality no longer holds
            if numvars <= 1 {
                continue;
            }
            let origin = if c.origin != K_HIGHS_IINF {
                -1
            } else {
                K_HIGHS_IINF
            };
            new_table.do_add_clique(
                &kept,
                if numvars != oldnumvars {
                    false
                } else {
                    c.equality
                },
                origin,
            );
        }
        *self = new_table;
    }

    /// `buildFrom(origModel, init)`: the table of `init` restricted to the columns
    /// that are binary in the original model (`orig_lb`, `orig_ub`), keeping `init`'s
    /// substitutions.
    pub fn build_from(&mut self, orig_lb: &[f64], orig_ub: &[f64], init: &CliqueTable) {
        assert_eq!(init.colsubstituted.len(), self.colsubstituted.len());
        let ncols = init.colsubstituted.len();
        let mut new_table = CliqueTable::new(ncols);
        new_table.set_presolve_flag(self.in_presolve);
        for c in &init.cliques {
            if c.start == -1 || c.num_active() <= 1 {
                continue;
            }
            let clq: Vec<CliqueVar> = init.cliqueentries[c.start as usize..c.end as usize]
                .iter()
                .copied()
                .filter(|v| !(orig_lb[v.c()] != 0.0 || orig_ub[v.c()] != 1.0))
                .collect();
            if clq.len() <= 1 {
                continue;
            }
            let origin = if c.origin != K_HIGHS_IINF {
                -1
            } else {
                K_HIGHS_IINF
            };
            new_table.do_add_clique(&clq, false, origin);
        }
        new_table.colsubstituted = init.colsubstituted.clone();
        new_table.substitutions = init.substitutions.clone();
        *self = new_table;
    }
}

#[cfg(test)]
mod tests {
    use super::super::domain::DomainModel;
    use super::super::objective::ObjectiveFunction;
    use super::*;
    use std::sync::Arc;

    /// A bare box: bounds and a change stack, no propagation.
    #[derive(Clone)]
    struct BoxDomain {
        lb: Vec<f64>,
        ub: Vec<f64>,
        is_int: Vec<bool>,
        infeasible: bool,
        stack: Vec<usize>,
        nnz: usize,
    }

    impl BoxDomain {
        fn new(lb: Vec<f64>, ub: Vec<f64>, is_int: Vec<bool>) -> Self {
            BoxDomain {
                lb,
                ub,
                is_int,
                infeasible: false,
                stack: Vec::new(),
                nnz: 0,
            }
        }
        fn binaries(n: usize) -> Self {
            Self::new(vec![0.0; n], vec![1.0; n], vec![true; n])
        }
    }

    impl CliqueDomain for BoxDomain {
        fn col_lower(&self) -> &[f64] {
            &self.lb
        }
        fn col_upper(&self) -> &[f64] {
            &self.ub
        }
        fn integrality(&self) -> &[bool] {
            &self.is_int
        }
        fn infeasible(&self) -> bool {
            self.infeasible
        }
        fn feastol(&self) -> f64 {
            1e-6
        }
        fn change_bound(&mut self, bt: BoundType, col: usize, val: f64, _r: CliqueReason) {
            match bt {
                BoundType::Lower => {
                    if val <= self.lb[col] {
                        return;
                    }
                    self.lb[col] = val;
                }
                BoundType::Upper => {
                    if val >= self.ub[col] {
                        return;
                    }
                    self.ub[col] = val;
                }
            }
            if self.lb[col] > self.ub[col] + 1e-6 {
                self.infeasible = true;
            }
            self.stack.push(col);
        }
        fn propagate(&mut self) {}
        fn domain_change_stack_len(&self) -> usize {
            self.stack.len()
        }
        fn domain_change_col(&self, pos: usize) -> usize {
            self.stack[pos]
        }
        fn num_model_nonzeros(&self) -> usize {
            self.nnz
        }
    }

    fn rows_of<'a>(
        start: &'a [usize],
        index: &'a [usize],
        value: &'a [f64],
        lo: &'a [f64],
        up: &'a [f64],
    ) -> CliqueRows<'a> {
        CliqueRows {
            ar_start: start,
            ar_index: index,
            ar_value: value,
            row_lower: lo,
            row_upper: up,
            num_orig_rows: lo.len(),
            feastol: 1e-6,
            epsilon: 1e-9,
        }
    }

    fn lit(c: usize, v: u8) -> CliqueVar {
        CliqueVar::new(c, v)
    }

    #[test]
    fn set_packing_row_yields_its_clique() {
        // x0 + x1 + x2 <= 1 ; x2 + x3 <= 1
        let start = [0, 3, 5];
        let index = [0, 1, 2, 2, 3];
        let value = [1.0; 5];
        let lo = [-INF, -INF];
        let up = [1.0, 1.0];
        let rows = rows_of(&start, &index, &value, &lo, &up);
        let mut dom = BoxDomain::binaries(4);
        let mut t = CliqueTable::new(4);
        t.extract_cliques(&mut dom, &rows, None, true);
        let cl = t.cliques();
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].1, &[lit(0, 1), lit(1, 1), lit(2, 1)]);
        assert_eq!(cl[0].3, 0, "origin is the row");
        assert!(!cl[0].2);
        assert_eq!(cl[1].1, &[lit(2, 1), lit(3, 1)]);
        assert!(t.have_common_clique(lit(0, 1), lit(2, 1)));
        assert!(t.have_common_clique(lit(3, 1), lit(2, 1)));
        assert!(!t.have_common_clique(lit(0, 1), lit(3, 1)));
        assert!(!t.have_common_clique(lit(0, 0), lit(1, 1)));
        assert_eq!(t.find_common_clique(lit(1, 1), lit(0, 1)).unwrap().len(), 3);
        assert_eq!(t.num_entries(), 5);
        // x2 = 1: implies x0 = x1 = x3 = 0. Counts: 2 cliques + (3-1)-1 for the size-3.
        assert_eq!(t.num_implications_val(2, true), 3);
        assert_eq!(t.num_implications(2), 3);
        assert_eq!(t.num_fixings(), 0);
    }

    #[test]
    fn knapsack_row_yields_the_clique_of_its_large_items() {
        // 3 x0 + 3 x1 + 2 x2 + x3 <= 4 : {x0, x1, x2} pairwise exceed 4, x3 fits with any.
        let start = [0, 4];
        let index = [0, 1, 2, 3];
        let value = [3.0, 3.0, 2.0, 1.0];
        let lo = [-INF];
        let up = [4.0];
        let rows = rows_of(&start, &index, &value, &lo, &up);
        let mut dom = BoxDomain::binaries(4);
        let mut t = CliqueTable::new(4);
        t.extract_cliques(&mut dom, &rows, None, true);
        let cl = t.cliques();
        assert_eq!(cl.len(), 1);
        let mut members: Vec<usize> = cl[0].1.iter().map(|v| v.c()).collect();
        members.sort();
        assert_eq!(members, vec![0, 1, 2]);
        assert_eq!(cl[0].3, K_HIGHS_IINF);
        // a complemented knapsack: 2 x0 - 2 x1 <= 0 (x0 <= x1) gives clique {x0, ~x1}
        let start = [0, 2];
        let index = [0, 1];
        let value = [2.0, -2.0];
        let lo = [-INF];
        let up = [0.0];
        let rows = rows_of(&start, &index, &value, &lo, &up);
        let mut dom = BoxDomain::binaries(2);
        let mut t = CliqueTable::new(2);
        t.extract_cliques(&mut dom, &rows, None, true);
        assert!(t.have_common_clique(lit(0, 1), lit(1, 0)));
    }

    #[test]
    fn clique_implies_fixings_for_a_bound_change() {
        // x0 + x1 + x2 + x3 = 1 (equality clique)
        let start = [0, 4];
        let index = [0, 1, 2, 3];
        let value = [1.0; 4];
        let lo = [1.0];
        let up = [1.0];
        let rows = rows_of(&start, &index, &value, &lo, &up);
        let mut dom = BoxDomain::binaries(5);
        let mut t = CliqueTable::new(5);
        t.extract_cliques(&mut dom, &rows, None, true);
        assert!(t.cliques()[0].2, "equality clique");
        // the domain fixes x2 = 1: the clique fixes x0, x1, x3 to 0
        let mut local = dom.clone();
        local.change_bound(BoundType::Lower, 2, 1.0, CliqueReason::Unspecified);
        t.add_implications(&mut local, 2, 1);
        assert_eq!(local.ub, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
        assert!(!local.infeasible);
        // equality: getNumImplications counts both directions
        assert_eq!(t.num_implications_val(2, true), 1 + 2 * 3 - 1);
        // a member already fixed to 1 in the global domain fixes the rest at addClique
        let mut dom = BoxDomain::binaries(3);
        dom.lb[1] = 1.0;
        let mut t = CliqueTable::new(3);
        t.add_clique(
            &mut dom,
            &[lit(0, 1), lit(1, 1), lit(2, 1)],
            false,
            K_HIGHS_IINF,
        );
        assert_eq!(dom.ub, vec![0.0, 1.0, 0.0]);
        assert_eq!(t.num_fixings(), 2);
        assert_eq!(t.num_cliques(), 0);
    }

    #[test]
    fn equality_edge_substitutes_and_conflicting_edges_fix() {
        // x0 + x1 = 1: x1 is substituted by ~x0
        let mut dom = BoxDomain::binaries(3);
        let mut t = CliqueTable::new(3);
        t.add_clique(&mut dom, &[lit(0, 1), lit(1, 1)], true, K_HIGHS_IINF);
        // adding x0 + x1 <= 1 first, then ~x0 + ~x1 <= 1 makes the cover
        let mut t2 = CliqueTable::new(3);
        let mut dom2 = BoxDomain::binaries(3);
        t2.add_clique(&mut dom2, &[lit(0, 1), lit(1, 1)], false, K_HIGHS_IINF);
        t2.add_clique(&mut dom2, &[lit(0, 0), lit(1, 0)], false, K_HIGHS_IINF);
        let s = t2.substitutions();
        assert_eq!(s.len(), 1);
        assert_eq!(
            s[0],
            Substitution {
                substcol: 1,
                replace: lit(0, 0)
            }
        );
        assert_eq!(t2.resolve_substitution(lit(1, 1)), lit(0, 0));
        // with x1 = ~x0, fixing x0 = 1 implies x1 = 0 through the substitution
        let mut local = dom2.clone();
        local.change_bound(BoundType::Upper, 1, 0.0, CliqueReason::Unspecified);
        t2.add_implications(&mut local, 1, 0);
        assert_eq!(local.lb[0], 1.0);
        // the first table took the equality path: its size-two equality clique alone
        // is no cover, so it is just stored
        assert!(t.have_common_clique(lit(0, 1), lit(1, 1)));
        // x0 + x2 <= 1 and x0 + ~x2 <= 1: x0 = 0
        let mut dom3 = BoxDomain::binaries(3);
        let mut t3 = CliqueTable::new(3);
        t3.add_clique(&mut dom3, &[lit(0, 1), lit(2, 1)], false, K_HIGHS_IINF);
        t3.add_clique(&mut dom3, &[lit(0, 1), lit(2, 0)], false, K_HIGHS_IINF);
        assert_eq!(dom3.ub[0], 0.0);
        assert!(t3.num_fixings() >= 1);
    }

    #[test]
    fn objective_partition_plugs_into_the_objective_function() {
        // HiGHS partitions the literals (col, cost < 0): negative-cost columns as x,
        // positive-cost ones as ~x. Cliques {x0, x1, x2} and {~x3, ~x4}; x5 alone.
        let mut dom = BoxDomain::binaries(6);
        let mut t = CliqueTable::new(6);
        t.add_clique(&mut dom, &[lit(0, 1), lit(1, 1), lit(2, 1)], false, 0);
        t.add_clique(&mut dom, &[lit(3, 0), lit(4, 0)], false, 1);
        // a clique over x3, x4 themselves is irrelevant for positive costs
        t.add_clique(&mut dom, &[lit(4, 1), lit(5, 1)], false, 2);
        let cost = [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0];
        let is_int = [true; 6];
        let (lb, ub) = ([0.0; 6], [1.0; 6]);
        let mut f = ObjectiveFunction::new(&cost, &is_int, &lb, &ub);
        let binaries: Vec<usize> = f.nonzeros()[..f.num_binary()].to_vec();
        let (cols, starts) = t.objective_clique_partition(&cost, &binaries);
        assert_eq!(*starts.last().unwrap(), 6);
        let mut parts: Vec<Vec<usize>> = starts
            .windows(2)
            .map(|w| {
                let mut p = cols[w[0]..w[1]].to_vec();
                p.sort();
                p
            })
            .collect();
        parts.sort();
        assert_eq!(parts, vec![vec![0, 1, 2], vec![3, 4], vec![5]]);
        f.setup_clique_partition(&cols, &starts);
        assert_eq!(f.num_partitions(), 2);
        for c in 0..5 {
            assert!(f.col_partition_pos(c) >= 0);
        }
        assert_eq!(f.col_partition_pos(5), -1);
    }

    /// Small binary programs (plus one general integer), every feasible point
    /// enumerated. Asserts: no feasible point is cut off by any clique, equality
    /// clique, substitution, global fixing, implied fixing, separated cut or
    /// partition; probes count what each check saw (CLAUDE.md §6).
    fn run_random(use_real_domain: bool) -> [usize; 9] {
        let mut rng = HighsRandom::new(12345 + use_real_domain as u32);
        let mut cnt = [0usize; 9];
        let (mut instances, mut cliques_seen, mut fixings, mut impl_checks) = (0, 0, 0, 0);
        let (mut multi_partitions, mut cut_checks, mut substitutions, mut pair_checks) =
            (0, 0, 0, 0);
        let mut obj_partitions = 0;
        for _ in 0..400 {
            let nb = 5 + rng.integer_below(4) as usize;
            let has_int = rng.bit();
            let n = nb + has_int as usize;
            let mut is_int = vec![true; n];
            let lb0 = vec![0.0; n];
            let mut ub0 = vec![1.0; n];
            if has_int {
                ub0[nb] = 3.0;
            }
            // a continuous column sometimes replaces the integer one
            if has_int && rng.bit() {
                is_int[nb] = false;
                ub0[nb] = 2.0;
            }
            let nrows = 2 + rng.integer_below(4) as usize;
            let mut rows: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
            let (mut lo, mut up) = (Vec::new(), Vec::new());
            for _ in 0..nrows {
                let kind = rng.integer_below(4);
                let len = 2 + rng.integer_below(3) as usize;
                let mut cols: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut cols);
                cols.truncate(len.min(n));
                match kind {
                    0 | 1 => {
                        // set packing / partitioning, possibly with complemented literals
                        let mut vals = Vec::new();
                        let mut rhs = 1.0;
                        cols.retain(|&c| c < nb);
                        for _ in &cols {
                            if rng.integer_below(4) == 0 {
                                vals.push(-1.0);
                                rhs -= 1.0;
                            } else {
                                vals.push(1.0);
                            }
                        }
                        let eq = kind == 1 && rng.bit();
                        lo.push(if eq { rhs } else { -INF });
                        up.push(rhs);
                        rows.push((cols, vals));
                    }
                    _ => {
                        let vals: Vec<f64> = cols
                            .iter()
                            .map(|_| {
                                let a = 1 + rng.integer_below(6);
                                if rng.integer_below(3) == 0 {
                                    -(a as f64)
                                } else {
                                    a as f64
                                }
                            })
                            .collect();
                        let pos: f64 = vals.iter().filter(|v| **v > 0.0).sum();
                        let r = (pos * rng.fraction()).floor();
                        if rng.integer_below(4) == 0 {
                            lo.push(r - 2.0);
                            up.push(INF);
                        } else {
                            lo.push(-INF);
                            up.push(r);
                        }
                        rows.push((cols, vals));
                    }
                }
            }
            // enumerate feasible points (the continuous column on a 0.25 grid)
            let mut points: Vec<Vec<f64>> = Vec::new();
            let extra: Vec<f64> = if has_int {
                if is_int[nb] {
                    vec![0.0, 1.0, 2.0, 3.0]
                } else {
                    (0..=8).map(|k| 0.25 * k as f64).collect()
                }
            } else {
                vec![0.0]
            };
            for mask in 0..(1u32 << nb) {
                for &e in &extra {
                    let mut x: Vec<f64> = (0..nb).map(|j| ((mask >> j) & 1) as f64).collect();
                    if has_int {
                        x.push(e);
                    }
                    let ok = rows
                        .iter()
                        .zip(lo.iter().zip(&up))
                        .all(|((c, v), (&l, &u))| {
                            let a: f64 = c.iter().zip(v).map(|(&j, &w)| w * x[j]).sum();
                            a >= l - 1e-9 && a <= u + 1e-9
                        });
                    if ok {
                        points.push(x);
                    }
                }
            }
            if points.is_empty() {
                continue;
            }
            instances += 1;
            let model = Arc::new(DomainModel::from_rows(
                n,
                &rows,
                lo.clone(),
                up.clone(),
                is_int.clone(),
            ));
            let nnz: usize = rows.iter().map(|r| r.0.len()).sum();
            let mut real = Domain::new(Arc::clone(&model), &lb0, &ub0, 1e-6, 1e-9);
            let mut boxd = BoxDomain::new(lb0.clone(), ub0.clone(), is_int.clone());
            boxd.nnz = nnz;
            let mut table = CliqueTable::new(n);
            table.set_max_entries(nnz as i32);
            let mut vb = VarBoundStore::new(n, 1e-6);
            let rows_csr = (
                model.ar_start.clone(),
                model.ar_index.clone(),
                model.ar_value.clone(),
            );
            let cr = CliqueRows {
                ar_start: &rows_csr.0,
                ar_index: &rows_csr.1,
                ar_value: &rows_csr.2,
                row_lower: &lo,
                row_upper: &up,
                num_orig_rows: lo.len(),
                feastol: 1e-6,
                epsilon: 1e-9,
            };
            macro_rules! with_dom {
                ($d:ident, $body:expr) => {
                    if use_real_domain {
                        let mut view = GlobalDomainView {
                            dom: &mut real,
                            is_int: &is_int,
                            feastol: 1e-6,
                            num_model_nonzeros: nnz,
                        };
                        let $d = &mut view;
                        $body
                    } else {
                        let $d = &mut boxd;
                        $body
                    }
                };
            }
            with_dom!(d, {
                d.propagate();
                table.extract_cliques(d, &cr, Some(&mut vb), true);
                for (r, (c, v)) in rows.iter().enumerate() {
                    if d.infeasible() {
                        break;
                    }
                    if up[r] < INF {
                        table.extract_cliques_from_cut(d, Some(&mut vb), c, v, up[r], 1e-6);
                    }
                    if lo[r] > -INF {
                        let nv: Vec<f64> = v.iter().map(|w| -w).collect();
                        table.extract_cliques_from_cut(d, Some(&mut vb), c, &nv, -lo[r], 1e-6);
                    }
                }
                if !d.infeasible() {
                    table.run_clique_merging(d);
                }
                if !d.infeasible() {
                    table.cleanup_fixed(d);
                }
                assert!(
                    !d.infeasible(),
                    "table declared a feasible program infeasible"
                );
            });
            let (glb, gub) = if use_real_domain {
                (real.col_lower.clone(), real.col_upper.clone())
            } else {
                (boxd.lb.clone(), boxd.ub.clone())
            };
            // global fixings keep every feasible point
            for j in 0..n {
                if glb[j] > lb0[j] || gub[j] < ub0[j] {
                    fixings += 1;
                }
            }
            for x in &points {
                for j in 0..n {
                    assert!(
                        x[j] >= glb[j] - 1e-9 && x[j] <= gub[j] + 1e-9,
                        "global bound of x{j} cuts off {x:?}"
                    );
                }
            }
            // every clique is valid; equality cliques exactly one
            let cl: Vec<(Vec<CliqueVar>, bool)> = table
                .cliques()
                .iter()
                .map(|c| (c.1.to_vec(), c.2))
                .collect();
            cliques_seen += cl.len();
            for (members, eq) in &cl {
                for x in &points {
                    let ones = members.iter().filter(|v| v.value_at(x)).count();
                    assert!(ones <= 1, "clique {members:?} has {ones} true at {x:?}");
                    if *eq {
                        assert_eq!(ones, 1, "equality clique {members:?} at {x:?}");
                    }
                }
            }
            // substitutions hold
            for s in table.substitutions() {
                substitutions += 1;
                for x in &points {
                    assert_eq!(
                        x[s.substcol] > 0.5,
                        s.replace.value_at(x),
                        "substitution {s:?} at {x:?}"
                    );
                }
            }
            // pairwise queries agree with the points
            for a in 0..2 * nb {
                for b in 0..2 * nb {
                    let (va, vbb) = (lit(a / 2, (a % 2) as u8), lit(b / 2, (b % 2) as u8));
                    if table.have_common_clique(va, vbb) {
                        pair_checks += 1;
                        for x in &points {
                            assert!(
                                !(va.value_at(x) && vbb.value_at(x)),
                                "edge {va:?}-{vbb:?} at {x:?}"
                            );
                        }
                    }
                }
            }
            // implied fixings: fixing a binary at a feasible point's value keeps that point
            for x in &points {
                for c in 0..nb {
                    if glb[c] == gub[c] {
                        continue;
                    }
                    let val = (x[c] > 0.5) as u8;
                    let mut local = BoxDomain::new(glb.clone(), gub.clone(), is_int.clone());
                    local.change_bound(
                        if val == 1 {
                            BoundType::Lower
                        } else {
                            BoundType::Upper
                        },
                        c,
                        val as f64,
                        CliqueReason::Unspecified,
                    );
                    let before = local.stack.len();
                    table.add_implications(&mut local, c, val);
                    assert!(
                        !local.infeasible,
                        "implications of x{c}={val} infeasible at {x:?}"
                    );
                    for j in 0..n {
                        assert!(
                            x[j] >= local.lb[j] - 1e-9 && x[j] <= local.ub[j] + 1e-9,
                            "x{c}={val} implied a bound on x{j} cutting off {x:?}"
                        );
                    }
                    impl_checks += local.stack.len() - before;
                }
            }
            // clique partition of all live literals with random signs
            let mut clq: Vec<CliqueVar> = (0..nb)
                .filter(|&c| glb[c] != gub[c] && table.substitution(c).is_none())
                .map(|c| lit(c, rng.bit() as u8))
                .collect();
            let starts = table.clique_partition(&mut clq);
            assert_eq!(*starts.last().unwrap(), clq.len());
            for w in starts.windows(2) {
                let part = &clq[w[0]..w[1]];
                if part.len() >= 2 {
                    multi_partitions += 1;
                }
                for x in &points {
                    assert!(
                        part.iter().filter(|v| v.value_at(x)).count() <= 1,
                        "partition {part:?} at {x:?}"
                    );
                }
            }
            // objective partition through the objective function
            let cost: Vec<f64> = (0..n).map(|_| rng.integer_below(7) as f64 - 3.0).collect();
            let mut f = ObjectiveFunction::new(&cost, &is_int, &glb, &gub);
            if f.num_binary() > 1 {
                let bins: Vec<usize> = f.nonzeros()[..f.num_binary()].to_vec();
                let (cols, starts) = table.objective_clique_partition(&cost, &bins);
                f.setup_clique_partition(&cols, &starts);
                let ps = f.partition_starts().to_vec();
                for w in ps.windows(2) {
                    obj_partitions += 1;
                    let part = &f.nonzeros()[w[0]..w[1]];
                    for x in &points {
                        let ones = part
                            .iter()
                            .filter(|&&c| lit(c, (cost[c] < 0.0) as u8).value_at(x))
                            .count();
                        assert!(ones <= 1, "objective partition {part:?} at {x:?}");
                    }
                }
            }
            // separation at a random fractional point: every cut is valid
            let sol: Vec<f64> = (0..n)
                .map(|j| if j < nb { rng.fraction() } else { 0.0 })
                .collect();
            let integral: Vec<usize> = (0..nb).collect();
            let mut lq = 0i64;
            let cuts = table.separate_cliques(
                &boxd_view(&glb, &gub, &is_int),
                &sol,
                &integral,
                1e-6,
                nnz as i64,
                0,
                None,
                &mut lq,
            );
            for cut in &cuts {
                for x in &points {
                    cut_checks += 1;
                    let a: f64 = cut
                        .inds
                        .iter()
                        .zip(&cut.vals)
                        .map(|(&j, &w)| w * x[j])
                        .sum();
                    assert!(a <= cut.rhs + 1e-9, "clique cut {cut:?} cuts off {x:?}");
                }
            }
            // after separation's subsumption the table is still valid
            for c in table.cliques() {
                for x in &points {
                    assert!(c.1.iter().filter(|v| v.value_at(x)).count() <= 1);
                }
            }
        }
        cnt[0] = instances;
        cnt[1] = cliques_seen;
        cnt[2] = fixings;
        cnt[3] = impl_checks;
        cnt[4] = multi_partitions;
        cnt[5] = cut_checks;
        cnt[6] = substitutions;
        cnt[7] = pair_checks;
        cnt[8] = obj_partitions;
        cnt
    }

    fn boxd_view(lb: &[f64], ub: &[f64], is_int: &[bool]) -> BoxDomain {
        BoxDomain::new(lb.to_vec(), ub.to_vec(), is_int.to_vec())
    }

    fn check_counts(c: [usize; 9]) {
        eprintln!(
            "instances {} cliques {} fixings {} implied-fixings {} multi-partitions {} \
             cut-checks {} substitutions {} edge-checks {} obj-partitions {}",
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]
        );
        assert!(c[0] >= 200, "instances {}", c[0]);
        assert!(c[1] >= 300, "cliques {}", c[1]);
        assert!(c[2] >= 30, "fixings {}", c[2]);
        assert!(c[3] >= 2000, "implied fixings {}", c[3]);
        assert!(c[4] >= 100, "multi-member partitions {}", c[4]);
        assert!(c[5] >= 1000, "cut checks {}", c[5]);
        assert!(c[6] >= 20, "substitutions {}", c[6]);
        assert!(c[7] >= 1000, "edge checks {}", c[7]);
        assert!(c[8] >= 30, "objective partitions {}", c[8]);
    }

    #[test]
    fn random_binary_programs_box_domain() {
        check_counts(run_random(false));
    }

    #[test]
    fn random_binary_programs_translated_domain() {
        check_counts(run_random(true));
    }

    #[test]
    fn rebuild_and_build_from_keep_valid_cliques() {
        let mut dom = BoxDomain::binaries(5);
        let mut t = CliqueTable::new(5);
        t.add_clique(
            &mut dom,
            &[lit(0, 1), lit(1, 1), lit(2, 1), lit(3, 1)],
            true,
            7,
        );
        t.add_clique(&mut dom, &[lit(3, 0), lit(4, 1)], false, K_HIGHS_IINF);
        // column 1 removed by presolve; the rest renumbered 0..3
        let o2r = [0, -1, 1, 2, 3];
        let reduced = BoxDomain::binaries(4);
        let mut r = t.clone();
        r.rebuild(4, &reduced, &o2r, |_| true);
        let cl = r.cliques();
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].1, &[lit(0, 1), lit(1, 1), lit(2, 1)]);
        assert!(!cl[0].2, "a relaxed equality is no longer one");
        assert_eq!(cl[0].3, -1);
        assert_eq!(cl[1].1, &[lit(2, 0), lit(3, 1)]);
        // buildFrom drops columns that are not binary in the original model
        let mut b = CliqueTable::new(5);
        b.build_from(&[0.0; 5], &[1.0, 1.0, 1.0, 1.0, 2.0], &t);
        let cl = b.cliques();
        assert_eq!(cl.len(), 1);
        assert_eq!(cl[0].1.len(), 4);
        assert_eq!(cl[0].3, -1);
    }
}
