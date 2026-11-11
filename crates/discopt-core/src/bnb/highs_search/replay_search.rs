//! Replay of HiGHS search-control dumps (`HS_DUMP_SEARCH`, written by
//! `docs/dev/data/highs-parity/search/hs_dump_search.patch`; format in
//! `docs/dev/highs-mip-replacement-plan.md`, stage 1b).
//!
//! The translated control (`search.rs`) runs against [`Mock`], which answers every
//! component call with HiGHS's recorded outcome (node evaluation results and bounds,
//! branching decisions, domain prunes, pseudocost scores, whether heuristics ran) and,
//! at every point where the dump records a decision, requires the control's decision and
//! values to be HiGHS's. The node queue is the translated queue, fed the queue
//! operations the control makes and the ones components make (the root node, bounding
//! after a new incumbent, global pruning); every recorded queue state is compared.

use super::node_queue::{DomainChange, NodeQueue, OpenNode};
use super::search::{
    BranchOutcome, BranchStep, InstallKind, MainLoop, NodeData, NodeResult, QueueHandoff, Search,
    SearchEnv, NO_NODE_LIMIT,
};

fn hx(t: &str) -> f64 {
    crate::lp::highs_cutgen::replay::parse_hex_f64(t)
}

/// Bitwise equality of doubles (the dump writes `%a`).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a == b)
}

/// Replay counts.
#[derive(Debug, Default)]
pub(crate) struct SearchStats {
    pub rounds: usize,
    pub evaluations: usize,
    pub branchings: usize,
    pub siblings: usize,
    pub plunge_choices: usize,
    pub to_queue: usize,
    pub handoffs: usize,
    pub handoffs_current: usize,
    pub plunge_limits: usize,
    pub ancestors_closed: usize,
    pub commits: usize,
    pub limit_stops: usize,
    pub leftover_events: usize,
    pub installs: usize,
    pub queue_ops: usize,
    pub queue_states: usize,
    pub mismatches: Vec<String>,
}

pub(crate) struct Mock {
    lines: Vec<Vec<String>>,
    pos: usize,
    main_tag: String,
    queue: NodeQueue,
    cutoff: f64,
    optlim: f64,
    feastol: f64,
    pub s: SearchStats,
}

impl Mock {
    pub fn new(text: &str) -> Self {
        let lines: Vec<Vec<String>> = text
            .lines()
            .map(|l| l.split_ascii_whitespace().map(String::from).collect())
            .filter(|t: &Vec<String>| !t.is_empty())
            .collect();
        // The main queue: the one popped right after the root install (`MI R`).
        let mi = lines
            .iter()
            .position(|t| t[0] == "MI")
            .expect("no MI event");
        let main_tag = lines[mi..]
            .iter()
            .find(|t| t[0] == "QP")
            .map(|t| t[1].clone())
            .expect("no pop after MI");
        let ncol = lines
            .iter()
            .find(|t| t[0] == "QN" && t[1] == main_tag)
            .map(|t| t[2].parse().unwrap())
            .unwrap_or(0);
        let mut m = Mock {
            lines,
            pos: 0,
            main_tag,
            queue: NodeQueue::new(ncol),
            cutoff: f64::INFINITY,
            optlim: f64::INFINITY,
            feastol: 1e-6,
            s: SearchStats::default(),
        };
        m.refresh_limits();
        m
    }

    fn bad(&mut self, what: String) {
        if std::env::var("DISCOPT_HIGHS_SEARCH_REPLAY_VERBOSE").is_ok() {
            eprintln!("  mismatch at event {}: {what}", self.pos);
        }
        if self.s.mismatches.len() < 200 {
            let at = self.pos;
            self.s.mismatches.push(format!("event {at}: {what}"));
        }
    }

    /// A queue line of another queue (sub-MIPs), or a `QN`.
    fn foreign(&self, t: &[String]) -> bool {
        t[0].starts_with('Q') && t.len() > 1 && (t[1] != self.main_tag || t[0] == "QN")
    }

    /// Skip foreign queue lines; apply component queue operations (`QE`/`QP` are not
    /// component operations: the control makes them), checking each `QS`.
    fn skip(&mut self) {
        while self.pos < self.lines.len() {
            let t = self.lines[self.pos].clone();
            if self.foreign(&t) {
                self.pos += 1;
                continue;
            }
            match t[0].as_str() {
                "QB" => {
                    self.queue.set_optimality_limit(hx(&t[3]));
                    self.queue.perform_bounding(hx(&t[2]));
                    self.s.queue_ops += 1;
                }
                "QG" => {
                    let n: usize = t[2].parse().unwrap();
                    for v in &t[3..3 + n] {
                        self.queue.prune_node(v.parse().unwrap());
                    }
                    self.s.queue_ops += 1;
                }
                "QC" => {
                    self.queue.clear();
                    self.s.queue_ops += 1;
                }
                "QS" => self.check_state(&t),
                "QE" if self.pos < 2 || !self.port_owns_emplace() => {
                    // The root node, emplaced by the solver before the search.
                    self.apply_emplace(&t);
                }
                _ => return,
            }
            self.pos += 1;
        }
    }

    /// Whether the next `QE` is made by the control (after `SPQ`/`SQ`), not a component.
    fn port_owns_emplace(&self) -> bool {
        // Look back for the last non-queue event.
        let mut i = self.pos;
        while i > 0 {
            i -= 1;
            let t = &self.lines[i];
            if self.foreign(t) || t[0].starts_with('Q') {
                continue;
            }
            return t[0] == "SPQ" || t[0] == "SQ";
        }
        false
    }

    fn apply_emplace(&mut self, t: &[String]) {
        let (slot, d, b, lb, est, depth, opt) = Self::parse_qe(t);
        self.queue.set_optimality_limit(opt);
        let (got, _) = self.queue.emplace_node(d, b, lb, est, depth);
        if got != slot {
            self.bad(format!("component emplace slot {got} vs {slot}"));
        }
        self.s.queue_ops += 1;
    }

    #[allow(clippy::type_complexity)]
    fn parse_qe(t: &[String]) -> (i64, Vec<DomainChange>, Vec<i32>, f64, f64, i32, f64) {
        let slot: i64 = t[2].parse().unwrap();
        let (lb, est) = (hx(&t[3]), hx(&t[4]));
        let depth: i32 = t[5].parse().unwrap();
        let opt = hx(&t[6]);
        let n: usize = t[7].parse().unwrap();
        let mut k = 8;
        let mut d = Vec::with_capacity(n);
        for _ in 0..n {
            d.push(DomainChange {
                col: t[k].parse().unwrap(),
                upper: t[k + 1] == "1",
                val: hx(&t[k + 2]),
            });
            k += 3;
        }
        let nb: usize = t[k].parse().unwrap();
        let b: Vec<i32> = t[k + 1..k + 1 + nb]
            .iter()
            .map(|v| v.parse().unwrap())
            .collect();
        (slot, d, b, lb, est, depth, opt)
    }

    fn check_state(&mut self, t: &[String]) {
        let want: Vec<i64> = t[2..6].iter().map(|v| v.parse().unwrap()).collect();
        let q = &self.queue;
        let got = vec![
            q.num_active_nodes(),
            q.lower_min(),
            q.hybrid_estim_min(),
            q.num_suboptimal(),
        ];
        if got != want {
            self.bad(format!("queue state {got:?} vs {want:?}"));
        }
        self.s.queue_states += 1;
    }

    /// The next search event (after skipping component queue lines).
    fn peek(&mut self) -> Option<Vec<String>> {
        self.skip();
        self.lines.get(self.pos).cloned()
    }

    /// The first upcoming line with this tag before any line with a tag in `stop`.
    fn find_ahead(&self, tag: &str, stop: &[&str]) -> Option<Vec<String>> {
        for t in &self.lines[self.pos..] {
            if self.foreign(t) {
                continue;
            }
            if t[0] == tag {
                return Some(t.clone());
            }
            if stop.contains(&t[0].as_str()) {
                return None;
            }
        }
        None
    }

    /// Consume the next search event, which must be `tag`.
    fn take(&mut self, tag: &str) -> Vec<String> {
        match self.peek() {
            Some(t) if t[0] == tag => {
                self.pos += 1;
                self.note_limits(&t);
                t
            }
            other => {
                let msg = format!(
                    "expected {tag}, dump has {:?}",
                    other.as_ref().map(|t| t.join(" "))
                );
                self.bad(msg);
                panic!("replay diverged: {:?}", self.s.mismatches.last());
            }
        }
    }

    /// Track the cutoff and optimality limit from node lines (`SE`/`SB`/`SK`/`SP`/`SD`).
    fn note_limits(&mut self, t: &[String]) {
        if matches!(t[0].as_str(), "SE" | "SB" | "SK" | "SP" | "SD") && t.len() >= 13 {
            self.cutoff = hx(&t[11]);
            self.optlim = hx(&t[12]);
        }
        if t[0] == "SPQ" && t.len() >= 12 {
            self.feastol = hx(&t[11]);
        }
    }

    /// Before the first node line: take the limits from the first one ahead.
    fn refresh_limits(&mut self) {
        if let Some(t) = self.lines[self.pos..]
            .iter()
            .find(|t| matches!(t[0].as_str(), "SE" | "SB" | "SK" | "SP" | "SD") && t.len() >= 13)
            .cloned()
        {
            self.cutoff = hx(&t[11]);
            self.optlim = hx(&t[12]);
        }
    }

    /// Compare a node against a node line's fields.
    fn check_node(&mut self, what: &str, t: &[String], depth: i32, n: &NodeData) {
        let dd: i32 = t[2].parse().unwrap();
        if dd != depth {
            self.bad(format!("{what}: depth {depth} vs {dd}"));
        }
        if t.len() < 11 {
            return;
        }
        let checks = [
            ("lb", n.lower_bound, hx(&t[3])),
            ("est", n.estimate, hx(&t[4])),
        ];
        for (k, a, b) in checks {
            if !same(a, b) {
                self.bad(format!("{what}: {k} {a:e} vs {b:e}"));
            }
        }
        let open: u8 = t[6].parse().unwrap();
        if open != n.opensubtrees {
            self.bad(format!("{what}: opensubtrees {} vs {open}", n.opensubtrees));
        }
        let col: i32 = t[7].parse().unwrap();
        if col != n.decision.col {
            self.bad(format!(
                "{what}: decision column {} vs {col}",
                n.decision.col
            ));
        }
    }

    /// Set a node from a node line (a component's effect).
    fn set_node(t: &[String], n: &mut NodeData) {
        n.lower_bound = hx(&t[3]);
        n.estimate = hx(&t[4]);
        n.lp_objective = hx(&t[5]);
    }
}

impl SearchEnv for Mock {
    fn queue_emplace(
        &mut self,
        d: Vec<DomainChange>,
        b: Vec<i32>,
        lb: f64,
        est: f64,
        depth: i32,
    ) -> f64 {
        let t = self.take("QE");
        let (slot, _, _, wlb, west, wdepth, opt) = Self::parse_qe(&t);
        if !same(lb, wlb) || !same(est, west) || depth != wdepth {
            self.bad(format!(
                "emplace (lb {lb:e}, est {est:e}, depth {depth}) vs ({wlb:e}, {west:e}, {wdepth})"
            ));
        }
        self.queue.set_optimality_limit(opt);
        let (got, w) = self.queue.emplace_node(d, b, lb, est, depth);
        if got != slot {
            self.bad(format!("emplace slot {got} vs {slot}"));
        }
        self.s.queue_ops += 1;
        w
    }
    fn queue_pop_best(&mut self) -> OpenNode {
        let t = self.take("QP");
        if t[2] != "E" {
            self.bad(format!("pop best-estimate vs HiGHS's {}", t[2]));
        }
        let (got, n) = self.queue.pop_best_node();
        let want: i64 = t[3].parse().unwrap();
        if got != want {
            self.bad(format!("pop slot {got} vs {want}"));
        }
        self.s.queue_ops += 1;
        n
    }
    fn queue_pop_best_bound(&mut self) -> OpenNode {
        let t = self.take("QP");
        if t[2] != "B" {
            self.bad(format!("pop best-bound vs HiGHS's {}", t[2]));
        }
        let (got, n) = self.queue.pop_best_bound_node();
        let want: i64 = t[3].parse().unwrap();
        if got != want {
            self.bad(format!("pop slot {got} vs {want}"));
        }
        self.s.queue_ops += 1;
        n
    }
    fn queue_is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    fn queue_best_lower_bound(&self) -> f64 {
        self.queue.best_lower_bound()
    }
    fn queue_best_bound_stack_size(&self) -> i32 {
        self.queue.best_bound_domchg_stack_size()
    }

    fn cutoff_bound(&self) -> f64 {
        self.cutoff
    }
    fn optimality_limit(&self) -> f64 {
        self.optlim
    }
    fn feastol(&self) -> f64 {
        self.feastol
    }
    fn search_check_limits(&mut self, _nnodes: i64, _open_on_stack: usize) -> bool {
        false
    }
    fn mip_check_limits(&mut self) -> bool {
        false
    }

    fn domain_install(&mut self, _node: &OpenNode) {}
    fn domain_stack_len(&self) -> i32 {
        0
    }
    fn domain_backtrack(&mut self) {}
    fn domain_backtrack_to_global(&mut self) {}
    fn domain_propagate_after_backtrack(&mut self, depth: i32) -> bool {
        let t = self.take("SKP");
        let d: i32 = t[1].parse().unwrap();
        if d != depth {
            self.bad(format!("ancestor propagation at depth {depth} vs {d}"));
        }
        if t[2] == "1" {
            self.s.ancestors_closed += 1;
        }
        t[2] == "1"
    }
    fn domain_try_sibling(&mut self, _decision: DomainChange, _skip: bool) -> bool {
        match self.peek() {
            Some(t) if t[0] == "SKS" => t[6] == "1",
            other => {
                let msg = format!("expected SKS, dump has {:?}", other.map(|t| t.join(" ")));
                self.bad(msg);
                panic!("replay diverged");
            }
        }
    }
    fn domain_undo_sibling(&mut self) {}
    fn domain_propagate_for_queue(&mut self) -> bool {
        match self.peek() {
            Some(t) if t[0] == "SQ" => t[t.len() - 1] == "1",
            other => {
                let msg = format!("expected SQ, dump has {:?}", other.map(|t| t.join(" ")));
                self.bad(msg);
                panic!("replay diverged");
            }
        }
    }
    fn domain_reduced_stack(&mut self) -> (Vec<DomainChange>, Vec<i32>) {
        match self.find_ahead("QE", &["SE", "SB", "SK", "SP", "SI", "MI"]) {
            Some(t) => {
                let (_, d, b, ..) = Self::parse_qe(&t);
                (d, b)
            }
            None => {
                self.bad("a queue hand-off with no QE ahead".into());
                (Vec::new(), Vec::new())
            }
        }
    }
    fn domain_objective_lower_bound(&mut self) -> f64 {
        match self.peek() {
            Some(t) if t[0] == "SPQ" => hx(&t[5]),
            Some(t) if t[0] == "QE" => hx(&t[3]),
            _ => f64::NEG_INFINITY,
        }
    }
    fn global_propagate(&mut self) {}
    fn global_infeasible(&self) -> bool {
        false
    }

    fn evaluate(
        &mut self,
        node: &mut NodeData,
        _parent: Option<&NodeData>,
        _depth: i32,
    ) -> NodeResult {
        // HiGHS's recursion (`return evaluateNode()` after propagation changed columns)
        // prints the same `SE`/`SEF` pair once per level, innermost first. Consume the
        // run of identical pairs but the last, which `note_evaluated` checks. A
        // *different* `SE` that follows belongs to the next call (`branch`'s own
        // re-evaluations).
        let t = self.peek().expect("dump ended before an evaluation");
        if t[0] != "SE" {
            self.bad(format!("expected SE, dump has {}", t.join(" ")));
            panic!("replay diverged");
        }
        let pair = |m: &Mock, i: usize| -> (Vec<String>, Option<Vec<String>>) {
            let a = m.lines[i].clone();
            let b = m.lines.get(i + 1).filter(|x| x[0] == "SEF").cloned();
            (a, b)
        };
        loop {
            let here = pair(self, self.pos);
            let step = if here.1.is_some() { 2 } else { 1 };
            let next = self.pos + step;
            if next < self.lines.len() && self.lines[next][0] == "SE" && pair(self, next) == here {
                self.pos = next;
                continue;
            }
            break;
        }
        let t = self.lines[self.pos].clone();
        Self::set_node(&t, node);
        let r = NodeResult::from_i32(t[1].parse().unwrap());
        let open: u8 = t[6].parse().unwrap();
        // A suboptimal result with subtrees still open is the control's own test.
        if r == NodeResult::SubOptimal && open != 0 {
            NodeResult::Open
        } else {
            r
        }
    }
    fn branch(
        &mut self,
        _top: &NodeData,
        _parent: Option<&NodeData>,
        _depth: i32,
    ) -> BranchOutcome {
        // Everything up to `SB` is branch selection's: strong-branching commitments
        // (`SBD` + `SBS`) and re-evaluations of the top node (`SE`).
        let mut steps = Vec::new();
        let mut pending: Option<(DomainChange, f64)> = None;
        while let Some(t) = self.peek() {
            match t[0].as_str() {
                "SB" => break,
                "SBD" => {
                    pending = Some((
                        DomainChange {
                            col: t[1].parse().unwrap(),
                            upper: t[2] == "1",
                            val: hx(&t[3]),
                        },
                        hx(&t[4]),
                    ));
                }
                "SBS" => {
                    let (decision, branching_point) = pending.take().expect("SBS without SBD");
                    self.s.commits += 1;
                    steps.push(BranchStep::Commit {
                        decision,
                        branching_point,
                        pos: 0,
                        parent_open: t[1].parse().unwrap(),
                        parent_other_lb: hx(&t[2]),
                    });
                }
                "SE" => steps.push(BranchStep::Update {
                    lower_bound: hx(&t[3]),
                    estimate: hx(&t[4]),
                    lp_objective: hx(&t[5]),
                    opensubtrees: t[6].parse().unwrap(),
                }),
                _ => {}
            }
            self.pos += 1;
            self.note_limits(&t);
        }
        let t = self.peek().expect("dump ended inside a branching");
        let result = NodeResult::from_i32(t[1].parse().unwrap());
        if result == NodeResult::Branched {
            let p = self.lines[self.pos + 1].clone();
            assert_eq!(p[0], "SBP", "branched without SBP");
            let decision = DomainChange {
                col: p[1].parse().unwrap(),
                upper: p[2] == "1",
                val: hx(&p[3]),
            };
            BranchOutcome {
                steps,
                result,
                // The child's bound as HiGHS created it stands in for `childLb`.
                branched: Some((
                    decision,
                    hx(&p[4]),
                    hx(&p[5]),
                    hx(&p[6]),
                    hx(&p[7]),
                    hx(&t[3]),
                    0,
                )),
                closed: None,
            }
        } else {
            BranchOutcome {
                steps,
                result,
                branched: None,
                closed: Some((
                    hx(&t[3]),
                    hx(&t[4]),
                    t[6].parse().unwrap(),
                    t[7].parse().unwrap(),
                )),
            }
        }
    }
    fn plunge_scores(
        &mut self,
        _node: (i32, f64, bool),
        ancestor: Option<(i32, f64, bool)>,
    ) -> (f64, Option<(f64, f64)>) {
        match self.peek() {
            Some(t) if t[0] == "SPQ" => {
                let anc: i32 = t[8].parse().unwrap();
                let a = (ancestor.is_some() && anc >= 0).then(|| (hx(&t[9]), hx(&t[10])));
                (hx(&t[6]), a)
            }
            other => {
                let msg = format!("expected SPQ, dump has {:?}", other.map(|t| t.join(" ")));
                self.bad(msg);
                panic!("replay diverged");
            }
        }
    }
    fn separate(&mut self) -> bool {
        self.skip();
        false
    }
    fn heuristics_allowed(&mut self) -> bool {
        self.peek().is_some_and(|t| t[0] == "MH")
    }
    fn run_heuristics(&mut self) {
        self.skip();
    }

    fn note_install(&mut self, kind: InstallKind) {
        let t = self.take("MI");
        let want = match t[1].as_str() {
            "R" => InstallKind::Root,
            "B" => InstallKind::BestBound,
            _ => InstallKind::BestEstimate,
        };
        if kind != want {
            self.bad(format!("install {kind:?} vs {want:?}"));
        }
        self.s.installs += 1;
    }
    fn note_installed(&mut self, depth: i32, node: &OpenNode) {
        let t = self.take("SI");
        let d: i32 = t[1].parse().unwrap();
        let n: usize = t[4].parse().unwrap();
        if d != depth
            || !same(node.lower_bound, hx(&t[2]))
            || !same(node.estimate, hx(&t[3]))
            || n != node.domchgstack.len()
        {
            self.bad(format!("installed node differs: {}", t.join(" ")));
        }
    }
    fn note_evaluated(&mut self, r: NodeResult, depth: i32, node: &NodeData) {
        let t = self.take("SE");
        let want = NodeResult::from_i32(t[1].parse().unwrap());
        if r != want {
            self.bad(format!("evaluate result {r:?} vs {want:?}"));
        }
        self.check_node("evaluate", &t, depth, node);
        if self.peek().is_some_and(|t| t[0] == "SEF") {
            self.pos += 1;
        }
        self.s.evaluations += 1;
    }
    fn note_branched(
        &mut self,
        r: NodeResult,
        depth: i32,
        top: &NodeData,
        parent: Option<&NodeData>,
        _lb0: f64,
    ) {
        let t = self.take("SB");
        let want = NodeResult::from_i32(t[1].parse().unwrap());
        if r != want {
            self.bad(format!("branch result {r:?} vs {want:?}"));
        }
        self.check_node("branch", &t, depth, top);
        if r == NodeResult::Branched {
            let p = self.take("SBP");
            let po: u8 = p[8].parse().unwrap();
            if parent.map(|x| x.opensubtrees) != Some(po) {
                self.bad("branch: parent's open subtrees".into());
            }
        }
        self.s.branchings += 1;
    }
    fn note_sibling(&mut self, depth: i32, decision: DomainChange, nodelb: f64, pruned: bool) {
        let t = self.take("SKS");
        let d: i32 = t[1].parse().unwrap();
        let col: i32 = t[2].parse().unwrap();
        let up = t[3] == "1";
        if d != depth
            || col != decision.col
            || up != decision.upper
            || !same(decision.val, hx(&t[4]))
            || !same(nodelb, hx(&t[5]))
            || pruned != (t[6] == "1")
        {
            self.bad(format!(
                "sibling (depth {depth}, col {}, upper {}, val {:e}, lb {nodelb:e}, pruned {pruned}) vs {}",
                decision.col, decision.upper, decision.val, t.join(" ")
            ));
        }
        self.s.siblings += 1;
    }
    fn note_plunge(
        &mut self,
        depth: i32,
        decision: DomainChange,
        nodelb: f64,
        score: f64,
        to_queue: bool,
        ancestor: i32,
    ) {
        let t = self.take("SPQ");
        let d: i32 = t[1].parse().unwrap();
        let tq = t[7] == "1";
        let anc: i32 = t[8].parse().unwrap();
        if d != depth
            || decision.col != t[2].parse::<i32>().unwrap()
            || !same(nodelb, hx(&t[5]))
            || !same(score, hx(&t[6]))
            || tq != to_queue
            || anc != ancestor
        {
            self.bad(format!("plunge choice (depth {depth}, lb {nodelb:e}, queue {to_queue}, ancestor {ancestor}) vs {}", t.join(" ")));
        }
        self.s.plunge_choices += 1;
        if to_queue {
            self.s.to_queue += 1;
        }
    }
    fn note_backtracked(&mut self, plunge: bool, r: bool, depth: i32, top: Option<&NodeData>) {
        let t = self.take(if plunge { "SP" } else { "SK" });
        if (t[1] == "1") != r {
            self.bad(format!("backtrack result {r} vs {}", t[1]));
        }
        if let Some(n) = top {
            self.check_node(
                if plunge {
                    "backtrackPlunge"
                } else {
                    "backtrack"
                },
                &t,
                depth,
                n,
            );
        }
    }
    fn note_queued(&mut self, kind: QueueHandoff, depth: i32, node: &NodeData, pruned: bool) {
        let t = self.take("SQ");
        let k = if kind == QueueHandoff::Current {
            "C"
        } else {
            "O"
        };
        let d: i32 = t[2].parse().unwrap();
        if t[1] != k
            || d != depth
            || !same(node.lower_bound, hx(&t[3]))
            || !same(node.estimate, hx(&t[4]))
            || (t[5] == "1") != pruned
        {
            self.bad(format!(
                "queue hand-off ({k}, depth {depth}, pruned {pruned}) vs {}",
                t.join(" ")
            ));
        }
        self.s.handoffs += 1;
        if kind == QueueHandoff::Current {
            self.s.handoffs_current += 1;
        }
    }
    fn note_dive(&mut self, end: Option<(NodeResult, i32, &NodeData)>) {
        match end {
            None => {
                self.take("SD0");
            }
            Some((r, depth, n)) => {
                let t = self.take("SD");
                let want = NodeResult::from_i32(t[1].parse().unwrap());
                if r != want {
                    self.bad(format!("dive result {r:?} vs {want:?}"));
                }
                if !n.lower_bound.is_nan() {
                    self.check_node("dive", &t, depth, n);
                }
            }
        }
    }
    fn note_separate(&mut self) {
        self.take("MS");
    }
    fn note_heuristics(&mut self) {
        self.take("MH");
    }
    fn note_plunge_limit(&mut self) {
        self.take("ML");
        self.s.plunge_limits += 1;
    }
    fn after_round(&mut self) -> bool {
        self.s.rounds += 1;
        // HiGHS records `MG` unless `checkLimits` ended the search first (the gap
        // closed); then the dump ends here.
        if self.peek().is_none() {
            self.s.limit_stops += 1;
            return true;
        }
        self.take("MG");
        false
    }
}

impl Mock {
    /// Root processing (before the first `MI`) is not the main loop: apply its queue
    /// operations (the root node's emplace among them) and pass over its search events
    /// (HiGHS's root uses the search object for its own dives).
    fn skip_root(&mut self) {
        let mi = self
            .lines
            .iter()
            .position(|t| t[0] == "MI")
            .expect("no MI event");
        while self.pos < mi {
            let t = self.lines[self.pos].clone();
            if !self.foreign(&t) {
                match t[0].as_str() {
                    "QE" => self.apply_emplace(&t),
                    "QB" | "QG" | "QC" | "QS" => {
                        self.skip();
                        continue;
                    }
                    _ => {}
                }
            }
            self.pos += 1;
        }
        self.refresh_limits();
    }
}

/// Replay one dump: the control's main loop until the queue empties.
pub(crate) fn replay(text: &str) -> SearchStats {
    let mut env = Mock::new(text);
    env.skip_root();
    let mut search = Search {
        count_tree_weight: true,
        ..Default::default()
    };
    let mut ml = MainLoop::default();
    let mut root = true;
    while !env.queue_is_empty() {
        if ml.round(&mut env, &mut search, root, NO_NODE_LIMIT) {
            break;
        }
        root = false;
        env.skip();
    }
    // Every recorded event must have been consumed by the control.
    env.skip();
    env.s.leftover_events = env.lines.len() - env.pos;
    if env.s.leftover_events > 0 {
        let rest = env.lines[env.pos].join(" ");
        env.bad(format!(
            "{} events left over, first: {rest}",
            env.s.leftover_events
        ));
    }
    env.s
}

#[test]
#[ignore]
fn full_search_replay() {
    let p = std::env::var("DISCOPT_HIGHS_SEARCH_REPLAY").expect("set DISCOPT_HIGHS_SEARCH_REPLAY");
    let md = std::fs::metadata(&p).unwrap();
    let files: Vec<std::path::PathBuf> = if md.is_dir() {
        let mut v: Vec<_> = std::fs::read_dir(&p)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        v.sort();
        v
    } else {
        vec![p.into()]
    };
    let mut bad = 0usize;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let s = replay(&text);
        eprintln!("{}: {s:?}", f.file_name().unwrap().to_string_lossy());
        for m in s.mismatches.iter().take(10) {
            eprintln!("  {m}");
        }
        bad += s.mismatches.len();
    }
    assert_eq!(bad, 0);
}

/// The six complete HiGHS solves of `tests/data/highs_search_replay/` (bell5, bppc8-02,
/// flugpl, gesa2, mik-250-20-75-1, p0201; HiGHS 1.15.1, threads=1, presolve, restarts
/// and symmetry off): the translated control takes every one of HiGHS's decisions.
#[test]
fn search_fixture_replays_exactly() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/highs_search_replay"
    );
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("search fixture directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 6, "{files:?}");
    let mut tot = SearchStats::default();
    for f in &files {
        let s = replay(&std::fs::read_to_string(f).unwrap());
        assert!(
            s.mismatches.is_empty(),
            "{}: {:#?}",
            f.display(),
            s.mismatches
        );
        tot.rounds += s.rounds;
        tot.evaluations += s.evaluations;
        tot.branchings += s.branchings;
        tot.siblings += s.siblings;
        tot.plunge_choices += s.plunge_choices;
        tot.to_queue += s.to_queue;
        tot.handoffs_current += s.handoffs_current;
        tot.plunge_limits += s.plunge_limits;
        tot.ancestors_closed += s.ancestors_closed;
        tot.commits += s.commits;
        tot.limit_stops += s.limit_stops;
        tot.queue_states += s.queue_states;
    }
    // §6: every kind of decision must have been exercised.
    assert!(
        tot.rounds >= 250 && tot.evaluations >= 1500 && tot.branchings >= 850,
        "{tot:?}"
    );
    assert!(
        tot.siblings >= 750 && tot.plunge_choices >= 450 && tot.to_queue >= 100,
        "{tot:?}"
    );
    assert!(
        tot.handoffs_current >= 1 && tot.plunge_limits >= 1 && tot.ancestors_closed >= 1,
        "{tot:?}"
    );
    assert!(
        tot.commits >= 500 && tot.limit_stops >= 1 && tot.queue_states >= 1000,
        "{tot:?}"
    );
}
