//! Persistent in-tree bound tightening (B3 of issue #51).
//!
//! Runs lightweight FBBT on a B&B node's local bounds, returning the
//! tightened intervals. Tightenings persist by virtue of the B&B
//! contract: any child node inherits its parent's bounds, so a
//! tightening applied at a node automatically propagates to its
//! subtree.
//!
//! ## Why this is cheap
//!
//! The expression DAG and constraint structure are identical at every
//! node — only the variable bounds change. So FBBT at a child node
//! re-uses all of the topology and only re-evaluates intervals on
//! shifted leaves. The marginal work per node is proportional to the
//! number of variables that *changed* relative to the parent (in
//! principle); this kernel runs the full pass for now and leaves the
//! incremental optimisation to a follow-up.
//!
//! ## Scheduling
//!
//! In-tree FBBT is gated by [`InTreePresolveOptions::depth_stride`] —
//! the pass runs only when `node_depth % depth_stride == 0`, so the
//! caller can amortise the cost over the tree without paying it at
//! every node. `depth_stride = 1` runs at every node;
//! `depth_stride = 0` disables the pass.

use crate::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, IndexElem, IndexSpec,
    ModelRepr, ObjectiveSense, UnOp, VarInfo, VarType,
};
use crate::presolve::fbbt::{
    any_empty_beyond, fbbt_with_cutoff, repair_subtol_crossings, Interval, FEAS_TOL,
};
use crate::presolve::probing::probe_node_bounds;
use std::sync::{Arc, Mutex};

/// Options controlling persistent in-tree bound tightening.
#[derive(Debug, Clone)]
pub struct InTreePresolveOptions {
    /// Run the pass at every `depth_stride`-th tree depth. `0` disables
    /// the pass entirely; `1` runs at every node.
    pub depth_stride: u32,
    /// FBBT inner-loop iteration cap.
    pub max_iter: usize,
    /// FBBT inner-loop convergence tolerance.
    pub tol: f64,
    /// Run per-node probing (P3 branch-and-reduce) after FBBT. Probing
    /// tentatively fixes each discrete variable at a bound and re-runs FBBT,
    /// contracting the domain on any proven-infeasible fixing. Off by default
    /// (it costs O(discrete) extra FBBT solves per node); sound when on.
    pub probing: bool,
    /// Cap on the number of discrete variables probed per node (budget).
    pub probe_max_vars: usize,
    /// Expand array-valued rows elementwise for the per-scalar kernel (#1568,
    /// [`expand_rows_for_fbbt`]). `None` reads `DISCOPT_IN_TREE_ARRAY_ROWS`
    /// ([`array_rows_enabled`], default ON; `=0` opts out); `Some(b)` overrides it. Only
    /// consulted by [`run_in_tree_presolve_scalar`] on a non-scalar layout.
    pub expand_array_rows: Option<bool>,
}

impl Default for InTreePresolveOptions {
    fn default() -> Self {
        Self {
            depth_stride: 4,
            max_iter: 8,
            tol: 1e-6,
            probing: false,
            probe_max_vars: 32,
            expand_array_rows: None,
        }
    }
}

/// Per-node tightening result.
#[derive(Debug, Clone, Default)]
pub struct InTreeDelta {
    /// Tightened lower bounds (one per variable).
    pub lb: Vec<f64>,
    /// Tightened upper bounds (one per variable).
    pub ub: Vec<f64>,
    /// Number of variables whose bounds tightened (either side).
    pub bounds_tightened: u32,
    /// True if the kernel detected infeasibility (empty interval).
    pub infeasible: bool,
    /// How many sub-`FEAS_TOL` bound crossings were repaired (#907).
    ///
    /// Surfaced rather than absorbed: a rising count is a numerical smell, and
    /// before #907 each of these events could have fathomed a live node.
    pub subtol_repaired: usize,
    /// True iff the schedule actually ran the pass at this node.
    pub ran: bool,
    /// How the per-scalar kernel saw array-valued rows (#1568): `None` when the
    /// model has a scalar layout (no view at all) or the expansion is off;
    /// otherwise expanded, partial (some rows kept their proxy-view form) or
    /// declined (the proxy view ran). Surfaced so the caller can count it --
    /// never a silent fallback.
    pub array_rows: Option<ArrayRowsOutcome>,
    /// Scalar rows the expansion put in place of array-structured constraints
    /// at this call ([`ArrayRowStats::rows_added`]); 0 when it is off,
    /// declined, or the schedule skipped the call.
    pub array_rows_added: usize,
    /// Array-structured constraints left in their proxy (hull) form at this
    /// call ([`ArrayRowStats::on_hull`]); 0 when off, declined or skipped.
    pub array_rows_on_hull: usize,
}

/// Run in-tree FBBT at a node with the given local bounds.
///
/// `model` is the **root** model (variable bounds inside it are
/// ignored — `node_lb`/`node_ub` override them). Returns an
/// [`InTreeDelta`] containing the post-tightening bounds.
///
/// `incumbent` is forwarded verbatim to [`fbbt_with_cutoff`] and
/// [`probe_node_bounds`], so it is in the **model's own objective space** —
/// `f(x_inc)`, positive-as-written for a maximize — and NOT the caller's
/// internal minimization space. See `fbbt_with_cutoff` for what the mismatch
/// costs: an emptied box that the caller consumes as a rigorous fathom, hence a
/// false `optimal` (issue #1373).
///
/// The pass is a no-op (returns `ran = false`, copies `node_lb` /
/// `node_ub` unchanged) when the schedule says to skip this depth.
pub fn run_in_tree_presolve(
    model: &ModelRepr,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> InTreeDelta {
    assert_eq!(node_lb.len(), model.variables.len());
    assert_eq!(node_ub.len(), model.variables.len());

    if opts.depth_stride == 0 || (node_depth as u32) % opts.depth_stride != 0 {
        return InTreeDelta {
            lb: node_lb.to_vec(),
            ub: node_ub.to_vec(),
            bounds_tightened: 0,
            infeasible: false,
            subtol_repaired: 0,
            ran: false,
            ..Default::default()
        };
    }

    // #907. Sanitize the INCOMING node box before anything reads it. A caller
    // upstream (or an earlier `in_tree_presolve` on a parent node) may hand us a
    // box already inverted by rounding noise; patching that straight onto
    // `VarInfo` would seed FBBT from an inverted domain and manufacture the very
    // emptiness we are trying not to over-read.
    let mut node_box: Vec<Interval> = (0..node_lb.len())
        .map(|i| Interval::new(node_lb[i], node_ub[i]))
        .collect();
    let mut subtol_repaired = repair_subtol_crossings(&mut node_box, FEAS_TOL);

    // Patch the model's variable bounds with the node-local bounds.
    // We clone only the lightweight `variables` Vec, not the arena.
    let mut patched = model.clone();
    for (i, vinfo) in patched.variables.iter_mut().enumerate() {
        if !vinfo.lb.is_empty() {
            vinfo.lb[0] = node_box[i].lo;
        }
        if !vinfo.ub.is_empty() {
            vinfo.ub[0] = node_box[i].hi;
        }
    }

    // #907. `infeasible` is consumed by the B&B loop as a RIGOROUS FATHOM — the
    // subtree is pruned outright — so it must never be set by floating-point
    // noise. Repair sub-`FEAS_TOL` crossings, then conclude infeasibility only
    // beyond the tolerance, exactly as `fbbt`, `fbbt_fp` and `probing` do.
    //
    // Loosening a fathom is the SOUND direction: the node is explored rather
    // than discarded. Genuine detections are unaffected — corpus instrumentation
    // found every real fathom carried either the `[inf, -inf]` empty sentinel or
    // a crossing of exactly 1.0 (binary domain wipeout), 6+ orders above
    // `FEAS_TOL`.
    let mut bounds: Vec<Interval> = fbbt_with_cutoff(&patched, opts.max_iter, opts.tol, incumbent);
    subtol_repaired += repair_subtol_crossings(&mut bounds, FEAS_TOL);
    let mut infeasible = any_empty_beyond(&bounds, FEAS_TOL);

    let mut new_lb: Vec<f64> = node_box.iter().map(|b| b.lo).collect();
    let mut new_ub: Vec<f64> = node_box.iter().map(|b| b.hi).collect();
    let mut tightened = 0u32;
    if !infeasible {
        for i in 0..bounds.len() {
            let iv = bounds[i];
            // Floor with the node's bounds — never relax.
            if iv.lo > new_lb[i] + opts.tol {
                new_lb[i] = iv.lo;
                tightened += 1;
            }
            if iv.hi < new_ub[i] - opts.tol {
                new_ub[i] = iv.hi;
                tightened += 1;
            }
        }
    }

    // P3 probing pass: contract discrete-variable domains by tentatively fixing
    // each at a bound and re-running FBBT (proven-infeasible fixings only).
    // Runs on the FBBT-tightened box; folds its (subset) result back, never
    // loosening. `patched` carries the node bounds; probing re-seeds fully from
    // the explicit interval box, so the two boxes agree.
    if opts.probing && !infeasible {
        let node_box: Vec<Interval> = (0..new_lb.len())
            .map(|i| Interval::new(new_lb[i], new_ub[i]))
            .collect();
        let pr = probe_node_bounds(
            &patched,
            &node_box,
            opts.probe_max_vars,
            opts.max_iter,
            opts.tol,
            incumbent,
            None,
        );
        if pr.infeasible {
            infeasible = true;
        } else {
            for i in 0..pr.tightened_bounds.len().min(new_lb.len()) {
                let iv = pr.tightened_bounds[i];
                if iv.lo > new_lb[i] + opts.tol {
                    new_lb[i] = iv.lo;
                    tightened += 1;
                }
                if iv.hi < new_ub[i] - opts.tol {
                    new_ub[i] = iv.hi;
                    tightened += 1;
                }
                // #907. NO infeasibility verdict here. This loop used to test
                // `new_lb[i] > new_ub[i] + opts.tol`, but `opts.tol` is the FBBT
                // *convergence* tolerance — independently settable and smaller
                // than `FEAS_TOL` in practice — so a crossing in
                // `(opts.tol, FEAS_TOL]` set the rigorous-fathom flag on exactly
                // the noise this fix exists to tolerate. The single exit below
                // repairs and then decides, at `FEAS_TOL`, for every path.
            }
        }
    }

    // #907. Final sanitation at the single exit. The probing branch above gates
    // its own emptiness test on `opts.tol` (a different, smaller tolerance), so it
    // can fold back a box that is inverted by a sub-`FEAS_TOL` amount WITHOUT
    // setting `infeasible`. Returning that inverted box would push `lo > hi` onto
    // an LP column bound downstream, reproducing the false infeasibility one layer
    // down — declining to *declare* emptiness is not enough on its own.
    let mut out: Vec<Interval> = (0..new_lb.len())
        .map(|i| Interval::new(new_lb[i], new_ub[i]))
        .collect();
    subtol_repaired += repair_subtol_crossings(&mut out, FEAS_TOL);
    if !infeasible && any_empty_beyond(&out, FEAS_TOL) {
        infeasible = true;
    }
    for (i, b) in out.iter().enumerate() {
        new_lb[i] = b.lo;
        new_ub[i] = b.hi;
    }
    debug_assert!(
        infeasible || new_lb.iter().zip(&new_ub).all(|(l, u)| l <= u),
        "#907: in_tree_presolve returned an inverted box without declaring infeasible"
    );

    InTreeDelta {
        lb: new_lb,
        ub: new_ub,
        bounds_tightened: tightened,
        infeasible,
        subtol_repaired,
        ran: true,
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────
// Per-scalar node boxes (issue #1513)
// ─────────────────────────────────────────────────────────────
//
// `run_in_tree_presolve` takes one interval per variable BLOCK, but every B&B
// node box is one interval per SCALAR. For a model whose blocks are all size 1
// the two coincide; for a model with a `shape=(n,)` variable they do not, and
// both Python node loops used to skip the kernel on that length mismatch --
// silently -- so array models never saw in-tree FBBT, cutoff FBBT or
// branch-and-reduce.
//
// The per-block kernel cannot simply be handed a per-scalar box: FBBT seeds an
// array block from the hull of its elements, reads `x[i]` as that hull, and
// never tightens a size>1 block (`backward_propagate` writes only size-1
// `Variable` nodes). So the fix hands FBBT a model in which every *element* is
// its own size-1 block: [`ScalarFbbtView`].

/// A per-scalar rewrite of a [`ModelRepr`] for FBBT (issue #1513).
///
/// * Variable slot `j < n_scalar` is the model's flat scalar `j` (block `b`,
///   element `k` maps to `offset_b + k`), a size-1 block.
/// * `x[i]` / `x[i, j]` on an array variable -- an `Index` whose spec selects
///   exactly one element on every axis -- becomes a size-1 `Variable` node on
///   that scalar slot, so FBBT reads and tightens the element itself.
/// * A reference that stays array-valued (the whole array in `sum(x)` or
///   `A @ x`, a slice `x[1:3]`, a partial index `X[i]` of a matrix) points at a
///   **proxy** block, one per array variable so used, appended after the
///   scalars. A proxy is seeded with the HULL of its elements' node bounds,
///   keeps `size > 1` (so FBBT never tightens it) and is typed continuous (so
///   probing never fixes it: fixing a proxy would fix every element at once,
///   which proves nothing about any single element). That is exactly what the
///   per-block kernel did for the whole block, so array-valued uses lose
///   nothing relative to it.
///
/// Node ids are preserved one-for-one (each node is rewritten in place), so the
/// objective and constraint bodies need no remapping and the per-node shapes
/// that `shapes_of` derives are unchanged (a full-arity element `Index` and a
/// size-1 `Variable` are both rank 0).
///
/// Soundness: the view denotes the same function of the same scalars. A proxy
/// is a sound OUTER enclosure of every element it stands for at the start of
/// the pass; the elements only tighten during FBBT, so the stale hull stays a
/// superset. The kernel's result is a subset of the node box, as before.
#[derive(Debug, Clone)]
pub struct ScalarFbbtView {
    model: ModelRepr,
    n_scalar: usize,
    /// `(view variable index, flat offset, size)` per proxy block.
    proxies: Vec<(usize, usize, usize)>,
}

impl ScalarFbbtView {
    /// Number of scalar slots (the length of a node box).
    pub fn n_scalar(&self) -> usize {
        self.n_scalar
    }

    /// Number of proxy blocks (array variables referenced as arrays).
    pub fn n_proxies(&self) -> usize {
        self.proxies.len()
    }

    /// The rewritten model (scalar blocks, then proxies).
    pub fn model(&self) -> &ModelRepr {
        &self.model
    }
}

/// True when every block is a single scalar, i.e. the per-block and per-scalar
/// layouts are the SAME vector (block `i` is scalar `i`). On such a model
/// [`run_in_tree_presolve_scalar`] calls [`run_in_tree_presolve`] on the model
/// itself, so it is identical to the per-block kernel by construction (the
/// bound-neutral argument for every scalar-only model, e.g. anything parsed
/// from `.nl`). `offset` is not consulted: the per-block kernel never reads it,
/// and the pre-#1513 callers accepted exactly the models this accepts.
pub fn is_scalar_layout(model: &ModelRepr) -> bool {
    model.n_vars == model.variables.len() && model.variables.iter().all(|v| v.size == 1)
}

/// Flat row-major index of an `Index` spec that selects exactly one element on
/// every axis of `shape`; `None` for anything array-valued or out of range.
fn single_element_flat(spec: &IndexSpec, shape: &[usize]) -> Option<usize> {
    let idx: Vec<usize> = match spec {
        IndexSpec::Scalar(i) => vec![*i],
        IndexSpec::Tuple(v) => v.clone(),
        IndexSpec::Multi(elems) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    IndexElem::Scalar(i) => v.push(*i),
                    // A newaxis makes the result array-shaped; an ellipsis is
                    // resolved against the base rank elsewhere (#1516). Both
                    // take the conservative array-reference (proxy) route.
                    IndexElem::Slice { .. } | IndexElem::NewAxis | IndexElem::Ellipsis => {
                        return None
                    }
                }
            }
            v
        }
    };
    // Numpy semantics: an index of lower arity than the rank keeps the trailing
    // axes, i.e. it is array-valued. Only a full-arity index is one element.
    if shape.is_empty() || idx.len() != shape.len() {
        return None;
    }
    let mut flat = 0usize;
    for (i, d) in idx.iter().zip(shape) {
        if i >= d {
            return None;
        }
        flat = flat * d + i;
    }
    Some(flat)
}

/// Build the per-scalar view of `model` (issue #1513).
///
/// Refuses -- `Err` naming the reason, never a guess -- when the variable
/// layout is not the contiguous `offset_b = sum_{c<b} size_c` layout a node box
/// is indexed by, or when a `Variable` node disagrees with its block.
pub fn scalarize_for_fbbt(model: &ModelRepr) -> Result<ScalarFbbtView, String> {
    check_contiguous_layout(model)?;
    let n_scalar = model.n_vars;
    let mut variables = scalar_slots(model);
    let mut proxy_of: Vec<Option<usize>> = vec![None; model.variables.len()];
    let mut proxies: Vec<(usize, usize, usize)> = Vec::new();

    let n_nodes = model.arena.len();
    let mut arena = ExprArena::with_capacity(n_nodes);
    for id in 0..n_nodes {
        let node = model.arena.get(ExprId(id));
        let rewritten = match node {
            ExprNode::Variable {
                name,
                index,
                size,
                shape,
            } => {
                let blk = model.variables.get(*index).ok_or_else(|| {
                    format!("Variable node {id} ('{name}') references missing block {index}")
                })?;
                if blk.size != *size {
                    return Err(format!(
                        "Variable node {id} ('{name}') has size {size}, block {index} has {}",
                        blk.size
                    ));
                }
                if *size == 1 {
                    ExprNode::Variable {
                        name: name.clone(),
                        index: blk.offset,
                        size: 1,
                        shape: shape.clone(),
                    }
                } else {
                    let p = match proxy_of[*index] {
                        Some(p) => p,
                        None => {
                            let p = variables.len();
                            let lo = blk.lb.iter().copied().fold(f64::INFINITY, f64::min);
                            let hi = blk.ub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            variables.push(VarInfo {
                                name: blk.name.clone(),
                                // Never probed -- see the struct docs.
                                var_type: VarType::Continuous,
                                offset: blk.offset,
                                size: blk.size,
                                shape: blk.shape.clone(),
                                // One hull interval; the node patch overwrites it.
                                lb: vec![lo],
                                ub: vec![hi],
                            });
                            proxies.push((p, blk.offset, blk.size));
                            proxy_of[*index] = Some(p);
                            p
                        }
                    };
                    ExprNode::Variable {
                        name: name.clone(),
                        index: p,
                        size: *size,
                        shape: shape.clone(),
                    }
                }
            }
            ExprNode::Index { base, index } => {
                let element = match model.arena.get(*base) {
                    ExprNode::Variable {
                        name,
                        index: vi,
                        size,
                        shape,
                    } if *size > 1 => single_element_flat(index, shape).and_then(|f| {
                        let blk = model.variables.get(*vi)?;
                        (f < blk.size).then(|| ExprNode::Variable {
                            name: format!("{name}[{f}]"),
                            index: blk.offset + f,
                            size: 1,
                            shape: vec![],
                        })
                    }),
                    _ => None,
                };
                element.unwrap_or_else(|| node.clone())
            }
            other => other.clone(),
        };
        arena.add(rewritten);
    }

    Ok(ScalarFbbtView {
        model: ModelRepr {
            arena,
            objective: model.objective,
            objective_sense: model.objective_sense,
            constraints: model.constraints.clone(),
            n_vars: variables.len(),
            variables,
        },
        n_scalar,
        proxies,
    })
}

/// `DISCOPT_IN_TREE_ARRAY_ROWS` -- **default ON** since the #1568 graduation
/// panel (CLAUDE.md §5); `=0` (or `false`/`off`/`no`) is the opt-out and
/// restores the legacy proxy view exactly. Every other value -- unset, empty,
/// `1`, or anything unrecognised -- means ON: only an explicit opt-out turns the
/// expansion off, so a typo cannot silently disable it.
///
/// ON: [`run_in_tree_presolve_scalar`] hands FBBT [`expand_rows_for_fbbt`]'s
/// view, in which every array-valued constraint row is one scalar row per
/// element, instead of [`scalarize_for_fbbt`]'s, in which an array-valued
/// reference is a never-tightened hull proxy. It is bound-changing only on models
/// with an array-valued row or reference: a scalar-layout model never builds a
/// view, so the whole `.nl` corpus is unaffected by construction (measured: 66/66
/// in-repo `.nl` instances identical ON vs OFF, bar one wall-clock-limited run).
/// Graduation panel (`discopt_benchmarks/scripts/array_rows_graduation_panel.py
/// --scale all`, 58 array instances, 30 s): 0 violations over 282 checks,
/// certificates +5 / -0, wall 1350 s -> 1244 s, FBBT 3.2% of ON wall.
pub fn array_rows_enabled() -> bool {
    match std::env::var("DISCOPT_IN_TREE_ARRAY_ROWS") {
        Err(_) => true,
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
    }
}

/// Per-scalar `VarInfo`s for a contiguous layout: slot `j` is flat scalar `j`.
fn scalar_slots(model: &ModelRepr) -> Vec<VarInfo> {
    let mut variables: Vec<VarInfo> = Vec::with_capacity(model.n_vars);
    for v in &model.variables {
        for k in 0..v.size {
            variables.push(VarInfo {
                name: if v.size == 1 {
                    v.name.clone()
                } else {
                    format!("{}[{k}]", v.name)
                },
                var_type: v.var_type,
                offset: v.offset + k,
                size: 1,
                shape: vec![],
                lb: vec![v.lb[k]],
                ub: vec![v.ub[k]],
            });
        }
    }
    variables
}

/// The contiguous `offset_b = sum_{c<b} size_c` layout a node box is indexed by,
/// or the reason `model` does not have it.
fn check_contiguous_layout(model: &ModelRepr) -> Result<(), String> {
    let mut run = 0usize;
    for (b, v) in model.variables.iter().enumerate() {
        if v.offset != run {
            return Err(format!(
                "variable block {b} ('{}') has offset {} but the contiguous layout puts it at {run}",
                v.name, v.offset
            ));
        }
        if v.lb.len() != v.size || v.ub.len() != v.size {
            return Err(format!(
                "variable block {b} ('{}') has size {} but {} lower / {} upper bounds",
                v.name,
                v.size,
                v.lb.len(),
                v.ub.len()
            ));
        }
        run += v.size;
    }
    if run != model.n_vars {
        return Err(format!(
            "variable blocks cover {run} scalars but the model declares n_vars = {}",
            model.n_vars
        ));
    }
    Ok(())
}

/// What [`expand_rows_for_fbbt`] did with the model's rows (#1568).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArrayRowStats {
    /// Scalar rows that replaced array-STRUCTURED constraints -- constraints
    /// whose body, in the proxy view, reads an array-valued node or a hull
    /// proxy (`z - sigmoid(w) == 0` over vectors, `sum(x) <= 5`). A constraint
    /// over individual elements only (`x[0] + x[1] <= 5`) already reads scalar
    /// slots in the proxy view and is not counted.
    pub rows_added: usize,
    /// Array-structured constraints whose body did not expand and so kept their
    /// original (proxy-view, hull) form.
    pub on_hull: usize,
    /// The first array-structured refusal (constraint or objective), when any.
    pub first_refusal: Option<String>,
}

/// How the per-scalar kernel saw array-valued rows at one call (#1568).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayRowsOutcome {
    /// Every array-structured row (and the objective) expanded elementwise.
    Expanded,
    /// Some array-structured rows expanded; the rest kept their proxy-view
    /// form. Carries the first refusal.
    Partial(String),
    /// No expanded view was built and the proxy view ran (the pre-#1568
    /// behaviour). Carries the reason.
    Declined(String),
}

/// Cap on the scalar instructions one expanded view may hold (#1568).
///
/// The view is built once per model (see [`ArrayRowViewCache`]) but FBBT
/// sweeps every node of it at every call, so a huge vectorised model (a dense
/// `A @ x` with 1e5 rows) must not turn one FBBT call into millions of nodes.
/// Over the budget the whole expansion is declined -- the proxy view runs,
/// sound and looser, and the decline is counted.
pub const ARRAY_ROW_INSTRUCTION_BUDGET: usize = 250_000;

/// The per-scalar FBBT view of `model` with every array-valued constraint row
/// expanded to one scalar row per element wherever it can be (issue #1568).
///
/// Built from [`crate::expand::expand_rowwise`] -- the same per-node fan-out as
/// [`crate::expand::expand`], which the AD tape and the `.nl` writer use, with a
/// per-row failure mode and the FBBT function set (`FuncSet::Fbbt`: `asinh`,
/// `acosh`, `atanh`, `erf`, `entropy` on top of the export set) -- by lowering
/// each scalar instruction back into an arena node. Variable references become
/// the size-1 slot of their flat scalar, so FBBT reads and tightens each element
/// on its own. Each source constraint's sense and right-hand side apply to every
/// row it expands to, which is what an elementwise `body sense rhs` means.
///
/// * Every row expands: a pure scalar view with no proxies.
/// * Some row (or the objective) does not: a HYBRID view -- the proxy view of
///   [`scalarize_for_fbbt`] (original node ids, hull proxies), with the lowered
///   instructions appended and each expanded constraint's body REPLACED by its
///   rows. A row that failed keeps its original proxy-view form; its refusal is
///   counted in [`ArrayRowStats::on_hull`], never approximated.
///
/// Soundness: each lowered row is an exact restatement of one element of its
/// source row (the expansion refuses rather than approximates, and the lowering
/// is one node per instruction), and a kept row is the proxy-view row, itself
/// sound (see [`ScalarFbbtView`]). The view denotes the same feasible set over
/// the same scalars, so FBBT on it is valid for the node box and only ever
/// intersects it.
///
/// Refuses (`Err` naming why; the caller falls back to the proxy view) when the
/// layout is not contiguous, the expansion exceeds
/// [`ARRAY_ROW_INSTRUCTION_BUDGET`], or the program is internally inconsistent.
pub fn expand_rows_for_fbbt(model: &ModelRepr) -> Result<(ScalarFbbtView, ArrayRowStats), String> {
    use crate::expand::{
        expand_rowwise, func_from_code, operands, shapes_of_partial, FuncSet, OP_ABS, OP_ADD,
        OP_CONST, OP_DIV, OP_FUNC_BASE, OP_MUL, OP_NEG, OP_POW, OP_SUB, OP_SUMOVER, OP_VAR,
    };
    check_contiguous_layout(model)?;
    let n_scalar = model.n_vars;
    let rw = expand_rowwise(model, FuncSet::Fbbt);
    let prog = &rw.program;
    if rw.rows.len() != model.constraints.len() {
        return Err(format!(
            "expand returned {} row groups for {} constraints",
            rw.rows.len(),
            model.constraints.len()
        ));
    }
    let n_inst = prog.op.len();
    if n_inst > ARRAY_ROW_INSTRUCTION_BUDGET {
        return Err(format!(
            "the expansion has {n_inst} scalar instructions, over the budget of \
             {ARRAY_ROW_INSTRUCTION_BUDGET}"
        ));
    }

    // Which nodes read an array in the PROXY view: an array-valued node, or
    // anything above one. A full-arity element of an array variable is a scalar
    // slot there, not an array read.
    let shapes = shapes_of_partial(&model.arena);
    let mut arrayish = vec![false; model.arena.len()];
    for i in 0..model.arena.len() {
        let node = model.arena.get(ExprId(i));
        let is_element = match node {
            ExprNode::Index { base, index } => match model.arena.get(*base) {
                ExprNode::Variable { size, shape, .. } if *size > 1 => {
                    single_element_flat(index, shape).is_some()
                }
                _ => false,
            },
            _ => false,
        };
        arrayish[i] = !is_element
            && (shapes[i].as_ref().map_or(true, |s| !s.is_empty())
                || operands(node).iter().any(|c| arrayish[c.0]));
    }

    let mut stats = ArrayRowStats::default();
    for (ci, (c, r)) in model.constraints.iter().zip(&rw.rows).enumerate() {
        if !arrayish[c.body.0] {
            continue;
        }
        match r {
            Ok(rows) => stats.rows_added += rows.len(),
            Err(e) => {
                stats.on_hull += 1;
                stats
                    .first_refusal
                    .get_or_insert_with(|| format!("constraint {ci}: {e}"));
            }
        }
    }
    if let Err(e) = &rw.objective {
        if arrayish[model.objective.0] {
            stats
                .first_refusal
                .get_or_insert_with(|| format!("objective: {e}"));
        }
    }

    // A failure anywhere (array-structured or not) means some original node
    // must stay, so the base is the proxy view with its original ids.
    let hybrid = rw.objective.is_err() || rw.rows.iter().any(|r| r.is_err());
    let (mut arena, variables, proxies) = if hybrid {
        let pv = scalarize_for_fbbt(model)?;
        (pv.model.arena, pv.model.variables, pv.proxies)
    } else {
        (
            ExprArena::with_capacity(n_inst),
            scalar_slots(model),
            Vec::new(),
        )
    };

    let mut ids: Vec<ExprId> = Vec::with_capacity(n_inst);
    let operand = |ids: &Vec<ExprId>, i: usize, j: i64| -> Result<ExprId, String> {
        if j < 0 || (j as usize) >= i {
            return Err(format!(
                "instruction {i} has operand {j}, not an earlier instruction"
            ));
        }
        Ok(ids[j as usize])
    };
    for i in 0..n_inst {
        let op = prog.op[i];
        let node = match op {
            OP_CONST => ExprNode::Constant(prog.k[i]),
            OP_VAR => {
                let k = prog.k[i];
                if !(k >= 0.0 && k.fract() == 0.0 && (k as usize) < n_scalar) {
                    return Err(format!("instruction {i} references variable slot {k}"));
                }
                let slot = k as usize;
                ExprNode::Variable {
                    name: variables[slot].name.clone(),
                    index: slot,
                    size: 1,
                    shape: vec![],
                }
            }
            OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_POW => ExprNode::BinaryOp {
                op: match op {
                    OP_ADD => BinOp::Add,
                    OP_SUB => BinOp::Sub,
                    OP_MUL => BinOp::Mul,
                    OP_DIV => BinOp::Div,
                    _ => BinOp::Pow,
                },
                left: operand(&ids, i, prog.a[i])?,
                right: operand(&ids, i, prog.b[i])?,
            },
            OP_NEG | OP_ABS => ExprNode::UnaryOp {
                op: if op == OP_NEG { UnOp::Neg } else { UnOp::Abs },
                operand: operand(&ids, i, prog.a[i])?,
            },
            OP_SUMOVER => {
                let (lo, hi) = (prog.args_ptr[i] as usize, prog.args_ptr[i + 1] as usize);
                let mut terms = Vec::with_capacity(hi - lo);
                for &j in &prog.args_flat[lo..hi] {
                    terms.push(operand(&ids, i, j)?);
                }
                ExprNode::SumOver { terms }
            }
            c if c >= OP_FUNC_BASE => {
                let func = func_from_code(c - OP_FUNC_BASE)
                    .ok_or_else(|| format!("instruction {i} has unknown function code {c}"))?;
                ExprNode::FunctionCall {
                    func,
                    args: vec![operand(&ids, i, prog.a[i])?],
                }
            }
            other => return Err(format!("instruction {i} has unsupported opcode {other}")),
        };
        ids.push(arena.add(node));
    }
    let root = |r: i64| -> Result<ExprId, String> {
        if r < 0 || (r as usize) >= n_inst {
            return Err(format!("root {r} is not an instruction"));
        }
        Ok(ids[r as usize])
    };

    let mut constraints = Vec::with_capacity(prog.row_roots.len() + model.constraints.len());
    for (c, r) in model.constraints.iter().zip(&rw.rows) {
        match r {
            Ok(rows) => {
                let n_rows = rows.len();
                for (k, &rr) in rows.iter().enumerate() {
                    constraints.push(ConstraintRepr {
                        body: root(rr)?,
                        sense: c.sense,
                        rhs: c.rhs,
                        name: c.name.as_ref().map(|n| {
                            if n_rows == 1 {
                                n.clone()
                            } else {
                                format!("{n}[{k}]")
                            }
                        }),
                    });
                }
            }
            // Kept as written: its ids are the proxy view's, which is the base
            // whenever anything failed.
            Err(_) => constraints.push(c.clone()),
        }
    }
    let objective = match rw.objective {
        Ok(r) => root(r)?,
        Err(_) => model.objective,
    };

    Ok((
        ScalarFbbtView {
            model: ModelRepr {
                arena,
                objective,
                objective_sense: model.objective_sense,
                constraints,
                n_vars: variables.len(),
                variables,
            },
            n_scalar,
            proxies,
        },
        stats,
    ))
}

/// What a built expanded view depends on, compared exactly.
///
/// The view reads the arena, the objective, the constraints and each block's
/// layout and type -- never the variable BOUNDS, which the node box overrides.
/// `ExprArena` is append-only (its nodes are private; `add`/`intern` only
/// append), so an arena of the same length holds the same nodes: the length
/// identifies its content exactly, with no hashing and no collision risk.
#[derive(Debug, Clone, PartialEq)]
struct ViewKey {
    arena_len: usize,
    objective: usize,
    objective_sense: ObjectiveSense,
    constraints: Vec<(usize, ConstraintSense, u64)>,
    variables: Vec<(usize, usize, VarType, Vec<usize>)>,
    n_vars: usize,
}

impl ViewKey {
    fn of(model: &ModelRepr) -> Self {
        ViewKey {
            arena_len: model.arena.len(),
            objective: model.objective.0,
            objective_sense: model.objective_sense,
            constraints: model
                .constraints
                .iter()
                .map(|c| (c.body.0, c.sense, c.rhs.to_bits()))
                .collect(),
            variables: model
                .variables
                .iter()
                .map(|v| (v.offset, v.size, v.var_type, v.shape.clone()))
                .collect(),
            n_vars: model.n_vars,
        }
    }
}

type BuiltView = Arc<Result<(ScalarFbbtView, ArrayRowStats), String>>;

/// One memoised [`expand_rows_for_fbbt`] result per model (#1568).
///
/// Building the expanded view costs far more than one FBBT pass over it, and
/// the view is the same at every node -- only the box changes. The cache holds
/// the last build with the [`ViewKey`] it was built from and rebuilds whenever
/// the key differs, so a stale view is never served. A clone starts empty.
#[derive(Default)]
pub struct ArrayRowViewCache {
    slot: Mutex<Option<(ViewKey, BuiltView)>>,
    n_builds: std::sync::atomic::AtomicUsize,
}

impl Clone for ArrayRowViewCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for ArrayRowViewCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayRowViewCache")
            .field("builds", &self.builds())
            .finish()
    }
}

impl ArrayRowViewCache {
    /// The expanded view of `model`, built on first use or when `model`'s
    /// structure changed since the last build.
    fn get(&self, model: &ModelRepr) -> BuiltView {
        let key = ViewKey::of(model);
        // A poisoned lock only means another thread panicked mid-build; the
        // slot is then discarded and rebuilt, never trusted.
        let mut slot = self.slot.lock().unwrap_or_else(|p| {
            let mut g = p.into_inner();
            *g = None;
            g
        });
        if let Some((k, built)) = slot.as_ref() {
            if *k == key {
                return Arc::clone(built);
            }
        }
        let built: BuiltView = Arc::new(expand_rows_for_fbbt(model));
        *slot = Some((key, Arc::clone(&built)));
        self.n_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        built
    }

    /// How many times a view has been built through this cache.
    pub fn builds(&self) -> usize {
        self.n_builds.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// [`run_in_tree_presolve`] on a PER-SCALAR node box (issue #1513).
///
/// `node_lb` / `node_ub` have one entry per scalar variable (`model.n_vars`),
/// exactly a B&B node box, and so does the returned delta. On a scalar-layout
/// model ([`is_scalar_layout`]) this IS `run_in_tree_presolve(model, ..)`; on
/// any other it runs the same kernel on the [`ScalarFbbtView`].
///
/// Errors -- never a silent skip -- when the box length is not `model.n_vars`
/// or the model cannot be scalarized.
///
/// Builds the expanded view (when [`InTreePresolveOptions::expand_array_rows`]
/// resolves ON) afresh at every call; [`run_in_tree_presolve_scalar_cached`]
/// memoises it.
pub fn run_in_tree_presolve_scalar(
    model: &ModelRepr,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> Result<InTreeDelta, String> {
    run_in_tree_presolve_scalar_cached(
        model,
        &ArrayRowViewCache::default(),
        node_lb,
        node_ub,
        node_depth,
        incumbent,
        opts,
    )
}

/// [`run_in_tree_presolve_scalar`] with the expanded view memoised in `cache`
/// (#1568). Identical to building it afresh: the cache only ever returns the
/// view [`expand_rows_for_fbbt`] built for a structurally identical model.
pub fn run_in_tree_presolve_scalar_cached(
    model: &ModelRepr,
    cache: &ArrayRowViewCache,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> Result<InTreeDelta, String> {
    if node_lb.len() != model.n_vars || node_ub.len() != model.n_vars {
        return Err(format!(
            "node box has {} lower / {} upper bounds but the model has {} scalar variables",
            node_lb.len(),
            node_ub.len(),
            model.n_vars
        ));
    }
    if is_scalar_layout(model) {
        return Ok(run_in_tree_presolve(
            model, node_lb, node_ub, node_depth, incumbent, opts,
        ));
    }
    let expand_on = opts.expand_array_rows.unwrap_or_else(array_rows_enabled);
    if !expand_on {
        // The legacy proxy view, exactly as before #1568 (and uncached).
        let view = scalarize_for_fbbt(model)?;
        return Ok(run_in_tree_presolve_view(
            &view, node_lb, node_ub, node_depth, incumbent, opts,
        ));
    }
    let built = cache.get(model);
    let d = match built.as_ref() {
        Ok((view, stats)) => {
            let mut d =
                run_in_tree_presolve_view(view, node_lb, node_ub, node_depth, incumbent, opts);
            d.array_rows = Some(match &stats.first_refusal {
                None => ArrayRowsOutcome::Expanded,
                Some(why) => ArrayRowsOutcome::Partial(why.clone()),
            });
            if d.ran {
                d.array_rows_added = stats.rows_added;
                d.array_rows_on_hull = stats.on_hull;
            }
            d
        }
        Err(why) => {
            // The proxy view is the pre-#1568 behaviour, sound and looser; the
            // reason travels back on the delta so the decline is counted.
            let view = scalarize_for_fbbt(model)?;
            let mut d =
                run_in_tree_presolve_view(&view, node_lb, node_ub, node_depth, incumbent, opts);
            d.array_rows = Some(ArrayRowsOutcome::Declined(why.clone()));
            d
        }
    };
    Ok(d)
}

/// Run the kernel on a prebuilt [`ScalarFbbtView`] (per-scalar box in and out).
pub fn run_in_tree_presolve_view(
    view: &ScalarFbbtView,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> InTreeDelta {
    let n = view.n_scalar;
    assert_eq!(node_lb.len(), n);
    assert_eq!(node_ub.len(), n);
    let mut full_lb = node_lb.to_vec();
    let mut full_ub = node_ub.to_vec();
    // Proxy seeds: the hull of the node box over the proxy's elements.
    for &(_, off, size) in &view.proxies {
        let lo = node_lb[off..off + size]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = node_ub[off..off + size]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        full_lb.push(lo);
        full_ub.push(hi);
    }
    debug_assert_eq!(full_lb.len(), view.model.variables.len());
    let mut d = run_in_tree_presolve(&view.model, &full_lb, &full_ub, node_depth, incumbent, opts);
    d.lb.truncate(n);
    d.ub.truncate(n);
    if d.ran && !d.infeasible {
        // Count scalar half-bounds only, so the count cannot include a proxy.
        let mut t = 0u32;
        for i in 0..n {
            if d.lb[i] > node_lb[i] + opts.tol {
                t += 1;
            }
            if d.ub[i] < node_ub[i] - opts.tol {
                t += 1;
            }
        }
        d.bounds_tightened = t;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr,
        ObjectiveSense, VarInfo, VarType,
    };

    fn scalar_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    fn x_plus_y_le_5() -> ModelRepr {
        // x + y <= 5, x ∈ [0, 10], y ∈ [0, 10], min x+y
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        }
    }

    #[test]
    fn tightens_at_node_with_branching_bound() {
        // Branch: x ∈ [3, 10] in the node. The constraint x+y≤5 then
        // forces y ≤ 2.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(delta.bounds_tightened >= 1);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
        // Lower bounds are not relaxed.
        assert_eq!(delta.lb[0], 3.0);
    }

    #[test]
    fn infers_indicator_binary_at_node() {
        // Guard x ≤ 10·b, x ∈ [0, 10], b binary. At a node where branching has
        // tightened x to [3, 10], FBBT infers b ≥ 0.3 and snaps it to b = 1 —
        // per-node indicator propagation (issue #230). This is the integration
        // the root-only probing pass cannot deliver inside the tree.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let b = scalar_var(&mut arena, "b", 1);
        let m = arena.add(ExprNode::Constant(10.0));
        let mb = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: m,
            right: b,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: mb,
        });
        let mut bvar = vinfo("b", 0.0, 1.0);
        bvar.var_type = VarType::Binary;
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), bvar],
            n_vars: 2,
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "binary should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn skips_when_depth_stride_zero() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 0,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!delta.ran);
        assert_eq!(delta.bounds_tightened, 0);
        assert_eq!(delta.lb, vec![3.0, 0.0]);
        assert_eq!(delta.ub, vec![10.0, 10.0]);
    }

    #[test]
    fn skips_off_schedule_depths() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 4,
            ..Default::default()
        };
        // depth=1 is not a multiple of 4 ⇒ skipped.
        let d = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!d.ran);
        // depth=4 ⇒ runs.
        let d4 = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 4, None, &opts);
        assert!(d4.ran);
        assert!(d4.bounds_tightened >= 1);
    }

    #[test]
    fn detects_infeasibility() {
        // Branch: x ∈ [10, 10] AND y ∈ [10, 10]. x+y=20 > 5 — infeasible.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(delta.infeasible);
    }

    #[test]
    fn never_relaxes_input_bounds() {
        // Bounds tighter than what FBBT alone would derive must be kept.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        // Caller-supplied tighter ub on x.
        let delta = run_in_tree_presolve(&model, &[0.0, 0.0], &[1.0, 10.0], 0, None, &opts);
        assert!(delta.ran);
        // The ub on x must remain at 1.0 (or tighter), never relax to 5.
        assert!(delta.ub[0] <= 1.0 + 1e-9);
    }

    #[test]
    fn probing_fixes_binary_at_node() {
        // x ≤ 10·b, x ∈ [0,10], b binary; node branch x ∈ [3,10] ⇒ b = 1.
        // The probing pass (opts.probing = true) must fix b to 1 at the node.
        let model = {
            let mut arena = ExprArena::new();
            let x = scalar_var(&mut arena, "x", 0);
            let b = scalar_var(&mut arena, "b", 1);
            let m = arena.add(ExprNode::Constant(10.0));
            let mb = arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: m,
                right: b,
            });
            let body = arena.add(ExprNode::BinaryOp {
                op: BinOp::Sub,
                left: x,
                right: mb,
            });
            let mut bvar = vinfo("b", 0.0, 1.0);
            bvar.var_type = VarType::Binary;
            ModelRepr {
                arena,
                objective: x,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![ConstraintRepr {
                    body,
                    sense: ConstraintSense::Le,
                    rhs: 0.0,
                    name: None,
                }],
                variables: vec![vinfo("x", 0.0, 10.0), bvar],
                n_vars: 2,
            }
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: true,
            probe_max_vars: 32,
            expand_array_rows: None,
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "b should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn probing_off_by_default_is_byte_neutral() {
        // With probing disabled (default), the delta matches the FBBT-only path.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        assert!(!opts.probing);
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
    }

    // ── #907: a sub-FEAS_TOL crossing must not FATHOM a node ─────────────
    //
    // `InTreeDelta::infeasible` is consumed by the B&B loop as a *rigorous
    // fathom*: the subtree is pruned outright. Setting it from a crossing of
    // 8.5e-14 discards a region that may contain feasible points, violating the
    // zero-slack `incorrect_count <= 0` gate with no flag set.

    /// A node box inverted by rounding noise must be explored, not fathomed,
    /// and must not be returned inverted.
    #[test]
    fn subtol_inverted_node_box_is_not_fathomed() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
            expand_array_rows: None,
        };
        // x fixed at 2.5 by two derivations disagreeing in the last ulps.
        let lo = [2.5, 0.0];
        let hi = [2.5 - 1e-14, 10.0];
        let d = run_in_tree_presolve(&model, &lo, &hi, 0, None, &opts);

        assert!(d.ran);
        assert!(
            !d.infeasible,
            "a 1e-14 crossing fathomed a live node — #907 regressed"
        );
        assert_eq!(d.subtol_repaired, 1);
        // The returned box must be well-formed: an inverted interval reaching an
        // LP column bound reproduces the false infeasibility one layer down.
        for i in 0..d.lb.len() {
            assert!(
                d.lb[i] <= d.ub[i],
                "returned an inverted box at var{i}: [{}, {}]",
                d.lb[i],
                d.ub[i]
            );
        }
        // Repair widens to contain both endpoints, so the feasible point x=2.5
        // survives.
        assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0]);
    }

    /// ANTI-PERMISSIVENESS CONTROL: a genuinely empty node box must STILL
    /// fathom. Without this the change is a tolerance-tweak, not a fix.
    #[test]
    fn genuine_empty_node_box_still_fathoms() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
            expand_array_rows: None,
        };
        // x >= 10 AND y >= 10 with x + y <= 5 — infeasible by 15, not by noise.
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(d.infeasible, "a genuine infeasibility stopped fathoming");
        assert_eq!(d.subtol_repaired, 0);
    }

    /// The repair must never cut a point the caller's box contained: sweep a
    /// feasible point through many noise-inverted boxes and assert containment
    /// survives. Prints nothing, but the assertion count is the point (§6).
    #[test]
    fn repair_never_cuts_a_contained_feasible_point() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
            expand_array_rows: None,
        };
        let mut checked = 0usize;
        for k in 0..40 {
            let v = 0.1 * k as f64; // feasible x value in [0, 3.9]
            for eps in [1e-16, 1e-14, 1e-12, 1e-9, 1e-7] {
                let d = run_in_tree_presolve(&model, &[v, 0.0], &[v - eps, 10.0], 0, None, &opts);
                assert!(!d.infeasible, "fathomed a live node at eps={eps}");
                assert!(
                    d.lb[0] <= v && v <= d.ub[0],
                    "repair cut x={v} at eps={eps}: [{}, {}]",
                    d.lb[0],
                    d.ub[0]
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 200,
            "probe did not execute the comparisons it claims"
        );
    }

    /// #907, probing path enabled: a node box inverted by sub-`FEAS_TOL` noise
    /// must not be fathomed, and must not come back inverted or with the point
    /// cut. Runs with `opts.tol` two orders BELOW `FEAS_TOL` so the two
    /// tolerances are distinguishable.
    ///
    /// SCOPE, stated honestly: this covers the incoming-box sanitation on the
    /// probing path (it fails on pre-#907 `main`). It does NOT isolate the
    /// fold-back verdict that used to read `new_lb[i] > new_ub[i] + opts.tol` —
    /// reverting that one line alone leaves this test green, because reaching it
    /// requires `probe_node_bounds` to itself return an interval inverted by
    /// `(opts.tol, FEAS_TOL]`, which no toy model here produces. That line is
    /// changed on consistency grounds — `opts.tol` is a convergence tolerance and
    /// must not gate an infeasibility verdict — and is UNCOVERED by a
    /// fail-before test.
    #[test]
    fn probing_path_does_not_fathom_subtol_inverted_box() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8, // << FEAS_TOL (1e-6): the window the old test fathomed in
            probing: true,
            probe_max_vars: 8,
            expand_array_rows: None,
        };
        let mut checked = 0usize;
        for eps in [1e-7, 5e-7, 9e-7] {
            // Crossing strictly inside (opts.tol, FEAS_TOL] — noise, not a proof.
            let d = run_in_tree_presolve(&model, &[2.5, 0.0], &[2.5 - eps, 10.0], 0, None, &opts);
            assert!(d.ran);
            assert!(
                !d.infeasible,
                "probing path fathomed a live node at a {eps:e} crossing (< FEAS_TOL)"
            );
            assert!(d.lb[0] <= d.ub[0], "probing path returned an inverted box");
            assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0], "probing path cut x=2.5");
            checked += 1;
        }
        assert_eq!(
            checked, 3,
            "probe did not execute the comparisons it claims"
        );
    }

    /// ANTI-PERMISSIVENESS CONTROL for the probing path: a genuine infeasibility
    /// must still fathom with probing on and a small `opts.tol`.
    #[test]
    fn probing_path_still_fathoms_genuine_infeasibility() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8,
            probing: true,
            probe_max_vars: 8,
            expand_array_rows: None,
        };
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(
            d.infeasible,
            "probing path stopped fathoming a real infeasibility"
        );
    }

    // ── #1513: per-scalar node boxes (array variable blocks) ─────────────

    fn arr_var(arena: &mut ExprArena, name: &str, idx: usize, shape: Vec<usize>) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: shape.iter().product(),
            shape,
        })
    }

    fn elem(arena: &mut ExprArena, base: ExprId, i: usize) -> ExprId {
        arena.add(ExprNode::Index {
            base,
            index: IndexSpec::Scalar(i),
        })
    }

    fn arr_vinfo(name: &str, offset: usize, shape: Vec<usize>, lb: f64, ub: f64) -> VarInfo {
        let n: usize = shape.iter().product();
        VarInfo {
            name: name.to_string(),
            var_type: VarType::Continuous,
            offset,
            size: n,
            shape,
            lb: vec![lb; n],
            ub: vec![ub; n],
        }
    }

    fn opts1() -> InTreePresolveOptions {
        InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
            // Pinned, so a `DISCOPT_IN_TREE_ARRAY_ROWS` in the environment
            // cannot change what these tests measure.
            expand_array_rows: Some(false),
        }
    }

    /// x ∈ [0,10]^2 as ONE block of shape (2,); x[0] + x[1] <= 5.
    fn array_x0_plus_x1_le_5() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arr_var(&mut arena, "x", 0, vec![2]);
        let x0 = elem(&mut arena, x, 0);
        let x1 = elem(&mut arena, x, 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: x1,
        });
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![arr_vinfo("x", 0, vec![2], 0.0, 10.0)],
            n_vars: 2,
        }
    }

    /// The array form of `tightens_at_node_with_branching_bound`: the per-scalar
    /// kernel derives x[1] <= 2 from x[0] >= 3. The per-block kernel on the same
    /// model sees only the block hull [0, 10] and derives nothing -- the root
    /// cause of #1513, pinned here as the control.
    #[test]
    fn per_scalar_box_tightens_array_element() {
        let model = array_x0_plus_x1_le_5();
        let d = run_in_tree_presolve_scalar(&model, &[3.0, 0.0], &[10.0, 10.0], 0, None, &opts1())
            .expect("array model must be accepted");
        assert!(d.ran && !d.infeasible);
        assert_eq!(d.lb.len(), 2);
        assert_eq!(d.lb[0], 3.0);
        assert!((d.ub[1] - 2.0).abs() <= 1e-6, "x[1] ub = {}", d.ub[1]);
        assert!(d.ub[0] <= 5.0 + 1e-6, "x[0] ub = {}", d.ub[0]);
        assert_eq!(d.bounds_tightened, 2);

        // Control: the per-block kernel, handed the block hull, gets nothing.
        let blk = run_in_tree_presolve(&model, &[0.0], &[10.0], 0, None, &opts1());
        assert_eq!(blk.bounds_tightened, 0);
    }

    /// An element-level infeasibility fathoms: x[0] >= 4 and x[1] >= 4.
    #[test]
    fn per_scalar_box_detects_array_infeasibility() {
        let model = array_x0_plus_x1_le_5();
        let d = run_in_tree_presolve_scalar(&model, &[4.0, 4.0], &[10.0, 10.0], 0, None, &opts1())
            .unwrap();
        assert!(d.ran && d.infeasible);
    }

    /// `y - x == 0` written as ONE array row over x, y ∈ [0,10]^2 (#1568), and
    /// `z - sigmoid(w) == 0` as one array row over w ∈ [-4,4]^2, z ∈ [0,1]^2.
    fn array_rows_model() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arr_var(&mut arena, "x", 0, vec![2]);
        let y = arr_var(&mut arena, "y", 1, vec![2]);
        let w = arr_var(&mut arena, "w", 2, vec![2]);
        let z = arr_var(&mut arena, "z", 3, vec![2]);
        let lin = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: y,
            right: x,
        });
        let sw = arena.add(ExprNode::FunctionCall {
            func: crate::expr::MathFunc::Sigmoid,
            args: vec![w],
        });
        let nl = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: z,
            right: sw,
        });
        let z0 = elem(&mut arena, z, 0);
        let eq = |body| ConstraintRepr {
            body,
            sense: ConstraintSense::Eq,
            rhs: 0.0,
            name: Some("row".to_string()),
        };
        ModelRepr {
            arena,
            objective: z0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq(lin), eq(nl)],
            variables: vec![
                arr_vinfo("x", 0, vec![2], 0.0, 10.0),
                arr_vinfo("y", 2, vec![2], 0.0, 10.0),
                arr_vinfo("w", 4, vec![2], -4.0, 4.0),
                arr_vinfo("z", 6, vec![2], 0.0, 1.0),
            ],
            n_vars: 8,
        }
    }

    fn sigmoid(t: f64) -> f64 {
        1.0 / (1.0 + (-t).exp())
    }

    /// #1568: through an array row the proxy view derives NOTHING per element
    /// (the control), and the expanded view tightens each element on its own.
    #[test]
    fn expanded_rows_tighten_through_array_rows() {
        let model = array_rows_model();
        // x ∈ [0,1]^2 narrows y; w0 ∈ [1,2], w1 ∈ [-2,-1] narrows z elementwise.
        let lb = [0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0];
        let ub = [1.0, 1.0, 10.0, 10.0, 2.0, -1.0, 1.0, 1.0];

        let proxy = scalarize_for_fbbt(&model).unwrap();
        assert_eq!(proxy.n_proxies(), 4);
        let p = run_in_tree_presolve_view(&proxy, &lb, &ub, 0, None, &opts1());
        assert!(p.ran && !p.infeasible);
        assert_eq!(
            p.bounds_tightened, 0,
            "control: hull proxies derive nothing"
        );

        let (view, stats) = expand_rows_for_fbbt(&model).expect("every node here expands");
        assert_eq!(view.n_proxies(), 0);
        assert_eq!(stats.rows_added, 4);
        assert_eq!(stats.on_hull, 0);
        assert_eq!(
            view.model().constraints.len(),
            4,
            "2 array rows x 2 elements"
        );
        let d = run_in_tree_presolve_view(&view, &lb, &ub, 0, None, &opts1());
        assert!(d.ran && !d.infeasible);
        for k in 0..2 {
            assert!(d.ub[2 + k] <= 1.0 + 1e-9, "y[{k}] ub = {}", d.ub[2 + k]);
        }
        assert!(d.lb[6] >= sigmoid(1.0) - 1e-9 && d.ub[6] <= sigmoid(2.0) + 1e-9);
        assert!(d.lb[7] >= sigmoid(-2.0) - 1e-9 && d.ub[7] <= sigmoid(-1.0) + 1e-9);
        // The two z elements land in DISJOINT intervals -- impossible for a hull.
        assert!(d.ub[7] < d.lb[6]);
    }

    /// Soundness and the differential bar on random node boxes: no feasible
    /// point inside the box is cut, and the expanded box is never looser than
    /// the proxy box.
    #[test]
    fn expanded_rows_never_cut_a_feasible_point() {
        let model = array_rows_model();
        let proxy = scalarize_for_fbbt(&model).unwrap();
        let (view, _) = expand_rows_for_fbbt(&model).unwrap();
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let root_lb = [0.0, 0.0, 0.0, 0.0, -4.0, -4.0, 0.0, 0.0];
        let root_ub = [10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 1.0, 1.0];
        let (mut points, mut boxes) = (0usize, 0usize);
        for _ in 0..300 {
            // A feasible point, then a random node box that contains it.
            let x = [10.0 * rnd(), 10.0 * rnd()];
            let w = [8.0 * rnd() - 4.0, 8.0 * rnd() - 4.0];
            let pt = [
                x[0],
                x[1],
                x[0],
                x[1],
                w[0],
                w[1],
                sigmoid(w[0]),
                sigmoid(w[1]),
            ];
            let mut lb = root_lb;
            let mut ub = root_ub;
            for j in 0..8 {
                lb[j] = root_lb[j] + (pt[j] - root_lb[j]) * rnd();
                ub[j] = pt[j] + (root_ub[j] - pt[j]) * rnd();
            }
            let d = run_in_tree_presolve_view(&view, &lb, &ub, 0, None, &opts1());
            let p = run_in_tree_presolve_view(&proxy, &lb, &ub, 0, None, &opts1());
            assert!(!d.infeasible, "a box holding a feasible point was fathomed");
            for j in 0..8 {
                assert!(
                    d.lb[j] <= pt[j] + 1e-7 && pt[j] <= d.ub[j] + 1e-7,
                    "slot {j}: feasible {} cut by [{}, {}]",
                    pt[j],
                    d.lb[j],
                    d.ub[j]
                );
                assert!(
                    d.lb[j] >= p.lb[j] - 1e-9 && d.ub[j] <= p.ub[j] + 1e-9,
                    "looser at {j}"
                );
                points += 1;
            }
            boxes += 1;
        }
        assert_eq!((boxes, points), (300, 2400));
    }

    /// On a scalar-layout model the per-scalar entry point IS the per-block
    /// kernel: every output field is bit-identical over a sweep of boxes (the
    /// bound-neutral argument for scalar models).
    #[test]
    fn scalar_layout_is_identical_to_per_block() {
        let model = x_plus_y_le_5();
        assert!(is_scalar_layout(&model));
        let mut checked = 0usize;
        for &(l0, u0, l1, u1) in &[
            (3.0, 10.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 10.0),
            (10.0, 10.0, 10.0, 10.0),
            (2.5, 2.5 - 1e-14, 0.0, 10.0),
            (-1.0, 4.0, 1.5, 3.0),
        ] {
            for probing in [false, true] {
                let mut o = opts1();
                o.probing = probing;
                o.probe_max_vars = 8;
                let a = run_in_tree_presolve(&model, &[l0, l1], &[u0, u1], 0, Some(4.0), &o);
                let b = run_in_tree_presolve_scalar(&model, &[l0, l1], &[u0, u1], 0, Some(4.0), &o)
                    .unwrap();
                assert_eq!(
                    a.lb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.lb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    a.ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(a.infeasible, b.infeasible);
                assert_eq!(a.bounds_tightened, b.bounds_tightened);
                assert_eq!(a.subtol_repaired, b.subtol_repaired);
                assert_eq!(a.ran, b.ran);
                checked += 1;
            }
        }
        assert_eq!(
            checked, 10,
            "probe did not execute the comparisons it claims"
        );
    }

    /// A box of the wrong length is an error, not a silent skip.
    #[test]
    fn wrong_box_length_is_refused() {
        let model = array_x0_plus_x1_le_5();
        let e = run_in_tree_presolve_scalar(&model, &[0.0], &[10.0], 0, None, &opts1());
        assert!(
            e.is_err(),
            "a per-block box on an array model must be refused"
        );
    }

    /// A non-contiguous layout (block offsets that do not tile 0..n_vars) is
    /// refused by name rather than scalarized into the wrong slots.
    #[test]
    fn noncontiguous_layout_is_refused() {
        let mut model = array_x0_plus_x1_le_5();
        model.variables[0].offset = 1;
        model.n_vars = 3;
        let e = scalarize_for_fbbt(&model);
        assert!(e.is_err());
        assert!(e.unwrap_err().contains("offset"));
    }

    /// Full-arity element indices are rewritten to scalar slots; array-valued
    /// references (the whole matrix, a row `X[1]` of a rank-2 block, a slice)
    /// stay array-valued on the hull proxy.
    #[test]
    fn scalarize_rewrites_elements_and_keeps_arrays_on_proxy() {
        let mut arena = ExprArena::new();
        let t = scalar_var(&mut arena, "t", 0);
        let x = arr_var(&mut arena, "X", 1, vec![2, 2]);
        let x10 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Tuple(vec![1, 0]),
        });
        let row = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(1),
        });
        let sl = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Multi(vec![IndexElem::Scalar(0), IndexElem::FULL]),
        });
        let model = ModelRepr {
            arena,
            objective: t,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![
                vinfo("t", 0.0, 1.0),
                arr_vinfo("X", 1, vec![2, 2], -1.0, 2.0),
            ],
            n_vars: 5,
        };
        let view = scalarize_for_fbbt(&model).unwrap();
        assert_eq!(view.n_scalar(), 5);
        assert_eq!(view.n_proxies(), 1);
        let m = view.model();
        assert_eq!(m.variables.len(), 6);
        // t stays slot 0.
        assert!(matches!(
            m.arena.get(t),
            ExprNode::Variable {
                index: 0,
                size: 1,
                ..
            }
        ));
        // X[1, 0] -> flat 2 -> slot 1 + 2 = 3.
        assert!(matches!(
            m.arena.get(x10),
            ExprNode::Variable {
                index: 3,
                size: 1,
                ..
            }
        ));
        // The whole X -> the proxy (slot 5), still size 4.
        assert!(matches!(
            m.arena.get(x),
            ExprNode::Variable {
                index: 5,
                size: 4,
                ..
            }
        ));
        assert_eq!(m.variables[5].var_type, VarType::Continuous);
        // Row and slice stay Index nodes on the proxy.
        assert!(matches!(m.arena.get(row), ExprNode::Index { .. }));
        assert!(matches!(m.arena.get(sl), ExprNode::Index { .. }));
    }

    /// Probing an array binary block PER ELEMENT. z ∈ {0,1}^2 as one block,
    /// z[0] + z[1] == 1. Fixing the whole block (what a per-block hull box
    /// does) makes both fixings infeasible and "proves" a feasible node empty;
    /// per element, the root box survives and z[0] = 1 forces z[1] = 0.
    #[test]
    fn probing_array_binary_block_is_per_element() {
        let mut arena = ExprArena::new();
        let z = arr_var(&mut arena, "z", 0, vec![2]);
        let z0 = elem(&mut arena, z, 0);
        let z1 = elem(&mut arena, z, 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: z0,
            right: z1,
        });
        let mut zi = arr_vinfo("z", 0, vec![2], 0.0, 1.0);
        zi.var_type = VarType::Binary;
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![zi],
            n_vars: 2,
        };
        let mut o = opts1();
        o.probing = true;
        o.probe_max_vars = 8;
        let root =
            run_in_tree_presolve_scalar(&model, &[0.0, 0.0], &[1.0, 1.0], 0, None, &o).unwrap();
        assert!(
            !root.infeasible,
            "a feasible array-binary node was fathomed"
        );
        let node =
            run_in_tree_presolve_scalar(&model, &[1.0, 0.0], &[1.0, 1.0], 0, None, &o).unwrap();
        assert!(!node.infeasible);
        assert!(
            node.ub[1] <= 1e-9,
            "z[1] should be forced to 0, got {}",
            node.ub[1]
        );
    }

    /// Feasible-point sampling on a model mixing element references and a
    /// whole-array reduction (the proxy path): x ∈ [0,4]^3 as one block,
    ///   sum(x) <= 5,  x[0] * x[1] >= 1,  x[2] - x[0] <= 1.
    /// For many random sub-boxes and random feasible points inside them, the
    /// kernel must neither fathom the box nor cut the point. Deterministic LCG.
    #[test]
    fn per_scalar_kernel_never_cuts_a_feasible_point() {
        let mut arena = ExprArena::new();
        let x = arr_var(&mut arena, "x", 0, vec![3]);
        let s = arena.add(ExprNode::Sum {
            operand: x,
            axis: None,
        });
        let x0 = elem(&mut arena, x, 0);
        let x1 = elem(&mut arena, x, 1);
        let x2 = elem(&mut arena, x, 2);
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x1,
        });
        let diff = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x2,
            right: x0,
        });
        let model = ModelRepr {
            arena,
            objective: s,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: s,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: prod,
                    sense: ConstraintSense::Ge,
                    rhs: 1.0,
                    name: None,
                },
                ConstraintRepr {
                    body: diff,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
            ],
            variables: vec![arr_vinfo("x", 0, vec![3], 0.0, 4.0)],
            n_vars: 3,
        };
        let mut seed: u64 = 0x1513;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut checked = 0usize;
        let mut tightened_any = 0usize;
        for _ in 0..4000 {
            let p = [4.0 * rnd(), 4.0 * rnd(), 4.0 * rnd()];
            let feasible = p[0] + p[1] + p[2] <= 5.0 && p[0] * p[1] >= 1.0 && p[2] - p[0] <= 1.0;
            if !feasible {
                continue;
            }
            let mut lb = [0.0; 3];
            let mut ub = [0.0; 3];
            for k in 0..3 {
                lb[k] = p[k] - (p[k]) * rnd();
                ub[k] = p[k] + (4.0 - p[k]) * rnd();
            }
            for probing in [false, true] {
                let mut o = opts1();
                o.probing = probing;
                let d = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &o).unwrap();
                assert!(!d.infeasible, "fathomed a box containing feasible {p:?}");
                for k in 0..3 {
                    assert!(
                        d.lb[k] <= p[k] + 1e-9 && p[k] <= d.ub[k] + 1e-9,
                        "cut feasible {p:?} at x[{k}]: [{}, {}]",
                        d.lb[k],
                        d.ub[k]
                    );
                }
                if d.bounds_tightened > 0 {
                    tightened_any += 1;
                }
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} boxes checked");
        // The kernel must actually DO something on this class, else the
        // soundness check above is vacuous.
        assert!(tightened_any > 100, "only {tightened_any} tightenings");
    }

    // ── #1568: per-row expansion, the view cache, ported #1579 cases ─────

    use crate::expr::{IndexElem, MathFunc};

    fn opts_rows(expand: bool) -> InTreePresolveOptions {
        InTreePresolveOptions {
            expand_array_rows: Some(expand),
            ..opts1()
        }
    }

    fn bin(arena: &mut ExprArena, op: BinOp, left: ExprId, right: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp { op, left, right })
    }

    fn row(body: ExprId, sense: ConstraintSense, rhs: f64) -> ConstraintRepr {
        ConstraintRepr {
            body,
            sense,
            rhs,
            name: None,
        }
    }

    // A fixed 2-3-1 sigmoid net, the shape `discopt.ml`'s full-space form emits.
    const WT: [[f64; 2]; 3] = [[0.9, -1.3], [-0.4, 0.7], [1.6, 0.2]];
    const B: [f64; 3] = [0.1, -0.2, 0.05];
    const V: [f64; 3] = [1.2, -0.8, 0.5];
    const C: f64 = -0.3;

    /// Blocks: x (2,) in [-1,1], zh (3,) in [-5,5], z (3,) in [0,1] (flat
    /// slots 0-1, 2-4, 5-7). Rows, every one made only of WHOLE-array
    /// references:
    ///   zh - (WT @ x + b) == 0          (`matmul`)  or
    ///   zh - (sum(WT * x, axis=1) + b) == 0  (`!matmul`, `W.T * x` + axis sum)
    ///   z - sigmoid(zh) == 0
    /// objective: min sum(v * z) + c (array-structured, scalar-valued).
    fn sigmoid_net(matmul: bool) -> ModelRepr {
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![2]);
        let zh = arr_var(&mut a, "zh", 1, vec![3]);
        let z = arr_var(&mut a, "z", 2, vec![3]);
        let wt = a.add(ExprNode::ConstantArray(
            WT.iter().flatten().copied().collect(),
            vec![3, 2],
        ));
        let b = a.add(ExprNode::ConstantArray(B.to_vec(), vec![3]));
        let lin = if matmul {
            a.add(ExprNode::MatMul { left: wt, right: x })
        } else {
            let prod = bin(&mut a, BinOp::Mul, wt, x);
            a.add(ExprNode::Sum {
                operand: prod,
                axis: Some(1),
            })
        };
        let pre = bin(&mut a, BinOp::Add, lin, b);
        let r1 = bin(&mut a, BinOp::Sub, zh, pre);
        let sig = a.add(ExprNode::FunctionCall {
            func: MathFunc::Sigmoid,
            args: vec![zh],
        });
        let r2 = bin(&mut a, BinOp::Sub, z, sig);
        let v = a.add(ExprNode::ConstantArray(V.to_vec(), vec![3]));
        let vz = bin(&mut a, BinOp::Mul, v, z);
        let s = a.add(ExprNode::Sum {
            operand: vz,
            axis: None,
        });
        let c = a.add(ExprNode::Constant(C));
        let obj = bin(&mut a, BinOp::Add, s, c);
        ModelRepr {
            arena: a,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                row(r1, ConstraintSense::Eq, 0.0),
                row(r2, ConstraintSense::Eq, 0.0),
            ],
            variables: vec![
                arr_vinfo("x", 0, vec![2], -1.0, 1.0),
                arr_vinfo("zh", 2, vec![3], -5.0, 5.0),
                arr_vinfo("z", 5, vec![3], 0.0, 1.0),
            ],
            n_vars: 8,
        }
    }

    /// The SAME net written one scalar row per element, over `x[i]` element
    /// references -- the reference the expansion must equal.
    fn sigmoid_net_scalar_rows() -> ModelRepr {
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![2]);
        let zh = arr_var(&mut a, "zh", 1, vec![3]);
        let z = arr_var(&mut a, "z", 2, vec![3]);
        let xs: Vec<ExprId> = (0..2).map(|i| elem(&mut a, x, i)).collect();
        let mut cons = Vec::new();
        let mut zs = Vec::new();
        for k in 0..3 {
            let terms: Vec<ExprId> = (0..2)
                .map(|j| {
                    let w = a.add(ExprNode::Constant(WT[k][j]));
                    bin(&mut a, BinOp::Mul, w, xs[j])
                })
                .collect();
            let dot = a.add(ExprNode::SumOver { terms });
            let bk = a.add(ExprNode::Constant(B[k]));
            let pre = bin(&mut a, BinOp::Add, dot, bk);
            let zhk = elem(&mut a, zh, k);
            let r1 = bin(&mut a, BinOp::Sub, zhk, pre);
            cons.push(row(r1, ConstraintSense::Eq, 0.0));
            let zk = elem(&mut a, z, k);
            let sig = a.add(ExprNode::FunctionCall {
                func: MathFunc::Sigmoid,
                args: vec![zhk],
            });
            let r2 = bin(&mut a, BinOp::Sub, zk, sig);
            cons.push(row(r2, ConstraintSense::Eq, 0.0));
            let vk = a.add(ExprNode::Constant(V[k]));
            zs.push(bin(&mut a, BinOp::Mul, vk, zk));
        }
        let s = a.add(ExprNode::SumOver { terms: zs });
        let c = a.add(ExprNode::Constant(C));
        let obj = bin(&mut a, BinOp::Add, s, c);
        ModelRepr {
            arena: a,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: cons,
            variables: vec![
                arr_vinfo("x", 0, vec![2], -1.0, 1.0),
                arr_vinfo("zh", 2, vec![3], -5.0, 5.0),
                arr_vinfo("z", 5, vec![3], 0.0, 1.0),
            ],
            n_vars: 8,
        }
    }

    fn net_root_box() -> (Vec<f64>, Vec<f64>) {
        let lb = vec![-1.0, -1.0, -5.0, -5.0, -5.0, 0.0, 0.0, 0.0];
        let ub = vec![1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0];
        (lb, ub)
    }

    /// The issue's measurement, pinned (ported from #1579): with `x0` branched
    /// to [0, 1] and `x1` to [-1, 0], the vectorised rows tighten NOTHING
    /// without the expansion (every reference is a hull proxy) and, with it,
    /// give exactly the box of the same rows written per element -- for both
    /// the `@` and the `W.T * x` + axis-sum forms of the linear layer.
    #[test]
    fn array_rows_match_scalar_rows_on_sigmoid_net() {
        let (mut lb, mut ub) = net_root_box();
        lb[0] = 0.0;
        ub[1] = 0.0;
        let reference = run_in_tree_presolve_scalar(
            &sigmoid_net_scalar_rows(),
            &lb,
            &ub,
            0,
            None,
            &opts_rows(false),
        )
        .unwrap();
        assert!(reference.ran && !reference.infeasible);
        for k in 2..8 {
            assert!(
                reference.ub[k] - reference.lb[k] < 0.75 * (ub[k] - lb[k]),
                "reference did not tighten slot {k}"
            );
        }
        let mut checks = 0;
        for matmul in [true, false] {
            let model = sigmoid_net(matmul);
            let off =
                run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(false)).unwrap();
            assert_eq!(
                off.bounds_tightened, 0,
                "hull path tightened (matmul={matmul})"
            );
            assert_eq!(off.array_rows_added, 0);
            assert_eq!(off.array_rows, None);
            let on =
                run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
            assert!(on.ran && !on.infeasible);
            assert_eq!(on.array_rows, Some(ArrayRowsOutcome::Expanded));
            assert_eq!(
                on.array_rows_added, 6,
                "3 rows per layer row (matmul={matmul})"
            );
            assert_eq!(on.array_rows_on_hull, 0);
            for k in 0..8 {
                assert!(
                    (on.lb[k] - reference.lb[k]).abs() <= 1e-9
                        && (on.ub[k] - reference.ub[k]).abs() <= 1e-9,
                    "slot {k} (matmul={matmul}): on [{}, {}] vs scalar rows [{}, {}]",
                    on.lb[k],
                    on.ub[k],
                    reference.lb[k],
                    reference.ub[k]
                );
                checks += 1;
            }
        }
        assert_eq!(checks, 16);
    }

    /// Differential bound test + feasible-point sampling on the net (ported
    /// from #1579), with and without an incumbent cutoff (which reads the
    /// EXPANDED objective) and with probing on a third of the boxes:
    ///   * ON never cuts a feasible point and never fathoms its box;
    ///   * ON is never looser than OFF on any slot;
    ///   * ON is strictly tighter often enough that the comparison is not
    ///     vacuous.
    #[test]
    fn array_rows_never_looser_and_never_cut_a_feasible_point() {
        let mut seed: u64 = 0x1568;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let (rlb, rub) = net_root_box();
        let mut checked = 0usize;
        let mut strictly_tighter = 0usize;
        for matmul in [true, false] {
            let model = sigmoid_net(matmul);
            let cache = ArrayRowViewCache::default();
            for trial in 0..1500 {
                let xp = [2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0];
                let mut p = vec![xp[0], xp[1]];
                let mut obj = C;
                let mut zs = Vec::new();
                for k in 0..3 {
                    let zh = WT[k][0] * xp[0] + WT[k][1] * xp[1] + B[k];
                    p.push(zh);
                    zs.push(sigmoid(zh));
                    obj += V[k] * sigmoid(zh);
                }
                p.extend(zs);
                let mut lb = vec![0.0; 8];
                let mut ub = vec![0.0; 8];
                for k in 0..8 {
                    lb[k] = p[k] - (p[k] - rlb[k]) * rnd();
                    ub[k] = p[k] + (rub[k] - p[k]) * rnd();
                }
                let cutoff = (trial % 2 == 1).then(|| obj + 0.05 * rnd());
                let mut on_o = opts_rows(true);
                let mut off_o = opts_rows(false);
                on_o.probing = trial % 3 == 0;
                off_o.probing = on_o.probing;
                let on =
                    run_in_tree_presolve_scalar_cached(&model, &cache, &lb, &ub, 0, cutoff, &on_o)
                        .unwrap();
                let off = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, cutoff, &off_o).unwrap();
                assert!(
                    !on.infeasible,
                    "ON fathomed a box containing feasible {p:?}"
                );
                assert!(!off.infeasible);
                let mut tighter = false;
                for k in 0..8 {
                    assert!(
                        on.lb[k] <= p[k] + 1e-9 && p[k] <= on.ub[k] + 1e-9,
                        "ON cut feasible {p:?} at slot {k}: [{}, {}]",
                        on.lb[k],
                        on.ub[k]
                    );
                    assert!(
                        on.lb[k] >= off.lb[k] - 1e-9 && on.ub[k] <= off.ub[k] + 1e-9,
                        "ON looser than OFF at slot {k}: [{}, {}] vs [{}, {}]",
                        on.lb[k],
                        on.ub[k],
                        off.lb[k],
                        off.ub[k]
                    );
                    if on.lb[k] > off.lb[k] + 1e-7 || on.ub[k] < off.ub[k] - 1e-7 {
                        tighter = true;
                    }
                }
                strictly_tighter += tighter as usize;
                checked += 1;
            }
            assert_eq!(cache.builds(), 1, "the view is built once per model");
        }
        assert_eq!(checked, 3000);
        assert!(
            strictly_tighter > 1000,
            "only {strictly_tighter} boxes tighter"
        );
    }

    /// `sum(x) <= 5` over a whole-array reference: the hull path derives
    /// nothing; the expanded row bounds each element, and a branch on `x[0]`
    /// flows to the others.
    #[test]
    fn sum_row_tightens_each_element() {
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![3]);
        let s = a.add(ExprNode::Sum {
            operand: x,
            axis: None,
        });
        let model = ModelRepr {
            arena: a,
            objective: s,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![row(s, ConstraintSense::Le, 5.0)],
            variables: vec![arr_vinfo("x", 0, vec![3], 0.0, 10.0)],
            n_vars: 3,
        };
        let (lb, ub) = ([3.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let off =
            run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(false)).unwrap();
        assert_eq!(off.bounds_tightened, 0);
        let on = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
        assert_eq!(on.array_rows_added, 1);
        assert!((on.ub[0] - 5.0).abs() <= 1e-9, "x0 ub {}", on.ub[0]);
        assert!((on.ub[1] - 2.0).abs() <= 1e-9, "x1 ub {}", on.ub[1]);
        assert!((on.ub[2] - 2.0).abs() <= 1e-9, "x2 ub {}", on.ub[2]);
    }

    /// Indexing that stays array-valued: a slice `x[1:3]` and a partial index
    /// `X[1]` of a matrix expand to exactly the selected elements, in
    /// row-major order, and nothing else is touched. The array-valued
    /// objective cannot be a scalar root, so it stays in proxy form and the
    /// call reports `Partial` -- while both rows still expand.
    #[test]
    fn slice_and_partial_index_rows_select_the_right_elements() {
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![3]);
        let xm = arr_var(&mut a, "X", 1, vec![2, 2]);
        let sl = a.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Multi(vec![IndexElem::Slice {
                start: Some(1),
                stop: Some(3),
                step: None,
            }]),
        });
        let row1 = a.add(ExprNode::Index {
            base: xm,
            index: IndexSpec::Scalar(1),
        });
        // X[1] - [0.5, 2.0] <= 0  (per element: X[1,0] <= 0.5, X[1,1] <= 2.0)
        let cap = a.add(ExprNode::ConstantArray(vec![0.5, 2.0], vec![2]));
        let r2 = bin(&mut a, BinOp::Sub, row1, cap);
        let model = ModelRepr {
            arena: a,
            objective: sl,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                row(sl, ConstraintSense::Le, 1.0),
                row(r2, ConstraintSense::Le, 0.0),
            ],
            variables: vec![
                arr_vinfo("x", 0, vec![3], 0.0, 10.0),
                arr_vinfo("X", 3, vec![2, 2], 0.0, 10.0),
            ],
            n_vars: 7,
        };
        let lb = [0.0; 7];
        let ub = [10.0; 7];
        let on = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
        assert_eq!(on.array_rows_added, 4);
        assert_eq!(on.array_rows_on_hull, 0);
        match &on.array_rows {
            Some(ArrayRowsOutcome::Partial(why)) => assert!(why.starts_with("objective"), "{why}"),
            other => panic!("expected Partial(objective ..), got {other:?}"),
        }
        let want_ub = [10.0, 1.0, 1.0, 10.0, 10.0, 0.5, 2.0];
        for k in 0..7 {
            assert!(
                (on.ub[k] - want_ub[k]).abs() <= 1e-9,
                "slot {k}: ub {} want {}",
                on.ub[k],
                want_ub[k]
            );
            assert_eq!(on.lb[k], 0.0);
        }
    }

    /// `norm2` and `prod` over a vector expand to their exact scalar forms
    /// (`sqrt(sum x_i * x_i)`, a product chain): the box equals the box of a
    /// scalar twin written in exactly those forms. The squares are lowered as
    /// `x_i * x_i`, which FBBT's product arm cannot invert through zero, so the
    /// norm row tightens `x` no further than the twin does -- the expansion is
    /// faithful, not stronger. NOTE: the chain is a sequence of binary
    /// products, so this row never reaches FBBT's single-array-argument
    /// `Prod` arm -- a defect in that arm is invisible here.
    #[test]
    fn norm2_and_prod_rows_expand_exactly() {
        let build = |twin: bool| {
            let mut a = ExprArena::new();
            let x = arr_var(&mut a, "x", 0, vec![2]);
            let y = arr_var(&mut a, "y", 1, vec![2]);
            let (n2, pr) = if twin {
                let xs: Vec<ExprId> = (0..2).map(|i| elem(&mut a, x, i)).collect();
                let ys: Vec<ExprId> = (0..2).map(|i| elem(&mut a, y, i)).collect();
                let sq: Vec<ExprId> = xs.iter().map(|&t| bin(&mut a, BinOp::Mul, t, t)).collect();
                let tot = bin(&mut a, BinOp::Add, sq[0], sq[1]);
                let n2 = a.add(ExprNode::FunctionCall {
                    func: MathFunc::Sqrt,
                    args: vec![tot],
                });
                (n2, bin(&mut a, BinOp::Mul, ys[0], ys[1]))
            } else {
                let n2 = a.add(ExprNode::FunctionCall {
                    func: MathFunc::Norm2,
                    args: vec![x],
                });
                let pr = a.add(ExprNode::FunctionCall {
                    func: MathFunc::Prod,
                    args: vec![y],
                });
                (n2, pr)
            };
            ModelRepr {
                arena: a,
                objective: n2,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![
                    row(n2, ConstraintSense::Le, 1.0),
                    row(pr, ConstraintSense::Ge, 4.0),
                ],
                variables: vec![
                    arr_vinfo("x", 0, vec![2], -5.0, 5.0),
                    arr_vinfo("y", 2, vec![2], 0.1, 8.0),
                ],
                n_vars: 4,
            }
        };
        let lb = [-5.0, -5.0, 0.1, 0.1];
        let ub = [5.0, 5.0, 8.0, 8.0];
        let on = run_in_tree_presolve_scalar(&build(false), &lb, &ub, 0, None, &opts_rows(true))
            .unwrap();
        let twin = run_in_tree_presolve_scalar(&build(true), &lb, &ub, 0, None, &opts_rows(false))
            .unwrap();
        assert_eq!(on.array_rows_added, 2);
        assert_eq!(on.array_rows_on_hull, 0);
        for k in 0..4 {
            assert!(
                (on.lb[k] - twin.lb[k]).abs() <= 1e-12,
                "lb {k}: {} vs {}",
                on.lb[k],
                twin.lb[k]
            );
            assert!(
                (on.ub[k] - twin.ub[k]).abs() <= 1e-12,
                "ub {k}: {} vs {}",
                on.ub[k],
                twin.ub[k]
            );
        }
        // y0 * y1 >= 4 with y in [0.1, 8] forces each y_i >= 0.5.
        for k in 2..4 {
            assert!(on.lb[k] >= 0.5 - 1e-9, "y[{}] lb {}", k - 2, on.lb[k]);
        }
    }

    /// A row with a node that has no exact element-wise form -- `norm1` and
    /// `max` over an array, which the shape pass types as element-wise but are
    /// reductions -- keeps its proxy-view form: no scalar row is added for it,
    /// it is counted, and the box equals the hull path's. An expandable row
    /// next to it IS expanded, replacing its array row (the per-row mode).
    #[test]
    fn unexpandable_rows_stay_on_the_hull_path() {
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![2]);
        let n1 = a.add(ExprNode::FunctionCall {
            func: MathFunc::Norm1,
            args: vec![x],
        });
        let s = a.add(ExprNode::Sum {
            operand: x,
            axis: None,
        });
        let mx = a.add(ExprNode::FunctionCall {
            func: MathFunc::Max,
            args: vec![x],
        });
        let mixed = bin(&mut a, BinOp::Add, s, mx);
        let base = ModelRepr {
            arena: a,
            objective: s,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                row(n1, ConstraintSense::Le, 1.0),
                row(mixed, ConstraintSense::Le, 3.0),
            ],
            variables: vec![arr_vinfo("x", 0, vec![2], 0.0, 4.0)],
            n_vars: 2,
        };
        let (lb, ub) = ([0.0, 0.0], [4.0, 4.0]);
        let (view, st) = expand_rows_for_fbbt(&base).unwrap();
        assert_eq!((st.rows_added, st.on_hull), (0, 2));
        assert!(st
            .first_refusal
            .as_deref()
            .unwrap()
            .starts_with("constraint 0"));
        assert_eq!(view.model().constraints.len(), 2);
        assert_eq!(view.n_proxies(), 1, "the kept rows still read the hull");
        let on = run_in_tree_presolve_scalar(&base, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
        let off = run_in_tree_presolve_scalar(&base, &lb, &ub, 0, None, &opts_rows(false)).unwrap();
        assert_eq!(on.array_rows_on_hull, 2);
        assert!(matches!(on.array_rows, Some(ArrayRowsOutcome::Partial(_))));
        assert_eq!((on.lb, on.ub), (off.lb, off.ub));

        // Add an expandable row; only it is expanded, and it REPLACES its row.
        let mut m = base.clone();
        m.constraints.push(row(s, ConstraintSense::Le, 1.0));
        let (view, st) = expand_rows_for_fbbt(&m).unwrap();
        assert_eq!((st.rows_added, st.on_hull), (1, 2));
        assert_eq!(view.model().constraints.len(), 3);
        // The two kept rows are the originals, ids unchanged.
        assert_eq!(view.model().constraints[0].body, n1);
        assert_eq!(view.model().constraints[1].body, mixed);
        assert!(view.model().constraints[2].body.0 >= m.arena.len());
        let on = run_in_tree_presolve_scalar(&m, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
        assert!(on.ub[0] <= 1.0 + 1e-9 && on.ub[1] <= 1.0 + 1e-9);
    }

    /// With the expansion OFF the per-scalar kernel is the pre-#1568 proxy
    /// view, field for field; ON with every row expandable it is a pure view
    /// (no proxies) whose rows REPLACE the array rows.
    #[test]
    fn expansion_off_is_the_legacy_view() {
        let model = sigmoid_net(true);
        let legacy = scalarize_for_fbbt(&model).unwrap();
        let (lb, ub) = net_root_box();
        let mut checks = 0;
        for depth in [0usize, 1, 2] {
            let mut o = opts_rows(false);
            o.depth_stride = 2;
            let off = run_in_tree_presolve_scalar(&model, &lb, &ub, depth, Some(0.1), &o).unwrap();
            let leg = run_in_tree_presolve_view(&legacy, &lb, &ub, depth, Some(0.1), &o);
            assert_eq!((off.lb, off.ub), (leg.lb, leg.ub));
            assert_eq!(
                (off.bounds_tightened, off.infeasible, off.ran),
                (leg.bounds_tightened, leg.infeasible, leg.ran)
            );
            assert_eq!(
                (off.array_rows, off.array_rows_added, off.array_rows_on_hull),
                (None, 0, 0)
            );
            checks += 1;
        }
        assert_eq!(checks, 3);
        let (on, st) = expand_rows_for_fbbt(&model).unwrap();
        assert_eq!(on.n_proxies(), 0);
        assert_eq!(on.model().constraints.len(), 6);
        assert_eq!((st.rows_added, st.on_hull, st.first_refusal), (6, 0, None));
    }

    /// The cache is bound-neutral: a cached call returns exactly what a fresh
    /// build returns, on every box, and rebuilds when the structure changes.
    #[test]
    fn cached_view_is_identical_to_a_fresh_build() {
        let model = sigmoid_net(false);
        let cache = ArrayRowViewCache::default();
        let (rlb, rub) = net_root_box();
        let mut seed: u64 = 0xC0FFEE;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut compared = 0usize;
        for _ in 0..200 {
            let mut lb = rlb.clone();
            let mut ub = rub.clone();
            for k in 0..8 {
                let a = rlb[k] + (rub[k] - rlb[k]) * rnd();
                let b = rlb[k] + (rub[k] - rlb[k]) * rnd();
                lb[k] = a.min(b);
                ub[k] = a.max(b);
            }
            let o = opts_rows(true);
            let fresh = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &o).unwrap();
            let cached =
                run_in_tree_presolve_scalar_cached(&model, &cache, &lb, &ub, 0, None, &o).unwrap();
            assert_eq!(fresh.lb, cached.lb);
            assert_eq!(fresh.ub, cached.ub);
            assert_eq!(
                (
                    fresh.infeasible,
                    fresh.bounds_tightened,
                    fresh.array_rows_added
                ),
                (
                    cached.infeasible,
                    cached.bounds_tightened,
                    cached.array_rows_added
                )
            );
            compared += 1;
        }
        assert_eq!(compared, 200);
        assert_eq!(cache.builds(), 1);
        // A different structure through the same cache is rebuilt, not served
        // the stale view.
        let other = sigmoid_net(true);
        let (lb, ub) = net_root_box();
        run_in_tree_presolve_scalar_cached(&other, &cache, &lb, &ub, 0, None, &opts_rows(true))
            .unwrap();
        assert_eq!(cache.builds(), 2);
        // Expansion OFF never touches the cache.
        run_in_tree_presolve_scalar_cached(&other, &cache, &lb, &ub, 0, None, &opts_rows(false))
            .unwrap();
        assert_eq!(cache.builds(), 2);
    }

    /// The five pointwise functions added to the FBBT expansion set (`asinh`,
    /// `acosh`, `atanh`, `erf`, `entropy` = x ln x) expand ELEMENTWISE: for
    /// `y - f(x) == 0` over vectors, ON equals the same rows written per
    /// element exactly, OFF derives nothing, and no feasible point is cut.
    #[test]
    fn pointwise_fbbt_functions_expand_elementwise() {
        fn erf(t: f64) -> f64 {
            libm::erf(t)
        }
        fn xlogx(t: f64) -> f64 {
            crate::expr::xlogx(t)
        }
        let cases: [(MathFunc, f64, f64, fn(f64) -> f64); 5] = [
            (MathFunc::Asinh, -3.0, 3.0, f64::asinh),
            (MathFunc::Acosh, 1.0, 4.0, f64::acosh),
            (MathFunc::Atanh, -0.9, 0.9, libm::atanh),
            (MathFunc::Erf, -2.0, 2.0, erf),
            (MathFunc::Entropy, 0.05, 3.0, xlogx),
        ];
        let (mut assertions, mut tightened) = (0usize, 0usize);
        for (func, xlo, xhi, f) in cases {
            // y - f(x) == 0 with x, y of shape (3,), as one array row ...
            let mut a = ExprArena::new();
            let x = arr_var(&mut a, "x", 0, vec![3]);
            let y = arr_var(&mut a, "y", 1, vec![3]);
            let fx = a.add(ExprNode::FunctionCall {
                func,
                args: vec![x],
            });
            let r = bin(&mut a, BinOp::Sub, y, fx);
            let obj = elem(&mut a, y, 0);
            let vars = vec![
                arr_vinfo("x", 0, vec![3], xlo, xhi),
                arr_vinfo("y", 3, vec![3], -10.0, 10.0),
            ];
            let model = ModelRepr {
                arena: a,
                objective: obj,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![row(r, ConstraintSense::Eq, 0.0)],
                variables: vars.clone(),
                n_vars: 6,
            };
            // ... and as three element rows.
            let mut a = ExprArena::new();
            let x = arr_var(&mut a, "x", 0, vec![3]);
            let y = arr_var(&mut a, "y", 1, vec![3]);
            let mut cons = Vec::new();
            for k in 0..3 {
                let xk = elem(&mut a, x, k);
                let yk = elem(&mut a, y, k);
                let fk = a.add(ExprNode::FunctionCall {
                    func,
                    args: vec![xk],
                });
                cons.push(row(
                    bin(&mut a, BinOp::Sub, yk, fk),
                    ConstraintSense::Eq,
                    0.0,
                ));
            }
            let obj = elem(&mut a, y, 0);
            let twin = ModelRepr {
                arena: a,
                objective: obj,
                objective_sense: ObjectiveSense::Minimize,
                constraints: cons,
                variables: vars,
                n_vars: 6,
            };
            let (_, st) = expand_rows_for_fbbt(&model).unwrap();
            assert_eq!((st.rows_added, st.on_hull), (3, 0), "{func:?}");

            let mut seed: u64 = 0x5EED ^ (assertions as u64);
            let mut rnd = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64) / ((1u64 << 53) as f64)
            };
            for _ in 0..100 {
                let xp: Vec<f64> = (0..3).map(|_| xlo + (xhi - xlo) * rnd()).collect();
                let mut p = xp.clone();
                p.extend(xp.iter().map(|&t| f(t)));
                let (mut lb, mut ub) = (vec![0.0; 6], vec![0.0; 6]);
                for k in 0..6 {
                    let (rl, ru) = if k < 3 { (xlo, xhi) } else { (-10.0, 10.0) };
                    lb[k] = p[k] - (p[k] - rl) * rnd();
                    ub[k] = p[k] + (ru - p[k]) * rnd();
                }
                let on = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(true))
                    .unwrap();
                let tw = run_in_tree_presolve_scalar(&twin, &lb, &ub, 0, None, &opts_rows(false))
                    .unwrap();
                let off = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(false))
                    .unwrap();
                assert_eq!(on.array_rows, Some(ArrayRowsOutcome::Expanded), "{func:?}");
                assert!(!on.infeasible, "{func:?}: fathomed a box holding {p:?}");
                assert_eq!(off.bounds_tightened, 0, "{func:?}: hull control tightened");
                for k in 0..6 {
                    assert!(
                        on.lb[k] <= p[k] + 1e-9 && p[k] <= on.ub[k] + 1e-9,
                        "{func:?}: cut feasible {p:?} at {k}: [{}, {}]",
                        on.lb[k],
                        on.ub[k]
                    );
                    assert!(
                        (on.lb[k] - tw.lb[k]).abs() <= 1e-12
                            && (on.ub[k] - tw.ub[k]).abs() <= 1e-12,
                        "{func:?} slot {k}: on [{}, {}] vs element rows [{}, {}]",
                        on.lb[k],
                        on.ub[k],
                        tw.lb[k],
                        tw.ub[k]
                    );
                    assertions += 2;
                }
                tightened += (on.bounds_tightened > 0) as usize;
            }
        }
        assert_eq!(assertions, 5 * 100 * 6 * 2);
        assert!(tightened > 250, "only {tightened} boxes tightened");
    }

    /// The strict export expansion (tape, `.nl` writer) still refuses the five
    /// FBBT-only functions: widening the FBBT set must not change what is
    /// exported.
    #[test]
    fn export_expansion_still_refuses_fbbt_only_functions() {
        for func in [
            MathFunc::Asinh,
            MathFunc::Acosh,
            MathFunc::Atanh,
            MathFunc::Erf,
            MathFunc::Entropy,
        ] {
            let mut a = ExprArena::new();
            let x = arr_var(&mut a, "x", 0, vec![2]);
            let fx = a.add(ExprNode::FunctionCall {
                func,
                args: vec![x],
            });
            let s = a.add(ExprNode::Sum {
                operand: fx,
                axis: None,
            });
            let model = ModelRepr {
                arena: a,
                objective: s,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![row(s, ConstraintSense::Le, 1.0)],
                variables: vec![arr_vinfo("x", 0, vec![2], 0.1, 0.5)],
                n_vars: 2,
            };
            assert!(crate::expand::expand(&model).is_err(), "{func:?}");
            let rw = crate::expand::expand_rowwise(&model, crate::expand::FuncSet::Export);
            assert!(rw.objective.is_err() && rw.rows[0].is_err(), "{func:?}");
            let rw = crate::expand::expand_rowwise(&model, crate::expand::FuncSet::Fbbt);
            assert!(rw.objective.is_ok() && rw.rows[0].is_ok(), "{func:?}");
        }
    }

    /// Where `expand` succeeds, `expand_rowwise` with the export set emits the
    /// very same program: the per-row mode re-uses `expand_node` and changes
    /// nothing about a row that expands.
    #[test]
    fn rowwise_equals_strict_expand_when_everything_expands() {
        let mut compared = 0;
        for model in [sigmoid_net(true), sigmoid_net(false), array_rows_model()] {
            let strict = crate::expand::expand(&model).unwrap();
            let rw = crate::expand::expand_rowwise(&model, crate::expand::FuncSet::Export);
            let p = &rw.program;
            assert_eq!(strict.op, p.op);
            assert_eq!(strict.a, p.a);
            assert_eq!(strict.b, p.b);
            assert_eq!(
                strict.k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                p.k.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(strict.args_flat, p.args_flat);
            assert_eq!(strict.args_ptr, p.args_ptr);
            assert_eq!(strict.objective_root, p.objective_root);
            assert_eq!(strict.row_roots, p.row_roots);
            assert_eq!(strict.rows_per_constraint, p.rows_per_constraint);
            compared += 1;
        }
        assert_eq!(compared, 3);
    }

    /// Over the instruction budget the whole expansion is declined, the proxy
    /// view runs, and the decline is reported -- never a silent fallback.
    #[test]
    fn over_budget_expansion_is_declined_loudly() {
        let n = 400usize; // 400x400 matmul: 160k products + 160k adds > budget
        let mut a = ExprArena::new();
        let x = arr_var(&mut a, "x", 0, vec![n]);
        let w = a.add(ExprNode::ConstantArray(vec![1.0; n * n], vec![n, n]));
        let mm = a.add(ExprNode::MatMul { left: w, right: x });
        let obj = elem(&mut a, x, 0);
        let model = ModelRepr {
            arena: a,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![row(mm, ConstraintSense::Le, 1.0)],
            variables: vec![arr_vinfo("x", 0, vec![n], 0.0, 1.0)],
            n_vars: n,
        };
        let why = expand_rows_for_fbbt(&model).unwrap_err();
        assert!(why.contains("budget"), "{why}");
        let lb = vec![0.0; n];
        let ub = vec![1.0; n];
        let d = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(true)).unwrap();
        assert!(matches!(d.array_rows, Some(ArrayRowsOutcome::Declined(_))));
        let off =
            run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &opts_rows(false)).unwrap();
        assert_eq!((d.lb, d.ub), (off.lb, off.ub));
    }
}
