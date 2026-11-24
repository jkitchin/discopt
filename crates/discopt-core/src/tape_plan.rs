//! Where a POUNCE problem may be cut into chunks without changing it (#2324).
//!
//! `pounce.NlProblemBuilder` (pounce#1263) copies constraint rows into a
//! problem a chunk at a time. POUNCE shares a subexpression when it counts two
//! references to it, and the builder counts per chunk, reusing a body an
//! earlier chunk shared. So a body that one `build_nl_problem` call would share
//! is inlined by the builder when its references straddle chunks and the first
//! chunk that reaches it reaches it only once. Values and Jacobians cannot move
//! when that happens, but the Hessian of POUNCE's shared-prelude tape can, in
//! the last bits.
//!
//! [`plan_chunks`] finds the cuts that cannot cause this. It is a pure function
//! over a DAG given in compressed form, so the frontend decides what a node's
//! operands are (the Python side flattens additive chains before lowering them)
//! and which nodes are leaves POUNCE never shares; this module only does the
//! graph sweeps, which are too slow in Python at 10^6 nodes. The rule, and why
//! each count is taken from the safe side, is documented on the Python wrapper
//! (`discopt._arena_tape.plan_chunks`). This is its implementation.

/// The result of [`plan_chunks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Start row of each chunk, then the row count: chunk `c` is rows
    /// `starts[c]..starts[c + 1]`. Empty when no exact cut into two or more
    /// chunks exists.
    pub starts: Vec<i64>,
    /// First row reaching each node (`n_rows` for the objective alone), or
    /// `-1` for a node no root reaches.
    pub first_row: Vec<i64>,
    /// Last row reaching each node (`n_rows` when the objective reaches it),
    /// or `-1` for a node no root reaches.
    pub last_row: Vec<i64>,
}

/// Plan the chunks.
///
/// * `ptr`, `idx`: node `i`'s operands are `idx[ptr[i]..ptr[i+1]]`, every one
///   below `i` (an operand list may repeat a node).
/// * `poly[i]`: node `i`'s own operation is one a row POUNCE might keep as a
///   per-row quadratic tree can contain.
/// * `leaf[i]`: node `i` is a variable or a constant, which POUNCE clones into
///   each operand and never shares.
/// * `row_root[r]`: the root of row `r`, or `-1` for a row with no tape (a
///   linear row passed as coefficients).
/// * `obj_root`: the objective's root.
/// * `target_rows`: a chunk is cut at the first admissible boundary once it
///   holds at least this many rows.
///
/// # Panics
///
/// On malformed input (an operand at or above its parent, an index out of
/// range). The caller builds the arrays from the arena; a panic there is a
/// bug in that construction, not a property of a model.
pub fn plan_chunks(
    ptr: &[i64],
    idx: &[i64],
    poly: &[bool],
    leaf: &[bool],
    row_root: &[i64],
    obj_root: i64,
    target_rows: usize,
) -> ChunkPlan {
    let n = poly.len();
    assert_eq!(
        ptr.len(),
        n + 1,
        "ptr must have one entry per node plus one"
    );
    assert_eq!(leaf.len(), n, "leaf must have one entry per node");
    let n_rows = row_root.len();
    let big = i64::MAX;
    let obj = obj_root as usize;
    assert!(obj < n, "objective root out of range");
    let ops = |i: usize| &idx[ptr[i] as usize..ptr[i + 1] as usize];

    // Which nodes reach an operation outside the polynomial set: a row whose
    // root does is certainly materialized through POUNCE's shared maps.
    let mut nonpoly = vec![false; n];
    for i in 0..n {
        let mut np = !poly[i];
        if !np {
            for &c in ops(i) {
                assert!((c as usize) < i, "operand {c} of node {i} is not below it");
                if nonpoly[c as usize] {
                    np = true;
                    break;
                }
            }
        }
        nonpoly[i] = np;
    }

    let r_obj = n_rows as i64; // the objective's position, after every row
    let mut first = vec![big; n];
    let mut first_l = vec![big; n];
    let mut last = vec![-1i64; n];
    let mut deg = vec![0u32; n];
    let mut obj_reach = vec![false; n];
    for (r, &root) in row_root.iter().enumerate() {
        if root < 0 {
            continue;
        }
        let v = root as usize;
        let r = r as i64;
        deg[v] += 1;
        first[v] = first[v].min(r);
        last[v] = last[v].max(r);
        if nonpoly[v] {
            first_l[v] = first_l[v].min(r);
        }
    }
    deg[obj] += 1;
    first[obj] = first[obj].min(r_obj);
    last[obj] = r_obj;
    obj_reach[obj] = true;
    if nonpoly[obj] {
        first_l[obj] = first_l[obj].min(r_obj);
    }

    // Backward sweep. `min1`/`second`: the smallest and second smallest
    // `first_l` over a node's distinct parents. `stamp` dedups an operand a
    // parent lists twice, so `deg` counts distinct parents.
    let mut min1 = vec![big; n];
    let mut second = vec![big; n];
    let mut stamp = vec![usize::MAX; n];
    for i in (0..n).rev() {
        if first[i] == big {
            continue;
        }
        let (fi, fli, li, oi) = (first[i], first_l[i], last[i], obj_reach[i]);
        for &c in ops(i) {
            let c = c as usize;
            if stamp[c] == i {
                continue;
            }
            stamp[c] = i;
            deg[c] += 1;
            first[c] = first[c].min(fi);
            first_l[c] = first_l[c].min(fli);
            last[c] = last[c].max(li);
            obj_reach[c] |= oi;
            if fli < min1[c] {
                second[c] = min1[c];
                min1[c] = fli;
            } else if fli < second[c] {
                second[c] = fli;
            }
        }
    }
    drop((stamp, min1, first_l, nonpoly));

    let unreached = |v: &mut Vec<i64>| {
        for x in v.iter_mut() {
            if *x == big {
                *x = -1;
            }
        }
    };
    let no_plan = |mut first: Vec<i64>, last: Vec<i64>| {
        unreached(&mut first);
        ChunkPlan {
            starts: Vec::new(),
            first_row: first,
            last_row: last,
        }
    };
    if n_rows < 2 {
        return no_plan(first, last);
    }

    // cover[r] > 0: rows r and r + 1 must share a chunk.
    let mut cover = vec![0i64; n_rows + 1];
    for v in 0..n {
        if leaf[v] || deg[v] < 2 || first[v] == big {
            continue;
        }
        let f = first[v];
        if f >= r_obj {
            continue; // the objective alone reaches it: counted as one call counts it
        }
        let mut end = second[v];
        if end >= r_obj {
            if obj_reach[v] {
                // Only the objective's chunk could hold its second reference,
                // and that chunk cannot be merged: one call is the exact build.
                return no_plan(first, last);
            }
            end = r_obj - 1;
        }
        if end > f {
            cover[f as usize] += 1;
            cover[end as usize] -= 1;
        }
    }
    let mut running = 0i64;
    let mut starts = vec![0i64];
    let mut size = 0usize;
    for (r, &c) in cover.iter().take(n_rows - 1).enumerate() {
        running += c;
        size += 1;
        if size >= target_rows && running == 0 {
            starts.push(r as i64 + 1);
            size = 0;
        }
    }
    starts.push(n_rows as i64);
    if starts.len() < 3 {
        return no_plan(first, last);
    }
    unreached(&mut first);
    ChunkPlan {
        starts,
        first_row: first,
        last_row: last,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DAG from per-node operand lists.
    fn csr(ops: &[&[i64]]) -> (Vec<i64>, Vec<i64>) {
        let mut ptr = vec![0i64];
        let mut idx = Vec::new();
        for o in ops {
            idx.extend_from_slice(o);
            ptr.push(idx.len() as i64);
        }
        (ptr, idx)
    }

    /// Node 0 is a variable, node 1 a non-polynomial body `b = exp(x)` over it,
    /// then one product `b * x` per row (nodes 2..2+rows), each a row root,
    /// then an objective node on the variable alone.
    fn body_rows(rows: usize) -> (Vec<i64>, Vec<i64>, Vec<bool>, Vec<bool>, Vec<i64>, i64) {
        let mut ops: Vec<Vec<i64>> = vec![vec![], vec![0]];
        for _ in 0..rows {
            ops.push(vec![1, 0]);
        }
        ops.push(vec![0]);
        let refs: Vec<&[i64]> = ops.iter().map(|o| o.as_slice()).collect();
        let (ptr, idx) = csr(&refs);
        let n = ops.len();
        let mut poly = vec![true; n];
        poly[1] = false;
        let mut leaf = vec![false; n];
        leaf[0] = true;
        let roots = (0..rows as i64).map(|r| r + 2).collect();
        (ptr, idx, poly, leaf, roots, n as i64 - 1)
    }

    #[test]
    fn a_body_reached_twice_in_the_first_chunk_lets_every_later_cut_through() {
        let (ptr, idx, poly, leaf, roots, obj) = body_rows(8);
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &roots, obj, 2);
        assert_eq!(plan.starts, vec![0, 2, 4, 6, 8]);
        assert_eq!(plan.first_row[1], 0);
        assert_eq!(plan.last_row[1], 7);
        assert_eq!(plan.first_row[0], 0, "the variable is reached by row 0");
        assert_eq!(plan.last_row[0], 8, "and by the objective");
    }

    #[test]
    fn a_first_chunk_reaching_a_shared_body_once_is_merged_with_the_next() {
        let (ptr, idx, poly, leaf, roots, obj) = body_rows(8);
        // One row a chunk would reach `b` once in chunk 0: rows 0 and 1 must
        // share a chunk, every later boundary is free.
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &roots, obj, 1);
        assert_eq!(plan.starts, vec![0, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn a_body_only_polynomial_rows_reach_is_never_cut() {
        let (ptr, idx, mut poly, leaf, roots, obj) = body_rows(8);
        poly[1] = true; // every row may be kept as its own tree: nothing is counted
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &roots, obj, 1);
        assert!(plan.starts.is_empty());
    }

    #[test]
    fn a_body_the_objective_shares_with_one_row_has_no_exact_cut() {
        // Rows: r0 = b * x, r1 = x * x (no b); objective = b * x again (node 4).
        let ops: Vec<&[i64]> = vec![&[], &[0], &[1, 0], &[0, 0], &[1, 0]];
        let (ptr, idx) = csr(&ops);
        let poly = vec![true, false, true, true, true];
        let leaf = vec![true, false, false, false, false];
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &[2, 3], 4, 1);
        assert!(plan.starts.is_empty());
        assert!(plan.first_row[1] == 0 && plan.last_row[1] == 2);
    }

    #[test]
    fn a_parent_listing_an_operand_twice_counts_once() {
        // Rows: r0 = b * b, r1 = b * x. Distinct parents of b: 2, so b needs
        // both rows in chunk 0; with b * b counted twice it would not.
        let ops: Vec<&[i64]> = vec![&[], &[0], &[1, 1], &[1, 0], &[0]];
        let (ptr, idx) = csr(&ops);
        let poly = vec![true, false, true, true, true];
        let leaf = vec![true, false, false, false, false];
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &[2, 3, -1], 4, 1);
        assert_eq!(
            plan.starts,
            vec![0, 2, 3],
            "rows 0 and 1 together, row 2 apart"
        );
    }

    #[test]
    fn unreached_nodes_and_untaped_rows() {
        let ops: Vec<&[i64]> = vec![&[], &[0], &[0], &[1]];
        let (ptr, idx) = csr(&ops);
        let poly = vec![true, false, false, true];
        let leaf = vec![true, false, false, false];
        let plan = plan_chunks(&ptr, &idx, &poly, &leaf, &[-1, 1, -1], 3, 1);
        assert_eq!(plan.first_row[2], -1);
        assert_eq!(plan.last_row[2], -1);
        assert_eq!(plan.first_row[1], 1);
        assert_eq!(plan.last_row[1], 3);
    }
}
