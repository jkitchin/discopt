//! Connected and strongly-connected components.
//!
//! [`connected_components`] answers the single cheapest structural question the
//! advisor asks: *is the model already block-diagonal?* Two variables land in
//! the same component iff a chain of shared constraints connects them, so the
//! component count is the number of independent blocks that could be solved in
//! parallel with no coordination at all.
//!
//! [`strongly_connected_components`] is its directed cousin, for the stage and
//! dual-dependency graphs where a cycle means two stages/duals are mutually
//! dependent and cannot be ordered — which forbids a naive nested (sequential)
//! decomposition and forces a simultaneous method instead.

use super::csr::CsrGraph;

/// Connected components of an undirected graph via union-find with path
/// halving and union by size.
///
/// Returns `(label, count)` where `label[v]` is the component id of vertex `v`
/// and `count` is the number of components. Labels are assigned so that a
/// component's id is the order in which its **smallest-indexed vertex** is first
/// encountered (ascending vertex scan). This makes the labeling deterministic
/// and matches the block-ordering convention of
/// `discopt.decomposition.structure` (blocks numbered in declared-variable
/// order). Complexity is `O(n + e·α(n))`.
pub fn connected_components(g: &CsrGraph) -> (Vec<u32>, usize) {
    let n = g.n();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size = vec![1u32; n];

    for v in 0..n as u32 {
        for &w in g.neighbors(v) {
            union(&mut parent, &mut size, v, w);
        }
    }

    // Relabel roots in ascending first-seen order for determinism.
    let mut label = vec![u32::MAX; n];
    let mut count = 0u32;
    for v in 0..n as u32 {
        let r = find(&mut parent, v);
        if label[r as usize] == u32::MAX {
            label[r as usize] = count;
            count += 1;
        }
        label[v as usize] = label[r as usize];
    }
    (label, count as usize)
}

/// Find with path halving.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        parent[i as usize] = parent[parent[i as usize] as usize];
        i = parent[i as usize];
    }
    i
}

/// Union by size.
fn union(parent: &mut [u32], size: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return;
    }
    let (big, small) = if size[ra as usize] >= size[rb as usize] {
        (ra, rb)
    } else {
        (rb, ra)
    };
    parent[small as usize] = big;
    size[big as usize] += size[small as usize];
}

/// Strongly-connected components of a **directed** graph via an iterative
/// formulation of Tarjan's algorithm.
///
/// Returns `(comp, count)` where `comp[v]` is the SCC id of vertex `v`. SCCs are
/// numbered in the order their roots are finalized, which is a reverse
/// topological order of the condensation (a source SCC gets a higher id than a
/// sink it reaches). The traversal is iterative, so arbitrarily deep graphs are
/// safe. Complexity is `O(n + e)`.
pub fn strongly_connected_components(g: &CsrGraph) -> (Vec<u32>, usize) {
    let n = g.n();
    let mut index = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut comp = vec![u32::MAX; n];
    let mut scc_stack: Vec<u32> = Vec::new();
    let mut idx = 0u32;
    let mut ncomp = 0u32;

    // Each work-stack frame is (vertex, next neighbor index to visit).
    for s in 0..n as u32 {
        if index[s as usize] != u32::MAX {
            continue;
        }
        let mut work: Vec<(u32, usize)> = vec![(s, 0)];
        while let Some(&(v, ci)) = work.last() {
            if ci == 0 {
                // First visit to v: assign its index and push it on the SCC stack.
                index[v as usize] = idx;
                low[v as usize] = idx;
                idx += 1;
                scc_stack.push(v);
                on_stack[v as usize] = true;
            }
            let nbrs = g.neighbors(v);
            if ci < nbrs.len() {
                work.last_mut().unwrap().1 += 1;
                let w = nbrs[ci];
                if index[w as usize] == u32::MAX {
                    work.push((w, 0));
                } else if on_stack[w as usize] && index[w as usize] < low[v as usize] {
                    low[v as usize] = index[w as usize];
                }
            } else {
                // Finished exploring v.
                if low[v as usize] == index[v as usize] {
                    // v is an SCC root: pop the stack down to and including v.
                    loop {
                        let w = scc_stack.pop().unwrap();
                        on_stack[w as usize] = false;
                        comp[w as usize] = ncomp;
                        if w == v {
                            break;
                        }
                    }
                    ncomp += 1;
                }
                work.pop();
                // Nested (not a let-chain): let-chains require edition 2024 and
                // this crate targets edition 2021 / MSRV 1.75.
                #[allow(clippy::collapsible_if)]
                if let Some(&(p, _)) = work.last() {
                    if low[v as usize] < low[p as usize] {
                        low[p as usize] = low[v as usize];
                    }
                }
            }
        }
    }
    (comp, ncomp as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_two_disjoint_blocks() {
        // {0,1,2} and {3,4}
        let g = CsrGraph::from_edges_undirected(5, &[(0, 1), (1, 2), (3, 4)]);
        let (label, count) = connected_components(&g);
        assert_eq!(count, 2);
        assert_eq!(label[0], label[1]);
        assert_eq!(label[1], label[2]);
        assert_eq!(label[3], label[4]);
        assert_ne!(label[0], label[3]);
        // deterministic ascending labeling: block containing vertex 0 is id 0.
        assert_eq!(label[0], 0);
        assert_eq!(label[3], 1);
    }

    #[test]
    fn cc_isolated_vertices_are_own_components() {
        let g = CsrGraph::from_edges_undirected(3, &[]);
        let (label, count) = connected_components(&g);
        assert_eq!(count, 3);
        assert_eq!(label, vec![0, 1, 2]);
    }

    #[test]
    fn cc_single_block() {
        let g = CsrGraph::from_edges_undirected(4, &[(0, 1), (1, 2), (2, 3)]);
        let (_, count) = connected_components(&g);
        assert_eq!(count, 1);
    }

    #[test]
    fn scc_single_cycle_is_one_component() {
        let g = CsrGraph::from_edges_directed(3, &[(0, 1), (1, 2), (2, 0)]);
        let (comp, count) = strongly_connected_components(&g);
        assert_eq!(count, 1);
        assert_eq!(comp, vec![0, 0, 0]);
    }

    #[test]
    fn scc_dag_each_vertex_own_component() {
        // 0 -> 1 -> 2, acyclic
        let g = CsrGraph::from_edges_directed(3, &[(0, 1), (1, 2)]);
        let (comp, count) = strongly_connected_components(&g);
        assert_eq!(count, 3);
        // distinct components
        assert_ne!(comp[0], comp[1]);
        assert_ne!(comp[1], comp[2]);
        assert_ne!(comp[0], comp[2]);
    }

    #[test]
    fn scc_two_cycles_joined_by_bridge() {
        // cycle {0,1} and cycle {2,3}, arc 1->2 linking them (one-way).
        let g = CsrGraph::from_edges_directed(4, &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);
        let (comp, count) = strongly_connected_components(&g);
        assert_eq!(count, 2);
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[2], comp[3]);
        assert_ne!(comp[0], comp[2]);
    }
}
