//! Articulation points and bridges — exact small-separator detection.
//!
//! An **articulation point** is a vertex whose removal increases the number of
//! connected components; a **bridge** is an edge with the same property. In an
//! optimization model these are exactly the cheapest structural cuts:
//!
//! - A bridge in the variable dependency graph is a single dependency whose
//!   removal splits the model — the linear-algebra analogue of a linking edge
//!   worth dualizing.
//! - An articulation variable is one whose fixing disconnects the model — a
//!   prime *complicating variable* for a Benders split (fix it in the master,
//!   the subproblems fall apart). This generalizes the current
//!   bridge-constraint heuristic in `discopt.decomposition.structure` from
//!   "single linking constraint" to "single linking entity of either kind".
//!
//! Both are found in one depth-first pass using discovery times and low-link
//! values (Tarjan). The traversal is iterative so deep graphs are safe, and it
//! assumes a **simple** graph — which [`super::csr::CsrGraph::from_edges_undirected`]
//! guarantees by de-duplicating edges. Complexity is `O(n + e)`.

use super::csr::CsrGraph;

/// Result of [`articulation_and_bridges`].
#[derive(Clone, Debug, Default)]
pub struct ArticulationResult {
    /// `is_articulation[v]` is true iff vertex `v` is a cut vertex.
    pub is_articulation: Vec<bool>,
    /// Cut edges, each as `(min, max)` endpoint pair, in discovery order.
    pub bridges: Vec<(u32, u32)>,
}

impl ArticulationResult {
    /// Vertices flagged as articulation points, ascending.
    pub fn articulation_points(&self) -> Vec<u32> {
        (0..self.is_articulation.len() as u32)
            .filter(|&v| self.is_articulation[v as usize])
            .collect()
    }
}

/// Compute all articulation points and bridges of an undirected simple graph in
/// a single iterative DFS.
///
/// The graph **must** be simple (no parallel edges); build it with
/// [`super::csr::CsrGraph::from_edges_undirected`], which de-duplicates. Passing
/// a multigraph would make the parent-edge skip under-count and can report false
/// bridges.
pub fn articulation_and_bridges(g: &CsrGraph) -> ArticulationResult {
    let n = g.n();
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut is_articulation = vec![false; n];
    let mut bridges: Vec<(u32, u32)> = Vec::new();
    let mut timer = 0u32;

    // Work-stack frame: (vertex, parent, next neighbor index). Parent is i64 so
    // the root can use -1.
    for s in 0..n as u32 {
        if disc[s as usize] != u32::MAX {
            continue;
        }
        disc[s as usize] = timer;
        low[s as usize] = timer;
        timer += 1;
        let mut stack: Vec<(u32, i64, usize)> = vec![(s, -1, 0)];
        let mut root_children = 0u32;

        while let Some(&(u, parent, ci)) = stack.last() {
            let nbrs = g.neighbors(u);
            if ci < nbrs.len() {
                stack.last_mut().unwrap().2 += 1;
                let w = nbrs[ci];
                if w as i64 == parent {
                    // Skip the single edge back to the parent (simple graph).
                    continue;
                }
                if disc[w as usize] == u32::MAX {
                    if parent == -1 {
                        root_children += 1;
                    }
                    disc[w as usize] = timer;
                    low[w as usize] = timer;
                    timer += 1;
                    stack.push((w, u as i64, 0));
                } else if disc[w as usize] < low[u as usize] {
                    // Back edge: pull u's low-link up to the ancestor.
                    low[u as usize] = disc[w as usize];
                }
            } else {
                // Finished u; propagate its low-link to its parent and test the
                // parent for articulation / the tree edge for bridge-ness.
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    if low[u as usize] < low[p as usize] {
                        low[p as usize] = low[u as usize];
                    }
                    // Non-root parent p is a cut vertex when a child cannot reach
                    // above p. (The root is handled via root_children below.)
                    if p != s && low[u as usize] >= disc[p as usize] {
                        is_articulation[p as usize] = true;
                    }
                    // The tree edge (p,u) is a bridge when u cannot reach p or above.
                    if low[u as usize] > disc[p as usize] {
                        let (lo, hi) = if p < u { (p, u) } else { (u, p) };
                        bridges.push((lo, hi));
                    }
                }
            }
        }
        // The DFS root is a cut vertex iff it has more than one DFS child.
        if root_children > 1 {
            is_articulation[s as usize] = true;
        }
    }

    ArticulationResult {
        is_articulation,
        bridges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_bridges(mut b: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
        b.sort_unstable();
        b
    }

    #[test]
    fn path_every_internal_vertex_and_edge_is_a_cut() {
        // 0 - 1 - 2 - 3 : vertices 1,2 are articulation; all edges are bridges.
        let g = CsrGraph::from_edges_undirected(4, &[(0, 1), (1, 2), (2, 3)]);
        let r = articulation_and_bridges(&g);
        assert_eq!(r.articulation_points(), vec![1, 2]);
        assert_eq!(sorted_bridges(r.bridges), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn cycle_has_no_cuts() {
        // triangle: 2-connected, so nothing is a cut.
        let g = CsrGraph::from_edges_undirected(3, &[(0, 1), (1, 2), (2, 0)]);
        let r = articulation_and_bridges(&g);
        assert!(r.articulation_points().is_empty());
        assert!(r.bridges.is_empty());
    }

    #[test]
    fn two_triangles_joined_by_a_bridge() {
        // triangle {0,1,2} - bridge (2,3) - triangle {3,4,5}
        let g = CsrGraph::from_edges_undirected(
            6,
            &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)],
        );
        let r = articulation_and_bridges(&g);
        // 2 and 3 are the articulation points; (2,3) is the only bridge.
        assert_eq!(r.articulation_points(), vec![2, 3]);
        assert_eq!(sorted_bridges(r.bridges), vec![(2, 3)]);
    }

    #[test]
    fn star_center_is_articulation_all_spokes_are_bridges() {
        // center 0 connected to 1,2,3
        let g = CsrGraph::from_edges_undirected(4, &[(0, 1), (0, 2), (0, 3)]);
        let r = articulation_and_bridges(&g);
        assert_eq!(r.articulation_points(), vec![0]);
        assert_eq!(sorted_bridges(r.bridges), vec![(0, 1), (0, 2), (0, 3)]);
    }

    #[test]
    fn disconnected_graph_handled_per_component() {
        // component A: edge 0-1 (bridge). component B: triangle 2,3,4 (no cut).
        let g = CsrGraph::from_edges_undirected(5, &[(0, 1), (2, 3), (3, 4), (4, 2)]);
        let r = articulation_and_bridges(&g);
        assert!(r.articulation_points().is_empty());
        assert_eq!(sorted_bridges(r.bridges), vec![(0, 1)]);
    }
}
