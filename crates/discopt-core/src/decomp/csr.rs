//! Compressed-sparse-row adjacency — the shared substrate for the graph kernels.
//!
//! An optimization model induces several graphs (variable–constraint bipartite,
//! variable dependency, directed stage / dual-dependency). They differ only in
//! *which edges* they contain, not in representation, so all of them are stored
//! as a [`CsrGraph`]. Building each other graph is then a transform of one
//! canonical incidence, and adjacency arrays can be shared behind an `Arc` by
//! the Python view layer.

/// An adjacency graph in compressed-sparse-row form.
///
/// Vertices are `0..n`. `indptr` has length `n + 1`; the neighbors of vertex
/// `v` occupy `indices[indptr[v] .. indptr[v + 1]]`.
///
/// The representation is direction-agnostic:
/// [`CsrGraph::from_edges_undirected`] stores each edge in *both* directions
/// (and de-duplicates, yielding a simple graph — required by
/// [`crate::decomp::articulation_and_bridges`]), whereas
/// [`CsrGraph::from_edges_directed`] stores each arc once. The same struct thus
/// backs both the undirected dependency graph and the directed stage / dual
/// graphs.
#[derive(Clone, Debug)]
pub struct CsrGraph {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
}

impl CsrGraph {
    /// Build an **undirected, simple** adjacency from an edge list.
    ///
    /// Each edge is canonicalized to `(min, max)`, de-duplicated, and then
    /// stored in both directions. Self-loops are dropped. De-duplication makes
    /// the articulation/bridge kernel correct (its parent-edge skip assumes a
    /// simple graph). Cost is `O(e log e)` for the sort; use this when a true
    /// graph topology is needed (dependency graph, bipartite incidence).
    pub fn from_edges_undirected(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut canon: Vec<(u32, u32)> = Vec::with_capacity(edges.len());
        for &(a, b) in edges {
            if a == b {
                continue;
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            canon.push((lo, hi));
        }
        canon.sort_unstable();
        canon.dedup();

        let mut deg = vec![0usize; n];
        for &(a, b) in &canon {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        let indptr = prefix_sum(&deg);
        let mut indices = vec![0u32; indptr[n]];
        let mut cursor = indptr[..n].to_vec();
        for &(a, b) in &canon {
            indices[cursor[a as usize]] = b;
            cursor[a as usize] += 1;
            indices[cursor[b as usize]] = a;
            cursor[b as usize] += 1;
        }
        Self { n, indptr, indices }
    }

    /// Build a **directed** adjacency from an arc list (`from -> to`, each arc
    /// stored once). Self-loops are dropped; parallel arcs are kept (harmless
    /// for [`crate::decomp::strongly_connected_components`]). Used for the stage
    /// and dual-dependency graphs where edge direction carries meaning.
    pub fn from_edges_directed(n: usize, arcs: &[(u32, u32)]) -> Self {
        let mut deg = vec![0usize; n];
        for &(a, b) in arcs {
            if a == b {
                continue;
            }
            deg[a as usize] += 1;
        }
        let indptr = prefix_sum(&deg);
        let mut indices = vec![0u32; indptr[n]];
        let mut cursor = indptr[..n].to_vec();
        for &(a, b) in arcs {
            if a == b {
                continue;
            }
            indices[cursor[a as usize]] = b;
            cursor[a as usize] += 1;
        }
        Self { n, indptr, indices }
    }

    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored adjacency entries. For an undirected graph built via
    /// [`CsrGraph::from_edges_undirected`] each edge counts twice; for a
    /// directed graph it is the number of arcs.
    pub fn num_adjacency_entries(&self) -> usize {
        self.indices.len()
    }

    /// Neighbors of `v` (out-neighbors, for a directed graph).
    pub fn neighbors(&self, v: u32) -> &[u32] {
        &self.indices[self.indptr[v as usize]..self.indptr[v as usize + 1]]
    }

    /// Degree of `v` (out-degree, for a directed graph).
    pub fn degree(&self, v: u32) -> usize {
        self.indptr[v as usize + 1] - self.indptr[v as usize]
    }
}

/// Exclusive prefix sum: returns a vector of length `deg.len() + 1` where entry
/// `i` is the sum of `deg[..i]`. The CSR row-pointer construction.
fn prefix_sum(deg: &[usize]) -> Vec<usize> {
    let mut p = vec![0usize; deg.len() + 1];
    for i in 0..deg.len() {
        p[i + 1] = p[i] + deg[i];
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = CsrGraph::from_edges_undirected(0, &[]);
        assert_eq!(g.n(), 0);
        assert_eq!(g.num_adjacency_entries(), 0);
    }

    #[test]
    fn undirected_is_symmetric_and_deduped() {
        // duplicate edge (0,1) twice + reversed (1,0) collapses to one edge.
        let g = CsrGraph::from_edges_undirected(3, &[(0, 1), (1, 0), (0, 1), (1, 2)]);
        assert_eq!(g.neighbors(0), &[1]);
        let mut n1 = g.neighbors(1).to_vec();
        n1.sort_unstable();
        assert_eq!(n1, vec![0, 2]);
        assert_eq!(g.neighbors(2), &[1]);
        // 2 undirected edges -> 4 directed entries.
        assert_eq!(g.num_adjacency_entries(), 4);
    }

    #[test]
    fn self_loops_dropped() {
        let g = CsrGraph::from_edges_undirected(2, &[(0, 0), (0, 1)]);
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.degree(0), 1);
    }

    #[test]
    fn directed_stores_arcs_once() {
        let g = CsrGraph::from_edges_directed(3, &[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[2]);
        assert_eq!(g.neighbors(2), &[0]);
        assert_eq!(g.num_adjacency_entries(), 3);
    }
}
