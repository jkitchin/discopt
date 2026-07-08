"""Regression tests for C-44 (#567): column-identity-safe cut-pool inheritance.

A root cut-pool row is stated by *column position* over the ROOT build's lifted
column layout. Per node the relaxation is re-built / re-lifted, which can produce
the SAME column count with DIFFERENT column semantics — a given position that was
``x_2·x_5·x_8`` at the root can become ``x_3·x_4`` at a node (measured on nvs22:
16–24 of 69 columns remap while the count is unchanged). The pre-C-44 gate
(``sparse_cols == n_total`` — a count check) then appended the pool row onto the
WRONG lifted variables → an invalid cut → a node holding the true optimum could
be falsely Farkas-fathomed (the C-43 nvs22 mechanism).

C-44 tags each pool row with the *identities* of the lifted columns it references
(``column_identities``) and, per node, remaps each row's coefficients from
root-column-identity → the node's current position for the SAME identity
(``_remap_pool_rows``); a row referencing a lifted term the node does not carry
is skipped (sound — fewer cuts only loosen). These tests pin the mechanism.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.mccormick_lp import _remap_pool_rows, column_identities


def test_column_identities_tags_orig_and_structural_aux():
    """The identity vector labels original columns ``("orig", k)`` (always
    stable) and structurally-keyed aux columns by their term key; unclaimed aux
    columns are ``("opaque", k)`` (position-locked, never remaps)."""
    # n_orig = 3 originals; cols 3,4 are aux (bilinear (0,1) and monomial (2,2));
    # col 5 is an unclaimed aux.
    varmap = {
        "original": {0: 0, 1: 1, 2: 2},
        "bilinear": {(0, 1): 3},
        "monomial": {(2, 2): 4},
    }
    idents = column_identities(varmap, n_total=6, n_orig=3)
    assert idents[0] == ("orig", 0)
    assert idents[1] == ("orig", 1)
    assert idents[2] == ("orig", 2)
    assert idents[3] == ("bilinear", (0, 1))
    assert idents[4] == ("monomial", (2, 2))
    assert idents[5] == ("opaque", 5)  # unclaimed aux → opaque, position-locked


def test_remap_moves_coefficients_to_the_matching_node_column():
    """A pool row stated over the ROOT layout must be remapped so each column's
    coefficient lands on the NODE position carrying the SAME identity — not the
    same index. Here the bilinear ``(0,1)`` aux sits at root col 3 but node col 4:
    the remapped row must put the aux coefficient at col 4, else it would
    constrain whatever the node lifted into col 3 (an invalid cut)."""
    # root layout: [orig0, orig1, orig2, bilinear(0,1)@3]
    root_idents = (("orig", 0), ("orig", 1), ("orig", 2), ("bilinear", (0, 1)))
    # node layout (re-lifted): the SAME bilinear now sits at col 4; col 3 is a
    # DIFFERENT lifted term (bilinear (1,2)).
    node_idents = (
        ("orig", 0),
        ("orig", 1),
        ("orig", 2),
        ("bilinear", (1, 2)),  # position 3 means something else now
        ("bilinear", (0, 1)),  # the identity the row references, remapped here
    )
    # pool row: 1.0*x0 - 2.0*aux(0,1) <= 5   (aux coeff at root position 3)
    a = np.array([[1.0, 0.0, 0.0, -2.0]])
    b = np.array([5.0])
    A_rm, b_rm, n_kept, n_skip = _remap_pool_rows(a, b, root_idents, node_idents, ncol=5)
    assert n_kept == 1 and n_skip == 0
    # The aux coefficient must move to node col 4 (its identity), NOT stay at col 3.
    assert A_rm[0, 3] == 0.0, "coefficient left on the WRONG (remapped) node column"
    assert A_rm[0, 4] == pytest.approx(-2.0)
    assert A_rm[0, 0] == pytest.approx(1.0)
    assert b_rm[0] == pytest.approx(5.0)


def test_remap_skips_a_row_whose_lifted_term_is_absent_at_the_node():
    """If a pool row references a lifted term the node does not carry, the row is
    inapplicable and must be SKIPPED (never appended over the wrong columns).
    Skipping an optional cut is always sound."""
    root_idents = (("orig", 0), ("orig", 1), ("bilinear", (0, 1)))
    # node lifted a DIFFERENT product; the (0,1) product is absent here.
    node_idents = (("orig", 0), ("orig", 1), ("bilinear", (0, 1) + (99,)))
    a = np.array([[1.0, 0.0, -2.0]])  # references bilinear(0,1) at col 2
    b = np.array([5.0])
    A_rm, b_rm, n_kept, n_skip = _remap_pool_rows(a, b, root_idents, node_idents, ncol=3)
    assert n_kept == 0 and n_skip == 1
    assert A_rm is None


def test_naive_positional_inheritance_would_cut_a_feasible_point_but_remap_does_not():
    """The soundness crux, concretely: a root pool row that is VALID over the root
    layout becomes an INVALID cut if appended by position onto a node whose columns
    remapped — it cuts a feasible lifted point. The identity remap must fix this.

    Root layout: col 2 = ``aux = x0*x1``. Row ``aux <= 6`` is valid when the box
    admits ``x0*x1 <= 6``. At the node, position 2 instead holds ``aux2 = x0*x0``
    (a re-lift), while ``x0*x1`` moved to col 3. The feasible node point
    ``x0=3, x1=1`` gives ``x0*x1 = 3 (<= 6, ok)`` but ``x0*x0 = 9 (> 6)``. So the
    NAIVE positional row ``z[2] <= 6`` cuts this feasible point; the REMAPPED row
    ``z[3] <= 6`` (the real ``x0*x1``) does not."""
    root_idents = (("orig", 0), ("orig", 1), ("bilinear", (0, 1)))
    node_idents = (
        ("orig", 0),
        ("orig", 1),
        ("monomial", (0, 2)),  # position 2 is now x0**2, NOT x0*x1
        ("bilinear", (0, 1)),  # x0*x1 moved to position 3
    )
    a = np.array([[0.0, 0.0, 1.0]])  # aux(0,1) <= 6, over the ROOT layout
    b = np.array([6.0])

    # The feasible lifted point at the node: x0=3, x1=1, x0**2=9, x0*x1=3.
    z = np.array([3.0, 1.0, 9.0, 3.0])

    # Naive positional inheritance (the pre-C-44 bug) would append the row as-is:
    naive_lhs = float(a[0] @ z[:3])  # uses z[2] = x0**2 = 9
    assert naive_lhs > b[0] + 1e-9, "the naive positional row must cut this feasible point"

    # C-44 remap: the row lands on the node's real x0*x1 column (3), value 3 <= 6.
    A_rm, b_rm, n_kept, n_skip = _remap_pool_rows(a, b, root_idents, node_idents, ncol=4)
    assert n_kept == 1 and n_skip == 0
    remapped_lhs = float(A_rm[0] @ z)
    assert remapped_lhs <= b_rm[0] + 1e-9, (
        "the REMAPPED row must NOT cut the feasible point (it now addresses x0*x1)"
    )
    assert A_rm[0, 2] == 0.0 and A_rm[0, 3] == pytest.approx(1.0)


def test_univariate_square_identity_resolves_through_its_base():
    """A ``univariate_square`` key is ``(base_col, 2)``; the identity must resolve
    the base column to ITS identity so a square of a lifted aux is stably tagged
    (a raw base index would be unstable across re-lifts)."""
    varmap = {
        "original": {0: 0, 1: 1},
        "bilinear": {(0, 1): 2},
        # square of the bilinear aux (col 2): key (2, 2), lifted to col 3.
        "univariate_square": {(2, 2): 3},
    }
    idents = column_identities(varmap, n_total=4, n_orig=2)
    # col 3's identity must nest the bilinear identity of its base col 2.
    assert idents[3][0] == "univariate_square"
    base_id, power = idents[3][1]
    assert base_id == ("bilinear", (0, 1))
    assert power == 2
