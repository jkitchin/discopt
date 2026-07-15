"""R1.2 boundary faithfulness guard (issue #632).

The canonical atom model must faithfully cover the federation's claim boundary
before the univariate cutover can be byte-identity-safe: every raw-expression node
the federation claims via an expr-id owner family (univariate / composite) must
canonicalize to a genuine nonlinear atom node — never to an ``opaque`` node (which
the cutover would silently drop to the fallback) and never to an ``affine`` node (a
real disagreement about what is nonlinear). This is the standing regression form of
the R1.2 entry census (``scripts/r12_boundary_census.py``): 0 opaque, 0 affine
across the vendored corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from discopt._jax.canonical_expr import canonicalize, is_affine
from discopt._jax.claim_audit import audit_build
from discopt.modeling.core import from_nl

# slow: builds every corpus relaxation (some — e.g. st_e36 — take 100s+), so this
# runs in the serial claim-boundary CI job (generous timeout), not the parallel
# python-fast job (120s/test).
pytestmark = [pytest.mark.claim_boundary, pytest.mark.slow]

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
_CORPUS = sorted(p.stem for p in _NL_DIR.glob("*.nl"))


@pytest.mark.parametrize("name", _CORPUS)
def test_claimed_nodes_are_covered_nonlinear_atoms(name):
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    report = audit_build(model)
    dag = canonicalize(model)
    opaque_hits = []
    affine_hits = []
    for family, ids in report.claimed_expr_ids.items():
        for eid in ids:
            node = dag._memo.get(eid)
            if node is None:
                continue  # issue-267 distributed-node claim (R2.3 scope), not R1.2
            if node.is_opaque:
                opaque_hits.append((family, node.key))
            elif is_affine(node):
                affine_hits.append((family, node.key))
    assert not opaque_hits, (
        f"[{name}] federation-claimed nodes canonicalize to opaque: {opaque_hits}"
    )
    assert not affine_hits, (
        f"[{name}] federation-claimed nodes canonicalize to affine: {affine_hits}"
    )
